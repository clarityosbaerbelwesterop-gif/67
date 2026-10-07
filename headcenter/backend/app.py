"""HTTP + WebSocket API of the headcenter.

    GET  /                       iPad frontend (static; it authenticates its API calls)
    GET  /webgpu/                the WebGPU runtime: run a checkpoint on the iPad's GPU
    GET  /models/{token}/{run}/{file}
                                 model.safetensors, state.json (the run's FINAL, else
                                 newest checkpoint) and tokenizer.json; the token sits
                                 in the path because the runtime fetches a base URL
    GET  /api/health             liveness, no auth
    GET  /api/rsi                controlled-RSI rounds and pending approvals per model
    POST /api/rsi/{model}/approve  {"round": n, "approved": bool}  (only while pending)
    POST /api/rsi/{model}/stop   the RSI loop halts at its next check
    GET  /api/longrun            runs/longrun/state.json + history and runs/champions.json
                                 (the current champion of every family), as written
    POST /api/swarm/{group}/open, POST /api/swarm/join, GET /api/swarm/{group}/assignment,
    POST /api/swarm/{group}/report, GET /api/swarm, DELETE /api/swarm/{group}
                                 rendezvous for real machines joining forge diloco
Restricted runs (headcenter/acl.json, e.g. "rouge-"): checkpoint files are
served only to listed company tokens (hash compare), never to the operator
token alone; every attempt is logged to runs/headcenter/access.jsonl.
    GET  /api/results            measured results: report, Darus merge search, benchmarks
    GET  /api/runs               snapshot of every run, agent modes, recent decisions
    POST /api/runs/{run}/cmd     body: a trainer command, e.g. {"cmd":"pause"}
    POST /api/agents/{agent}     body: {"mode": "act" | "advise" | "off"}
    WS   /ws?token=...           snapshot, then every new record and decision live;
                                 accepts {"type":"cmd","run":..,"cmd":{..},"id":..}
                                 and {"type":"mode","agent":..,"mode":..,"id":..}

With a token configured every API call needs it (`Authorization: Bearer`, or
`?token=` where headers are impossible, i.e. WebSockets).
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import time
from collections import deque
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from .agents import MODES, Agent, Decision, JitOptimizer, SiliconWatchdog
from .access import Access
from .control import RUN_RE, CommandError, Controller, NotLive
from .rsi import RSI, RSIError, load_json, read_longrun
from .swarm import Swarm, SwarmError
from .telemetry import RunState, Tail, load_max_steps, parse

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "headcenter" / "frontend" / "index.html"
WEBGPU = {"": ("index.html", "text/html"), "forge-webgpu.mjs": ("forge-webgpu.mjs", "text/javascript")}
CHECKPOINT_FILES = ("model.safetensors", "state.json")


def checkpoint_dir(runs_dir: Path, run: str) -> Path | None:
    """The run's FINAL checkpoint, else its newest one; always inside runs_dir."""
    if not RUN_RE.match(run):
        return None
    root = runs_dir.resolve()
    cand = runs_dir / run / "FINAL"
    if not cand.exists():
        try:
            cand = Path(json.loads((runs_dir / run / "latest.json").read_text())["dir"])
        except (OSError, ValueError, KeyError, TypeError):
            return None
        if not cand.is_absolute():  # forge records it relative to its working directory
            cand = next((c for c in (runs_dir.parent / cand, runs_dir / cand) if c.exists()), cand)
    cand = cand.resolve()
    if root not in cand.parents or not (cand / "model.safetensors").is_file():
        return None
    return cand


# Files in runs/ that are results, not run logs.
NOT_RUNS = {"benchmarks"}


def read_results(runs_dir: Path) -> dict:
    """Everything the finish chain measured, as written (never recomputed here)."""

    def load(path: Path):
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return None

    search = load(runs_dir / "darus-1" / "search.json")
    if search:
        search = {k: search.get(k) for k in ("selection", "best_parent", "best")} | {
            "candidates": len(search.get("candidates") or [])
        }
    bench = []
    try:
        for line in (runs_dir / "benchmarks.jsonl").read_text().splitlines():
            rec = parse(line)
            if rec and "suite" in rec:
                bench.append(rec)
    except OSError:
        pass
    return {"report": load(runs_dir / "report.json"), "search": search, "benchmarks": bench}


def results_stamp(runs_dir: Path) -> tuple:
    out = []
    for p in (runs_dir / "report.json", runs_dir / "darus-1" / "search.json", runs_dir / "benchmarks.jsonl"):
        try:
            st = p.stat()
            out.append((st.st_mtime_ns, st.st_size))
        except OSError:
            out.append(None)
    return tuple(out)


class Hub:
    """Owns run state, agents, the control channel and connected clients."""

    def __init__(
        self,
        runs_dir: Path,
        configs_dir: Path | None = None,
        modes: dict[str, str] | None = None,
        poll: float = 0.5,
    ):
        self.runs_dir = Path(runs_dir)
        self.configs_dir = Path(configs_dir) if configs_dir else None
        self.poll = poll
        self.controller = Controller(self.runs_dir)
        modes = modes or {}
        self.agents: dict[str, Agent] = {
            a.name: a
            for a in (
                SiliconWatchdog(modes.get(SiliconWatchdog.name, "act")),
                JitOptimizer(modes.get(JitOptimizer.name, "act")),
            )
        }
        self.runs: dict[str, RunState] = {}
        self.tails: dict[str, Tail] = {}
        self.decisions: deque = deque(maxlen=500)
        self.clients: set[asyncio.Queue] = set()
        self.audit = self.runs_dir / "headcenter" / "decisions.jsonl"
        self.stamp: tuple | None = None

    # ---- ingestion -------------------------------------------------------

    def scan(self) -> list[dict]:
        """Read new lines of every run log; returns the messages to broadcast."""
        msgs: list[dict] = []
        for path in sorted(self.runs_dir.glob("*.jsonl")):
            run = path.stem
            if run in NOT_RUNS:
                continue
            first = run not in self.tails
            if first:
                self.tails[run] = Tail(path)
                self.runs[run] = RunState(run, load_max_steps(self.configs_dir, run))
            state = self.runs[run]
            for line in self.tails[run].read():
                rec = parse(line)
                if rec is None:
                    continue
                state.ingest(rec)
                if not first:
                    msgs.append({"type": "record", "run": run, "record": rec, "summary": self.brief(state)})
                for agent in self.agents.values():
                    if agent.mode != "off":
                        for d in agent.observe(state, rec, live=not first):
                            msgs.append(self.decide(agent, d))
            if first:
                msgs.append({"type": "run", "run": state.summary()})
        return msgs

    def tick(self, now: float | None = None) -> list[dict]:
        now = time.time() if now is None else now
        return [
            self.decide(agent, d)
            for agent in self.agents.values()
            if agent.mode != "off"
            for d in agent.tick(self.runs, now)
        ]

    def decide(self, agent: Agent, d: Decision) -> dict:
        d.mode = agent.mode
        if d.action is not None and agent.mode == "act":
            try:
                d.action = self.controller.send(d.run, d.action)
                d.executed = True
            except (CommandError, NotLive, OSError) as e:
                d.error = str(e)
        rec = d.to_json()
        self.decisions.append(rec)
        try:
            self.audit.parent.mkdir(parents=True, exist_ok=True)
            with self.audit.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass
        return rec

    @staticmethod
    def brief(state: RunState) -> dict:
        s = state.summary()
        for k in ("history", "evals", "events", "checkpoints", "autotune", "last", "history_columns"):
            s.pop(k)
        return s

    def snapshot(self) -> dict:
        return {
            "type": "snapshot",
            "runs": {name: st.summary() for name, st in self.runs.items()},
            "agents": {name: a.mode for name, a in self.agents.items()},
            "decisions": list(self.decisions)[-100:],
            "results": read_results(self.runs_dir),
        }

    # ---- control ---------------------------------------------------------

    def command(self, run: str, cmd: Any) -> dict:
        canon = self.controller.send(run, cmd)
        rec = {"type": "command", "run": run, "cmd": canon, "ts": time.time()}
        self.broadcast([rec])
        return canon

    def set_mode(self, agent: str, mode: str) -> None:
        if agent not in self.agents:
            raise CommandError(f"unknown agent {agent!r}")
        if mode not in MODES:
            raise CommandError(f"mode must be one of {MODES}")
        self.agents[agent].mode = mode
        self.broadcast([{"type": "agents", "agents": {n: a.mode for n, a in self.agents.items()}}])

    # ---- clients ---------------------------------------------------------

    def broadcast(self, msgs: list[dict]) -> None:
        for q in list(self.clients):
            for m in msgs:
                try:
                    q.put_nowait(m)
                except asyncio.QueueFull:  # a client that cannot keep up is dropped
                    self.clients.discard(q)
                    break

    async def run(self) -> None:
        last_tick = 0.0
        while True:
            msgs = self.scan()
            now = time.time()
            if now - last_tick >= 5.0:
                last_tick = now
                msgs += self.tick(now)
                stamp = results_stamp(self.runs_dir)
                if stamp != self.stamp:
                    if self.stamp is not None:
                        msgs.append({"type": "results", "results": read_results(self.runs_dir)})
                    self.stamp = stamp
            if msgs:
                self.broadcast(msgs)
            await asyncio.sleep(self.poll)


def create_app(
    runs_dir: Path,
    configs_dir: Path | None = None,
    token: str | None = None,
    modes: dict[str, str] | None = None,
    poll: float = 0.5,
    tokenizer: Path | None = None,
    acl: Path | None = None,
) -> FastAPI:
    hub = Hub(runs_dir, configs_dir, modes, poll)
    runs_path = Path(runs_dir)
    access = Access(Path(acl) if acl else ROOT / "headcenter" / "acl.json", runs_path / "headcenter" / "access.jsonl")
    rsi = RSI(runs_path)
    swarm = Swarm(runs_path, [ROOT / "training" / "configs", runs_path], ROOT)
    tokenizer = Path(tokenizer) if tokenizer else ROOT / "data" / "stores" / "base" / "tokenizer.json"

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        task = asyncio.create_task(hub.run())
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    app = FastAPI(title="forge headcenter", lifespan=lifespan)
    app.state.hub = hub

    def authorised(supplied: str | None) -> bool:
        return token is None or (supplied is not None and hmac.compare_digest(supplied, token))

    def check(request: Request) -> None:
        auth = request.headers.get("authorization", "")
        supplied = auth[7:] if auth.lower().startswith("bearer ") else request.query_params.get("token")
        if not authorised(supplied):
            raise HTTPException(401, "missing or wrong token")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(FRONTEND, media_type="text/html")

    @app.get("/webgpu/{name:path}")
    def webgpu(name: str) -> FileResponse:
        if name not in WEBGPU:
            raise HTTPException(404)
        file, media = WEBGPU[name]
        return FileResponse(ROOT / "webgpu" / file, media_type=media)

    @app.get("/models/{key}/{run}/{file}")
    def model_file(key: str, run: str, file: str) -> FileResponse:
        if file != "tokenizer.json":
            ck = checkpoint_dir(hub.runs_dir, run) if file in CHECKPOINT_FILES else None
            names = [run] + (list(ck.relative_to(hub.runs_dir.resolve()).parts) if ck is not None else [])
            prefixes = access.prefixes(*names)
            if prefixes or access.error is not None:
                ok, company, reason = access.authorise(key, prefixes)
                access.log(run=run, file=file, allowed=ok, company=company, reason=reason)
                if not ok:
                    raise HTTPException(403, "restricted model: this key is not licensed for it")
                if ck is None:
                    raise HTTPException(404)
                return FileResponse(ck / file)
        if token is not None and not hmac.compare_digest(key, token):
            raise HTTPException(401, "missing or wrong token")
        if file == "tokenizer.json":
            if not tokenizer.is_file():
                raise HTTPException(404)
            return FileResponse(tokenizer, media_type="application/json")
        ck = checkpoint_dir(hub.runs_dir, run) if file in CHECKPOINT_FILES else None
        if ck is None:
            raise HTTPException(404)
        return FileResponse(ck / file)

    @app.get("/api/results")
    def results(request: Request) -> dict:
        check(request)
        return read_results(hub.runs_dir)

    @app.get("/api/models")
    def models(request: Request) -> dict:
        check(request)
        out = {}
        for d in sorted(p for p in hub.runs_dir.iterdir() if p.is_dir()):
            ck = checkpoint_dir(hub.runs_dir, d.name)
            if ck is not None:
                out[d.name] = {"dir": str(ck.relative_to(hub.runs_dir.resolve())), "final": (d / "FINAL").exists(),
                               "restricted": access.restricted(d.name)}
        return {"models": out}

    def call(fn, *args):
        try:
            return fn(*args)
        except (RSIError, SwarmError) as e:
            raise HTTPException(getattr(e, "status", 400), str(e)) from None

    @app.get("/api/rsi")
    def rsi_overview(request: Request) -> dict:
        check(request)
        return rsi.overview()

    @app.post("/api/rsi/{model}/approve")
    def rsi_approve(model: str, request: Request, body: dict = Body(...)) -> dict:
        check(request)
        return call(rsi.decide, model, body.get("round"), body.get("approved"), body.get("candidate_sha256"))

    @app.post("/api/rsi/{model}/stop")
    def rsi_stop(model: str, request: Request) -> dict:
        check(request)
        return call(rsi.stop, model)

    @app.get("/api/longrun")
    def longrun(request: Request) -> dict:
        check(request)
        return {**read_longrun(hub.runs_dir), "champions": load_json(hub.runs_dir / "champions.json")}

    @app.get("/api/swarm")
    def swarm_overview(request: Request) -> dict:
        check(request)
        return swarm.overview()

    @app.post("/api/swarm/join")
    def swarm_join(request: Request, body: dict = Body(...)) -> dict:
        check(request)
        return call(swarm.join, body)

    @app.post("/api/swarm/{group}/open")
    def swarm_open(group: str, request: Request, body: dict = Body(...)) -> dict:
        check(request)
        return call(swarm.open, group, body)

    @app.get("/api/swarm/{group}/assignment")
    def swarm_assignment(group: str, worker_id: str, request: Request) -> JSONResponse:
        check(request)
        status, body = call(swarm.assignment, group, worker_id)
        return JSONResponse(body, status_code=status)

    @app.post("/api/swarm/{group}/report")
    def swarm_report(group: str, request: Request, body: dict = Body(...)) -> dict:
        check(request)
        return call(swarm.report, group, body)

    @app.delete("/api/swarm/{group}")
    def swarm_delete(group: str, request: Request) -> dict:
        check(request)
        call(swarm.delete, group)
        return {"ok": True}

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "runs": len(hub.runs)}

    @app.get("/api/runs")
    def runs(request: Request) -> dict:
        check(request)
        return hub.snapshot()

    @app.post("/api/runs/{run}/cmd")
    def command(run: str, request: Request, cmd: Any = Body(...)) -> JSONResponse:
        check(request)
        try:
            return JSONResponse({"ok": True, "cmd": hub.command(run, cmd)})
        except CommandError as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
        except NotLive as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=409)

    @app.post("/api/agents/{agent}")
    def mode(agent: str, request: Request, body: dict = Body(...)) -> JSONResponse:
        check(request)
        try:
            hub.set_mode(agent, body.get("mode"))
        except CommandError as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
        return JSONResponse({"ok": True})

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        if not authorised(socket.query_params.get("token")):
            await socket.close(code=4401)
            return
        await socket.accept()
        # One sender per socket; the snapshot is queued in the same step the
        # client is registered, so no record falls between the two.
        queue: asyncio.Queue = asyncio.Queue(maxsize=5000)
        queue.put_nowait(hub.snapshot())
        hub.clients.add(queue)

        async def pump() -> None:
            while True:
                await socket.send_json(await queue.get())

        sender = asyncio.create_task(pump())
        try:
            while True:
                msg = await socket.receive_json()
                try:
                    queue.put_nowait(handle(msg))
                except asyncio.QueueFull:
                    break
        except (WebSocketDisconnect, json.JSONDecodeError, RuntimeError):
            pass
        finally:
            hub.clients.discard(queue)
            sender.cancel()

    def handle(msg: Any) -> dict:
        mid = msg.get("id") if isinstance(msg, dict) else None
        try:
            if not isinstance(msg, dict):
                raise CommandError("message must be a JSON object")
            if msg.get("type") == "cmd":
                return {"type": "ack", "id": mid, "ok": True, "cmd": hub.command(msg.get("run"), msg.get("cmd"))}
            if msg.get("type") == "mode":
                hub.set_mode(msg.get("agent"), msg.get("mode"))
                return {"type": "ack", "id": mid, "ok": True}
            raise CommandError("unknown message type")
        except (CommandError, NotLive) as e:
            return {"type": "ack", "id": mid, "ok": False, "error": str(e)}

    return app
