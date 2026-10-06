"""python -m headcenter.backend --runs runs --configs training/configs [--host H --port P]

Binding to anything but loopback requires a token (HEADCENTER_TOKEN or
--token): the API can pause, stop and retune training runs.
"""

from __future__ import annotations

import argparse
import ipaddress
import os
import secrets
from pathlib import Path

import uvicorn

from .agents import MODES
from .app import create_app


def main() -> None:
    p = argparse.ArgumentParser(prog="headcenter")
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--configs", type=Path, default=Path("training/configs"))
    p.add_argument("--tokenizer", type=Path, default=Path("data/stores/base/tokenizer.json"))
    p.add_argument("--acl", type=Path, default=Path("headcenter/acl.json"), help="restricted-model ACL (see headcenter/acl.example.json)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8067)
    p.add_argument("--token", default=os.environ.get("HEADCENTER_TOKEN"))
    p.add_argument("--new-token", action="store_true", help="generate and print a random token")
    p.add_argument("--watchdog", choices=MODES, default="act")
    p.add_argument("--jit", choices=MODES, default="act")
    a = p.parse_args()
    token = secrets.token_urlsafe(24) if a.new_token else a.token
    try:
        loopback = ipaddress.ip_address(a.host).is_loopback
    except ValueError:
        loopback = a.host == "localhost"
    if not loopback and not token:
        p.error("binding beyond loopback needs --token, HEADCENTER_TOKEN or --new-token")
    if token and a.new_token:
        print(f"headcenter: open http://{a.host}:{a.port}/#token={token}", flush=True)
    modes = {"silicon-watchdog": a.watchdog, "jit-optimizer": a.jit}
    app = create_app(a.runs, a.configs, token, modes, tokenizer=a.tokenizer, acl=a.acl)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
