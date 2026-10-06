"""Restricted model access: runs only selected companies may download.

The ACL is a JSON file (default `headcenter/acl.json`, gitignored; see
`headcenter/acl.example.json`):

    {"restricted": {"rouge-": ["sha256:<hex>", ...]},
     "companies":  {"sha256:<hex>": "Company name"}}

A run whose name (or any directory of its resolved checkpoint path inside
`runs/`) starts with a restricted prefix is served by `/models/{key}/{run}/{file}`
only when sha256(key) is listed for every matching prefix. The operator token
opens nothing here unless its hash is listed too. Only hashes are stored; the
comparison is constant-time over every listed hash. Without an ACL file the
built-in policy restricts `rouge-` to nobody. A malformed ACL file closes every
model download until it is fixed (fail closed).

Every decision on a restricted run (and every request that presents a known
company token) is appended to `runs/headcenter/access.jsonl`.

CLI (the token is printed once and never stored):

    python3 -m headcenter.backend.access add --prefix rouge- --company 'Name'
    python3 -m headcenter.backend.access list
    python3 -m headcenter.backend.access revoke --company 'Name' [--prefix rouge-]
    python3 -m headcenter.backend.access restrict --prefix darus-
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import secrets
import sys
import tempfile
import threading
import time
from pathlib import Path

PREFIX_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
DEFAULT_PATH = Path("headcenter/acl.json")
# Rouge is the secret model: with no ACL file it is restricted to nobody.
DEFAULT_POLICY = {"restricted": {"rouge-": []}, "companies": {}}


class ACLError(ValueError):
    """The ACL file is malformed."""


def token_hash(token: str) -> str:
    return "sha256:" + hashlib.sha256(token.encode()).hexdigest()


def validate(doc: object) -> dict:
    """Normalised ACL, or ACLError. Keys starting with '_' are comments."""
    if not isinstance(doc, dict):
        raise ACLError("ACL must be a JSON object")
    restricted = doc.get("restricted", {})
    companies = doc.get("companies", {})
    if not isinstance(restricted, dict) or not isinstance(companies, dict):
        raise ACLError('"restricted" and "companies" must be objects')
    out_r: dict[str, list[str]] = {}
    for prefix, hashes in restricted.items():
        if prefix.startswith("_"):
            continue
        if not PREFIX_RE.match(prefix):
            raise ACLError(f"bad run prefix {prefix!r}")
        if not isinstance(hashes, list) or not all(isinstance(h, str) and HASH_RE.match(h) for h in hashes):
            raise ACLError(f'restricted[{prefix!r}] must be a list of "sha256:<64 hex>"')
        out_r[prefix] = list(dict.fromkeys(hashes))
    out_c: dict[str, str] = {}
    for h, name in companies.items():
        if h.startswith("_"):
            continue
        if not HASH_RE.match(h) or not isinstance(name, str) or not name.strip() or len(name) > 200:
            raise ACLError(f"bad company entry {h!r}")
        out_c[h] = name.strip()
    return {"restricted": out_r, "companies": out_c}


def load_file(path: Path) -> dict:
    try:
        return validate(json.loads(path.read_text()))
    except json.JSONDecodeError as e:
        raise ACLError(f"{path}: {e}") from None


def save_file(path: Path, acl: dict) -> None:
    """Atomic write, owner-only permissions."""
    acl = validate(acl)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".acl-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(acl, f, indent=2, sort_keys=True)
            f.write("\n")
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class Access:
    """The live ACL (re-read when the file changes) plus the access audit log."""

    def __init__(self, path: Path | None, audit: Path | None):
        self.path = Path(path) if path else None
        self.audit = Path(audit) if audit else None
        self.lock = threading.Lock()
        self.stamp: tuple | None = ("init",)
        self.acl = validate(DEFAULT_POLICY)
        self.error: str | None = None
        self.refresh()

    def refresh(self) -> None:
        if self.path is None:
            return
        try:
            st = self.path.stat()
            stamp = (st.st_mtime_ns, st.st_size, st.st_ino)
        except FileNotFoundError:
            stamp = None
        with self.lock:
            if stamp == self.stamp:
                return
            self.stamp = stamp
            if stamp is None:
                self.acl, self.error = validate(DEFAULT_POLICY), None
                return
            try:
                self.acl, self.error = load_file(self.path), None
            except (OSError, ACLError) as e:
                self.error = str(e)  # fail closed until the file is fixed

    def prefixes(self, *names: str) -> list[str]:
        """Restricted prefixes matched by any of the names (case-insensitive)."""
        self.refresh()
        out = []
        for prefix in self.acl["restricted"]:
            p = prefix.casefold()
            if any(n.casefold().startswith(p) for n in names if n):
                out.append(prefix)
        return out

    def restricted(self, *names: str) -> bool:
        return self.error is not None or bool(self.prefixes(*names))

    def company(self, key: str) -> str | None:
        h = token_hash(key)
        name = None
        for listed, company in self.acl["companies"].items():
            if hmac.compare_digest(h, listed):
                name = company
        return name

    def authorise(self, key: str, prefixes: list[str]) -> tuple[bool, str | None, str]:
        """(allowed, company, reason): the key's hash must be listed for every prefix."""
        self.refresh()
        if self.error is not None:
            return False, None, "ACL unreadable (fail closed)"
        h = token_hash(key)
        allowed = bool(prefixes)
        for prefix in prefixes:
            hit = False
            for listed in self.acl["restricted"].get(prefix, []):
                hit |= hmac.compare_digest(h, listed)  # no early exit
            allowed &= hit
        company = self.company(key)
        if allowed:
            return True, company, "listed"
        return False, company, "not listed for " + ", ".join(prefixes) if prefixes else "no prefix"

    def log(self, **rec) -> None:
        if self.audit is None:
            return
        rec = {"ts": time.time(), **rec}
        try:
            self.audit.parent.mkdir(parents=True, exist_ok=True)
            with self.lock, self.audit.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass


# ---------------------------------------------------------------- CLI


def _load_or_default(path: Path) -> dict:
    if path.exists():
        return load_file(path)
    return validate(DEFAULT_POLICY)


def cmd_add(path: Path, prefix: str, company: str) -> str:
    if not PREFIX_RE.match(prefix):
        raise ACLError(f"bad prefix {prefix!r}")
    if not company.strip():
        raise ACLError("company name must not be empty")
    acl = _load_or_default(path)
    token = secrets.token_urlsafe(32)
    h = token_hash(token)
    acl["restricted"].setdefault(prefix, []).append(h)
    acl["companies"][h] = company.strip()
    save_file(path, acl)
    return token


def cmd_revoke(path: Path, company: str, prefix: str | None) -> int:
    acl = _load_or_default(path)
    hashes = {h for h, n in acl["companies"].items() if n == company.strip()}
    removed = 0
    for p, listed in acl["restricted"].items():
        if prefix is not None and p != prefix:
            continue
        keep = [h for h in listed if h not in hashes]
        removed += len(listed) - len(keep)
        acl["restricted"][p] = keep
    still_used = {h for listed in acl["restricted"].values() for h in listed}
    for h in hashes - still_used:
        del acl["companies"][h]
    save_file(path, acl)
    return removed


def cmd_restrict(path: Path, prefix: str) -> None:
    if not PREFIX_RE.match(prefix):
        raise ACLError(f"bad prefix {prefix!r}")
    acl = _load_or_default(path)
    acl["restricted"].setdefault(prefix, [])
    save_file(path, acl)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m headcenter.backend.access", description=__doc__.split("\n\n")[0])
    ap.add_argument("--acl", type=Path, default=DEFAULT_PATH, help="ACL file (default headcenter/acl.json)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add", help="issue a new company token for a restricted prefix (printed once)")
    a.add_argument("--prefix", required=True)
    a.add_argument("--company", required=True)
    a.add_argument("--acl", type=Path, default=argparse.SUPPRESS)
    r = sub.add_parser("revoke", help="remove every token of a company (optionally only for one prefix)")
    r.add_argument("--company", required=True)
    r.add_argument("--prefix")
    r.add_argument("--acl", type=Path, default=argparse.SUPPRESS)
    s = sub.add_parser("restrict", help="restrict a prefix (to nobody until tokens are added)")
    s.add_argument("--prefix", required=True)
    s.add_argument("--acl", type=Path, default=argparse.SUPPRESS)
    ls = sub.add_parser("list", help="show prefixes and companies (hashes abbreviated)")
    ls.add_argument("--acl", type=Path, default=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "add":
            token = cmd_add(args.acl, args.prefix, args.company)
            print(f"company: {args.company.strip()}\nprefix:  {args.prefix}\ntoken:   {token}")
            print(f"download: /models/{token}/<run>/model.safetensors  (shown once; only its sha256 is stored in {args.acl})")
        elif args.cmd == "revoke":
            n = cmd_revoke(args.acl, args.company, args.prefix)
            print(f"revoked {n} grant(s) of {args.company!r}")
        elif args.cmd == "restrict":
            cmd_restrict(args.acl, args.prefix)
            print(f"{args.prefix} is restricted")
        else:
            acl = _load_or_default(args.acl)
            src = str(args.acl) if args.acl.exists() else "built-in default (no ACL file)"
            print(f"ACL: {src}")
            for prefix, listed in sorted(acl["restricted"].items()):
                names = [f"{acl['companies'].get(h, '?')} ({h[7:19]}…)" for h in listed]
                print(f"  {prefix}*  ->  {', '.join(names) if names else 'nobody'}")
    except (ACLError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
