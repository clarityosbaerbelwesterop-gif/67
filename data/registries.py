"""Package-registry sources for corpus v3 (data/build_corpus_v3.py).

Every source downloads one package's published archive through the registry's
official API or index, reads the licence from the registry's own metadata and
returns (entry, files): `entry` is the provenance record that lands in
data/manifests/v3/sources.json (version, licence and where it was read, archive
URL + sha256, authors/homepage for attribution, status) and `files` the
(path, kind, text) of the package's useful text files.

  registry  how the popular packages are chosen                     licence read from
  pypi      hugovk/top-pypi-packages (30-day PyPI downloads)         JSON API license / classifiers
  npm       registry search API, popularity-weighted, many keywords  package.json "license"
  crates    crates.io API /api/v1/crates?sort=downloads              Cargo.toml inside the .crate
  go        curated list of widely imported modules (no index API)   root LICENSE text, classified
  maven     curated list of widely used artifacts (Central's search  POM <licenses>, following <parent>
            API is unreachable here); version from maven-metadata.xml
  rubygems  /api/v1/search.json over keywords, ranked by "downloads" API "licenses"
  hackage   /packages/top (JSON) ranked by recent downloads          .cabal "license" field

Requests are sequential, send USER_AGENT, keep a per-host minimum interval
(crates.io asks for <= 1 request/s) and back off on 429/503 (Retry-After).
Nothing is scraped from HTML pages. Licences follow the v1/v2 allow-list; the
deny-list additionally catches spelled-out and versioned copyleft names
("GNU General Public License", "GPLv3", "EPL-2.0", ...), so it only gets stricter.
"""
from __future__ import annotations

import hashlib
import html.parser
import importlib.util
import io
import json
import os
import re
import tarfile
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("build_corpus_v2", ROOT / "data" / "build_corpus_v2.py")
v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v2)

USER_AGENT = "forge-corpus-v3/1.0 (licensed training-corpus builder; sequential, rate-limited)" + (
    f" contact: {os.environ['FORGE_CORPUS_CONTACT']}" if os.environ.get("FORGE_CORPUS_CONTACT") else "")
MIN_INTERVAL = {"crates.io": 1.0, "rubygems.org": 0.2, "hackage.haskell.org": 0.3, "repo1.maven.org": 0.25}
DEFAULT_INTERVAL = 0.05
MAX_DOWNLOAD = 100_000_000
_last: dict[str, float] = {}


def log(msg: str) -> None:
    print(f"[corpus-v3 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch(url: str, timeout: int = 90, accept: str | None = None, max_bytes: int = MAX_DOWNLOAD) -> bytes | None:
    """GET `url` politely; None when missing, refused, oversized or still failing after retries."""
    if not url or not url.startswith("https://"):  # archive URLs come from registry metadata: never file:// etc.
        return None
    host = urllib.parse.urlsplit(url).hostname or ""
    headers = {"User-Agent": USER_AGENT} | ({"Accept": accept} if accept else {})
    for attempt in range(5):
        wait = _last.get(host, 0.0) + MIN_INTERVAL.get(host, DEFAULT_INTERVAL) - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last[host] = time.monotonic()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:  # noqa: S310 - registry hosts
                if int(r.headers.get("Content-Length") or 0) > max_bytes:
                    return None
                data = r.read(max_bytes + 1)
                return data if len(data) <= max_bytes else None
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):
                retry = e.headers.get("Retry-After", "") if e.headers else ""
                time.sleep(min(120, int(retry) if retry.isdigit() else 5 * 2 ** attempt))
                continue
            if e.code in (400, 401, 403, 404, 410, 451):
                return None
        except Exception:  # noqa: BLE001 - network errors are retried, then reported as missing
            pass
        time.sleep(min(60, 2 ** attempt))
    return None


def reachable(url: str, timeout: int = 12) -> bool:
    """True when a HEAD request to `url` succeeds (2xx/3xx)."""
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
            return r.status < 400
    except Exception:  # noqa: BLE001 - blocked, refused or timed out
        return False


# ---------------------------------------------------------------- licences

COPYLEFT = re.compile(r"\b(?:A|L)?GPL-?v?\d|General Public License|Mozilla Public|Eclipse Public|\bEPL\b|\bEPL-|\bCDDL|"
                      r"Common Development and Distribution|European Union Public|Server Side Public|Share-?Alike|\bMPL-?v?\d", re.I)
CABAL_LICENCES = {"bsd2": "BSD-2-Clause", "bsd3": "BSD-3-Clause", "bsd4": "BSD-4-Clause", "publicdomain": "Public Domain"}


def licence_ok(lic: str | None) -> bool:
    """v2's allow-list (and deny-list), plus a stricter deny on spelled-out / versioned copyleft names."""
    return v2.licence_ok(lic) and not COPYLEFT.search(lic)


def cabal_fields(text: str) -> dict[str, str]:
    """Top-level `field: value` pairs of a .cabal file (lower-cased names, first occurrence wins)."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = re.match(r"([A-Za-z][\w-]*)\s*:\s*(.*?)\s*$", line)
        if m and m.group(1).lower() not in out:
            out[m.group(1).lower()] = m.group(2)
    return out


def cabal_licence(text: str) -> str | None:
    """Licence of a .cabal file, legacy names (BSD3, PublicDomain) mapped to SPDX-like ones."""
    lic = cabal_fields(text).get("license", "").strip()
    return CABAL_LICENCES.get(lic.lower(), lic) or None


def _xml(data: bytes) -> ET.Element | None:
    """Parse registry XML without DTDs/entities, namespaces stripped."""
    if b"<!DOCTYPE" in data or b"<!ENTITY" in data:
        return None
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return None
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


def _text(el: ET.Element | None, path: str) -> str | None:
    x = el.find(path) if el is not None else None
    return x.text.strip() if x is not None and x.text and x.text.strip() else None


def pom_info(data: bytes) -> dict | None:
    """Licences, parent coordinates and attribution fields of a POM."""
    root = _xml(data)
    if root is None:
        return None
    lics = [_text(x, "name") or _text(x, "url") for x in root.findall("licenses/license")]
    parent = root.find("parent")
    devs = [_text(x, "name") or _text(x, "id") for x in root.findall("developers/developer")]
    return {"licenses": [x for x in lics if x], "name": _text(root, "name"), "url": _text(root, "url") or _text(root, "scm/url"),
            "authors": ", ".join(x for x in devs if x)[:200] or _text(root, "organization/name"),
            "parent": (_text(parent, "groupId"), _text(parent, "artifactId"), _text(parent, "version")) if parent is not None else None}


# ---------------------------------------------------------------- archives

CODE_EXT = v2.CODE_EXT + (".java", ".kt", ".scala", ".rb", ".rake", ".hs", ".lhs")
DOC_EXT = v2.DOC_EXT + (".rdoc", ".markdown", ".adoc")
JAVADOC_HTML = ("package.html", "overview.html")
SKIP = re.compile(v2.SKIP.pattern + r"|(^|/)meta-inf/")
NOT_PROSE = re.compile(r"^(licen[cs]e|copying|notice|changelog|changes|history|news|release[-_ ]?notes)", re.I)


def kind_of(path: str) -> str | None:
    """v2's file selection, extended to Java/Kotlin/Scala, Ruby and Haskell sources and their prose docs."""
    low = path.lower()
    if SKIP.search(low):
        return None
    base = low.rsplit("/", 1)[-1]
    if low.endswith(CODE_EXT):
        return "code"
    # Gem and jar paths are root-relative, so "/test" is also matched at the start.
    if (low.endswith(DOC_EXT) or base in JAVADOC_HTML) and "/test" not in "/" + low and not NOT_PROSE.match(base):
        return "general"
    return None


class _HTMLText(html.parser.HTMLParser):
    BLOCK = {"p", "div", "br", "li", "ul", "ol", "pre", "h1", "h2", "h3", "h4", "h5", "h6", "table", "tr", "dt", "dd", "blockquote"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):  # noqa: ANN001
        self.skip += tag in ("script", "style")
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):  # noqa: ANN001
        if tag in ("script", "style") and self.skip:
            self.skip -= 1
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):  # noqa: ANN001
        if not self.skip:
            self.parts.append(data)


def html_text(src: str) -> str:
    """Plain text of a javadoc package.html / overview.html."""
    p = _HTMLText()
    p.feed(src)
    lines = (re.sub(r"[ \t\r\f\v]+", " ", x).strip() for x in "".join(p.parts).splitlines())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def extract(members) -> list[tuple[str, str, str]]:
    """(path, kind, text) of the useful text files of an archive, size-capped like v2."""
    out, total = [], 0
    for name, size, read in members:
        kind = kind_of(name)
        if kind is None or size > v2.MAX_FILE:
            continue
        try:
            text = read().decode("utf-8")
        except (UnicodeDecodeError, OSError, KeyError, EOFError, tarfile.TarError, zipfile.BadZipFile):
            continue
        if name.lower().rsplit("/", 1)[-1] in JAVADOC_HTML:
            text = html_text(text)
        total += len(text)
        if total > v2.MAX_PACKAGE:
            break
        out.append((name, kind, text))
    return out


def gem_members(data: bytes):
    """Files of a .gem: a plain tar whose data.tar.gz holds the package."""
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tf:
            inner = tf.extractfile("data.tar.gz").read()
    except (tarfile.TarError, KeyError, AttributeError, EOFError, OSError):
        return iter(())
    return v2.tar_members(inner)


def _entry(registry: str, name: str) -> dict:
    return {"source": f"{registry}:{name}", "registry": registry, "name": name}


def _finish(entry: dict, data: bytes, url: str) -> dict:
    entry.update(url=url, sha256=hashlib.sha256(data).hexdigest())
    who = entry.get("authors") or f"the {entry['name']} authors"
    entry["attribution"] = f"{entry['name']} {entry.get('version') or ''} by {who}, {entry.get('license')}; {entry.get('homepage') or url}"[:400]
    return entry


def _rejected(entry: dict) -> tuple[dict, list]:
    return entry | {"status": "rejected-licence"}, []


# ---------------------------------------------------------------- sources


def pypi(name: str) -> tuple[dict, list]:
    entry = _entry("pypi", name)
    meta = fetch(f"https://pypi.org/pypi/{urllib.parse.quote(name)}/json")
    if not meta:
        return entry | {"status": "unavailable"}, []
    m = json.loads(meta)
    info = m.get("info") or {}
    lic = " | ".join(x for x in [info.get("license_expression") or "", (info.get("license") or "")[:300]] +
                     [c for c in info.get("classifiers") or [] if c.startswith("License ::")] if x and x.upper() != "UNKNOWN")
    urls = info.get("project_urls") or {}
    entry.update(version=info.get("version"), license=lic[:300] or None, license_from="pypi json api",
                 authors=(info.get("author") or info.get("maintainer") or "")[:200] or None,
                 homepage=urls.get("Source") or urls.get("Homepage") or info.get("home_page") or info.get("package_url"))
    if not licence_ok(lic):
        return _rejected(entry)
    sdist = next((u for u in m.get("urls") or [] if u.get("packagetype") == "sdist"), None)
    if not sdist or sdist.get("size", 0) > 60_000_000:
        return entry | {"status": "no-sdist"}, []
    data = fetch(sdist["url"])
    if not data:
        return entry | {"status": "unavailable"}, []
    return _finish(entry, data, sdist["url"]), extract(v2.zip_members(data) if sdist["url"].endswith(".zip") else v2.tar_members(data))


def npm(name: str) -> tuple[dict, list]:
    entry = _entry("npm", name)
    meta = fetch(f"https://registry.npmjs.org/{name.replace('/', '%2F')}/latest")
    if not meta:
        return entry | {"status": "unavailable"}, []
    m = json.loads(meta)
    lic = m.get("license") if isinstance(m.get("license"), str) else (m.get("license") or {}).get("type")
    author, repo = m.get("author"), m.get("repository")
    entry.update(version=m.get("version"), license=lic, license_from="package.json",
                 authors=(author.get("name") if isinstance(author, dict) else author if isinstance(author, str) else None),
                 homepage=m.get("homepage") or (repo.get("url") if isinstance(repo, dict) else repo))
    if not licence_ok(lic):
        return _rejected(entry)
    url = (m.get("dist") or {}).get("tarball", "")
    data = fetch(url)
    if not data:
        return entry | {"status": "unavailable"}, []
    return _finish(entry, data, url), extract(v2.tar_members(data))


def crates(name: str) -> tuple[dict, list]:
    entry = _entry("crates", name)
    idx = fetch(f"https://index.crates.io/{v2.crate_index_path(name)}")
    if not idx:
        return entry | {"status": "unavailable"}, []
    lines = [json.loads(x) for x in idx.decode().splitlines() if x.strip()]
    rel = [x for x in lines if not x.get("yanked") and "-" not in x["vers"]] or lines
    vers = rel[-1]["vers"]
    url = f"https://static.crates.io/crates/{name}/{name}-{vers}.crate"
    entry["version"] = vers
    data = fetch(url)
    if not data:
        return entry | {"status": "unavailable"}, []
    pkg = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            cargo = tf.extractfile(f"{name}-{vers}/Cargo.toml")
            pkg = (tomllib.loads(cargo.read().decode()).get("package") or {}) if cargo else {}
    except (tarfile.TarError, KeyError, tomllib.TOMLDecodeError, OSError, UnicodeDecodeError, EOFError):
        pass
    authors = pkg.get("authors") if isinstance(pkg.get("authors"), list) else []
    lic = pkg.get("license") if isinstance(pkg.get("license"), str) else None
    entry.update(license=lic, license_from="Cargo.toml", authors=", ".join(map(str, authors))[:200] or None,
                 homepage=pkg.get("repository") if isinstance(pkg.get("repository"), str) else f"https://crates.io/crates/{name}")
    _finish(entry, data, url)
    if not licence_ok(lic):
        return _rejected(entry)
    return entry, extract(v2.tar_members(data))


def go(module: str) -> tuple[dict, list]:
    entry = _entry("go", module)
    esc = re.sub(r"[A-Z]", lambda m: "!" + m.group(0).lower(), module)
    latest = fetch(f"https://proxy.golang.org/{esc}/@latest")
    if not latest:
        return entry | {"status": "unavailable"}, []
    vers = json.loads(latest)["Version"]
    url = f"https://proxy.golang.org/{esc}/@v/{vers}.zip"
    entry.update(version=vers, homepage=f"https://pkg.go.dev/{module}")
    data = fetch(url)
    if not data:
        return entry | {"status": "unavailable"}, []
    lic = None
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            root = f"{module}@{vers}/"
            lf = next((n for n in zf.namelist() if n.startswith(root) and "/" not in n[len(root):]
                       and n[len(root):].upper().startswith(("LICENSE", "LICENCE", "COPYING"))), None)
            lic = v2.classify_licence_text(zf.read(lf).decode("utf-8", "replace")) if lf else None
    except (zipfile.BadZipFile, OSError):
        pass
    entry.update(license=lic, license_from="LICENSE file (classified)")
    _finish(entry, data, url)
    if not licence_ok(lic):
        return _rejected(entry)
    return entry, extract(v2.zip_members(data))


MAVEN = "https://repo1.maven.org/maven2"
PRERELEASE = re.compile(r"snapshot|alpha|beta|[.-]rc|rc\d*$|[.-]m\d+$|milestone|preview|[.-]ea$|[.-]dev|incubat|[.-]cr\d", re.I)


def maven_version(metadata: bytes) -> str | None:
    """Newest stable version listed in maven-metadata.xml (<release> when it is stable)."""
    root = _xml(metadata)
    if root is None:
        return None
    rel = _text(root, "versioning/release")
    if rel and not PRERELEASE.search(rel):
        return rel
    stable = [x.text.strip() for x in root.findall("versioning/versions/version") if x.text and not PRERELEASE.search(x.text)]
    return stable[-1] if stable else None


def maven_pom_url(g: str, a: str, v: str) -> str:
    return f"{MAVEN}/{g.replace('.', '/')}/{a}/{v}/{a}-{v}.pom"


def maven_licence(g: str, a: str, v: str, depth: int = 0) -> tuple[str | None, dict]:
    """Licence of an artifact from its POM <licenses>, else from its <parent> chain (up to 4 levels)."""
    pom = fetch(maven_pom_url(g, a, v))
    info = pom_info(pom) if pom else None
    if info is None:
        return None, {}
    if info["licenses"]:
        # Several <license> entries in a POM are alternatives (Maven POM reference).
        return " OR ".join(info["licenses"])[:300], info
    p = info["parent"]
    if depth < 4 and p and all(p) and "${" not in "".join(p):
        lic, pinfo = maven_licence(*p, depth=depth + 1)
        return lic, {k: info.get(k) or pinfo.get(k) for k in ("name", "url", "authors")} | {"parent": p}
    return None, info


def maven(coord: str) -> tuple[dict, list]:
    entry = _entry("maven", coord)
    g, a = coord.split(":", 1)
    base = f"{MAVEN}/{g.replace('.', '/')}/{a}"
    meta = fetch(f"{base}/maven-metadata.xml")
    vers = maven_version(meta) if meta else None
    if not vers:
        return entry | {"status": "unavailable"}, []
    lic, info = maven_licence(g, a, vers)
    entry.update(version=vers, license=lic, license_from="pom.xml <licenses>" + (" (parent)" if info.get("parent") else ""),
                 authors=info.get("authors"), homepage=info.get("url") or f"https://central.sonatype.com/artifact/{g}/{a}")
    if not licence_ok(lic):
        return _rejected(entry)
    url = f"{base}/{vers}/{a}-{vers}-sources.jar"
    data = fetch(url, max_bytes=60_000_000)
    if not data:
        return entry | {"status": "no-sources"}, []
    return _finish(entry, data, url), extract(v2.zip_members(data))


def rubygems(name: str) -> tuple[dict, list]:
    entry = _entry("rubygems", name)
    meta = fetch(f"https://rubygems.org/api/v1/gems/{urllib.parse.quote(name)}.json")
    if not meta:
        return entry | {"status": "unavailable"}, []
    m = json.loads(meta)
    lic = " OR ".join(x for x in m.get("licenses") or [] if isinstance(x, str)) or None
    entry.update(version=m.get("version"), license=lic, license_from="rubygems api licenses",
                 authors=(m.get("authors") or "")[:200] or None, homepage=m.get("source_code_uri") or m.get("homepage_uri") or m.get("project_uri"))
    if not licence_ok(lic):
        return _rejected(entry)
    url = m.get("gem_uri") or f"https://rubygems.org/gems/{name}-{m.get('version')}.gem"
    data = fetch(url)
    if not data:
        return entry | {"status": "unavailable"}, []
    if m.get("sha") and hashlib.sha256(data).hexdigest() != m["sha"]:
        return entry | {"status": "checksum-mismatch"}, []
    return _finish(entry, data, url), extract(gem_members(data))


def _vkey(v: str) -> tuple:
    return tuple(int(x) if x.isdigit() else 0 for x in v.split("."))


def hackage(name: str) -> tuple[dict, list]:
    entry = _entry("hackage", name)
    pref = fetch(f"https://hackage.haskell.org/package/{name}/preferred", accept="application/json")
    try:
        versions = json.loads(pref).get("normal-version") or [] if pref else []
    except ValueError:
        versions = []
    if not versions:
        return entry | {"status": "unavailable"}, []
    vers = max(versions, key=_vkey)
    url = f"https://hackage.haskell.org/package/{name}-{vers}/{name}-{vers}.tar.gz"
    entry.update(version=vers, homepage=f"https://hackage.haskell.org/package/{name}")
    data = fetch(url)
    if not data:
        return entry | {"status": "unavailable"}, []
    fields: dict[str, str] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            cab = next((m for m in tf.getmembers() if m.isfile() and m.name.count("/") == 1 and m.name.endswith(".cabal")), None)
            text = tf.extractfile(cab).read().decode("utf-8", "replace") if cab else ""
            fields = cabal_fields(text)
    except (tarfile.TarError, OSError, EOFError, AttributeError):
        text = ""
    lic = cabal_licence(text) if text else None
    entry.update(license=lic, license_from=".cabal license field", authors=(fields.get("author") or fields.get("maintainer") or "")[:200] or None,
                 homepage=fields.get("homepage") or entry["homepage"])
    _finish(entry, data, url)
    if not licence_ok(lic):
        return _rejected(entry)
    return entry, extract(v2.tar_members(data))


SOURCES = {"pypi": pypi, "npm": npm, "crates": crates, "go": go, "maven": maven, "rubygems": rubygems, "hackage": hackage}

# ---------------------------------------------------------------- package lists

GO = """github.com/stretchr/objx github.com/davecgh/go-spew github.com/pmezard/go-difflib github.com/kr/pretty github.com/kr/text
github.com/google/btree github.com/google/gofuzz github.com/google/go-querystring github.com/google/shlex github.com/google/pprof
github.com/google/wire github.com/google/gopacket github.com/google/go-containerregistry github.com/google/s2a-go
github.com/googleapis/gax-go/v2 github.com/golang/mock github.com/golang/groupcache github.com/golang/glog go.uber.org/mock
go.uber.org/fx go.uber.org/dig go.uber.org/goleak go.uber.org/ratelimit github.com/modern-go/reflect2 github.com/modern-go/concurrent
github.com/gogo/protobuf github.com/grpc-ecosystem/grpc-gateway/v2 github.com/grpc-ecosystem/go-grpc-middleware/v2
github.com/bufbuild/protocompile github.com/emicklei/go-restful/v3 github.com/go-openapi/swag github.com/go-openapi/jsonpointer
github.com/go-logr/logr github.com/go-logr/zapr github.com/go-logr/stdr github.com/munnerz/goautoneg github.com/josharian/intern
github.com/mailru/easyjson github.com/pelletier/go-toml/v2 github.com/magiconair/properties github.com/spf13/afero github.com/spf13/cast
github.com/subosito/gotenv github.com/fxamacker/cbor/v2 github.com/x448/float16 github.com/ugorji/go/codec github.com/bytedance/sonic
github.com/goccy/go-json github.com/goccy/go-yaml github.com/leodido/go-urn github.com/gabriel-vasile/mimetype
github.com/go-playground/locales github.com/go-playground/universal-translator github.com/klauspost/cpuid/v2
github.com/andybalholm/brotli github.com/valyala/bytebufferpool github.com/valyala/fasttemplate github.com/labstack/gommon
github.com/mattn/go-shellwords github.com/atotto/clipboard github.com/charmbracelet/bubbles github.com/charmbracelet/glamour
github.com/charmbracelet/log github.com/muesli/termenv github.com/lucasb-eyer/go-colorful github.com/rivo/uniseg modernc.org/sqlite
github.com/jackc/puddle/v2 github.com/jackc/pgpassfile github.com/uptrace/bun entgo.io/ent github.com/Masterminds/squirrel
github.com/doug-martin/goqu/v9 github.com/golang-migrate/migrate/v4 github.com/pressly/goose/v3 github.com/gocql/gocql
go.mongodb.org/mongo-driver github.com/elastic/go-elasticsearch/v8 github.com/olivere/elastic/v7 github.com/rabbitmq/amqp091-go
github.com/streadway/amqp github.com/twmb/franz-go github.com/hibiken/asynq github.com/go-co-op/gocron
github.com/opentracing/opentracing-go go.opentelemetry.io/otel go.opentelemetry.io/otel/sdk github.com/prometheus/client_model
github.com/prometheus/procfs github.com/beorn7/perks github.com/zeebo/xxh3 github.com/minio/sha256-simd github.com/minio/highwayhash
github.com/dgryski/go-rendezvous github.com/bits-and-blooms/bitset github.com/bits-and-blooms/bloom/v3 github.com/RoaringBitmap/roaring
github.com/tidwall/btree github.com/tidwall/buntdb github.com/tidwall/match github.com/tidwall/pretty github.com/huandu/xstrings
github.com/iancoleman/strcase github.com/gertd/go-pluralize github.com/jinzhu/inflection github.com/jinzhu/now github.com/gosimple/slug
github.com/microcosm-cc/bluemonday github.com/yuin/gopher-lua github.com/dop251/goja github.com/robertkrimen/otto github.com/traefik/yaegi
github.com/naoina/toml sigs.k8s.io/yaml gopkg.in/ini.v1 github.com/mitchellh/go-homedir github.com/mitchellh/copystructure
github.com/mitchellh/reflectwalk dario.cat/mergo github.com/jedib0t/go-pretty/v6 github.com/briandowns/spinner github.com/manifoldco/promptui
github.com/AlecAivazis/survey/v2 github.com/c-bata/go-prompt github.com/peterh/liner github.com/chzyer/readline github.com/creack/pty
github.com/kballard/go-shellquote github.com/google/go-jsonnet cuelang.org/go github.com/docker/go-units github.com/docker/go-connections
github.com/opencontainers/go-digest github.com/opencontainers/image-spec github.com/moby/term github.com/coreos/go-semver
github.com/coreos/go-systemd/v22 github.com/godbus/dbus/v5 github.com/vishvananda/netlink github.com/miekg/dns github.com/quic-go/quic-go
github.com/caddyserver/certmagic github.com/libp2p/go-libp2p github.com/ipfs/go-cid github.com/multiformats/go-multiaddr
github.com/btcsuite/btcd github.com/gorilla/schema github.com/gorilla/sessions github.com/gorilla/handlers github.com/gorilla/securecookie
github.com/julienschmidt/httprouter github.com/go-resty/resty/v2 github.com/imroc/req/v3 github.com/sony/gobreaker github.com/afex/hystrix-go
github.com/ulule/limiter/v3 golang.org/x/mod golang.org/x/tools golang.org/x/term golang.org/x/image golang.org/x/vuln gonum.org/v1/gonum
gonum.org/v1/plot github.com/montanaflynn/stats github.com/ajstarks/svgo github.com/fogleman/gg github.com/disintegration/imaging
github.com/nfnt/resize github.com/sergi/go-diff github.com/hexops/gotextdiff github.com/onsi/ginkgo/v2 github.com/onsi/gomega
github.com/smartystreets/goconvey github.com/matryer/is github.com/frankban/quicktest gotest.tools/v3 github.com/maxbrunsfeld/counterfeiter/v6
github.com/vektra/mockery/v2 github.com/DATA-DOG/go-sqlmock github.com/jarcoal/httpmock github.com/h2non/gock github.com/brianvoe/gofakeit/v6
github.com/go-faker/faker/v4 mvdan.cc/gofumpt mvdan.cc/sh/v3 honnef.co/go/tools github.com/mgechev/revive github.com/kisielk/errcheck
github.com/securego/gosec/v2 github.com/evanw/esbuild github.com/tdewolff/minify/v2 github.com/gohugoio/hugo github.com/rs/xid
github.com/oklog/ulid/v2 github.com/segmentio/ksuid github.com/lithammer/shortuuid/v4 github.com/sqids/sqids-go github.com/go-git/go-git/v5
github.com/xanzy/go-gitlab github.com/slack-go/slack github.com/bwmarrin/discordgo github.com/go-telegram-bot-api/telegram-bot-api/v5
github.com/aws/aws-lambda-go github.com/aws/smithy-go cloud.google.com/go/storage github.com/Azure/azure-sdk-for-go/sdk/azcore
github.com/casbin/casbin/v2 github.com/markbates/goth github.com/coreos/go-oidc/v3 github.com/lestrrat-go/jwx/v2 github.com/go-jose/go-jose/v4
github.com/pquerna/otp github.com/gofrs/uuid github.com/gofrs/flock gopkg.in/natefinch/lumberjack.v2 github.com/lmittmann/tint
github.com/phuslu/log github.com/apex/log github.com/inconshreveable/mousetrap github.com/cpuguy83/go-md2man/v2
github.com/russross/blackfriday/v2 github.com/alecthomas/participle/v2 github.com/alecthomas/units github.com/alecthomas/kingpin/v2
github.com/jessevdk/go-flags github.com/peterbourgon/ff/v3 github.com/nxadm/tail github.com/radovskyb/watcher github.com/panjf2000/ants/v2
github.com/panjf2000/gnet/v2 github.com/alitto/pond github.com/gammazero/workerpool github.com/Jeffail/tunny github.com/eapache/go-resiliency
github.com/eapache/queue github.com/emirpasic/gods github.com/zyedidia/generic github.com/elliotchance/orderedmap/v2
github.com/wk8/go-ordered-map/v2 github.com/deckarep/golang-set/v2 github.com/orcaman/concurrent-map/v2 github.com/puzpuzpuz/xsync/v3
github.com/allegro/bigcache/v3 github.com/coocood/freecache github.com/VictoriaMetrics/fastcache github.com/maypok86/otter
github.com/jellydator/ttlcache/v3 github.com/bradfitz/gomemcache github.com/gomodule/redigo github.com/alicebob/miniredis/v2
github.com/syndtr/goleveldb github.com/cockroachdb/pebble go.etcd.io/etcd/client/v3 github.com/boltdb/bolt github.com/nutsdb/nutsdb""".split()

MAVEN_ARTIFACTS = """com.google.guava:guava com.google.code.gson:gson com.google.protobuf:protobuf-java com.google.inject:guice
com.google.dagger:dagger com.google.auto.value:auto-value com.google.auto.service:auto-service com.google.truth:truth
com.google.jimfs:jimfs com.google.googlejavaformat:google-java-format com.google.errorprone:error_prone_annotations com.google.zxing:core
com.google.flogger:flogger com.google.http-client:google-http-client com.google.api-client:google-api-client
com.google.cloud:google-cloud-storage com.google.j2objc:j2objc-annotations com.google.code.findbugs:jsr305 com.google.crypto.tink:tink
org.apache.commons:commons-lang3 org.apache.commons:commons-collections4 org.apache.commons:commons-text org.apache.commons:commons-math3
org.apache.commons:commons-compress org.apache.commons:commons-csv org.apache.commons:commons-configuration2 org.apache.commons:commons-pool2
org.apache.commons:commons-dbcp2 org.apache.commons:commons-exec org.apache.commons:commons-jexl3 org.apache.commons:commons-vfs2
org.apache.commons:commons-numbers-core org.apache.commons:commons-rng-simple commons-io:commons-io
commons-codec:commons-codec commons-cli:commons-cli commons-logging:commons-logging commons-beanutils:commons-beanutils
commons-validator:commons-validator commons-net:commons-net org.apache.httpcomponents:httpclient org.apache.httpcomponents:httpcore
org.apache.httpcomponents.client5:httpclient5 org.apache.httpcomponents.core5:httpcore5 org.apache.logging.log4j:log4j-api
org.apache.logging.log4j:log4j-core org.apache.kafka:kafka-clients org.apache.zookeeper:zookeeper org.apache.lucene:lucene-core
org.apache.lucene:lucene-analysis-common org.apache.lucene:lucene-queryparser org.apache.poi:poi org.apache.pdfbox:pdfbox
org.apache.tika:tika-core org.apache.avro:avro org.apache.arrow:arrow-vector org.apache.arrow:arrow-memory-core org.apache.thrift:libthrift
org.apache.velocity:velocity-engine-core org.freemarker:freemarker org.apache.curator:curator-framework org.apache.sshd:sshd-core
org.apache.mina:mina-core org.apache.shiro:shiro-core org.apache.calcite:calcite-core org.apache.datasketches:datasketches-java
org.apache.maven:maven-core org.apache.ant:ant org.apache.tomcat.embed:tomcat-embed-core org.apache.camel:camel-api
org.apache.flink:flink-core org.apache.parquet:parquet-column org.apache.orc:orc-core org.apache.hadoop:hadoop-common
org.apache.spark:spark-core_2.13 org.apache.pekko:pekko-actor_2.13 org.apache.activemq:activemq-client org.apache.groovy:groovy
org.apache.opennlp:opennlp-tools com.fasterxml.jackson.core:jackson-core com.fasterxml.jackson.core:jackson-databind
com.fasterxml.jackson.core:jackson-annotations com.fasterxml.jackson.dataformat:jackson-dataformat-yaml
com.fasterxml.jackson.dataformat:jackson-dataformat-xml com.fasterxml.jackson.dataformat:jackson-dataformat-csv
com.fasterxml.jackson.dataformat:jackson-dataformat-cbor com.fasterxml.jackson.datatype:jackson-datatype-jsr310
com.fasterxml.jackson.datatype:jackson-datatype-jdk8 com.fasterxml.jackson.module:jackson-module-kotlin
com.fasterxml.jackson.module:jackson-module-parameter-names com.fasterxml.woodstox:woodstox-core com.fasterxml:classmate
com.squareup.okhttp3:okhttp com.squareup.okhttp3:mockwebserver com.squareup.okio:okio-jvm com.squareup.retrofit2:retrofit
com.squareup.moshi:moshi com.squareup:javapoet com.squareup:kotlinpoet-jvm com.squareup.wire:wire-runtime-jvm io.netty:netty-common
io.netty:netty-buffer io.netty:netty-codec io.netty:netty-codec-http io.netty:netty-codec-http2 io.netty:netty-handler
io.netty:netty-transport io.netty:netty-resolver io.projectreactor:reactor-core io.projectreactor.netty:reactor-netty-core
io.reactivex.rxjava3:rxjava org.reactivestreams:reactive-streams io.micrometer:micrometer-core io.dropwizard.metrics:metrics-core
io.opentelemetry:opentelemetry-api io.opentelemetry:opentelemetry-sdk io.prometheus:prometheus-metrics-core io.prometheus:simpleclient
io.grpc:grpc-api io.grpc:grpc-core io.grpc:grpc-stub io.grpc:grpc-netty io.grpc:grpc-protobuf org.springframework:spring-core
org.springframework:spring-context org.springframework:spring-beans org.springframework:spring-web org.springframework:spring-webmvc
org.springframework:spring-webflux org.springframework:spring-jdbc org.springframework:spring-aop org.springframework:spring-tx
org.springframework:spring-expression org.springframework:spring-messaging org.springframework.boot:spring-boot
org.springframework.boot:spring-boot-autoconfigure org.springframework.boot:spring-boot-actuator org.springframework.data:spring-data-commons
org.springframework.data:spring-data-jpa org.springframework.security:spring-security-core org.springframework.security:spring-security-web
org.springframework.kafka:spring-kafka org.springframework.batch:spring-batch-core org.hibernate.validator:hibernate-validator
jakarta.validation:jakarta.validation-api jakarta.inject:jakarta.inject-api javax.inject:javax.inject org.mockito:mockito-core
org.assertj:assertj-core org.hamcrest:hamcrest org.testng:testng org.awaitility:awaitility org.wiremock:wiremock
org.testcontainers:testcontainers io.rest-assured:rest-assured org.easymock:easymock io.cucumber:cucumber-core
com.tngtech.archunit:archunit nl.jqno.equalsverifier:equalsverifier org.xmlunit:xmlunit-core org.skyscreamer:jsonassert
org.jetbrains.kotlin:kotlin-stdlib org.jetbrains.kotlin:kotlin-reflect org.jetbrains.kotlinx:kotlinx-coroutines-core-jvm
org.jetbrains.kotlinx:kotlinx-serialization-core-jvm org.jetbrains.kotlinx:kotlinx-serialization-json-jvm
org.jetbrains.kotlinx:kotlinx-datetime-jvm org.jetbrains.kotlinx:kotlinx-collections-immutable-jvm org.jetbrains:annotations
io.ktor:ktor-server-core-jvm io.ktor:ktor-client-core-jvm io.ktor:ktor-http-jvm com.github.ajalt.clikt:clikt-jvm io.arrow-kt:arrow-core-jvm
org.jetbrains.exposed:exposed-core io.insert-koin:koin-core-jvm io.kotest:kotest-assertions-core-jvm io.mockk:mockk-jvm
org.scala-lang:scala-library org.scala-lang:scala3-library_3 org.typelevel:cats-core_3 org.typelevel:cats-effect_3 co.fs2:fs2-core_3
io.circe:circe-core_3 com.lihaoyi:os-lib_3 com.lihaoyi:upickle_3 com.lihaoyi:fastparse_3 org.http4s:http4s-core_3 dev.zio:zio_3
org.scalameta:munit_3 com.softwaremill.sttp.client4:core_3 org.apache.pekko:pekko-actor_3
org.scala-lang.modules:scala-parser-combinators_3 com.github.pureconfig:pureconfig-core_3 com.zaxxer:HikariCP org.postgresql:postgresql
org.xerial:sqlite-jdbc org.flywaydb:flyway-core org.liquibase:liquibase-core org.jooq:jooq org.mybatis:mybatis redis.clients:jedis
io.lettuce:lettuce-core org.mongodb:mongodb-driver-core org.mongodb:bson org.apache.cassandra:java-driver-core
co.elastic.clients:elasticsearch-java org.elasticsearch.client:elasticsearch-rest-client io.r2dbc:r2dbc-spi org.jdbi:jdbi3-core
com.github.ben-manes.caffeine:caffeine com.lmax:disruptor org.jctools:jctools-core it.unimi.dsi:fastutil org.roaringbitmap:RoaringBitmap
net.bytebuddy:byte-buddy org.ow2.asm:asm org.ow2.asm:asm-tree org.objenesis:objenesis com.esotericsoftware:kryo org.msgpack:msgpack-core
com.github.luben:zstd-jni org.lz4:lz4-java org.xerial.snappy:snappy-java com.jayway.jsonpath:json-path com.networknt:json-schema-validator
info.picocli:picocli com.beust:jcommander org.jline:jline org.antlr:antlr4-runtime com.typesafe:config org.codehaus.plexus:plexus-utils
io.swagger.core.v3:swagger-core org.openapitools:jackson-databind-nullable org.immutables:value org.projectlombok:lombok
org.mapstruct:mapstruct org.checkerframework:checker-qual joda-time:joda-time org.joda:joda-convert org.threeten:threeten-extra
org.hdrhistogram:HdrHistogram com.github.oshi:oshi-core com.jcraft:jsch com.github.mwiede:jsch org.quartz-scheduler:quartz
com.vladsch.flexmark:flexmark org.commonmark:commonmark com.thoughtworks.xstream:xstream org.dom4j:dom4j org.jdom:jdom2
org.yaml:snakeyaml org.snakeyaml:snakeyaml-engine org.jsoup:jsoup io.jsonwebtoken:jjwt-api io.jsonwebtoken:jjwt-impl com.auth0:java-jwt
com.nimbusds:nimbus-jose-jwt org.luaj:luaj-jse org.agrona:agrona io.aeron:aeron-client org.codehaus.janino:janino org.mvel:mvel2
io.vavr:vavr org.slf4j:slf4j-api org.slf4j:slf4j-simple org.tinylog:tinylog-api com.googlecode.libphonenumber:libphonenumber
io.undertow:undertow-core io.javalin:javalin com.sparkjava:spark-core io.micronaut:micronaut-core io.quarkus:quarkus-core
io.dropwizard:dropwizard-core com.linecorp.armeria:armeria org.asynchttpclient:async-http-client com.konghq:unirest-java-core
software.amazon.awssdk:s3 software.amazon.awssdk:sdk-core software.amazon.awssdk:dynamodb com.amazonaws:aws-java-sdk-core
com.amazonaws:aws-lambda-java-core com.azure:azure-core com.azure:azure-storage-blob org.ejml:ejml-ddense ai.djl:api""".split()

NPM_QUERIES = v2.NPM_QUERIES + [f"keywords:{k}" for k in """webpack babel eslint plugin vue angular svelte express middleware database
mongodb sql orm graphql websocket crypto hash image markdown template logger log config env color terminal ansi fs file path glob url
query math number object function events emitter queue cache redis aws cloud api sdk client server rest fetch request xml yaml csv
parse format i18n mock assert benchmark lint build bundler rollup vite postcss sass svg canvas chart d3 three game audio video pdf zip
compression buffer binary encoding unicode regex ast typescript-types""".split()]

RUBY_QUERIES = """rails active json http test rspec rack aws google api cli parser yaml xml html markdown logger cache redis sql database
async thread time date string crypto ssl auth oauth jwt template erb web server client rake bundler rubocop lint format image file io net
ssh git docker kubernetes graphql grpc protobuf sidekiq job queue mail i18n config env debug benchmark profiler memory math matrix
statistics csv excel pdf""".split()

DEFAULT_MAX = {"pypi": 6000, "npm": 4000, "crates": 1500, "go": len(GO), "maven": len(MAVEN_ARTIFACTS), "rubygems": 1500, "hackage": 2000}


def pypi_names(n: int, skip: set[str]) -> list[str]:
    raw = fetch("https://raw.githubusercontent.com/hugovk/top-pypi-packages/main/top-pypi-packages.min.json")
    if not raw:
        log("top-pypi list unavailable; no new PyPI packages")
        return []
    return [r["project"] for r in json.loads(raw)["rows"] if r["project"] not in skip][:n]


def npm_names(n: int, skip: set[str]) -> list[str]:
    names: dict[str, None] = {}
    for q in NPM_QUERIES:
        for start in range(0, 2000, 250):
            raw = fetch(f"https://registry.npmjs.org/-/v1/search?text={urllib.parse.quote(q)}&size=250&from={start}"
                        "&popularity=1.0&quality=0.0&maintenance=0.0")
            objs = json.loads(raw).get("objects", []) if raw else []
            names.update({o["package"]["name"]: None for o in objs if o["package"]["name"] not in skip})
            if len(objs) < 250 or len(names) >= n:
                break
        if len(names) >= n:
            break
    return list(names)[:n]


def crates_names(n: int, skip: set[str]) -> list[str]:
    names: list[str] = []
    for page in range(1, 200):
        raw = fetch(f"https://crates.io/api/v1/crates?sort=downloads&per_page=100&page={page}")
        crates_ = json.loads(raw).get("crates", []) if raw else []
        names += [c["name"] for c in crates_ if c["name"] not in skip and c["name"] not in names]
        if len(crates_) < 100 or len(names) >= n:
            break
    return names[:n]


def rubygems_names(n: int, skip: set[str]) -> list[str]:
    """Gems found by keyword search, ranked by total downloads (the API has no top list any more)."""
    found: dict[str, int] = {}
    for q in RUBY_QUERIES:
        for page in (1, 2, 3):
            raw = fetch(f"https://rubygems.org/api/v1/search.json?query={urllib.parse.quote(q)}&page={page}")
            gems = json.loads(raw) if raw else []
            found.update({g["name"]: int(g.get("downloads") or 0) for g in gems if g.get("name") and g["name"] not in skip})
            if len(gems) < 30:
                break
        if len(found) >= 2 * n:
            break
    return sorted(found, key=lambda g: -found[g])[:n]


def hackage_names(n: int, skip: set[str]) -> list[str]:
    raw = fetch("https://hackage.haskell.org/packages/top", accept="application/json")
    rows = json.loads(raw) if raw else []
    rows = sorted((r for r in rows if r.get("packageName") and r["packageName"] not in skip), key=lambda r: -int(r.get("downloads") or 0))
    return [r["packageName"] for r in rows[:n]]


LISTS = {"pypi": pypi_names, "npm": npm_names, "crates": crates_names, "rubygems": rubygems_names, "hackage": hackage_names,
         "go": lambda n, skip: [m for m in GO if m not in skip][:n],
         "maven": lambda n, skip: [c for c in MAVEN_ARTIFACTS if c not in skip][:n]}
