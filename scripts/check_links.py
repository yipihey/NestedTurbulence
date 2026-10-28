#!/usr/bin/env python3
"""Check the provenance shortcodes of the text. Exit 1 on failure.

Julia projects (quadrant: language: julia in _quarto.yml):
  {{< src name >}}     a definition of `name` exists under quadrant.src (function, @kernel function, one-line method,
                       struct, const), the same patterns the quadrant extension resolves at render time;
  {{< test "name" >}}  a `@testset "name…"` exists under quadrant.tests.
Rust projects (the quadrant default): rustdoc pages under api/ and `fn name(` under core/tests.
Every {{< lab "hash" >}} must use keys of the state schema in app/js/state.js and the '_' list separator."""
import re, sys, glob, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
cfg = {"language": "rust", "api": "api", "tests": "core/tests", "src": "core/src", "crate": None}
qy = (ROOT / "_quarto.yml").read_text() if (ROOT / "_quarto.yml").exists() else ""
m = re.search(r"^quadrant:\s*\n((?:[ \t]+.*\n?)+)", qy, re.M)
if m:
    for line in m.group(1).splitlines():
        k, _, v = line.strip().partition(":")
        if k in cfg and v.strip():
            cfg[k] = v.strip().strip('"').strip("'")
JULIA = cfg["language"] == "julia"
if JULIA and cfg["tests"] == "core/tests":
    cfg["tests"] = "core/test"

def julia_def(name, src):
    n = re.escape(name)
    pats = [rf"^\s*(?:@\w+\s+)*function\s+{n}\s*[({{]", rf"^\s*(?:@\w+\s+)*{n}\(.*?\)\s*(?:where\s*\{{[^}}]*\}}\s*)?=",
            rf"^\s*(?:mutable\s+)?struct\s+{n}\b", rf"^\s*Base\.@kwdef\s+struct\s+{n}\b", rf"^\s*const\s+{n}\s*="]
    return any(re.search(p, src, re.M) for p in pats)

if JULIA:
    src_text = "\n".join(p.read_text() for p in sorted((ROOT / cfg["src"]).glob("*.jl")))
    tests_text = "\n".join(p.read_text() for p in sorted((ROOT / cfg["tests"]).glob("*.jl")))
    resolve_src = lambda path: julia_def(path, src_text)
    has_test = lambda name: f'@testset "{name}' in tests_text
else:
    KINDS = ["struct", "enum", "trait", "type", "union"]
    API = ROOT / cfg["api"]
    if not cfg["crate"]:
        cands = [d.name for d in API.iterdir() if d.is_dir() and (d / "index.html").exists() and d.name not in ("src", "static.files")] if API.exists() else []
        cfg["crate"] = cands[0] if cands else "core"
    CRATE_DIR = API / cfg["crate"]
    def resolve_src(path):
        parts = path.split("::"); item = parts[-1]
        if item[:1].isupper():
            d = CRATE_DIR.joinpath(*parts[:-1]); return any((d / f"{k}.{item}.html").exists() for k in KINDS)
        if len(parts) >= 2 and parts[-2][:1].isupper():
            d = CRATE_DIR.joinpath(*parts[:-2])
            for k in KINDS:
                f = d / f"{k}.{parts[-2]}.html"
                if f.exists(): return f'id="method.{item}"' in f.read_text(errors="ignore")
            return False
        d = CRATE_DIR.joinpath(*parts[:-1]); return (d / f"fn.{item}.html").exists() or (d / item / "index.html").exists()
    tests_text = "\n".join(p.read_text() for p in (ROOT / cfg["tests"]).glob("*.rs")) if (ROOT / cfg["tests"]).exists() else ""
    has_test = lambda name: re.search(rf"fn\s+{re.escape(name)}\s*\(", tests_text) is not None

keys = set()
for sj in glob.glob(str(ROOT / "app" / "js" / "**" / "state.js"), recursive=True):
    keys |= set(re.findall(r"\['[A-Za-z0-9]+', '([A-Za-z0-9]+)', '", pathlib.Path(sj).read_text()))

bad = 0; n_src = n_test = n_lab = 0
files = sorted(glob.glob(str(ROOT / "chapters" / "*.qmd"))) + sorted(glob.glob(str(ROOT / "*.qmd")))
for qmd in files:
    text = pathlib.Path(qmd).read_text(); rel = pathlib.Path(qmd).relative_to(ROOT)
    for mm in re.finditer(r"\{\{<\s*src\s+\"?([A-Za-z0-9_:!]+)", text):
        n_src += 1
        if not resolve_src(mm.group(1)):
            print(f"{rel}: src {mm.group(1)} has no definition under {cfg['src'] if JULIA else cfg['api']}"); bad += 1
    for mm in re.finditer(r"\{\{<\s*test\s+\"([^\"]+)\"|\{\{<\s*test\s+([A-Za-z0-9_]+)", text):
        n_test += 1; name = mm.group(1) or mm.group(2)
        if not has_test(name):
            print(f"{rel}: test {name!r} not found under {cfg['tests']}"); bad += 1
    for mm in re.finditer(r"\{\{<\s*lab\s+\"([^\"]*)\"", text):
        n_lab += 1
        if "%2C" in mm.group(1) or "," in mm.group(1):
            print(f"{rel}: lab hash {mm.group(1)!r} uses commas; lists are joined with '_'"); bad += 1
        if keys:
            for kv in mm.group(1).split("&"):
                if kv and kv.split("=")[0] not in keys:
                    print(f"{rel}: lab hash key {kv.split('=')[0]!r} is not in the state schema"); bad += 1
print(f"provenance links ({n_src} src, {n_test} test, {n_lab} lab):", "OK" if bad == 0 else f"{bad} problem(s)")
sys.exit(1 if bad else 0)
