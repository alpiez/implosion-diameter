"""Which source modules does each test file import at collection time?

If a patch makes module M fail at import, every test file whose import pulls
in M fails to collect, so all of its tests fail (synthetic s0090: an IndexError
in jsonpath.parse, run while importing generators/duckdb.py, broke 1124 tests
in 40+ files). The trace parent preloads everything, so this is measured
separately: one FRESH interpreter per test file imports only that file (after
pytest/unittest, as under pytest) and reports the sqlglot/ and tests/ modules
now loaded. Lazy imports through sqlglot.dialects.__getattr__ that run at module
level are included, because they actually happen.

Also measures a bare `import sqlglot` (what test_lazy_load's subprocess runs).

Runs inside br-dev:
  docker run --rm -v "$PWD/tools:/tools:ro" -v "$PWD/work:/out" br-dev \
      python /tools/import_closure.py --tests /out/tests.json --out /out/trace/closure.json
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = "/opt/sqlglot"

PROBE = r"""
import sys, json
import pytest, unittest  # as under pytest (sqlglot reads sys.modules at import)
target = sys.argv[1]
err = None
try:
    if target != "sqlglot":
        __import__(target)
    else:
        import sqlglot
except BaseException as e:
    err = repr(e)[:300]
root = %r + "/"
files = sorted({m.__file__[len(root):] for m in list(sys.modules.values())
                if getattr(m, "__file__", None) and m.__file__.startswith(root)})
print(json.dumps({"files": files, "err": err}))
""" % ROOT


def probe(mod: str) -> tuple[str, dict]:
    env = dict(os.environ, PYTHONPATH=ROOT, OPENBLAS_NUM_THREADS="1", SKIP_INTEGRATION="0")
    r = subprocess.run([sys.executable, "-c", PROBE, mod], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    try:
        return mod, json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        return mod, {"files": [], "err": "probe failed: " + r.stderr[-500:]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    tests = json.loads(Path(a.tests).read_text())["tests"]
    mods = sorted({t["file"][:-3].replace("/", ".") for t in tests})
    with ThreadPoolExecutor(a.jobs) as ex:
        res = dict(ex.map(probe, mods + ["sqlglot"]))
    out = {"test_files": {m.replace(".", "/") + ".py": r for m, r in res.items() if m != "sqlglot"},
           "bare_import": res["sqlglot"]}
    a.out.write_text(json.dumps(out))
    errs = {m: r["err"] for m, r in res.items() if r["err"]}
    sizes = sorted(len(r["files"]) for r in res.values())
    print(f"{len(res)} probes, {len(errs)} errors {errs}, modules per file: "
          f"min {sizes[0]} median {sizes[len(sizes) // 2]} max {sizes[-1]}")


if __name__ == "__main__":
    main()
