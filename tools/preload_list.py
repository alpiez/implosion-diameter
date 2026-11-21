"""List the standard-library modules that sqlglot and its tests import (maps/preload.json).

The verifier's main process imports these once before forking, so each verification
child stops paying for them (about 67 ms per child, mostly unittest.mock -> asyncio).
Standard-library modules are never patched, so sharing them cannot change a verdict.
Runs a fresh interpreter in the submission image against the pinned checkout:

  docker run --rm -v "$PWD":/w:ro -v "$PWD/maps":/maps br-sub python /w/tools/preload_list.py --out /maps/preload.json
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROBE = r'''
import json, os, sys, sysconfig, importlib, pkgutil
import unittest  # first, as under pytest (sqlglot.expressions.core.UNITTEST)
before = set(sys.modules)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import sqlglot
for info in pkgutil.walk_packages(sqlglot.__path__, prefix="sqlglot."):
    if info.name != "sqlglot.__main__":
        try:
            importlib.import_module(info.name)
        except Exception:
            pass
for p in sorted(os.listdir(os.path.join(ROOT, "tests"))) + ["dialects/" + q for q in sorted(os.listdir(os.path.join(ROOT, "tests", "dialects")))]:
    if p.endswith(".py") and not p.startswith("__") and p.split("/")[-1].startswith("test_"):
        try:
            importlib.import_module("tests." + p[:-3].replace("/", "."))
        except Exception:
            pass
std = {os.path.realpath(sysconfig.get_paths()[k]) for k in ("stdlib", "platstdlib")}
out = []
for name, mod in list(sys.modules.items()):
    if name in before or mod is None or name.startswith(("sqlglot", "tests", "__main__")):
        continue
    f = getattr(mod, "__file__", None)
    if f is None:
        ok = name in sys.builtin_module_names
    else:
        rf = os.path.realpath(f)
        ok = any(rf.startswith(s + os.sep) for s in std) and "site-packages" not in rf
    if ok:
        out.append(name)
print(json.dumps(out))
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/opt/sqlglot")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    r = subprocess.run([sys.executable, "-c", f"ROOT = {a.root!r}\n" + PROBE],
                       capture_output=True, text=True, env=env, check=True)
    mods = json.loads(r.stdout.strip().splitlines()[-1])
    a.out.write_text(json.dumps(mods, indent=0))
    os.chmod(a.out, 0o644)
    print(f"{len(mods)} standard-library modules", file=sys.stderr)


if __name__ == "__main__":
    main()
