"""Ground-truth labeller: which tests does a patch break?

Runs INSIDE the br-dev image (dev/Dockerfile). For every patch: copy the pinned
checkout (/opt/sqlglot) to a scratch dir, `git apply` the patch, run the whole
suite with pytest (+timeout, +xdist), and print one JSON line:

  {"commit": id, "failing": [test ids], "collect_errors": [...],
   "missing": [ids in tests.json never reported], "error": str|null, "wall": s}

A collection error on tests/x.py marks every tests.json test of that file as
failing (the problem statement: "its file cannot be imported").

Usage (see tools/run_oracle.sh for the docker wrapper):
  python /tools/oracle.py --tests /data/tests.json [--jobs 3] [--workers 4]
         [--timeout 90] PATCH...
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = Path("/opt/sqlglot")
TOOLS = Path(__file__).resolve().parent


def touched_paths(patch_text: str) -> list[str]:
    paths = []
    for line in patch_text.splitlines():
        if line.startswith(("--- a/", "+++ b/")):
            paths.append(line[6:].split("\t")[0])
    return sorted(set(paths))


def label(patch: Path, tests: dict, args) -> dict:
    t0 = time.time()
    commit = patch.stem
    res = {"commit": commit, "failing": [], "collect_errors": [], "missing": [], "error": None}
    with tempfile.TemporaryDirectory(prefix=f"oracle-{commit}-") as tmp:
        work = Path(tmp) / "sqlglot"
        shutil.copytree(BASE, work, symlinks=True)
        text = patch.read_text()
        ap = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", str(patch.resolve())],
            cwd=work, capture_output=True, text=True,
        )
        if ap.returncode:
            res["error"] = "apply: " + ap.stderr.strip()
            return res
        # Drop stale bytecode of touched files (pyc validation is mtime+size).
        for rel in touched_paths(text):
            p = work / rel
            cache = p.parent / "__pycache__"
            if cache.is_dir():
                for pyc in cache.glob(p.stem + ".*.pyc"):
                    pyc.unlink()
        out = Path(tmp) / "result.json"
        env = dict(os.environ)
        env.update(
            PYTHONPATH=f"{work}{os.pathsep}{TOOLS}",
            SKIP_INTEGRATION="0",
            ORACLE_OUT=str(out),
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            PYTHON_CPU_COUNT=str(args.pool),
        )
        env.pop("PYTHONDONTWRITEBYTECODE", None)
        base = [
            sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
            "-p", "oracle_plugin", "--continue-on-collection-errors",
            f"--timeout={args.timeout}", "-o", "console_output_style=classic",
        ]
        cmd = base + ["tests"]
        if args.workers > 1:
            cmd += ["-n", str(args.workers)]
        try:
            pr = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True,
                                timeout=args.wall)
        except subprocess.TimeoutExpired:
            res["error"] = f"wall timeout {args.wall}s"
            return res
        if not out.exists():
            res["error"] = f"no result (rc={pr.returncode}): " + (pr.stdout + pr.stderr)[-2000:]
            return res
        r = json.loads(out.read_text())
        first_why = r.get("why", {})
        # Confirm: rerun the failing tests together in one fresh process (no xdist).
        # Under heavy parallel load the pool-based optimizer tests occasionally fail
        # spuriously; only failures that reproduce count. Skipped for huge sets.
        flaky = []
        if r["failed"] and len(r["failed"]) <= args.confirm_max:
            out2 = Path(tmp) / "confirm.json"
            env2 = dict(env, ORACLE_OUT=str(out2))
            cmd2 = base + ["-p", "no:xdist", *r["failed"]]
            try:
                subprocess.run(cmd2, cwd=work, env=env2, capture_output=True, text=True,
                               timeout=args.wall)
                if out2.exists():
                    r2 = json.loads(out2.read_text())
                    still = set(r2["failed"]) | {n for n in r["failed"] if n not in set(r2["seen"])}
                    flaky = sorted(set(r["failed"]) - still)
                    r["failed"] = sorted(still)
            except subprocess.TimeoutExpired:
                pass
        res["flaky"] = flaky
        res["flaky_why"] = {k: first_why.get(k, "")[-800:] for k in flaky}
        failing = set(r["failed"])
        for nodeid in r["collect_errors"]:
            f = nodeid.split("::")[0]
            failing |= {tid for tid, file in tests.items() if file == f or nodeid in ("", ".", "tests")}
        seen = set(r["seen"])
        res["failing"] = sorted(t for t in failing if t in tests)
        res["unknown_failing"] = sorted(t for t in failing if t not in tests)
        res["collect_errors"] = r["collect_errors"]
        res["missing"] = sorted(t for t in tests if t not in seen and t not in failing)
        res["exitstatus"] = r["exitstatus"]
        res["why"] = r.get("why", {})
        if args.keep_log:
            res["log"] = pr.stdout[-5000:]
    res["wall"] = round(time.time() - t0, 1)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", default="/data/tests.json")
    ap.add_argument("--jobs", type=int, default=3, help="patches labelled concurrently")
    ap.add_argument("--workers", type=int, default=4, help="xdist workers per patch")
    ap.add_argument("--pool", type=int, default=2, help="PYTHON_CPU_COUNT (TestOptimizer pool)")
    ap.add_argument("--timeout", type=int, default=90, help="pytest-timeout per test (s)")
    ap.add_argument("--wall", type=int, default=1800, help="wall limit per patch (s)")
    ap.add_argument("--confirm-max", type=int, default=300,
                    help="rerun-confirm failures when at most this many failed")
    ap.add_argument("--keep-log", action="store_true")
    ap.add_argument("patches", nargs="+", type=Path)
    args = ap.parse_args()

    tests = {t["id"]: t["file"] for t in json.loads(Path(args.tests).read_text())["tests"]}
    # Precompile the base once so each scratch copy starts with valid bytecode.
    subprocess.run([sys.executable, "-m", "compileall", "-q", "-j0", str(BASE / "sqlglot"),
                    str(BASE / "tests")], capture_output=True)
    with ThreadPoolExecutor(args.jobs) as ex:
        for res in ex.map(lambda p: label(p, tests, args), args.patches):
            print(json.dumps(res), flush=True)
            print(f"[oracle] {res['commit']}: {len(res['failing'])} failing "
                  f"err={res['error']} wall={res.get('wall')}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
