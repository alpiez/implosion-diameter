"""Measure verification CPU per test on the clean pinned tree (inside the submission image).

Runs every test of tests.json through app/verify.py exactly as verification would
(one child per test file, class by class, inline pools), with a no-op change so the
overlay path is exercised. Writes {test id: cpu seconds} plus per-file overhead
(child CPU minus the tests' own CPU: imports and class fixtures). Repeated --reps
times; the median is kept. The result is a property of the pinned commit, not of
any patch (CLAUDE.md section 5).

  docker run --rm -v "$PWD":/w:ro -v "$PWD/work":/out br-sub \\
      python /w/tools/measure_vcpu.py --tests /w/sample/tests.json --out /out/vcpu.json
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
import verify  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--exec-cache", action="store_true", help="phase 6c executor caches on")
    ap.add_argument("--only-file", help="measure only this test file (and its split tests)")
    ap.add_argument("--merge", type=Path, help="update this vcpu.json instead of starting empty")
    a = ap.parse_args()
    if a.exec_cache:
        import exec_cache
        print("executor cache loaded:", exec_cache.load(), file=sys.stderr)

    def mk(key, tl):
        j = verify.Job(key, noop, tl)
        j.exec_ok = a.exec_cache
        return j
    tests = json.loads(a.tests.read_text())["tests"]
    by_file: dict[str, list] = {}
    for t in tests:
        if a.only_file and not t["id"].startswith(a.only_file + "::"):
            continue
        by_file.setdefault(t["id"].split("::")[0], []).append((t["id"], max(30.0, 5 * t["seconds"] + 10)))
    noop = {"sqlglot/py.typed": (verify.SRC / "sqlglot/py.typed").read_text()
            if (verify.SRC / "sqlglot/py.typed").exists() else ""}
    preload = json.loads((ROOT / "maps" / "preload.json").read_text())  # as at runtime
    v = verify.Verifier(parallel=a.parallel, gap_timeout=120, preload=preload)
    cpu: dict[str, list] = {}
    ovh: dict[str, list] = {}
    failed = set()
    for rep in range(a.reps):
        jobs = [mk(f, tl) for f, tl in by_file.items()]
        t0 = time.monotonic()
        v.run(jobs, time.monotonic() + 1e6)
        for j in jobs:
            failed |= j.failed
            for tid, c in j.tcpu.items():
                cpu.setdefault(tid, []).append(c)
            ovh.setdefault(j.key, []).append(max(0.0, j.child_cpu - j.cpu))
        print(f"rep {rep}: {time.monotonic() - t0:.0f}s wall, failed {len(failed)}", file=sys.stderr)
    # split tests: CPU of each unit alone (one child per test, every unit as its own test)
    import zlib
    units = json.loads(zlib.decompress((ROOT / "maps" / "units.bin").read_bytes()))
    if a.only_file:
        units = {t: d for t, d in units.items() if t.startswith(a.only_file + "::")}
    ucpu: dict[str, list] = {}
    for rep in range(a.reps):
        jobs = [mk(tid, [(tid, 120.0, [u]) for u in range(d["n"])]) for tid, d in units.items()]
        v.run(jobs, time.monotonic() + 1e6)
        for j in jobs:
            if len(j.icpu) != len(j.tests) or j.failed:
                print(f"units of {j.key}: incomplete ({j.why})", file=sys.stderr)
            for i, c in j.icpu.items():
                ucpu.setdefault(j.key, [[] for _ in j.tests])[i].append(c)
    out = {"units": {t: [round(statistics.median(c), 4) if c else None for c in cs]
                     for t, cs in sorted(ucpu.items())},
           "cpu": {t: round(statistics.median(c), 4) for t, c in sorted(cpu.items())},
           "file_overhead": {f: round(statistics.median(c), 4) for f, c in sorted(ovh.items())},
           "failed_clean": sorted(failed)}
    if a.merge:
        base = json.loads(a.merge.read_text())
        for k in ("units", "cpu", "file_overhead"):
            base[k].update(out[k])
        base["failed_clean"] = sorted(set(base.get("failed_clean", [])) | set(out["failed_clean"]))
        out = base
    a.out.write_text(json.dumps(out, indent=0, sort_keys=True))
    print(f"{len(out['cpu'])} tests, clean failures: {sorted(failed)[:10]}", file=sys.stderr)


if __name__ == "__main__":
    main()
