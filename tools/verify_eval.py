"""Run the phase-4 verifier on labelled patches inside br-dev and compare with labels.

For every commit: static candidates S (app/blast.py), then run ALL of S on the
patched overlay (app/verify.py) and record, per test, pass/fail and CPU, plus the
per-commit child CPU (imports, fixtures). Then compare with the oracle labels:
  unsafe   = labelled failing, but the verifier said it passed  (would be a miss)
  extra    = verifier failed it, label says it passes            (costs only)
Evaluation only. Output: one JSON line per commit.

  docker run --rm -v "$PWD":/w:ro -v "$PWD/work":/out br-dev \\
      python /w/tools/verify_eval.py --data /w/work/synth --answers /w/work/synth/answers.json \\
      --out /out/synth_verify.jsonl --parallel 8
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
import blast  # noqa: E402
import verify  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--answers", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--max-sum", type=float, default=1e9, help="skip commits whose S sums above this (s)")
    a = ap.parse_args()

    preload = json.loads((ROOT / "maps" / "preload.json").read_text())  # as at runtime
    v = verify.Verifier(parallel=a.parallel, gap_timeout=120, preload=preload)
    tests = json.loads((a.data / "tests.json").read_text())["tests"]
    num = {t["id"]: t["number"] for t in tests}
    by_num = {t["number"]: t for t in tests}
    ans = json.loads(a.answers.read_text())["commits"]
    maps = blast.Maps(blast.MAPS)
    sel = blast.Selector(maps)
    bitnum = [num.get(t) for t in maps.tests]
    jobs, meta = [], {}
    for c in sorted(ans):
        if a.only and c not in a.only:
            continue
        text = (a.data / "commits" / f"{c}.patch").read_text(errors="replace")
        bits = sel.select(text, [])
        S = sorted(bitnum[i] for i in range(len(bitnum)) if bits >> i & 1 and bitnum[i] is not None)
        F = set(ans[c]["failing"])
        m = {"commit": c, "kind": ans[c].get("kind"), "S": S, "F": sorted(F)}
        meta[c] = m
        if not S:
            m["skip"] = "empty"
            continue
        if sum(by_num[n]["seconds"] for n in S) > a.max_sum:
            m["skip"] = "too big"
            continue
        try:
            files = blast.parse_patch(text)
            if any(not (p or "").startswith(verify.VERIFY_ROOTS)
                   for fc in files for p in (fc.old, fc.new) if p is not None):
                raise verify.PatchError("outside roots")
            changes = verify.patched_files(files)
        except Exception as e:  # noqa: BLE001
            m["skip"] = f"patch: {e}"
            continue
        tl = [(by_num[n]["id"], max(10.0, 4 * by_num[n]["seconds"] + 5)) for n in S]
        jobs.append(verify.Job(c, changes, tl))
    t0 = time.monotonic()
    v.run(jobs, deadline=time.monotonic() + 1e6)
    with a.out.open("w") as out:
        for j in jobs:
            m = meta[j.key]
            failed = {num[t] for t in j.failed}
            passed = {num[t] for t in j.passed}
            F = set(m["F"])
            m.update(failed=sorted(failed), passed=sorted(passed),
                     unsafe=sorted(F & passed), extra=sorted(failed - F),
                     tcpu={str(num[t]): c for t, c in j.tcpu.items()},
                     child_cpu=round(j.child_cpu, 3), test_cpu=round(j.cpu, 3), why=j.why.strip())
        for c in sorted(meta):
            out.write(json.dumps(meta[c]) + "\n")
    unsafe = [(m["commit"], m["unsafe"]) for m in meta.values() if m.get("unsafe")]
    print(f"{len(jobs)} verified in {time.monotonic() - t0:.0f}s wall, child cpu "
          f"{sum(j.child_cpu for j in jobs):.0f}s; UNSAFE commits: {len(unsafe)} {unsafe[:10]}")


if __name__ == "__main__":
    main()
