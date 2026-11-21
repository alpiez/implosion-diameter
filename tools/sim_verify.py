"""Replay the runner's verification plan on verify_eval.py results (evaluation only).

For each commit: plan V with app/runner.plan_commit (CPU estimates from --vcpu),
then charge the REAL outcome recorded by verify_eval: tests of V that passed are
dropped, the rest stay; CPU = measured CPU of V's tests + per-file overhead.
Reports static vs verified totals (+ misses: labelled failures that would be dropped).

  python3 tools/sim_verify.py --data work/synth --verify work/synth_verify.jsonl --vcpu maps/vcpu.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
import runner  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--verify", type=Path, required=True)
    ap.add_argument("--vcpu", type=Path, default=ROOT / "maps" / "vcpu.json")
    ap.add_argument("--margin", type=int, default=runner.MARGIN_MS)
    ap.add_argument("--q-data", type=float)
    ap.add_argument("--rel", type=float)
    ap.add_argument("--q-code", type=float)
    ap.add_argument("--cpu-scale", type=float, default=1.0, help="scorer CPU / our CPU")
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    if a.rel is not None:
        runner.MARGIN_REL = a.rel
    if a.q_data is not None:
        runner.Q_DATA = a.q_data
    if a.q_code is not None:
        runner.Q_CODE = a.q_code
    tests = json.loads((a.data / "tests.json").read_text())["tests"]
    dur = {t["number"]: runner.blast.duration_ms(t["seconds"]) for t in tests}
    id_of = {t["number"]: t["id"] for t in tests}
    num_by_id = {t["id"]: t["number"] for t in tests}
    sys.path.insert(0, str(ROOT / "app"))
    import verify
    maps = runner.blast.Maps(runner.blast.MAPS)
    sel = runner.blast.Selector(maps)
    planner = runner.Planner(dur, id_of, num_by_id, maps, verify.SPLITS)
    if a.vcpu != ROOT / "maps" / "vcpu.json":
        vc = json.loads(a.vcpu.read_text())
        planner.cpu = {n: float(vc["cpu"].get(id_of[n], dur[n] / 1000)) for n in dur}
        planner.file_ovh = vc["file_overhead"]
    tot_s = tot_v = tot_o = tot_cpu = 0
    nver = misses = 0
    for line in a.verify.open():
        m = json.loads(line)
        text = (a.data / "commits" / f"{m['commit']}.patch").read_text(errors="replace")
        bits = sel.select(text, [])
        bn = [num_by_id.get(t) for t in maps.tests]
        S = sorted(bn[i] for i in range(len(bn)) if bits >> i & 1 and bn[i] is not None)
        F = set(m["F"])
        cs = runner.pack_cost([dur[n] for n in S])
        co = runner.pack_cost([dur[n] for n in F])
        if F - set(S):
            cs += 300000  # a static miss is a miss either way
        cv = cs
        if "passed" in m:
            gain, V, ccpu, cunits = planner.plan(S, runner.blast.parse_patch(text),
                                                 (sel.noncov, sel.covlines))
            if V and gain >= a.margin:
                V = sorted(V, key=lambda n: (ccpu[n], n))
                rest_d = [dur[n] for n in S if n not in set(V)]
                outcome_pass = set(m["passed"])
                passed, failed, done = set(), set(), set()
                real_cpu = 0.0
                for n in V:   # sequential replay with the runtime's stopping rule
                    ok = n in outcome_pass
                    real_cpu += ccpu[n] if (ok or n in cunits) else min(
                        ccpu[n], float(m["tcpu"].get(str(n), ccpu[n])))
                    (passed if ok else failed).add(n)
                    done.add(n)
                    if failed and not runner.worth_continuing(failed, done, V, rest_d, dur, ccpu):
                        break
                keep = [n for n in S if n not in passed]
                real_cpu += runner.overhead({planner.file_of[n] for n in V}, planner.file_ovh)
                real_cpu *= a.cpu_scale
                cv = runner.pack_cost([dur[n] for n in keep]) + int(1000 * real_cpu)
                nver += 1
                tot_cpu += real_cpu
                if F - set(keep):
                    cv += 300000
                    if not F - set(S):
                        misses += 1  # a new miss caused by verification
                if a.v:
                    nar = {id_of[n].split("::")[-1]: len(u) for n, u in cunits.items() if n in V}
                    print(f"{m['commit']} S={len(S)} V={len(V)} F={len(F)} static={cs/1000:.1f} "
                          f"-> {cv/1000:.1f} (cpu {real_cpu:.1f}, planned gain {gain/1000:.1f}) "
                          f"oracle={co/1000:.1f} {nar or ''}")
        if a.v and cv == cs and cs - co > 8000:
            print(f"{m['commit']} NOT VERIFIED S={len(S)} F={len(F)} static={cs/1000:.1f} oracle={co/1000:.1f}")
        tot_s += cs
        tot_v += cv
        tot_o += co
    print(f"{a.data.name}: static {tot_s/1000:.1f}s -> verified {tot_v/1000:.1f}s "
          f"(verify cpu {tot_cpu:.1f}s, {nver} commits verified, misses {misses}); oracle {tot_o/1000:.1f}s")


if __name__ == "__main__":
    main()
