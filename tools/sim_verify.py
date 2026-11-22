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
    ap.add_argument("--vcpu", type=Path, default=ROOT / "maps" / "vcpu.json",
                    help="CPU estimates the planner decides with")
    ap.add_argument("--real-vcpu", type=Path,
                    help="CPU actually charged for tests that run (default: --vcpu)")
    ap.add_argument("--margin", type=int, default=runner.MARGIN_MS)
    ap.add_argument("--q-data", type=float)
    ap.add_argument("--rel", type=float)
    ap.add_argument("--q-code", type=float)
    ap.add_argument("--cpu-scale", type=float, default=1.0, help="scorer CPU / our CPU")
    ap.add_argument("--ev-cap", type=float, help="phase 5c: evidence CPU cap (s); 0 = off")
    ap.add_argument("--ev-k", type=int)
    ap.add_argument("--ev-mode", choices=["cov", "all"])
    ap.add_argument("--ev-cover", type=int, choices=[0, 1])
    ap.add_argument("--ev-data", action="store_true", help="readers of a changed data file count as evidence")
    ap.add_argument("--ev-always", action="store_true", help="skip the worth check (analysis)")
    ap.add_argument("--report", type=Path, help="write per-commit 5c records (JSON lines)")
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    if a.ev_cap is not None:
        runner.EV_CAP = a.ev_cap
    if a.ev_k is not None:
        runner.EV_K = a.ev_k
    if a.ev_mode is not None:
        runner.EV_MODE = a.ev_mode
    if a.ev_cover is not None:
        runner.EV_COVER = bool(a.ev_cover)
    rep = a.report.open("w") if a.report else None
    ev_applied = ev_miss = ev_fired = 0
    ev_saving = 0
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
    planner = runner.Planner(dur, id_of, num_by_id, maps, verify.SPLITS,
                             json.loads(a.vcpu.read_text()))
    real = runner.Planner(dur, id_of, num_by_id, maps, verify.SPLITS,
                          json.loads((a.real_vcpu or a.vcpu).read_text()), plan_scale=False)
    bn = [num_by_id.get(t) for t in maps.tests]
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
            if not (V and gain >= a.margin):
                V = []
            nums = lambda bits: {bn[i] for i in range(len(bn)) if bits >> i & 1 and bn[i] is not None}
            direct = nums(sel.direct | (sel.databits if a.ev_data else 0))
            noncov = nums(sel.noncov)
            line_tests = []
            for f, ln in set(sel.covlines):
                pf = maps.file(f)
                si = pf["cov"].get(str(ln)) if pf else None
                if si is not None:
                    line_tests.append(nums(maps.s(pf, si)))
            E, D = runner.evidence_plan(S, V, direct, noncov, ccpu, line_tests)
            if E and not a.ev_always and not runner.evidence_worth(
                    S, V, E, D, dur, ccpu, planner.file_of, planner.file_ovh,
                    runner.prior_q(runner.blast.parse_patch(text))):
                E, D = [], []
            outcome = set(m["passed"])
            if E:
                ev_applied += 1
                e_ok = all(n in outcome for n in E)
                base_keep = [n for n in S if not (n in set(V) and n in outcome)]
                if e_ok:
                    ev_fired += 1
                    keep = [n for n in base_keep if n not in set(D) and not (n in set(E) and n in outcome)]
                else:
                    keep = [n for n in base_keep if not (n in set(E) and n in outcome)]
                e_cpu = sum(ccpu[n] for n in E if n not in set(V))
                files = {planner.file_of[n] for n in set(V) | set(E)}
                v_cpu = sum(ccpu[n] for n in V)
                cpu5 = (e_cpu + v_cpu + runner.overhead(files, planner.file_ovh)) * a.cpu_scale
                cv = runner.pack_cost([dur[n] for n in keep]) + int(1000 * cpu5)
                missed = bool(F - set(keep)) and not (F - set(S))
                if F - set(S):
                    cv += 300000  # static miss
                if missed:
                    cv += 300000
                    ev_miss += 1
                    misses += 1
                # what phase 4 alone would have cost (V only, all of V run)
                k4 = [n for n in S if not (n in set(V) and n in outcome)]
                c4 = (300000 if F - set(S) else 0) + runner.pack_cost([dur[n] for n in k4]) + (int(1000 * a.cpu_scale * (
                    v_cpu + runner.overhead({planner.file_of[n] for n in V}, planner.file_ovh))) if V else 0)
                ev_saving += c4 - cv
                tot_cpu += cpu5
                nver += 1
                if rep:
                    rep.write(json.dumps({"commit": m["commit"], "kind": m.get("kind"), "E": len(E),
                                          "D": [planner.id_of[n].split("::")[-1] for n in D],
                                          "e_ok": e_ok, "missed": missed, "saving": c4 - cv,
                                          "e_cpu": round(e_cpu, 2)}) + "\n")
                if a.v:
                    print(f"{m['commit']} 5c E={len(E)} D={len(D)} e_ok={e_ok} missed={missed} "
                          f"{c4/1000:.1f} -> {cv/1000:.1f}")
            elif V:
                V = sorted(V, key=lambda n: (ccpu[n], n))
                rest_d = [dur[n] for n in S if n not in set(V)]
                outcome_pass = set(m["passed"])
                passed, failed, done = set(), set(), set()
                real_cpu = 0.0
                why = m.get("why", "")
                rcpu = real.plan(S, runner.blast.parse_patch(text), (sel.noncov, sel.covlines))[2]
                for n in V:   # sequential replay with the runtime's stopping rule
                    ok = n in outcome_pass
                    if not ok and f"timeout:{id_of[n].split('::')[-1]}" in why:
                        # a timeout costs its budget; split tests with per-unit limits stop at
                        # the first slow unit (measured: about 0.2-1.0x the planned CPU)
                        lim = planner.unit_limits(n, cunits.get(n))
                        real_cpu += (0.6 * ccpu[n] + 1.0) if lim else runner.timeout_s(ccpu[n])
                    else:
                        real_cpu += rcpu[n] if (ok or n in cunits) else min(
                            rcpu[n], float(m["tcpu"].get(str(n), rcpu[n])))
                    (passed if ok else failed).add(n)
                    done.add(n)
                    if failed and not runner.worth_continuing(failed, done, V, rest_d, dur, ccpu):
                        break
                keep = [n for n in S if n not in passed]
                real_cpu += runner.overhead({planner.file_of[n] for n in V}, real.file_ovh)
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
    if runner.EV_CAP > 0:
        print(f"  5c: applied {ev_applied}, evidence all passed {ev_fired}, misses {ev_miss}, "
              f"saving vs phase 4 {ev_saving/1000:.1f}s")
    print(f"{a.data.name}: static {tot_s/1000:.1f}s -> verified {tot_v/1000:.1f}s "
          f"(verify cpu {tot_cpu:.1f}s, {nver} commits verified, misses {misses}); oracle {tot_o/1000:.1f}s")


if __name__ == "__main__":
    main()
