"""Evaluate a selector on a set of labelled patches.

  # run the submission image under the scorer's flags (default)
  python3 tools/eval.py --data sample --answers sample/answers.json --image blast-radius-entry
  # or run a local command; it must read $BR_DATA instead of /data
  python3 tools/eval.py --data sample --answers sample/answers.json --cmd "python3 selector.py"
  # or just score a saved output
  python3 tools/eval.py --data sample --answers sample/answers.json --output output.jsonl

--data is a dir with tests.json and commits/*.patch. Patches without an answer
are dropped (a temporary data dir is assembled). Reports simulated seconds,
misses (and which tests were missed), CPU, the total, and the oracle cost
(exact failing set, best packing) for reference.
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import score  # noqa: E402

RUN_FLAGS = ("--rm --platform linux/amd64 --network none --cpus 2 --memory 4096m "
             "--memory-swap 4096m --pids-limit 256 --read-only "
             "--tmpfs /tmp:rw,size=64m,mode=1777 --cap-drop ALL "
             "--security-opt no-new-privileges --user 65534:65534 --env HOME=/tmp").split()


def ms(sec: float) -> int:
    return score._duration_milliseconds(sec)


def best_pack_cost(durations: list[int]) -> int:
    if not durations:
        return 0
    durations = sorted(durations, reverse=True)
    best = None
    for k in range(1, min(8, len(durations)) + 1):
        loads = [0] * k
        for d in durations:
            i = loads.index(min(loads))
            loads[i] += d
        c = max(loads) + 2000 * k
        best = c if best is None or c < best else best
    return best


def image_cmd(image: str) -> list[str]:
    out = subprocess.run(["docker", "inspect", "--format",
                          "{{json .Config.Entrypoint}}|{{json .Config.Cmd}}", image],
                         capture_output=True, text=True, check=True).stdout.strip()
    ep, cmd = (json.loads(x) or [] for x in out.split("|"))
    return ep + cmd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--answers", type=Path, required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image")
    g.add_argument("--cmd")
    g.add_argument("--output", type=Path)
    ap.add_argument("--save", type=Path, help="save the selector's stdout here")
    ap.add_argument("-v", "--verbose", action="store_true", help="per-commit table")
    a = ap.parse_args()

    tests = json.loads((a.data / "tests.json").read_text())["tests"]
    answers = json.loads(a.answers.read_text())["commits"]
    dur = {t["number"]: ms(t["seconds"]) for t in tests}
    patches = {p.stem: p for p in (a.data / "commits").glob("*.patch") if p.stem in answers}
    answers = {k: v for k, v in answers.items() if k in patches}
    if not answers:
        sys.exit("no labelled patches")

    cpu = None
    stderr = ""
    if a.output:
        stdout = a.output.read_bytes()
    else:
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            (data / "commits").mkdir(parents=True)
            os.chmod(tmp, 0o755)
            os.chmod(data, 0o755)
            os.chmod(data / "commits", 0o755)
            for k, p in patches.items():
                (data / "commits" / f"{k}.patch").write_bytes(p.read_bytes())
            (data / "tests.json").write_bytes((a.data / "tests.json").read_bytes())
            if a.image:
                cmd = ["docker", "run", *RUN_FLAGS,
                       "-v", f"{data / 'commits'}:/data/commits:ro",
                       "-v", f"{data / 'tests.json'}:/data/tests.json:ro",
                       "-v", f"{ROOT / 'tools' / 'cpuwrap.py'}:/cpuwrap.py:ro",
                       "--entrypoint", "python", a.image, "/cpuwrap.py", *image_cmd(a.image)]
                env = None
            else:
                cmd = shlex.split(a.cmd)
                env = dict(os.environ, BR_DATA=str(data))
            import resource
            r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
            pr = subprocess.run(cmd, capture_output=True, env=env)
            r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
            stdout, stderr = pr.stdout, pr.stderr.decode(errors="replace")
            if a.cmd:
                cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
            m = re.search(r"@@CPUWRAP cpu=([\d.]+) wall=([\d.]+) maxrss_kb=(\d+)", stderr)
            if m:
                cpu = float(m.group(1))
                print(f"container: cpu {cpu:.3f}s wall {m.group(2)}s maxrss {int(m.group(3))//1024} MB",
                      file=sys.stderr)
            if pr.returncode:
                print(f"selector exited {pr.returncode}\n{stderr[-3000:]}", file=sys.stderr)
    if a.save:
        a.save.write_bytes(stdout)

    ev = score.evaluate_run(stdout, tests, answers)
    if not ev["ok"]:
        print(f"REJECTED: {ev['code']} line {ev['line']}")
        sys.exit(1)

    lines = {json.loads(l)["commit"]: json.loads(l)["workers"]
             for l in stdout.decode().splitlines()}
    oracle = sum(best_pack_cost([dur[n] for n in v["failing"]]) for v in answers.values())
    sel_total = ev["simulated_milliseconds"] - 300_000 * len(ev["missed_commits"])
    if a.verbose:
        print(f"{'commit':8} {'kind':8} {'fail':>4} {'sel':>5} {'k':>2} {'cost':>8} {'oracle':>8}")
    for c in sorted(answers):
        ws = [w for w in lines[c] if w]
        sel = {n for w in ws for n in w}
        cost = (max(sum(dur[n] for n in w) for w in ws) + 2000 * len(ws)) if ws else 0
        fail = answers[c]["failing"]
        miss = sorted(set(fail) - sel)
        if a.verbose or miss:
            print(f"{c:8} {answers[c].get('kind', ''):8} {len(fail):4} {len(sel):5} {len(ws):2} "
                  f"{cost / 1000:8.3f} {best_pack_cost([dur[n] for n in fail]) / 1000:8.3f}"
                  + (f"  MISSED {miss[:10]}{'...' if len(miss) > 10 else ''}" if miss else ""))
    n = len(answers)
    print(f"commits {n}  misses {len(ev['missed_commits'])}  "
          f"selected {sel_total / 1000:.3f}s  ({sel_total / 1000 / n:.2f}/commit)  "
          f"oracle {oracle / 1000:.3f}s")
    total = ev["simulated_milliseconds"] / 1000 + (cpu or 0)
    print(f"simulated {ev['simulated_milliseconds'] / 1000:.3f}s  cpu "
          f"{'n/a' if cpu is None else f'{cpu:.3f}s'}  TOTAL {total:.3f}s")
    if ev["missed_commits"]:
        print("missed commits:", " ".join(ev["missed_commits"]))


if __name__ == "__main__":
    main()
