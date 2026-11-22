"""Entry logic: static selection (blast.py), then optional verification (verify.py).

1. Static candidates S for every commit, exactly as in phase 3.
2. Plan: for each commit, the verification set V is S minus its m most expensive
   tests (m = 0..40), chosen to maximise
        gain = pack(S) - pack(S - V) - cpu(V) - overhead(V)
   using per-test CPU measured on the pinned commit (maps/vcpu.json). Commits whose
   best gain is under MARGIN are not verified. Verifying never adds risk beyond the
   harness itself: a test is dropped only if it ran and passed on the patched tree.
3. Verify the planned commits, best gain first, two at a time, within a wall budget.
4. Output S minus the tests that passed, packed (best k in 1..8, LPT).
Any failure in steps 2-3 leaves that commit (or all commits) at S.
"""

import json
import os
import sys
import time

import blast

T_START = time.monotonic()
MARGIN_MS = int(os.environ.get("BR_MARGIN_MS", "300"))
# prior probability that all verified candidates pass (tuned on synthetic patches)
Q_DATA = float(os.environ.get("BR_Q_DATA", "0.1"))   # only data files (fixtures) changed
Q_CODE = float(os.environ.get("BR_Q_CODE", "0.35"))  # any .py change
# the scorer's CPU may be slower than ours: a plan must also beat this fraction of its CPU
MARGIN_REL = float(os.environ.get("BR_MARGIN_REL", "0.15"))
MAX_REST = 40                      # largest number of expensive tests kept unverified
BASE_OVERHEAD_S = 0.10             # fork, overlay, import sqlglot (precompiled)
IMPORT_FLOOR_S = 0.048             # the smallest per-file overhead = the shared bare import
PARALLEL = int(os.environ.get("BR_PARALLEL", "2"))
BUDGET_PER_COMMIT_S = float(os.environ.get("BR_BUDGET_PER_COMMIT", "2.25"))  # limit: about 4.5 s wall per commit
BUDGET_CAP_S = float(os.environ.get("BR_BUDGET_CAP", "150"))


# A test running longer than this (wall) counts as failed (kept selected): never unsafe, but a
# timeout costs its whole budget in CPU. The labellers' limit is 90 s per test (about 8x the
# cached tpcds CPU), so a slow-down they would see as a timeout still trips this one.
TIMEOUT_MULT = float(os.environ.get("BR_TIMEOUT_MULT", "2.0"))
TIMEOUT_ADD_S = 3.0


def timeout_s(cpu: float) -> float:
    return TIMEOUT_MULT * cpu + TIMEOUT_ADD_S


def log(*a) -> None:
    print("[runner]", *a, file=sys.stderr, flush=True)


def pack_cost(durs: list[int]) -> int:
    if not durs:
        return 0
    durs = sorted(durs, reverse=True)
    best = None
    for k in range(1, min(blast.MAX_WORKERS, len(durs)) + 1):
        loads = [0] * k
        for d in durs:
            i = loads.index(min(loads))
            loads[i] += d
        c = max(loads) + blast.WORKER_MS * k
        if best is None or c < best:
            best = c
    return best


def overhead(files, file_ovh: dict) -> float:
    """Child CPU besides the tests: base + each test file's own imports and class fixtures
    (measured per file on the clean tree, minus the shared import of sqlglot)."""
    return BASE_OVERHEAD_S + sum(max(0.0, file_ovh.get(f, 0.3) - IMPORT_FLOOR_S) for f in files)


def pack_estimate(total: int, longest: int, count: int) -> int:
    """Lower bound of the packed cost (max(longest, total/k) + 2 s * k, best k): O(1),
    close to LPT for these sets. Used only for planning; the output uses blast.pack."""
    if count <= 0:
        return 0
    best = None
    for k in range(1, min(blast.MAX_WORKERS, count) + 1):
        c = max(longest, -(-total // k)) + blast.WORKER_MS * k
        if best is None or c < best:
            best = c
    return best


def plan_commit(S: list[int], dur: dict, cpu: dict, file_ovh: dict, file_of: dict,
                q: float = 1.0):
    """Best (expected gain ms, V) for candidate numbers S; V == [] means do not verify.

    Candidates are ordered by verification CPU per ms of selected duration (best value
    first) and every prefix V is scored:
        gain = pack(S) - E[pack after] - (1 + MARGIN_REL) * (cpu(V) + overhead(V))
    with E[pack after] = qv * pack(S - V) + (1 - qv) * pack(S - V + cheapest test of V):
    q is the prior that all of S passes, spread evenly over the candidates
    (qv = q ** (|V| / |S|)); if V does not all pass, at least one test of V stays."""
    if not S:
        return 0, []
    n = len(S)
    def value(x):  # CPU to verify (incl. its file's own overhead) per ms of duration
        fo = max(0.0, file_ovh.get(file_of[x], 0.3) - IMPORT_FLOOR_S)
        return ((cpu[x] + fo) / max(dur[x], 1), cpu[x], x)

    order = sorted(S, key=value)
    durs = [dur[x] for x in order]
    total = sum(durs)
    base = pack_estimate(total, max(durs), n)
    suffix_max = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_max[i] = max(suffix_max[i + 1], durs[i])
    best_gain, best_len = 0, 0
    pre_d = 0
    pre_cpu = 0.0
    min_v = None
    files: set = set()
    ovh = BASE_OVERHEAD_S
    for i, x in enumerate(order):
        pre_d += durs[i]
        pre_cpu += cpu[x]
        min_v = durs[i] if min_v is None else min(min_v, durs[i])
        f = file_of[x]
        if f not in files:
            files.add(f)
            ovh += max(0.0, file_ovh.get(f, 0.3) - IMPORT_FLOOR_S)
        rest_n = n - i - 1
        rest_total = total - pre_d
        clean = pack_estimate(rest_total, suffix_max[i + 1], rest_n)
        dirty = min(base, pack_estimate(rest_total + min_v, max(suffix_max[i + 1], min_v),
                                        rest_n + 1))
        qv = q ** ((i + 1) / n)  # failures spread evenly over the candidates
        exp_after = qv * clean + (1 - qv) * dirty
        gain = int(base - exp_after - 1000 * (1 + MARGIN_REL) * (pre_cpu + ovh))
        if gain > best_gain:
            best_gain, best_len = gain, i + 1
    return best_gain, order[:best_len]


def worth_continuing(job_failed: set, job_done: set, V: list, rest_d: list, dur: dict,
                     ccpu: dict) -> bool:
    """After a failure inside a job: is verifying the remaining tests still worth their CPU?
    (Dropping them only helps if they would all pass; failures tend to cluster: q = 0.5.)"""
    if not job_failed:
        return True
    left = [n for n in V if n not in job_done]
    if not left:
        return True
    kept = rest_d + [dur[n] for n in job_failed]
    now = pack_cost(kept + [dur[n] for n in left])
    best = pack_cost(kept)
    return 0.5 * (now - best) > 1000 * sum(ccpu[n] for n in left) + MARGIN_MS


# ---- phase 5c: cheap tests as evidence for expensive ones
EV_CAP = float(os.environ.get("BR_EV_CAP", "0"))      # max CPU (s) of an evidence test; 0 = off
EV_K = int(os.environ.get("BR_EV_K", "3"))            # min number of evidence tests
EV_MODE = os.environ.get("BR_EV_MODE", "cov")         # which unverified tests may be dropped
EV_COVER = os.environ.get("BR_EV_COVER", "1") == "1"  # evidence must run every changed line
EV_Q = float(os.environ.get("BR_EV_Q", "0"))          # P(evidence all passes); 0 = prior_q


def evidence_plan(S, V, direct: set, noncov: set, ccpu: dict, line_tests: list | None = None):
    """(E, D): evidence tests E (cheap tests that execute a changed line or read a changed
    attribute/key) and the tests D that may be dropped WITHOUT running them if every test
    of E (and of V) passes on the patched tree. ([], []) when the rule does not apply.
      cov mode: D = unverified tests selected only through runtime line coverage
      all mode: D = every unverified test
    line_tests: [(tests executing changed line 1), ...]; with EV_COVER every changed line
    that a test of D executes must also be executed by a test of E."""
    if EV_CAP <= 0:
        return [], []
    Vs = set(V)
    E = sorted(n for n in S if n in direct and ccpu.get(n, 1e9) <= EV_CAP)
    if len(E) < EV_K:
        return [], []
    Es = set(E)
    rest = [n for n in S if n not in Vs and n not in Es]
    if EV_MODE == "cov":
        D = [n for n in rest if n not in noncov]
    else:
        D = rest
    if not D:
        return [], []
    if EV_COVER and line_tests is not None:
        Ds = set(D)
        for lt in line_tests:
            if lt & Ds and not lt & Es:
                return [], []
    return E, D


def evidence_worth(S, V, E, D, dur: dict, ccpu: dict, file_of: dict, file_ovh: dict,
                   q: float) -> bool:
    """Is running E (besides V) worth it for the chance (q) of dropping D?"""
    Vs = set(V)
    extra = [n for n in E if n not in Vs]
    files_v = {file_of[n] for n in V}
    files_ve = files_v | {file_of[n] for n in extra}
    cpu = sum(ccpu[n] for n in extra) + overhead(files_ve, file_ovh) - (
        overhead(files_v, file_ovh) if V else 0.0)
    gone = Vs | set(E) | set(D)
    before = pack_cost([dur[n] for n in S if n not in Vs])
    after = pack_cost([dur[n] for n in S if n not in gone])
    q = EV_Q or q
    return q * (before - after) > 1000 * (1 + MARGIN_REL) * cpu + MARGIN_MS


def prior_q(files) -> float:
    """Prior that the candidates all pass: fixture corruptions nearly always break their
    reader, code edits often do not (equivalent or uncovered mutants)."""
    if files and all(not (fc.path or "").endswith(".py") for fc in files):
        return Q_DATA
    return Q_CODE


def main() -> None:
    data = blast.DATA
    tests = json.loads((data / "tests.json").read_text())["tests"]
    num_by_id, dur, id_of = {}, {}, {}
    for t in tests:
        num_by_id[t["id"]] = t["number"]
        id_of[t["number"]] = t["id"]
        dur[t["number"]] = blast.duration_ms(t.get("seconds", 0))
    all_numbers = sorted(dur)
    patches = sorted((data / "commits").glob("*.patch"))

    # ---- 1. static selection
    try:
        maps = blast.Maps(blast.MAPS)
        sel = blast.Selector(maps)
        bit_num = [num_by_id.get(tid) for tid in maps.tests]
        unknown = set(dur) - {n for n in bit_num if n is not None}
        if unknown:
            log(f"{len(unknown)} tests not in maps: always selected")
    except Exception as e:  # noqa: BLE001
        log(f"maps unavailable ({e!r}): selecting everything")
        maps = sel = None
        bit_num, unknown = [], set(all_numbers)

    static: dict[str, list[int]] = {}
    texts: dict[str, str] = {}
    info: dict[str, tuple[int, list]] = {}  # commit -> (non-coverage bits, coverage lines)
    probes: dict[str, list[str]] = {}      # commit -> modules to import-check (attribute rule)
    wide: dict[str, list[int]] = {}        # commit -> selection without the attribute rule

    def to_nums(bits: int) -> list[int]:
        nums = set(unknown) if bits else set()
        x = bits
        while x:
            low = x & -x
            i = low.bit_length() - 1
            if i < len(bit_num) and bit_num[i] is not None:
                nums.add(bit_num[i])
            x ^= low
        return sorted(nums)
    for patch in patches:
        commit = patch.stem
        why: list[str] = []
        try:
            if sel is None:
                nums = all_numbers
            else:
                text = patch.read_text(encoding="utf-8", errors="replace")
                texts[commit] = text
                bits = sel.select(text, why)
                info[commit] = (sel.noncov, sel.covlines)
                if EV_CAP > 0:
                    lt = []
                    for f, ln in set(sel.covlines):
                        pf = maps.file(f)
                        si = pf["cov"].get(str(ln)) if pf else None
                        if si is not None:
                            lt.append(set(to_nums(maps.s(pf, si))))
                    info[commit] += (set(to_nums(sel.direct)), set(to_nums(sel.noncov)), lt)
                nums = to_nums(bits)
                if sel.probe_files:
                    # the import-time rules assume the patched modules still import: keep the
                    # widened selection for when that cannot be checked
                    probes[commit] = [module_name(f) for f in sel.probe_files]
                    sel.assume_imports = False
                    try:
                        wide[commit] = to_nums(sel.select(text, []))
                    finally:
                        sel.assume_imports = True
        except Exception as e:  # noqa: BLE001
            why.append(f"error {e!r} -> all")
            nums = all_numbers
        static[commit] = nums
        log(f"{commit}: static {len(nums)} tests | " + "; ".join(why)[:200])

    # ---- 2.-3. verification
    final = dict(static)
    verified: dict = {}
    if os.environ.get("BR_VERIFY", "1") != "0" and patches:
        try:
            verified = verify_commits(static, texts, dur, id_of, num_by_id, len(patches),
                                      maps, info, probes)
        except Exception as e:  # noqa: BLE001
            log(f"verification disabled: {e!r}")
            verified = {}
    for commit, (passed, _) in verified.items():
        final[commit] = [n for n in static[commit] if n not in passed]
    for commit in probes:
        passed, ok = verified.get(commit, (set(), None))
        if ok is False:
            log(f"{commit}: patched module fails to import -> all tests")
            final[commit] = all_numbers
        elif ok is None:
            log(f"{commit}: import not checked -> wide selection")
            final[commit] = [n for n in wide.get(commit, all_numbers) if n not in passed]

    # ---- 4. output
    out = []
    for patch in patches:
        commit = patch.stem
        try:
            workers = blast.pack([(n, dur[n]) for n in final[commit]])
        except Exception:  # noqa: BLE001
            workers = blast.pack([(n, dur[n]) for n in all_numbers])
        out.append(json.dumps({"commit": commit, "workers": workers}, separators=(",", ":")))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))
    sys.stdout.flush()
    log(f"done in {time.monotonic() - T_START:.1f}s wall, main process cpu "
        f"{time.process_time():.2f}s")


# The executor tests' measured CPU (with the phase 6c caches) is scaled up for PLANNING only:
# their failures are the costliest to verify (slow-downs run into the time limit), and the
# measured cost alone made the planner verify them too eagerly. Tuned on calib + synth,
# checked on holdout + confirm (CHANGELOG.md, Phase 6e).
EXEC_PLAN_SCALE = float(os.environ.get("BR_EXEC_PLAN_SCALE", "1.25"))
EXEC_FILE = "tests/test_executor.py"


def scale_executor(vc: dict, k: float) -> dict:
    vc = dict(vc)
    pre = EXEC_FILE + "::"
    vc["cpu"] = {t: (c * k if t.startswith(pre) else c) for t, c in vc["cpu"].items()}
    vc["units"] = {t: ([None if x is None else x * k for x in u] if t.startswith(pre) else u)
                   for t, u in vc.get("units", {}).items()}
    vc["file_overhead"] = {f: (c * k if f == EXEC_FILE else c)
                           for f, c in vc["file_overhead"].items()}
    return vc


class Units:
    """Per-unit coverage of the split tests (maps/units.bin, verify.SPLITS)."""

    def __init__(self, maps, vc: dict, splits: dict):
        import zlib
        self.maps = maps
        try:
            raw = json.loads(zlib.decompress((blast.MAPS / "units.bin").read_bytes()))
        except Exception as e:  # noqa: BLE001
            log(f"units unavailable: {e!r}")
            raw = {}
        ucpu = vc.get("units", {})
        self.whole = vc.get("cpu", {})
        self.tests = {}
        index = {t: i for i, t in enumerate(maps.tests)}
        for tid, d in raw.items():
            c = ucpu.get(tid)
            if (tid in splits and d.get("ok") and tid in index and c and len(c) == d["n"]
                    and all(x is not None for x in c)):
                self.tests[tid] = (index[tid], d, c)

    def reachable(self, tid: str, noncov: int, covlines: list):
        """(unit indices, cpu) for this commit, or None when the whole test must run."""
        if tid not in self.tests:
            return None
        bit, d, ucpu = self.tests[tid]
        if noncov >> bit & 1:
            return None                      # selected through class/module/data users
        mask = 0
        found = False
        for f, ln in covlines:
            p = self.maps.file(f)
            si = p["cov"].get(str(ln)) if p else None
            if si is None or not (self.maps.s(p, si) >> bit & 1):
                continue
            found = True
            ub = d["lines"].get(f, {}).get(str(ln))
            if ub is None:
                return None                  # executed by the full run but by no single unit
            mask |= int(ub, 16)
        if not found or mask == (1 << d["n"]) - 1:
            return None
        units = [u for u in range(d["n"]) if mask >> u & 1]
        # units measured alone are cold (sum > whole test); scale to the whole test's CPU
        whole = float(self.whole.get(tid, sum(ucpu)))
        return units, whole * sum(ucpu[u] for u in units) / max(sum(ucpu), 1e-9)


class Planner:
    """Per-commit verification plan: which candidates to run (and which units of split tests)."""

    def __init__(self, dur: dict, id_of: dict, num_by_id: dict, maps, splits: dict,
                 vc: dict | None = None, plan_scale: bool = True):
        if vc is None:
            vc = json.loads((blast.MAPS / "vcpu.json").read_text())
        self._args = (dur, id_of, num_by_id, maps, splits, vc)
        self._nocache = None
        if plan_scale and EXEC_PLAN_SCALE != 1.0:
            vc = scale_executor(vc, EXEC_PLAN_SCALE)
        self.dur = dur
        self.id_of = id_of
        self.file_ovh = vc["file_overhead"]
        self.cpu = {n: float(vc["cpu"].get(id_of[n], max(0.05, dur[n] / 1000))) for n in dur}
        self.file_of = {n: id_of[n].split("::")[0] for n in dur}
        self.units = Units(maps, vc, splits) if maps is not None else None
        self.split_nums = {num_by_id[t]: t for t in (self.units.tests if self.units else {})
                           if t in num_by_id}

    def without_cache(self) -> "Planner":
        """The planner for a commit whose executor tests run WITHOUT the phase 6c caches
        (exec_cache.guard_ok fails, or no cache): their measured uncached CPU (vcpu.json
        "nocache"), unscaled, for both the plan and the per-unit time limits."""
        if self._nocache is None:
            dur, id_of, num_by_id, maps, splits, vc = self._args
            nc = vc.get("nocache") or {}
            vc = dict(vc)
            for key in ("cpu", "units", "file_overhead"):
                vc[key] = {**vc.get(key, {}), **nc.get(key, {})}
            self._nocache = Planner(dur, id_of, num_by_id, maps, splits, vc, plan_scale=False)
        return self._nocache

    def unit_limits(self, n: int, units) -> list | None:
        """Per-unit time limits (s) for a split test whose units report progress (verify
        PROGRESS): a slow unit fails the test early instead of after its whole budget."""
        tid = self.split_nums.get(n)
        if tid is None or tid not in self.units.tests:
            return None
        ucpu = self.units.tests[tid][2]
        order = list(units) if units is not None else list(range(len(ucpu)))
        lim = [TIMEOUT_MULT * float(ucpu[u]) + 1.0 for u in order]
        if lim:
            lim[-1] += 2.0 + 0.1 * len(lim)  # the result comparisons after the last unit
        return lim

    def plan(self, S: list[int], files: list, info) -> tuple[int, list, dict, dict]:
        """(expected gain ms, V, cpu estimates, {number: unit indices})."""
        ccpu, cunits = self.cpu, {}
        for n in self.split_nums.keys() & set(S):
            r = self.units.reachable(self.split_nums[n], *(info or (-1, []))[:2])
            if r is not None:
                if ccpu is self.cpu:
                    ccpu = dict(self.cpu)
                cunits[n], ccpu[n] = r
        gain, V = plan_commit(S, self.dur, ccpu, self.file_ovh, self.file_of, prior_q(files))
        return gain, sorted(V), ccpu, cunits


def module_name(f: str) -> str:
    m = f[:-3].replace("/", ".")
    return m[: -len(".__init__")] if m.endswith(".__init__") else m


def verify_commits(static, texts, dur, id_of, num_by_id, n_commits, maps=None,
                   info=None, probes=None) -> dict[str, tuple[set[int], bool | None]]:
    """{commit: (numbers of tests that ran and passed on the patched tree,
                 import probe result: True / False / None = not checked)}."""
    import verify
    if not (verify.SRC / "sqlglot" / "__init__.py").is_file():
        log(f"no checkout at {verify.SRC}: static only")
        return {}
    import exec_cache
    planner = Planner(dur, id_of, num_by_id, maps, verify.SPLITS)
    cache_ok = exec_cache.available()
    info = info or {}
    probes = probes or {}
    plans = []
    for commit, S in static.items():
        if commit not in texts or (not S and commit not in probes):
            continue
        try:
            files = blast.parse_patch(texts[commit])
            if any(not p.startswith(verify.VERIFY_ROOTS)
                   for fc in files for p in (fc.old, fc.new) if p is not None):
                continue
            inf = info.get(commit)
            # the executor caches apply only when the patch leaves their inputs alone
            exec_ok = cache_ok and exec_cache.guard_ok(
                [q for fc in files for q in (fc.old, fc.new) if q is not None])
            pl = planner if exec_ok else planner.without_cache()
            gain, V, ccpu, cunits = pl.plan(S, files, inf) if S else (0, [], {}, {})
            if gain < MARGIN_MS or not V:
                gain, V = 0, []
            E, D = [], []
            if S and inf is not None and len(inf) > 2:
                E, D = evidence_plan(S, V, inf[2], inf[3], ccpu, inf[4])
                if E and not evidence_worth(S, V, E, D, dur, ccpu, pl.file_of,
                                            pl.file_ovh, prior_q(files)):
                    E, D = [], []
                if E:
                    V = sorted(set(V) | set(E))
            if not V and commit not in probes:
                continue  # (else: import probe only)
            changes = verify.patched_files(files)
        except Exception as e:  # noqa: BLE001
            log(f"{commit}: not verifiable ({e})")
            continue
        V.sort(key=lambda n: (ccpu[n], n))  # cheapest first: failures show up early
        tl = [(id_of[n], timeout_s(ccpu[n]), cunits.get(n), pl.unit_limits(n, cunits.get(n)))
              for n in V]
        for n in V:
            if n in cunits:
                log(f"{commit}: {id_of[n].split('::')[-1]} narrowed to {len(cunits[n])} units")
        job = verify.Job(commit, changes, tl)
        rest_d = [dur[n] for n in S if n not in set(V)]

        def check(job, V=V, rest_d=rest_d, ccpu=ccpu):
            failed = {num_by_id[t] for t in job.failed if t in num_by_id}
            done = failed | {num_by_id[t] for t in job.passed if t in num_by_id}
            return worth_continuing(failed, done, V, rest_d, dur, ccpu)

        job.on_result = check
        job.probe = probes.get(commit, [])
        if any(t[0].startswith(EXEC_FILE + "::") for t in tl):
            job.exec_ok = exec_ok
        job.evidence = (E, D)
        plans.append((gain, commit, job))
    if not plans:
        return {}
    # import probes first (cheap, and they unlock the attribute rule), then best gain first
    plans.sort(key=lambda p: (not p[2].probe or bool(p[2].tests), -p[0], p[1]))
    if sys.flags.no_site:
        import site
        site.main()  # main.py runs with -S; the tests need site-packages
    try:
        preload = json.loads((blast.MAPS / "preload.json").read_text())
    except Exception:  # noqa: BLE001
        preload = []
    if any(p[2].exec_ok for p in plans):
        import exec_cache
        c0 = time.process_time()
        ok = exec_cache.load()  # before the forks: the children share it
        log(f"executor cache {'loaded' if ok else 'unavailable'} in "
            f"{time.process_time() - c0:.2f}s cpu")
        if not ok:  # planned with cached costs but running without: no unit limits, 2x time
            for _, _, job in plans:
                if job.exec_ok:
                    job.exec_ok = False
                    job.tests = [(t[0], 2 * t[1], t[2], None) if t[0].startswith(EXEC_FILE + "::")
                                 else t for t in job.tests]
    v = verify.Verifier(parallel=PARALLEL, preload=preload)
    budget = min(BUDGET_CAP_S, BUDGET_PER_COMMIT_S * n_commits)
    deadline = T_START + budget
    jobs = [p[2] for p in plans]
    log(f"verifying {len(jobs)} commits, planned gain {sum(p[0] for p in plans) / 1000:.1f}s, "
        f"budget {budget:.0f}s")
    v.run(jobs, deadline)
    out = {}
    for gain, commit, job in plans:
        passed = {num_by_id[t] for t in job.passed if t in num_by_id} - \
                 {num_by_id[t] for t in job.failed if t in num_by_id}
        E, D = job.evidence
        if E and all(n in passed for n in E):
            # phase 5c: every evidence test passed -> drop D without running it
            log(f"{commit}: evidence {len(E)} tests passed -> dropping {len(D)} unverified: "
                + ", ".join(id_of[n].split("::")[-1] for n in D)[:200])
            passed = passed | set(D)
        out[commit] = (passed, job.probe_ok if job.probe else None)
        log(f"{commit}: verified {len(job.tests)} -> passed {len(passed)}, failed {len(job.failed)}, "
            f"cpu {job.child_cpu:.2f}s tests {job.cpu:.2f}s (planned gain {gain / 1000:.1f}s){job.why}")
    log(f"verification cpu {v.cpu_children:.1f}s")
    log_components(plans, id_of, num_by_id)
    return out


def log_components(plans, id_of, num_by_id) -> None:
    """Phase 6a: verification CPU by component, summed over all jobs (stderr only)."""
    from collections import defaultdict
    cat = defaultdict(float)
    by_file = defaultdict(float)
    by_test = defaultdict(float)
    for _, commit, job in plans:
        known = 0.0
        for k, c in job.comp.items():
            kind, _, f = k.partition(":")
            cat[kind] += c
            known += c
            if f:
                by_file[f"{kind}:{f.split('/')[-1]}"] += c
        for tid, c in job.tcpu.items():
            by_test[tid.split("::")[-1]] += c
            known += c
        cat["tests"] += sum(job.tcpu.values())
        cat["untracked"] += max(0.0, job.child_cpu - known)  # fork/exit, stopped tails
        cat["jobs"] += 1
    log("cpu components: " + ", ".join(f"{k} {v:.1f}" for k, v in sorted(cat.items())))
    log("cpu top files: " + ", ".join(f"{k} {v:.1f}" for k, v in
                                      sorted(by_file.items(), key=lambda x: -x[1])[:10]))
    log("cpu top tests: " + ", ".join(f"{k} {v:.1f}" for k, v in
                                      sorted(by_test.items(), key=lambda x: -x[1])[:10]))
