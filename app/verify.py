"""Phase 4: run candidate tests against the patched tree inside the container.

The selector's static candidates are safe but wide. For a commit whose candidates
are cheap to run, we apply the patch to an overlay of the pinned checkout in /tmp
and run those tests; tests that pass are dropped, everything else stays selected.

  overlay   /tmp/brv/<n>/ mirrors /opt/sqlglot with symlinks; only directories on
            the path to a changed file are real, and changed files are real copies.
            Unchanged modules load their prebuilt .pyc through the symlinked
            __pycache__ (timestamp-checked, so patched files are recompiled).
  server    this process: imports unittest (so sqlglot's import-time UNITTEST flag
            is set exactly as under pytest) and, when needed, numpy/pandas/duckdb,
            but never sqlglot. One forked child per commit (re-forked after a hang).
  child     puts the overlay first on sys.path, imports the test modules, swaps
            the tests' ProcessPoolExecutors for an inline pool (same pickling, no
            spawned interpreters), and runs the tests class by class through
            unittest, reporting start/end of every test over a pipe.
  verdict   a test is dropped only if the child reported that it ran and passed.
            Timeouts, crashes, import errors, class-fixture errors and anything
            not reported keep the test selected.
"""

import os
import pickle
import select
import shutil
import signal
import sys
import time
from pathlib import Path

SRC = Path(os.environ.get("BR_SRC", "/opt/sqlglot"))
TMP = Path(os.environ.get("BR_TMP", "/tmp")) / "brv"
VERIFY_ROOTS = ("sqlglot/", "tests/")  # patches touching anything else are not verified
HEAVY = ("tests/test_executor.py", "tests/test_optimizer.py")  # import numpy/pandas/duckdb
MAX_RSS_TO_START = 1500 << 20  # do not start a second child while the others use more
                               # (test_max_nodes alone peaks at 1.7 GB; the limit is 4 GB)


# Expensive tests made of independent units (subtests over a list). The maps record
# which units execute each line (tools/trace_map.py traces every unit in its own cold
# child), so verification can run only the units a patch can reach; the other units
# never execute a changed line and behave exactly as on the pinned commit.
#   kind "cls": a list attribute set by setUpClass; "mod": a module-level iterable.
SPLITS = {
    "tests/test_executor.py::TestExecutor::test_execute_tpcds": ("cls", "tpcds_sqls"),
    "tests/test_executor.py::TestExecutor::test_execute_tpch": ("cls", "tpch_sqls"),
    "tests/test_executor.py::TestExecutor::test_optimized_tpch": ("cls", "tpch_sqls"),
    "tests/test_generator.py::TestGenerator::test_generator_memory": ("mod", "Dialects"),
}


def _unit_owner(test, tid: str):
    kind, attr = SPLITS[tid]
    owner = type(test) if kind == "cls" else sys.modules[type(test).__module__]
    return owner, attr


def unit_count(test, tid: str) -> int:
    owner, attr = _unit_owner(test, tid)
    return len(list(getattr(owner, attr)))


def restrict(test, tid: str, units) -> None:
    """Make this test instance run only the given unit indices (in their original order)."""
    import functools
    keep = set(units)
    name = test._testMethodName
    orig = getattr(test, name)

    @functools.wraps(orig)
    def run():
        owner, attr = _unit_owner(test, tid)
        saved = getattr(owner, attr)
        setattr(owner, attr, [x for i, x in enumerate(list(saved)) if i in keep])
        try:
            return orig()
        finally:
            setattr(owner, attr, saved)

    setattr(test, name, run)


def log(*a) -> None:
    print("[verify]", *a, file=sys.stderr, flush=True)


class PatchError(Exception):
    pass


# ------------------------------------------------------------------------- applying

def _find(lines: list[str], seq: list[str], start: int, lo: int) -> int:
    """Index where seq matches lines exactly: at start, else the nearest offset >= lo."""
    n = len(seq)
    if not n:
        return max(start, lo)
    if lines[start:start + n] == seq and start >= lo:
        return start
    for off in range(1, len(lines) + 1):
        for i in (start - off, start + off):
            if lo <= i <= len(lines) - n and lines[i:i + n] == seq:
                return i
        if start - off < lo and start + off > len(lines) - n:
            break
    raise PatchError("hunk does not apply")


def apply_hunks(orig: str, hunks: list) -> str:
    has_nl = orig.endswith("\n")
    lines = orig.split("\n")
    if has_nl or orig == "":
        lines.pop()
    out: list[str] = []
    pos = 0
    final_nl = has_nl or not lines
    for old_start, old_cnt, body, _old_noeol, new_noeol, new_cnt in hunks:
        old_seq = [t for tag, t in body if tag != "+"]
        new_seq = [t for tag, t in body if tag != "-"]
        if len(old_seq) != old_cnt or len(new_seq) != new_cnt:
            raise PatchError("truncated hunk")
        start = old_start - 1 if old_cnt else old_start
        idx = _find(lines, old_seq, max(start, 0), pos)
        out.extend(lines[pos:idx])
        out.extend(new_seq)
        pos = idx + len(old_seq)
        if pos >= len(lines):
            final_nl = not new_noeol
    out.extend(lines[pos:])
    return "\n".join(out) + ("\n" if final_nl and out else "")


def patched_files(files: list, src: Path = SRC) -> dict[str, str | None]:
    """{path: new text, or None if deleted} for parsed FileChanges; PatchError if unsure."""
    out: dict[str, str | None] = {}
    for fc in files:
        if fc.binary:
            raise PatchError("binary")
        old, new = fc.old, fc.new
        for p in (old, new):
            if p is not None and (p.startswith("/") or ".." in p.split("/") or "\0" in p):
                raise PatchError(f"bad path {p!r}")
        if old is not None:
            f = src / old
            if not f.is_file() or f.is_symlink():
                raise PatchError(f"missing {old}")
            text = f.read_bytes().decode("utf-8")
        else:
            text = ""
        if new is None:
            if fc.hunks and apply_hunks(text, fc.hunks) != "":
                raise PatchError(f"deletion of {old} does not match")
            out[old] = None
            continue
        text = apply_hunks(text, fc.hunks) if fc.hunks else text
        out[new] = text
        if old is not None and old != new:
            out[old] = None
    if not out or not any(fc.hunks or fc.old != fc.new for fc in files):
        raise PatchError("no changes")
    return out


def build_overlay(root: Path, changes: dict[str, str | None], src: Path = SRC) -> None:
    real_dirs = set()
    for p in changes:
        parts = p.split("/")
        for i in range(1, len(parts)):
            real_dirs.add("/".join(parts[:i]))
    root.mkdir(parents=True)

    def fill(sdir: Path, ddir: Path, rel: str) -> None:
        names = set()
        if sdir.is_dir():
            for e in os.scandir(sdir):
                names.add(e.name)
                r = rel + e.name
                if r in changes:
                    continue
                if r in real_dirs and e.is_dir(follow_symlinks=False):
                    (ddir / e.name).mkdir()
                    fill(sdir / e.name, ddir / e.name, r + "/")
                else:
                    os.symlink(e.path, ddir / e.name)
        for d in sorted(real_dirs):
            if d.startswith(rel) and "/" not in d[len(rel):] and d[len(rel):] not in names:
                (ddir / d[len(rel):]).mkdir()
                fill(sdir / d[len(rel):], ddir / d[len(rel):], d + "/")

    fill(src, root, "")
    for p, text in changes.items():
        if text is not None:
            (root / p).write_text(text, encoding="utf-8")


# ------------------------------------------------------------------------- child

def cap_duckdb() -> None:
    """duckdb starts `threads` + `async_threads` (4 x cores) threads per connection;
    the scorer allows 256 processes in total. Results do not depend on thread counts."""
    duckdb = sys.modules.get("duckdb")
    if duckdb is None or getattr(duckdb.connect, "_br_capped", False):
        return
    orig = duckdb.connect

    def connect(database=":memory:", read_only=False, config=None, **kw):
        cfg = dict(config or {})
        cfg.setdefault("threads", 1)
        cfg.setdefault("async_threads", 1)
        return orig(database, read_only, config=cfg, **kw)

    connect._br_capped = True
    duckdb.connect = connect


class InlinePool:
    """Drop-in for the tests' ProcessPoolExecutor: runs work in-process, pickling like a pool."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **kw):
        if initializer is not None:
            initializer(*pickle.loads(pickle.dumps(initargs)))

    def submit(self, fn, *args, **kwargs):
        from concurrent.futures import Future
        fut = Future()
        try:
            a, k = pickle.loads(pickle.dumps((args, kwargs)))
            fut.set_result(pickle.loads(pickle.dumps(fn(*a, **k))))
        except BaseException as e:  # noqa: BLE001 - mirror pool semantics
            fut.set_exception(e)
        return fut

    def map(self, fn, *its, **kw):
        return [self.submit(fn, *a).result() for a in zip(*its)]

    def shutdown(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _child(root: Path, tests: list[tuple[int, str, list | None]], wfd: int,
           probe: list | None = None, exec_ok: bool = False) -> None:
    """Run tests [(index, id, units)] from the overlay at root; report over wfd. Never returns.
    probe: modules to import first ("P 1" if they all import, "P 0" if one raises)."""
    code = 0
    try:
        out = os.fdopen(wfd, "w", buffering=1)

        def say(*a):
            out.write(" ".join(map(str, a)) + "\n")

        os.setpgid(0, 0)
        try:  # if a mutant eats memory, the kernel must kill this child, not the selector
            with open("/proc/self/oom_score_adj", "w") as f:
                f.write("1000")
        except OSError:
            pass
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        sys.dont_write_bytecode = True
        os.chdir(root)
        os.environ["PYTHONPATH"] = str(root)  # for tests that start "python -c ..."
        sys.path[:0] = [str(root)]
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        sys.stdout = open(os.devnull, "w")
        sys.stderr = sys.stdout
        import unittest

        # phase 6a: CPU by component ("C name seconds"); the child's clock starts at the fork
        say("C", "start", round(time.process_time(), 4))
        c0 = time.process_time()
        try:
            import sqlglot  # noqa: F401 - the test modules import it first thing anyway
        except BaseException:  # noqa: BLE001 - the test-module imports will fail the same way
            pass
        say("C", "sqlglot", round(time.process_time() - c0, 4))
        if probe:
            import importlib
            c0 = time.process_time()
            ok = 1
            for name in probe:
                try:
                    importlib.import_module(name)
                except BaseException:  # noqa: BLE001
                    ok = 0
            say("C", "probe", round(time.process_time() - c0, 4))
            say("P", ok)

        by_mod: dict[str, list] = {}
        units_of = {}
        for idx, tid, units in tests:
            path, cls, meth = tid.split("::")
            by_mod.setdefault(path, []).append((idx, path, cls, meth))
            if units is not None:
                units_of[idx] = (tid, units)
        for path, items in by_mod.items():  # one block per class (class fixtures run once)
            first = {}
            for k, it in enumerate(items):
                first.setdefault(it[2], k)
            items.sort(key=lambda it: first[it[2]])

        class Result(unittest.TestResult):
            def __init__(self):
                super().__init__()
                # failfast makes unittest stop a test at its first failing subtest
                # (subTest raises _ShouldStop); stop() is a no-op so the suite goes on
                self.failfast = True
                self.cur = None
                self.bad = False
                self.c0 = 0.0
                self.tsum = 0.0

            def startTest(self, test):
                super().startTest(test)
                self.cur = ids.get(id(test))
                self.bad = False
                self.c0 = time.process_time()
                if self.cur is not None:
                    say("S", self.cur)

            def stopTest(self, test):
                super().stopTest(test)
                dt = time.process_time() - self.c0
                self.tsum += dt
                if self.cur is not None:
                    say("E", self.cur, int(self.bad), round(dt, 4))
                self.cur = None

            def addError(self, test, err):
                super().addError(test, err)
                if ids.get(id(test)) is None:  # class/module fixture error: fail its tests
                    desc = getattr(test, "description", "")
                    scope = desc[desc.find("(") + 1:desc.rfind(")")] if "(" in desc else ""
                    say("X", desc.replace(" ", "_")[:80])
                    for idx, mod_name, cls in owners:
                        if scope in (mod_name, f"{mod_name}.{cls}") or not scope:
                            say("F", idx)
                self.bad = True

            def addFailure(self, test, err):
                super().addFailure(test, err)
                self.bad = True

            def stop(self):
                pass

            def addSubTest(self, test, subtest, err):
                if err is not None:
                    self.bad = True

            def addUnexpectedSuccess(self, test):
                super().addUnexpectedSuccess(test)
                self.bad = True

        ids: dict[int, int] = {}
        owners: list[tuple[int, str, str]] = []
        for path, items in by_mod.items():
            mod_name = path[:-3].replace("/", ".")
            owners = [(idx, mod_name, cls) for idx, _, cls, _ in items]
            c0 = time.process_time()
            try:
                mod = __import__(mod_name, fromlist=["_"])
                if hasattr(mod, "ProcessPoolExecutor"):
                    mod.ProcessPoolExecutor = InlinePool
                cap_duckdb()
                if exec_ok and path == "tests/test_executor.py":
                    import exec_cache
                    say("C", "exec_cache", int(exec_cache.install(mod)))
            except BaseException:  # noqa: BLE001 - collection error: the file's tests fail
                for idx, *_ in items:
                    say("F", idx)
                continue
            finally:
                say("C", "imp:" + path, round(time.process_time() - c0, 4))
            suite = unittest.TestSuite()
            for idx, _, cls, meth in items:
                try:
                    t = getattr(mod, cls)(meth)
                    if idx in units_of:
                        restrict(t, *units_of[idx])
                except BaseException:  # noqa: BLE001
                    say("F", idx)
                    continue
                ids[id(t)] = idx
                suite.addTest(t)
            res = Result()
            c0 = time.process_time()
            suite.run(res)
            # class/module fixtures (setUpClass, tearDownClass) = suite time - test time
            say("C", "fix:" + path, round(time.process_time() - c0 - res.tsum, 4))
        say("D")
        out.flush()
    except BaseException:  # noqa: BLE001
        code = 3
    os._exit(code)


# ------------------------------------------------------------------------- server

class Job:
    """One commit to verify: an overlay and a list of (index, test id, timeout seconds)."""

    def __init__(self, key, changes: dict, tests: list[tuple]):
        self.key = key
        self.changes = changes
        self.tests = tests              # [(test id, timeout s[, unit indices or None])]
        self.passed: set[str] = set()
        self.failed: set[str] = set()
        self.cpu = 0.0                  # CPU reported by the tests themselves
        self.child_cpu = 0.0            # CPU of the child processes (tests + imports + fixtures)
        self.comp: dict[str, float] = {}  # child CPU by component (phase 6a)
        self.exec_ok = False            # use the executor-test caches (phase 6c, exec_cache.py)
        self.tcpu: dict[str, float] = {}
        self.icpu: dict[int, float] = {}
        self.todo = list(range(len(tests)))
        self.root: Path | None = None
        self.proc = None                # (pid, read fd, file)
        self.buf = b""
        self.current = None
        self.deadline = 0.0
        self.done = False
        self.restarts = 0
        self.why = ""
        self.on_result = None           # callable(job) -> False to stop verifying this job
        self.stopped = False
        self.probe: list[str] = []      # modules whose import must be checked (phase 5a)
        self.probe_ok: bool | None = None


class Verifier:
    def __init__(self, parallel: int = 2, gap_timeout: float = 30.0, preload: list | None = None):
        self.parallel = parallel
        self.gap_timeout = gap_timeout
        self.heavy_loaded = False
        self.n = 0
        self.cpu_children = 0.0
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ["SKIP_INTEGRATION"] = "0"
        os.environ.setdefault("PYTHON_CPU_COUNT", "2")
        import unittest  # noqa: F401 - must precede sqlglot (sqlglot.expressions.core.UNITTEST)
        import concurrent.futures  # noqa: F401
        import importlib
        import pickle as _p  # noqa: F401
        # standard-library modules sqlglot and its tests import (maps/preload.json): imported
        # once here instead of in every child. They are never patched.
        for name in preload or ():
            try:
                importlib.import_module(name)
            except Exception:  # noqa: BLE001
                pass

    def preload_heavy(self) -> None:
        if not self.heavy_loaded:
            self.heavy_loaded = True
            try:
                import numpy  # noqa: F401
                import pandas  # noqa: F401
                import duckdb  # noqa: F401
                import pandas.testing  # noqa: F401
                cap_duckdb()
            except Exception as e:  # noqa: BLE001
                log(f"preload failed: {e!r}")

    def _start(self, job: Job) -> None:
        if job.root is None:
            self.n += 1
            job.root = TMP / str(self.n)
            build_overlay(job.root, job.changes)
        r, w = os.pipe()
        import gc
        gc.freeze()  # preloaded objects: not rescanned by every child's GC, fewer copy-on-write faults
        tests = [(i, job.tests[i][0], job.tests[i][2] if len(job.tests[i]) > 2 else None)
                 for i in job.todo]
        pid = os.fork()
        if pid == 0:
            os.close(r)
            _child(job.root, tests, w, job.probe if job.probe_ok is None else None, job.exec_ok)
        os.close(w)
        os.set_blocking(r, False)
        job.proc = (pid, r)
        job.buf = b""
        job.current = None
        job.deadline = time.monotonic() + self.gap_timeout

    @staticmethod
    def _children_rss() -> int:
        """Resident bytes of the running children's process groups (best effort)."""
        total = 0
        try:
            page = os.sysconf("SC_PAGE_SIZE")
            for d in os.listdir("/proc"):
                if d.isdigit() and int(d) != os.getpid():
                    try:
                        with open(f"/proc/{d}/statm") as f:
                            total += int(f.read().split()[1]) * page
                    except OSError:
                        pass
        except OSError:
            return 0
        return total

    def _reap(self, job: Job, kill: bool) -> None:
        pid, r = job.proc
        if kill:
            try:
                os.killpg(pid, signal.SIGKILL)
            except OSError:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
        try:
            _, status, ru = os.wait4(pid, 0)
            self.cpu_children += ru.ru_utime + ru.ru_stime
            job.child_cpu += ru.ru_utime + ru.ru_stime
        except ChildProcessError:
            pass
        os.close(r)
        job.proc = None

    def _finish(self, job: Job) -> None:
        job.done = True
        if job.root is not None:
            shutil.rmtree(job.root, ignore_errors=True)
            job.root = None

    def _lines(self, job: Job) -> bool:
        """Process complete lines; True if the child said it is done."""
        while b"\n" in job.buf and not job.stopped:  # after a stop, ignore the rest (deterministic)
            line, job.buf = job.buf.split(b"\n", 1)
            parts = line.decode().split()
            if not parts:
                continue
            tag = parts[0]
            if tag == "S":
                i = int(parts[1])
                job.current = i
                job.deadline = time.monotonic() + job.tests[i][1]
            elif tag == "E":
                i = int(parts[1])
                tid = job.tests[i][0]
                (job.failed if parts[2] != "0" else job.passed).add(tid)
                job.cpu += float(parts[3])
                job.tcpu[tid] = float(parts[3])
                job.icpu[i] = float(parts[3])
                if i in job.todo:
                    job.todo.remove(i)
                job.current = None
                job.deadline = time.monotonic() + self.gap_timeout
            elif tag == "F":
                i = int(parts[1])
                job.failed.add(job.tests[i][0])
                job.passed.discard(job.tests[i][0])
                if i in job.todo:
                    job.todo.remove(i)
            if tag in "EF" and job.failed and job.on_result is not None and not job.stopped:
                try:
                    if not job.on_result(job):
                        job.stopped = True
                        job.why += " stopped(not worth it)"
                except Exception as e:  # noqa: BLE001
                    job.stopped = True
                    job.why += f" stopped({e!r})"
            elif tag == "X":
                job.why += f" fixture-error:{parts[1] if len(parts) > 1 else '?'}"
            elif tag == "P":
                job.probe_ok = parts[1] == "1"
            elif tag == "C" and len(parts) == 3:
                job.comp[parts[1]] = job.comp.get(parts[1], 0.0) + float(parts[2])
            elif tag == "D":
                return True
        return False

    def _abort_current(self, job: Job, why: str) -> None:
        """The child hung or died: the running test fails; restart for the rest."""
        if job.probe and job.probe_ok is None and why != "budget":
            job.probe_ok = False  # died or hung while importing the patched modules
        if job.current is not None:
            i = job.current
            job.failed.add(job.tests[i][0])
            if i in job.todo:
                job.todo.remove(i)
            job.why += f" {why}:{job.tests[i][0].split('::')[-1]}"
        else:
            job.why += f" {why}:between-tests"
            # no test running: the failure belongs to a fixture or import; keep the rest
            for i in job.todo:
                job.failed.add(job.tests[i][0])
            job.todo = []
        job.restarts += 1

    def run(self, jobs: list[Job], deadline: float) -> None:
        """Run jobs (in order) until all are done or the wall-clock deadline passes."""
        try:
            self._run(jobs, deadline)
        finally:
            import gc
            gc.unfreeze()  # let the main process free everything normally at exit

    def _run(self, jobs: list[Job], deadline: float) -> None:
        if any(t[0].startswith(HEAVY) for j in jobs for t in j.tests):
            self.preload_heavy()
        queue = list(jobs)
        running: list[Job] = []
        while queue or running:
            now = time.monotonic()
            while (queue and len(running) < self.parallel and now < deadline
                   and (not running or self._children_rss() < MAX_RSS_TO_START)):
                job = queue.pop(0)
                try:
                    self._start(job)
                    running.append(job)
                except Exception as e:  # noqa: BLE001
                    job.why += f" start-error {e!r}"
                    self._finish(job)
            if now >= deadline:
                for job in queue:
                    job.why += " skipped(budget)"
                queue = []
            if not running:
                break
            fds = {job.proc[1]: job for job in running}
            timeout = max(0.0, min(min(j.deadline for j in running), deadline) - now)
            ready, _, _ = select.select(list(fds), [], [], min(timeout, 1.0))
            now = time.monotonic()
            for fd in ready:
                job = fds[fd]
                try:
                    chunk = os.read(fd, 65536)
                except BlockingIOError:
                    continue
                if chunk:
                    job.buf += chunk
                    if self._lines(job):
                        self._reap(job, kill=False)
                        # anything not reported (e.g. class fixture error) stays failed
                        for i in job.todo:
                            job.failed.add(job.tests[i][0])
                        job.todo = []
                        running.remove(job)
                        self._finish(job)
                    elif job.stopped:  # the rest is not worth verifying: leave it selected
                        self._reap(job, kill=True)
                        running.remove(job)
                        self._finish(job)
                else:  # EOF without "D": the child died
                    self._lines(job)
                    self._reap(job, kill=True)
                    self._abort_current(job, "crash")
                    running.remove(job)
                    if job.todo and job.restarts < 4 and now < deadline:
                        queue.insert(0, job)
                    else:
                        for i in job.todo:
                            job.failed.add(job.tests[i][0])
                        self._finish(job)
            for job in list(running):
                if job.proc is not None and (now > job.deadline or now > deadline):
                    self._reap(job, kill=True)
                    self._abort_current(job, "timeout" if now <= deadline else "budget")
                    running.remove(job)
                    if job.todo and job.restarts < 4 and now < deadline:
                        queue.insert(0, job)
                    else:
                        for i in job.todo:
                            job.failed.add(job.tests[i][0])
                        self._finish(job)
