"""Trace the pinned sqlglot test suite to build the raw data for our maps.

Runs INSIDE br-dev (dev/Dockerfile) against the pristine checkout at
/opt/sqlglot. Nothing here looks at patches: everything comes from running the
pinned commit's own tests (CLAUDE.md section 5: coverage/import/fixture maps
built from the pinned commit are allowed).

Design (see CHANGELOG.md, Phase 2):
  * A parent process imports sqlglot, every sqlglot submodule and every test
    module, with a sys.monitoring tracer that attributes each executed line to
    the module being imported ("import contexts").
  * Then each test runs in its own forked child (cold caches, setUpClass
    re-run), with the tracer recording every line of sqlglot/ and tests/ it
    executes. LINE events return DISABLE and are re-armed once per test with
    sys.monitoring.restart_events(), so each line costs one callback per test.
  * Per test we also record: classes used (MRO keys of every Dialect /
    Tokenizer / Parser / Generator / Expr class instantiated or looked up),
    files opened and directories listed (audit hook), whether a subprocess was
    started, and the unittest outcome.
  * The two ProcessPoolExecutors in the tests are replaced by an inline
    executor that round-trips arguments and results through pickle, so the
    pool work (and Expr.__reduce__ -> serde) is traced under the right test.

Output (in --out): meta.json (import contexts, import graph, class table,
core-import modules) and tests/<n>.json (one per test).

  docker run --rm -v "$PWD/tools:/tools:ro" -v "$PWD/work:/out" br-dev \
      python /tools/trace_map.py --tests /out/tests.json --out /out/trace --jobs 12
"""

import argparse
import functools
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

ROOT = "/opt/sqlglot"
ROOT_PREFIX = ROOT + "/"
mon = sys.monitoring
TOOL = mon.COVERAGE_ID

# ----------------------------------------------------------------- recording state
STATE = {
    "mode": "import",  # import | idle | test
    "stack": [],  # modules being imported (relative file paths)
}
TEST_LINES: dict[str, set[int]] = {}
IMPORT_LINES: dict[str, dict[str, set[int]]] = {}  # module file -> {file: lines}
IMPORT_EDGES: set[tuple[str, str]] = set()
TEST_FILES: set[str] = set()
IMPORT_FILES: dict[str, set[str]] = {}
USED_CLASSES: set[type] = set()
FLAGS: dict[str, bool] = {"subprocess": False}

_rel_cache: dict[str, str | None] = {}


def rel(filename: str) -> str | None:
    r = _rel_cache.get(filename, 0)
    if r == 0:
        r = None
        if filename.startswith(ROOT_PREFIX):
            p = filename[len(ROOT_PREFIX):]
            if p.startswith(("sqlglot/", "tests/")) and p.endswith(".py"):
                r = p
        _rel_cache[filename] = r
    return r


def on_line(code, line):
    f = rel(code.co_filename)
    if f is not None:
        stack = STATE["stack"]
        if stack:
            IMPORT_LINES.setdefault(stack[-1], {}).setdefault(f, set()).add(line)
        if STATE["mode"] == "test":
            s = TEST_LINES.get(f)
            if s is None:
                s = TEST_LINES[f] = set()
            s.add(line)
    return mon.DISABLE


def on_start(code, offset):
    if code.co_name == "<module>":
        f = rel(code.co_filename)
        if f is not None:
            stack = STATE["stack"]
            if stack:
                IMPORT_EDGES.add((stack[-1], f))
            stack.append(f)
            mon.restart_events()  # re-arm lines already seen while importing other modules
            return None
    return mon.DISABLE


def on_end(code, *args):
    if code.co_name == "<module>":
        f = rel(code.co_filename)
        if f is not None and STATE["stack"] and STATE["stack"][-1] == f:
            STATE["stack"].pop()
            mon.restart_events()
            return None
    return mon.DISABLE


def on_unwind(code, offset, exc):
    if code.co_name == "<module>":
        f = rel(code.co_filename)
        if f is not None and STATE["stack"] and STATE["stack"][-1] == f:
            STATE["stack"].pop()


def audit(event, args):
    if event == "open" or event in ("os.listdir", "os.scandir"):
        p = args[0] if args else None
        if isinstance(p, bytes):
            p = p.decode(errors="replace")
        if not isinstance(p, (str, os.PathLike)):
            return
        p = os.fspath(p)
        if p.startswith("<"):
            return
        p = os.path.abspath(p)
        if not p.startswith(ROOT_PREFIX):
            return
        r = p[len(ROOT_PREFIX):]
        if r.endswith((".pyc", ".py")) and event == "open":
            return
        if event != "open":
            r += "/"  # directory listing
        if STATE["mode"] == "test":
            TEST_FILES.add(r)
        if STATE["stack"]:
            IMPORT_FILES.setdefault(STATE["stack"][-1], set()).add(r)
    elif event == "subprocess.Popen" and STATE["mode"] == "test":
        FLAGS["subprocess"] = True


def install_tracer():
    mon.use_tool_id(TOOL, "blast-radius-map")
    mon.register_callback(TOOL, mon.events.LINE, on_line)
    mon.register_callback(TOOL, mon.events.PY_START, on_start)
    mon.register_callback(TOOL, mon.events.PY_RETURN, on_end)
    mon.register_callback(TOOL, mon.events.PY_UNWIND, on_unwind)
    mon.set_events(TOOL, mon.events.LINE | mon.events.PY_START | mon.events.PY_RETURN
                   | mon.events.PY_UNWIND)
    sys.addaudithook(audit)


# ----------------------------------------------------------------- inline pool

class InlinePool:
    """Drop-in for ProcessPoolExecutor: runs work in-process, pickling like a pool."""

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

    def shutdown(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# ----------------------------------------------------------------- class usage

def class_key(c: type) -> str:
    return f"{c.__module__}:{c.__qualname__}"


def all_subclasses(c: type) -> set[type]:
    out, todo = set(), [c]
    while todo:
        for s in todo.pop().__subclasses__():
            if s not in out:
                out.add(s)
                todo.append(s)
    return out


def install_class_tracking() -> list[type]:
    from sqlglot.dialects.dialect import Dialect, _Dialect
    from sqlglot.expressions.core import Expr
    from sqlglot.generator import Generator
    from sqlglot.parser import Parser
    from sqlglot.tokens import Tokenizer

    roots = [Dialect, Tokenizer, Parser, Generator, Expr]
    try:
        from sqlglot.jsonpath import JSONPathTokenizer  # noqa: F401
    except Exception:
        pass

    def wrap(c: type) -> None:
        orig = c.__dict__.get("__init__")
        if orig is None or getattr(orig, "_br_wrapped", False):
            return

        @functools.wraps(orig)
        def __init__(self, *a, **k):
            USED_CLASSES.add(type(self))
            return orig(self, *a, **k)

        __init__._br_wrapped = True
        setattr(c, "__init__", __init__)

    tracked = []
    for r in roots:
        for c in [r, *all_subclasses(r)]:
            tracked.append(c)
            wrap(c)
        if "__init__" not in r.__dict__:  # make sure instantiation is seen
            def __init__(self, *a, __sup=r, **k):
                USED_CLASSES.add(type(self))
                return super(__sup, self).__init__(*a, **k)
            r.__init__ = __init__

    # _Dialect.get / __getitem__ are classmethods of the metaclass: keep them classmethods
    for name in ("get", "__getitem__"):
        orig = getattr(_Dialect, name)  # bound to _Dialect already

        def make(orig):
            def f(mcs, *a, **k):
                res = orig(*a, **k)
                if isinstance(res, type):
                    USED_CLASSES.add(res)
                return res
            return classmethod(functools.wraps(orig.__func__)(f))

        setattr(_Dialect, name, make(orig))
    return tracked


def mro_keys(classes: set[type]) -> list[str]:
    keys = set()
    for c in classes:
        for b in c.__mro__:
            if b.__module__.startswith("sqlglot"):
                keys.add(class_key(b))
    return sorted(keys)


# ----------------------------------------------------------------- running a test

class Result:
    """Minimal unittest result: did anything fail (including subtests)?"""

    def __init__(self):
        import unittest
        self.r = unittest.TestResult()
        self.r.addSubTest = self._sub
        self.sub_fail = []

    def _sub(self, test, subtest, err):
        if err is not None:
            self.sub_fail.append(str(subtest)[:200] + ": " + "".join(
                traceback.format_exception_only(err[0], err[1]))[-300:])


def run_one(tid: str, out: Path, timeout: int) -> None:
    import signal
    import unittest

    path, cls_name, meth = tid.split("::")
    mod_name = path[:-3].replace("/", ".")
    t0 = time.perf_counter()
    c0 = time.process_time()
    err = None
    ok = False
    try:
        signal.alarm(timeout)
        mod = sys.modules.get(mod_name) or __import__(mod_name, fromlist=["x"])
        test_cls = getattr(mod, cls_name)
        suite = unittest.TestSuite([test_cls(meth)])
        res = Result()
        STATE["mode"] = "test"
        mon.restart_events()
        suite.run(res.r)
        STATE["mode"] = "idle"
        signal.alarm(0)
        ok = res.r.wasSuccessful() and not res.sub_fail
        if not ok:
            err = "; ".join([e[1][-500:] for e in res.r.failures + res.r.errors] + res.sub_fail)[:3000]
    except BaseException as e:  # noqa: BLE001
        STATE["mode"] = "idle"
        err = "harness: " + "".join(traceback.format_exception(e))[-2000:]
    data = {
        "id": tid,
        "ok": ok,
        "err": err,
        "wall": round(time.perf_counter() - t0, 3),
        "cpu": round(time.process_time() - c0, 3),
        "lines": {f: sorted(s) for f, s in sorted(TEST_LINES.items())},
        "classes": mro_keys(USED_CLASSES),
        "files": sorted(TEST_FILES),
        "subprocess": FLAGS["subprocess"],
        # the child starts with IMPORT_LINES empty, so anything here was imported lazily
        "lazy_imports": {m: {f: sorted(s) for f, s in d.items()} for m, d in IMPORT_LINES.items()},
    }
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.rename(out)



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", required=True, help="tests.json (ids to run)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--only", help="substring filter on test ids (debugging)")
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()

    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    (a.out / "tests").mkdir(parents=True, exist_ok=True)
    tests = [t["id"] for t in json.loads(Path(a.tests).read_text())["tests"]]
    index = {tid: i for i, tid in enumerate(tests)}
    if a.only:
        tests = [t for t in tests if a.only in t]

    install_tracer()
    t0 = time.perf_counter()

    # sqlglot/expressions/core.py:48 sets UNITTEST = "unittest" in sys.modules or "pytest" in
    # sys.modules AT IMPORT TIME (it enables strict arg validation). The labellers ran pytest,
    # so both must be imported before sqlglot.
    import pytest  # noqa: F401
    import unittest  # noqa: F401

    # 1. `import sqlglot` alone: what a bare import executes (subprocess tests use this)
    before = set(sys.modules)
    import sqlglot  # noqa: F401
    core_modules = sorted(m for m in set(sys.modules) - before if m.startswith("sqlglot"))

    # 2. every sqlglot module, then every test module
    import importlib
    import pkgutil
    failed_imports = {}
    for info in sorted(pkgutil.walk_packages(sqlglot.__path__, prefix="sqlglot."), key=lambda i: i.name):
        if info.name == "sqlglot.__main__":
            continue
        try:
            importlib.import_module(info.name)
        except Exception as e:  # noqa: BLE001
            failed_imports[info.name] = repr(e)[:300]
    test_mods = sorted({t.split("::")[0][:-3].replace("/", ".") for t in tests})
    for extra in sorted(Path(ROOT, "tests").rglob("*.py")):
        name = str(extra.relative_to(ROOT))[:-3].replace("/", ".")
        if name.endswith("__init__") or name.startswith("tests.sqlglot"):
            continue
        if name not in test_mods:
            test_mods.append(name)
    for m in test_mods:
        try:
            importlib.import_module(m)
        except Exception as e:  # noqa: BLE001
            failed_imports[m] = repr(e)[:300]
    STATE["mode"] = "idle"
    STATE["stack"].clear()
    import_secs = time.perf_counter() - t0

    # 3. patch the pools, install class tracking
    for m in ("tests.test_optimizer", "tests.test_executor"):
        if m in sys.modules and hasattr(sys.modules[m], "ProcessPoolExecutor"):
            sys.modules[m].ProcessPoolExecutor = InlinePool
    tracked = install_class_tracking()

    # class table: every class defined in sqlglot modules (for hierarchy / subclass closure)
    classes = {}
    seen = set()
    for mname, mod in list(sys.modules.items()):
        if not mname.startswith("sqlglot") or mod is None:
            continue
        todo = [v for v in vars(mod).values() if isinstance(v, type)]
        while todo:
            c = todo.pop()
            if c in seen or not c.__module__.startswith("sqlglot"):
                continue
            seen.add(c)
            todo.extend(v for v in vars(c).values() if isinstance(v, type))
            k = class_key(c)
            entry = classes.setdefault(k, {"bases": set(), "file": None, "line": None})
            entry["bases"] |= {class_key(b) for b in c.__bases__}
            try:
                import inspect
                entry["file"] = rel(inspect.getsourcefile(c) or "")
                entry["line"] = inspect.getsourcelines(c)[1]
            except Exception:  # noqa: BLE001 - dynamic classes have no source
                pass
    for c in tracked:  # dynamic classes made by the _Dialect metaclass
        k = class_key(c)
        classes.setdefault(k, {"bases": set(), "file": None, "line": None})["bases"] |= {
            class_key(b) for b in c.__bases__}
    from sqlglot.dialects.dialect import Dialect
    dialects = {}
    for d in [Dialect, *all_subclasses(Dialect)]:
        dialects[class_key(d)] = {
            part: class_key(getattr(d, part)) for part in
            ("tokenizer_class", "jsonpath_tokenizer_class", "parser_class", "generator_class")
            if isinstance(getattr(d, part, None), type)}

    meta = {
        "root": ROOT,
        "python": sys.version,
        "import_seconds": round(import_secs, 2),
        "core_modules": core_modules,
        "failed_imports": failed_imports,
        "import_lines": {m: {f: sorted(s) for f, s in d.items()} for m, d in sorted(IMPORT_LINES.items())},
        "import_files": {m: sorted(s) for m, s in sorted(IMPORT_FILES.items())},
        "import_edges": sorted(IMPORT_EDGES),
        "classes": {k: {"bases": sorted(v["bases"]), "file": v["file"], "line": v["line"]}
                    for k, v in sorted(classes.items())},
        "dialects": dialects,
        "tests": [t["id"] for t in json.loads(Path(a.tests).read_text())["tests"]],
    }
    (a.out / "meta.json").write_text(json.dumps(meta))
    print(f"[trace] imports done in {import_secs:.1f}s, {len(IMPORT_LINES)} modules, "
          f"{len(failed_imports)} failed, {len(tracked)} tracked classes", file=sys.stderr, flush=True)

    # 4. fork one child per test
    running: dict[int, str] = {}
    todo = [t for t in tests if not (a.out / "tests" / f"{index[t]}.json").exists()]
    done = 0
    t1 = time.perf_counter()

    def reap(block: bool) -> None:
        nonlocal done
        while running:
            pid, status = os.waitpid(-1, 0 if block else os.WNOHANG)
            if pid == 0:
                return
            tid = running.pop(pid, None)
            done += 1
            if tid and os.waitstatus_to_exitcode(status) != 0:
                print(f"[trace] child for {tid} exited {os.waitstatus_to_exitcode(status)}",
                      file=sys.stderr, flush=True)
            if done % 100 == 0:
                print(f"[trace] {done}/{len(todo)} tests, {time.perf_counter() - t1:.0f}s",
                      file=sys.stderr, flush=True)
            block = False

    for tid in todo:
        while len(running) >= a.jobs:
            reap(True)
        pid = os.fork()
        if pid == 0:
            try:
                TEST_LINES.clear()
                IMPORT_LINES.clear()
                run_one(tid, a.out / "tests" / f"{index[tid]}.json", a.timeout)
            except BaseException:  # noqa: BLE001
                traceback.print_exc()
                os._exit(3)
            os._exit(0)
        running[pid] = tid
    while running:
        reap(True)
    print(f"[trace] all {len(todo)} tests in {time.perf_counter() - t1:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
