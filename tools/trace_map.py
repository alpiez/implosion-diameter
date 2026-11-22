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
sys.path.insert(0, os.environ.get("BR_APP", "/app"))
import verify  # noqa: E402  (app/verify.py: SPLITS and restrict(), shared with the verifier)
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
IMPORT_INST: dict[str, set[type]] = {}  # module file -> classes instantiated while importing it
FLAGS: dict[str, bool] = {"subprocess": False}
# phase 5a: class-attribute reads per test (Dialect/Tokenizer/Parser/Generator families)
ATTR_READS: set[str] = set()   # "module:Qual.NAME" read (any access)
KEY_READS: set[str] = set()    # "module:Qual.NAME|keyid" looked up (hit or miss)
WHOLE_READS: set[str] = set()  # "module:Qual.NAME" iterated / copied / sized
# reads made while modules are imported, by functions or module-level code (class bodies are
# analysed statically by build_maps): these values are consumed at import time
IMPORT_ATTR: set[str] = set()
IMPORT_KEYS: set[str] = set()

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
        if event == "open" and (r.endswith(".pyc") or (r.endswith(".py") and STATE["mode"] != "test")):
            return  # .py opened during a test = read as text (e.g. tests/test_typing.py:60)
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


# The roots' __init__ are wrapped as soon as their module has executed, so instantiation
# during later imports (templates parsed at import, Literal.number(...) constants) is seen:
# a class-body edit to such a class can break that module's import (phase 5b finding).
EARLY_ROOTS = {"sqlglot.expressions.core": "Expr", "sqlglot.tokens": "Tokenizer",
               "sqlglot.parser": "Parser", "sqlglot.generator": "Generator",
               "sqlglot.dialects.dialect": "Dialect"}


def _record_instance(self) -> None:
    if STATE["mode"] == "test":
        USED_CLASSES.add(type(self))
    elif STATE["stack"]:
        IMPORT_INST.setdefault(STATE["stack"][-1], set()).add(type(self))


def _early_wrap(c) -> None:
    if not isinstance(c, type):
        return
    orig = c.__dict__.get("__init__")
    if orig is not None and getattr(orig, "_br_wrapped", False):
        return
    if orig is None:
        def __init__(self, *a, __sup=c, **k):
            _record_instance(self)
            return super(__sup, self).__init__(*a, **k)
    else:
        @functools.wraps(orig)
        def __init__(self, *a, **k):
            _record_instance(self)
            return orig(self, *a, **k)
    __init__._br_wrapped = True
    c.__init__ = __init__


class _EarlyFinder:
    """Meta path hook: wrap a root class's __init__ right after its module executes."""

    def find_spec(self, name, path, target=None):
        if name not in EARLY_ROOTS:
            return None
        import importlib.machinery
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return spec
        orig = spec.loader.exec_module

        def exec_module(module, orig=orig, cls=EARLY_ROOTS[name]):
            orig(module)
            _early_wrap(getattr(module, cls, None))
        spec.loader.exec_module = exec_module
        return spec


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
            _record_instance(self)
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
                _record_instance(self)
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


# ----------------------------------------------------------------- attribute reads (5a)

UPPER = __import__("re").compile(r"^[A-Z][A-Z0-9_]*$")


_COMPREHENSIONS = ("<listcomp>", "<dictcomp>", "<setcomp>", "<genexpr>", "<lambda>")


def _import_reader_counts() -> bool:
    """While importing: is the code reading the attribute a function or module-level code
    (consumption) rather than a class body (merges and derivations, handled statically)?"""
    f = sys._getframe(2)
    while f is not None and (f.f_code.co_filename == __file__ or f.f_code.co_name in _COMPREHENSIONS[:4]):
        f = f.f_back
    if f is None:
        return True
    code = f.f_code
    return code.co_name == "<module>" or bool(code.co_flags & 0x1)  # CO_OPTIMIZED: a function


def key_id(k) -> str:
    import enum
    if isinstance(k, str):
        return "s:" + k
    if isinstance(k, enum.Enum):
        return f"e:{type(k).__qualname__}.{k.name}"
    if isinstance(k, type):
        return f"t:{k.__module__}.{k.__qualname__}"
    return "r:" + repr(k)[:100]


class TrackDict(dict):
    """A class-level table that records which keys a test looks up (or whether it scans it)."""
    __slots__ = ("_k",)

    def _key(self, key):
        mode = STATE["mode"]
        if mode == "test":
            try:
                KEY_READS.add(self._k + "|" + key_id(key))
            except Exception:  # noqa: BLE001
                WHOLE_READS.add(self._k)
        elif mode == "import" and _import_reader_counts():
            try:
                IMPORT_KEYS.add(self._k + "|" + key_id(key))
            except Exception:  # noqa: BLE001
                IMPORT_ATTR.add(self._k)

    def _whole(self):
        mode = STATE["mode"]
        if mode == "test":
            WHOLE_READS.add(self._k)
        elif mode == "import" and _import_reader_counts():
            IMPORT_ATTR.add(self._k)

    def __getitem__(self, key):
        self._key(key)
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        self._key(key)
        return dict.get(self, key, default)

    def __contains__(self, key):
        self._key(key)
        return dict.__contains__(self, key)

    def pop(self, key, *a):
        self._key(key)
        return dict.pop(self, key, *a)

    def setdefault(self, key, default=None):
        self._key(key)
        return dict.setdefault(self, key, default)

    def __iter__(self):
        self._whole()
        return dict.__iter__(self)

    def keys(self):
        self._whole()
        return dict.keys(self)

    def items(self):
        self._whole()
        return dict.items(self)

    def values(self):
        self._whole()
        return dict.values(self)

    def __len__(self):
        self._whole()
        return dict.__len__(self)

    def copy(self):
        self._whole()
        return dict(dict.items(self))

    def __or__(self, other):
        self._whole()
        return dict(dict.items(self)) | other

    def __ror__(self, other):
        self._whole()
        return other | dict(dict.items(self))

    def __eq__(self, other):
        self._whole()
        return dict.__eq__(self, other)

    __hash__ = None


class TrackedAttr:
    """Non-data descriptor standing in for a class attribute: records the read, returns the value."""
    __slots__ = ("v", "k")

    def __init__(self, v, k):
        self.v = v
        self.k = k

    def __get__(self, obj, typ=None):
        mode = STATE["mode"]
        if mode == "test":
            ATTR_READS.add(self.k)
        elif mode == "import" and not isinstance(self.v, TrackDict) and _import_reader_counts():
            IMPORT_ATTR.add(self.k)  # scalars; tables record their own key or whole reads
        return self.v


_SHARED: dict[int, object] = {}


def install_attr_tracking(tracked_classes=None) -> int:
    """Replace UPPER_CASE data attributes of the parser/generator/tokenizer/dialect families
    by recording descriptors (dict tables become TrackDicts). Shared objects stay shared."""
    import types
    from sqlglot.dialects.dialect import Dialect
    from sqlglot.generator import Generator
    from sqlglot.parser import Parser
    from sqlglot.tokens import Tokenizer
    roots = [Dialect, Generator, Parser, Tokenizer]
    try:
        from sqlglot.jsonpath import JSONPathTokenizer
        roots.append(JSONPathTokenizer)
    except Exception:  # noqa: BLE001
        pass
    fam = set()
    for r in roots:
        fam.add(r)
        fam |= all_subclasses(r)
    for c in list(fam):
        fam.update(b for b in c.__mro__ if b.__module__.startswith("sqlglot"))
    shared = _SHARED
    n = 0
    for c in sorted(fam, key=class_key):
        for name, v in list(vars(c).items()):
            if not UPPER.match(name) or isinstance(v, (type, types.FunctionType, classmethod,
                                                      staticmethod, property, TrackedAttr)):
                continue
            k = f"{class_key(c)}.{name}"
            if isinstance(v, dict) and type(v) is dict:
                entry = shared.get(id(v))
                if entry is not None and entry[0] is v:
                    td = entry[1]
                else:
                    td = TrackDict(v)
                    td._k = k
                    shared[id(v)] = (v, td)  # keep the original alive: ids must not be reused
                v = td
            setattr(c, name, TrackedAttr(v, k))
            n += 1
    return n


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


def run_one(tid: str, out: Path, timeout: int, units: list | None = None) -> None:
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
        test = test_cls(meth)
        if units is not None:
            verify.restrict(test, tid, units)
        suite = unittest.TestSuite([test])
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
    n_units = None
    if tid in verify.SPLITS and units is None:
        try:
            n_units = verify.unit_count(test, tid)
        except Exception:  # noqa: BLE001
            pass
    data = {
        "id": tid,
        "units": units,
        "n_units": n_units,
        "ok": ok,
        "err": err,
        "wall": round(time.perf_counter() - t0, 3),
        "cpu": round(time.process_time() - c0, 3),
        "lines": {f: sorted(s) for f, s in sorted(TEST_LINES.items())},
        "classes": mro_keys(USED_CLASSES),
        "attr_reads": sorted(ATTR_READS),
        "key_reads": sorted(KEY_READS),
        "whole_reads": sorted(WHOLE_READS),
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
    sys.meta_path.insert(0, _EarlyFinder())
    import sqlglot  # noqa: F401
    core_modules = sorted(m for m in set(sys.modules) - before if m.startswith("sqlglot"))
    ATTRS = os.environ.get("BR_ATTRS", "1") != "0"
    if ATTRS:
        install_attr_tracking()  # from here on, import-time reads by functions are recorded

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
        if ATTRS:
            install_attr_tracking()  # classes this module created
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
        if ATTRS:
            install_attr_tracking()
    STATE["mode"] = "idle"
    STATE["stack"].clear()
    import_secs = time.perf_counter() - t0

    # 3. patch the pools, install class tracking
    for m in ("tests.test_optimizer", "tests.test_executor"):
        if m in sys.modules and hasattr(sys.modules[m], "ProcessPoolExecutor"):
            sys.modules[m].ProcessPoolExecutor = InlinePool
    tracked = install_class_tracking()
    if ATTRS:
        install_attr_tracking()  # dynamic classes made since (none expected)
        print(f"[trace] {len([1 for c in _SHARED])} tables, import-time reads: "
              f"{len(IMPORT_ATTR)} attributes, {len(IMPORT_KEYS)} keys", file=sys.stderr, flush=True)

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
    # each dialect's parts and their REAL ancestry (from the live classes, not from the class
    # table: dynamic parts made by the metaclass share one key, so their table bases merge)
    dialect_mros: dict[str, set[str]] = {}
    for d in [Dialect, *all_subclasses(Dialect)]:
        parts = {part: getattr(d, part) for part in
                 ("tokenizer_class", "jsonpath_tokenizer_class", "parser_class", "generator_class")
                 if isinstance(getattr(d, part, None), type)}
        dialects[class_key(d)] = {part: class_key(c) for part, c in parts.items()}
        acc = dialect_mros.setdefault(class_key(d), set())
        for c in parts.values():
            acc.update(class_key(b) for b in c.__mro__ if b.__module__.startswith("sqlglot"))

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
        "dialect_part_mros": {k: sorted(v) for k, v in sorted(dialect_mros.items())},
        "import_instances": {f: mro_keys(cs) for f, cs in sorted(IMPORT_INST.items())},
        "import_attr_reads": sorted(IMPORT_ATTR),
        "import_key_reads": sorted(IMPORT_KEYS),
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

    # 5. split tests: one cold child per unit (the full run above recorded the unit count)
    (a.out / "units").mkdir(exist_ok=True)
    unit_jobs = []
    for tid in tests:
        if tid not in verify.SPLITS:
            continue
        full = json.loads((a.out / "tests" / f"{index[tid]}.json").read_text())
        for u in range(full.get("n_units") or 0):
            dst = a.out / "units" / f"{index[tid]}_{u}.json"
            if not dst.exists():
                unit_jobs.append((tid, u, dst))
    t2 = time.perf_counter()
    for tid, u, dst in unit_jobs:
        while len(running) >= a.jobs:
            reap(True)
        pid = os.fork()
        if pid == 0:
            try:
                TEST_LINES.clear()
                IMPORT_LINES.clear()
                run_one(tid, dst, a.timeout, [u])
            except BaseException:  # noqa: BLE001
                traceback.print_exc()
                os._exit(3)
            os._exit(0)
        running[pid] = f"{tid}[{u}]"
    while running:
        reap(True)
    print(f"[trace] {len(unit_jobs)} units in {time.perf_counter() - t2:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
