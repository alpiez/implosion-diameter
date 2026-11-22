"""Phase 5a: which tests depend on a class-level attribute (or one entry of a class table).

Used by build_maps.py. Inputs, all from tracing the PINNED commit (tools/trace_map.py):
  * per test: class attributes read ("module:Qual.NAME"), table keys looked up
    ("...NAME|keyid", hits and misses) and tables scanned whole (iterated, copied, sized);
  * reads made at import time by functions or module-level code (import-time parsing,
    helper functions building constants): such values are CONSUMED at import;
  * the class table (bases), for subclass closure.
Static analysis of the sources adds:
  * C0: names read by class-creation hooks (`__init_subclass__`, metaclass `__new__`): they
    derive data (tries, escape sets, required_args) from the class's own attributes before
    any tracking can see it, so these attributes are consumed;
  * derivations in class bodies (`TABLE_ALIAS_TOKENS = ID_VAR_TOKENS - {...}`,
    `FUNCTIONS = {**Parser.FUNCTIONS, ...}`): an edit to Y also reaches every X whose
    defining expression mentions Y (name level, transitive).

For each UPPER_CASE assignment in a class body of the Dialect/Tokenizer/Parser/Generator
families, every line of the statement gets an entry
    [key_level_bits | None, attr_level_bits | None, is_first_line_of_entry, statement_start]
  attr level: tests that read the attribute (any access, key, scan) on the class or any
              subclass, plus readers of every attribute derived from it;
  key level:  for a line inside one `key: value` entry of a dict display whose key evaluates
              on the pinned commit: tests that looked that key up or scanned the table, on the
              class or a subclass, plus readers of derived attributes.
None means "consumed at import time or unresolvable": the selector keeps its wide rule.
"""

import ast
import enum
import importlib
import re
import sys
from collections import defaultdict
from pathlib import Path

UPPER = re.compile(r"^[A-Z][A-Z0-9_]*$")
ROOTS = ("sqlglot.dialects.dialect:Dialect", "sqlglot.tokens:Tokenizer", "sqlglot.parser:Parser",
         "sqlglot.generator:Generator", "sqlglot.jsonpath:JSONPathTokenizer")


def key_id(k) -> str:
    """Same encoding as tools/trace_map.py key_id()."""
    if isinstance(k, str):
        return "s:" + k
    if isinstance(k, enum.Enum):
        return f"e:{type(k).__qualname__}.{k.name}"
    if isinstance(k, type):
        return f"t:{k.__module__}.{k.__qualname__}"
    return "r:" + repr(k)[:100]


def _subclasses(classes: dict) -> dict:
    subs = defaultdict(set)
    for k, v in classes.items():
        for b in v["bases"]:
            subs[b].add(k)
    return subs


def _desc(k: str, subs: dict, memo: dict) -> frozenset:
    if k not in memo:
        out, todo = {k}, [k]
        while todo:
            for s in subs.get(todo.pop(), ()):
                if s not in out:
                    out.add(s)
                    todo.append(s)
        memo[k] = frozenset(out)
    return memo[k]


def _names_in(node, skip_pop: bool = False) -> set:
    """UPPER names mentioned under node. skip_pop: ignore `X.NAME.pop(...)`, which only
    deletes keys (the dialect metaclass prunes unsupported JSON path parts from TRANSFORMS)
    and derives nothing from the values; edits to surviving entries are seen by their readers."""
    pops = set()
    if skip_pop:
        for n in ast.walk(node):
            if isinstance(n, ast.Attribute) and n.attr == "pop" and isinstance(n.value, ast.Attribute):
                pops.add(id(n.value))
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and UPPER.match(n.id):
            out.add(n.id)
        elif isinstance(n, ast.Attribute) and UPPER.match(n.attr) and id(n) not in pops:
            out.add(n.attr)
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and UPPER.match(n.value):
            out.add(n.value)
    return out


def class_hook_names(root: Path) -> set:
    """C0: UPPER names read by __init_subclass__ / metaclass __new__ anywhere in sqlglot."""
    out = set()
    for f in sorted(root.glob("sqlglot/**/*.py")):
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            is_meta = any(isinstance(b, ast.Name) and b.id == "type" for b in node.bases)
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and (
                        fn.name == "__init_subclass__" or (is_meta and fn.name == "__new__")):
                    out |= _names_in(fn, skip_pop=True)
    return out


def _assignments(tree, module: str):
    """(class key, NAME, stmt node) for UPPER_CASE assignments in (nested) class bodies."""
    def visit(body, prefix):
        for st in body:
            if isinstance(st, ast.ClassDef):
                q = prefix + st.name
                for s2 in st.body:
                    tg = None
                    if isinstance(s2, ast.Assign) and len(s2.targets) == 1:
                        tg = s2.targets[0]
                    elif isinstance(s2, ast.AnnAssign) and s2.value is not None:
                        tg = s2.target
                    if isinstance(tg, ast.Name) and UPPER.match(tg.id):
                        yield f"{module}:{q}", tg.id, s2
                yield from visit(st.body, q + ".")
            elif isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
    yield from visit(tree.body, "")


def build(root: Path, sources: list, meta: dict, reads: dict, problems: list) -> dict:
    """{file: {line: [kbits|None, abits|None, first, stmt_start]}}.
    reads: {"attr": {k: bits}, "whole": {k: bits}, "key": {k|kid: bits}}"""
    classes = meta["classes"]
    subs = _subclasses(classes)
    memo: dict = {}
    family = set()
    for r in ROOTS:
        if r in classes:
            family |= _desc(r, subs, memo)
    imp_attr = set(meta.get("import_attr_reads", []))
    imp_keys = set(meta.get("import_key_reads", []))
    imp_tables = {k.split("|")[0] for k in imp_keys}
    c0 = class_hook_names(root)
    attr_b, whole_b, key_b = reads["attr"], reads["whole"], reads["key"]
    keyany = defaultdict(int)
    for k, bits in key_b.items():
        keyany[k.split("|")[0]] |= bits

    trees, assigns = {}, []
    for f in sources:
        if not f.startswith("sqlglot/"):
            continue
        module = f[:-3].replace("/", ".")
        if module.endswith(".__init__"):
            module = module[: -len(".__init__")]
        tree = trees[f] = ast.parse((root / f).read_text())
        for ck, name, st in _assignments(tree, module):
            if ck in family:
                assigns.append((f, module, ck, name, st))

    # name-level derivation graph: Y -> X when X's defining expression mentions Y
    derived = defaultdict(set)
    for _, _, _, x, st in assigns:
        for y in _names_in(st.value):
            if y != x:
                derived[y].add(x)

    def closure(name):
        out, todo = set(), [name]
        while todo:
            for x in derived.get(todo.pop(), ()):
                if x not in out and x != name:
                    out.add(x)
                    todo.append(x)
        return out

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    out: dict = defaultdict(dict)
    n_key = n_attr = n_consumed = 0
    for f, module, ck, name, st in assigns:
        ds = _desc(ck, subs, memo)
        others = closure(name)
        names = {name} | others
        consumed = bool(names & c0) or any(
            f"{d}.{n}" in imp_attr or f"{d}.{n}" in imp_tables for d in ds for n in names)
        attr_bits = 0
        for d in ds:
            for n in names:
                k = f"{d}.{n}"
                attr_bits |= attr_b.get(k, 0) | whole_b.get(k, 0) | keyany.get(k, 0)
        a = None if consumed else attr_bits
        n_consumed += consumed
        derived_bits = 0
        derived_consumed = bool(others & c0)
        for d in ds:
            for n in others:
                k = f"{d}.{n}"
                derived_bits |= attr_b.get(k, 0) | whole_b.get(k, 0) | keyany.get(k, 0)
                derived_consumed |= k in imp_attr or k in imp_tables
        s0 = st.lineno
        for ln in range(st.lineno, st.end_lineno + 1):
            out[f][ln] = [None, a, 0, s0]
        if a is not None:
            n_attr += 1
        if not isinstance(st.value, ast.Dict) or name in c0 or derived_consumed:
            continue
        # key level: one entry of a dict display
        mod = None
        try:
            mod = importlib.import_module(module)
        except Exception as e:  # noqa: BLE001
            problems.append(f"attr_maps: cannot import {module}: {e!r}"[:200])
        whole_any = any(f"{d}.{name}" in imp_attr for d in ds)
        for kn, vn in zip(st.value.keys, st.value.values):
            if kn is None or mod is None:
                continue  # `**merge` lines stay attribute level
            try:
                kid = key_id(eval(compile(ast.Expression(body=kn), f, "eval"), vars(mod)))
            except Exception:  # noqa: BLE001 - class-local names etc.: attribute level
                continue
            if whole_any or any(f"{d}.{name}|{kid}" in imp_keys for d in ds):
                continue  # this entry is consumed at import
            kbits = derived_bits
            for d in ds:
                kbits |= key_b.get(f"{d}.{name}|{kid}", 0) | whole_b.get(f"{d}.{name}", 0)
            for ln in range(kn.lineno, vn.end_lineno + 1):
                ent = out[f][ln]
                ent[0] = kbits
                ent[2] = 1 if ln == kn.lineno else 0
            n_key += 1
    print(f"attr_maps: {len(assigns)} class attributes, {n_attr} usable at attribute level, "
          f"{n_consumed} consumed at import, {n_key} table entries at key level, "
          f"{len(c0)} names read by class hooks", file=sys.stderr)
    return out
