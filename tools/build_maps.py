"""Build the committed maps (maps/) from the raw trace of tools/trace_map.py.

Runs INSIDE br-dev (needs the pristine sources at /opt/sqlglot for the AST
structure). Input: the trace dir (meta.json + tests/<n>.json). Output:

  maps/meta.json   small, always loaded:
      tests          test ids (bit i of every set = tests[i])
      sets           deduplicated test sets, base64 little-endian bitsets
      data_files     {path or dir/: set}  non-.py files opened / dirs listed per test
      importers      {module file: [files that import it]}   (import graph, reversed)
      core_modules   files executed by a bare `import sqlglot`
      subprocess     set of tests that start a subprocess (they re-import sqlglot)
      all            set of all tests
      files          {source file: [offset, length]} into files.bin
  maps/files.bin   one zlib(JSON) blob per source file (sqlglot/**.py, tests/**.py):
      cov            {line: set}  tests that executed the line at runtime
      imp            {line: [module files]}  modules whose import executed the line
      struct         [[start, end, kind, qualname, col, end_col, body_start], ...]
                     kind: c/f/l (class, def, lambda); start includes decorators;
                     body_start = first line of the body (lines before it are the header,
                     evaluated in the enclosing scope); columns only matter for lambdas
      strings        [[start, end, is_docstring], ...] string literals spanning lines
      any            set: tests that executed any line of this file at runtime
      class_users    {qualname: set}  tests whose used classes include this class in
                     their MRO (i.e. used it or a subclass)
      mod_users      set: any | all class_users of the file
      file_tests     set: tests defined in this file (test files only)
      collect_users  set: tests whose test FILE imports this module at collection time
                     (fresh-process probe, tools/import_closure.py): if the module fails
                     to import, all of them fail
      stmts          [[start, header_end], ...] statements whose header spans several
                     lines (a condition split over lines): an edit on any of these lines
                     can change the outcome for every test that reaches the statement
      sets           the file-local set table (indices above refer to it)

  docker run --rm -v "$PWD/tools:/tools:ro" -v "$PWD/work:/out" -v "$PWD/maps:/maps" br-dev \
      python /tools/build_maps.py --trace /out/trace --out /maps
"""

import argparse
import ast
import base64
import json
import os
import zlib
from collections import defaultdict
from pathlib import Path

ROOT = Path("/opt/sqlglot")


def b64(bits: int) -> str:
    return base64.b64encode(bits.to_bytes((bits.bit_length() + 7) // 8 or 1, "little")).decode()


class SetTable:
    def __init__(self):
        self.index: dict[int, int] = {}
        self.items: list[str] = []

    def add(self, bits: int) -> int:
        i = self.index.get(bits)
        if i is None:
            i = self.index[bits] = len(self.items)
            self.items.append(b64(bits))
        return i


def structure(path: Path) -> list:
    try:
        tree = ast.parse(path.read_text(), str(path))
    except SyntaxError:
        return []
    out = []

    def visit(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                q = prefix + child.name
                start = min([d.lineno for d in child.decorator_list] + [child.lineno])
                out.append([start, child.end_lineno, "c", q, 0, 0, child.body[0].lineno])
                visit(child, q + ".")
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                q = prefix + child.name
                start = min([d.lineno for d in child.decorator_list] + [child.lineno])
                out.append([start, child.end_lineno, "f", q, 0, 0, child.body[0].lineno])
                visit(child, q + ".<locals>.")
            elif isinstance(child, ast.Lambda):
                out.append([child.lineno, child.end_lineno, "l", prefix + "<lambda>",
                            child.col_offset, child.end_col_offset, child.body.lineno])
                visit(child, prefix)
            else:
                visit(child, prefix)

    visit(tree, "")
    return out


def multiline_strings(path: Path) -> list:
    """[start, end, is_docstring] of string literals (incl. f-strings) spanning lines."""
    try:
        tree = ast.parse(path.read_text(), str(path))
    except SyntaxError:
        return []
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                docs.add(id(body[0].value))
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes))) \
                or isinstance(node, ast.JoinedStr):
            if node.end_lineno > node.lineno:
                out.append([node.lineno, node.end_lineno, int(id(node) in docs)])
    return sorted(out)


def mro(key: str, classes: dict, memo: dict) -> list:
    """Approximate MRO (all sqlglot ancestors) of a class key from the class table."""
    if key in memo:
        return memo[key]
    memo[key] = [key]
    seen = [key]
    for b in classes.get(key, {}).get("bases", []):
        for k in mro(b, classes, memo):
            if k not in seen:
                seen.append(k)
    memo[key] = seen
    return seen


def multiline_statements(path: Path) -> list:
    """[start, header_end] of statements whose header spans more than one line."""
    try:
        tree = ast.parse(path.read_text(), str(path))
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.stmt):
            continue
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.stmt):
            # compound statement: header = from start to the line before the first body stmt
            # (decorators excluded: defs/classes get their own handling)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            end = body[0].lineno - 1
        else:
            end = node.end_lineno
        if end > node.lineno:
            out.append([node.lineno, end])
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    meta = json.loads((a.trace / "meta.json").read_text())
    tests: list[str] = meta["tests"]
    n = len(tests)
    tindex = {t: i for i, t in enumerate(tests)}

    cov: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))  # file -> line -> bits
    any_bits: dict[str, int] = defaultdict(int)
    class_bits: dict[str, int] = defaultdict(int)  # "module:qualname" -> bits
    data_bits: dict[str, int] = defaultdict(int)
    subprocess_bits = 0
    lazy = defaultdict(lambda: defaultdict(set))
    problems = []
    for i, tid in enumerate(tests):
        p = a.trace / "tests" / f"{i}.json"
        if not p.exists():
            problems.append(f"missing trace for {tid}")
            continue
        d = json.loads(p.read_text())
        if not d["ok"]:
            problems.append(f"not ok under trace: {tid}: {(d['err'] or '')[:200]}")
        bit = 1 << i
        for f, lines in d["lines"].items():
            fc = cov[f]
            for ln in lines:
                fc[ln] |= bit
            any_bits[f] |= bit
        for k in d["classes"]:
            class_bits[k] |= bit
        for f in d["files"]:
            data_bits[f] |= bit
        if d["subprocess"]:
            subprocess_bits |= bit
        for m, fl in d.get("lazy_imports", {}).items():
            for f, lines in fl.items():
                lazy[m][f].update(lines)

    # Dialect composition: the _Dialect metaclass reads the tokenizer/parser/generator
    # classes' attributes when a dialect is created (QUOTE_START, TRANSFORMS pruning, ...),
    # so users of a dialect are users of its parts and of the parts' ancestors.
    memo = {}
    for d, parts in meta.get("dialects", {}).items():
        dbits = class_bits.get(d, 0)
        if not dbits:
            continue
        for part in parts.values():
            for k in mro(part, meta["classes"], memo):
                class_bits[k] |= dbits

    # import contexts: file -> line -> modules whose import executed it
    imp: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    for m, fl in meta["import_lines"].items():
        for f, lines in fl.items():
            for ln in lines:
                imp[f][ln].add(m)
    for m, fl in lazy.items():
        for f, lines in fl.items():
            for ln in lines:
                imp[f][ln].add(m)
    importers: dict[str, set[str]] = defaultdict(set)
    for parent, child in meta["import_edges"]:
        importers[child].add(parent)
    # static imports too (an import that happened earlier is not re-executed, so the dynamic
    # edge list misses e.g. a second module importing sqlglot.helper)
    mod_file = {}
    for f in [*ROOT.glob("sqlglot/**/*.py"), *ROOT.glob("tests/**/*.py")]:
        r = str(f.relative_to(ROOT))
        name = r[:-3].replace("/", ".")
        if name.endswith(".__init__"):
            name = name[: -len(".__init__")]
        mod_file[name] = r
    for r in mod_file.values():
        try:
            tree = ast.parse((ROOT / r).read_text())
        except SyntaxError:
            continue
        pkg = r[:-3].replace("/", ".").rsplit(".", 1)[0] if not r.endswith("__init__.py") \
            else r[: -len("/__init__.py")].replace("/", ".")
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [al.name for al in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parts = pkg.split(".")
                    base = ".".join(parts[: len(parts) - node.level + 1] + ([base] if base else []))
                names = [base] + [f"{base}.{al.name}" for al in node.names]
            for nm in names:
                if nm in mod_file and mod_file[nm] != r:
                    importers[mod_file[nm]].add(r)

    # collection-time import closure per test file (tools/import_closure.py)
    collect_bits: dict[str, int] = defaultdict(int)
    closure_path = a.trace / "closure.json"
    if closure_path.exists():
        closure = json.loads(closure_path.read_text())
        for tf, r in closure["test_files"].items():
            bits = 0
            for i, tid in enumerate(tests):
                if tid.split("::")[0] == tf:
                    bits |= 1 << i
            for f in r["files"]:
                collect_bits[f] |= bits
            if r.get("err"):
                problems.append(f"closure probe error for {tf}: {r['err']}")
        bare = sorted(closure["bare_import"]["files"])
    else:
        problems.append("no closure.json: collect_users empty")
        bare = []

    # tests defined per test file
    file_test_bits: dict[str, int] = defaultdict(int)
    for i, tid in enumerate(tests):
        file_test_bits[tid.split("::")[0]] |= 1 << i

    gsets = SetTable()
    a.out.mkdir(parents=True, exist_ok=True)
    blobs = bytearray()
    index = {}
    sources = sorted({str(p.relative_to(ROOT)) for p in [*ROOT.glob("sqlglot/**/*.py"),
                                                         *ROOT.glob("tests/**/*.py")]})
    for f in sources:
        module = f[:-3].replace("/", ".")
        if module.endswith(".__init__"):
            module = module[: -len(".__init__")]
        st = SetTable()
        struct = structure(ROOT / f)
        cu = {}
        mod_users = any_bits.get(f, 0)
        for s in struct:
            if s[2] == "c":
                bits = class_bits.get(f"{module}:{s[3]}", 0)
                cu[s[3]] = st.add(bits)
                mod_users |= bits
        payload = {
            "cov": {str(ln): st.add(bits) for ln, bits in sorted(cov.get(f, {}).items())},
            "imp": {str(ln): sorted(ms) for ln, ms in sorted(imp.get(f, {}).items())},
            "struct": struct,
            "any": st.add(any_bits.get(f, 0)),
            "class_users": cu,
            "mod_users": st.add(mod_users),
            "file_tests": st.add(file_test_bits.get(f, 0)),
            "collect_users": st.add(collect_bits.get(f, 0)),
            "stmts": multiline_statements(ROOT / f),
            "strings": multiline_strings(ROOT / f),
        }
        payload["sets"] = st.items
        blob = zlib.compress(json.dumps(payload, separators=(",", ":")).encode(), 9)
        index[f] = [len(blobs), len(blob)]
        blobs += blob

    out_meta = {
        "pinned": "eb11aec4be8cb2122afe713a4816662c3f0fb2fa",
        "tests": tests,
        "data_files": {f: gsets.add(b) for f, b in sorted(data_bits.items())},
        "importers": {k: sorted(v) for k, v in sorted(importers.items())},
        "core_modules": bare or sorted({mod_file.get(m, m) for m in meta["core_modules"]}),
        "subprocess": gsets.add(subprocess_bits),
        "all": gsets.add((1 << n) - 1),
        "files": index,
        "problems": problems,
    }
    out_meta["sets"] = gsets.items
    (a.out / "files.bin").write_bytes(bytes(blobs))
    (a.out / "meta.json").write_text(json.dumps(out_meta, separators=(",", ":")))
    for p in (a.out / "files.bin", a.out / "meta.json"):
        os.chmod(p, 0o644)
    print(f"files.bin {len(blobs) / 1e6:.2f} MB, meta.json "
          f"{(a.out / 'meta.json').stat().st_size / 1e3:.0f} kB, {len(index)} files, "
          f"{len(problems)} problems")
    for p in problems[:20]:
        print("  ", p)


if __name__ == "__main__":
    main()
