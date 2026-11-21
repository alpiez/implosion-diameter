"""Hand-written edge-case patches (LOCAL EVALUATION ONLY, like mutate.py).

They cover the risks in CLAUDE.md section 4 that random mutants rarely hit:
comment-looking edits that do matter, whitespace-only edits, test-file and
helper edits, and new files.

  python3 tools/edge.py --src work/sqlglot --out work/edge
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mutate import udiff  # noqa: E402


def insert_before(text: str, needle: str, new_line: str, keep_indent=True) -> str:
    lines = text.splitlines(keepends=True)
    for i, l in enumerate(lines):
        if needle in l:
            indent = l[: len(l) - len(l.lstrip())] if keep_indent else ""
            return "".join(lines[:i] + [indent + new_line + "\n"] + lines[i:])
    raise KeyError(needle)


def replace_once(text: str, old: str, new: str) -> str:
    assert text.count(old) >= 1, old
    return text.replace(old, new, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    src = a.src

    def rd(rel):
        return (src / rel).read_text()

    first_sql = lambda rel: next(l for l in rd(rel).splitlines() if l.strip() and not l.startswith(("#", "--")))
    pp = "tests/fixtures/optimizer/pushdown_predicates.sql"
    cases = {
        # (kind, rel, new_text) ; kind is our own label for analysis
        "e01": ("fixture-comment", pp, insert_before(rd(pp), first_sql(pp), "-- harmless sql comment", False)),
        "e02": ("fixture-comment", pp, insert_before(rd(pp), first_sql(pp), "# a note without colon", False)),
        "e03": ("fixture-meta", pp, insert_before(rd(pp), first_sql(pp), "# dialect: tsql", False)),
        "e04": ("docstring", "sqlglot/optimizer/pushdown_predicates.py",
                replace_once(rd("sqlglot/optimizer/pushdown_predicates.py"), ">>> ", ">>>  ")),
        "e05": ("whitespace", "sqlglot/optimizer/simplify.py",
                insert_before(rd("sqlglot/optimizer/simplify.py"), "def simplify(", "", False)),
        "e06": ("test-edit", "tests/test_tokens.py",
                replace_once(rd("tests/test_tokens.py"), "self.assertEqual(", "self.assertNotEqual(")),
        "e07": ("test-helper", "tests/helpers.py",
                replace_once(rd("tests/helpers.py"), 'line and not line.startswith("--")',
                             'line and not line.startswith("-- ")')),
        "e08": ("fixture-comment", "tests/fixtures/identity.sql",
                insert_before(rd("tests/fixtures/identity.sql"), first_sql("tests/fixtures/identity.sql"),
                              "# a note", False)),
        "e09": ("fixture", "tests/fixtures/pretty.sql",
                replace_once(rd("tests/fixtures/pretty.sql"), "SELECT", "SELECT  ")),
        "e10": ("comment", "sqlglot/__init__.py",
                insert_before(rd("sqlglot/__init__.py"), "import ", "# module level note", False)),
        "e11": ("comment-in-string", "sqlglot/optimizer/simplify.py", None),  # filled below
    }
    # e11: a '#' added inside a multi-line string literal that is not a docstring, if any
    import ast
    for rel in ["sqlglot/optimizer/simplify.py", "sqlglot/generator.py", "sqlglot/parser.py",
                "sqlglot/dialects/dialect.py", "sqlglot/executor/python.py"]:
        text = rd(rel)
        tree = ast.parse(text)
        doc_ids = set()
        for n in ast.walk(tree):
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body \
                    and isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0], "value", None), ast.Constant):
                doc_ids.add(id(n.body[0].value))
        hit = None
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in doc_ids \
                    and n.end_lineno > n.lineno \
                    and text.splitlines()[n.lineno - 1][n.col_offset:].lstrip("rbuRBUfF")[:3] in ('"""', "'''"):
                hit = n
                break
        if hit:
            lines = text.splitlines(keepends=True)
            li = hit.lineno  # 0-based index of the second line of the literal
            lines.insert(li, "# looks like a comment\n")
            cases["e11"] = ("comment-in-string", rel, "".join(lines))
            break
    else:
        del cases["e11"]

    (a.out / "commits").mkdir(parents=True, exist_ok=True)
    kinds = {}
    for cid, (kind, rel, new) in cases.items():
        d = udiff(rel, rd(rel), new)
        assert d, cid
        (a.out / "commits" / f"{cid}.patch").write_text(d)
        kinds[cid] = kind
    # e12: a brand-new module (new-file diff)
    body = "X = 1\n"
    (a.out / "commits" / "e12.patch").write_text(
        "diff --git a/sqlglot/newmod.py b/sqlglot/newmod.py\nnew file mode 100644\n"
        "index 0000000..1111111\n--- /dev/null\n+++ b/sqlglot/newmod.py\n@@ -0,0 +1 @@\n+" + body)
    kinds["e12"] = "new-file"
    (a.out / "kinds.json").write_text(json.dumps(kinds, indent=1))
    print(f"wrote {len(kinds)} edge patches")


if __name__ == "__main__":
    main()
