"""Generate synthetic patches in the sample's styles, for LOCAL EVALUATION ONLY.

The patches (and their oracle labels) are used to measure our selector's recall
and cost. They are never shipped in the image and never turned into a lookup
table (see CLAUDE.md section 5).

Kinds (same names as answers.json):
  code     one small edit on one line of a sqlglot/*.py file: and/or, comparison
           flips, is/is not, in/not in, adding or dropping `not`, True/False,
           integer +-1, arithmetic +/-, occasionally a string literal tweak
  fixture  corrupt one SQL line of a tests/fixtures/**.sql file (insert a token,
           split an identifier, drop or swap a word)
  comment  add a `# ...` line inside a function/class body or between imports
  multi    2-3 of the above in different files (usually comment noise + a real edit)

  python3 tools/mutate.py --src work/sqlglot --out work/synth --n 300 --seed 1
writes <out>/commits/<id>.patch and <out>/kinds.json.
"""

import argparse
import ast
import difflib
import io
import json
import random
import re
import tokenize
from pathlib import Path

COMMENTS = [
    "Cheap check first", "Keep in sync with the generator", "Intentionally not cached",
    "Kept for backwards compatibility", "TODO: revisit", "Fast path", "See above",
    "Order matters here", "Handled by the caller", "NOTE: mutates in place",
    "Edge case: empty input", "Preserve original behaviour", "Avoid recursion",
]

SWAPS = {
    "and": ["or"], "or": ["and"], "True": ["False"], "False": ["True"],
    "==": ["!="], "!=": ["=="], "<": ["<=", ">"], "<=": ["<"], ">": [">=", "<"], ">=": [">"],
    "+": ["-"], "-": ["+"], "is": ["is not", "=="], "in": ["not in"],
}
PROFILE = {"name": "synth"}  # --profile; "sample" changes the mix (see main)

FIX_TOKENS = ["x", "1", "a", "AS", "SELECT", ",", "(", "FROM"]


def udiff(rel: str, old: str, new: str) -> str:
    a = old.splitlines(keepends=True)
    b = new.splitlines(keepends=True)
    body = "".join(difflib.unified_diff(a, b, f"a/{rel}", f"b/{rel}", n=3))
    if not body:
        return ""
    # difflib emits "--- a/x\n+++ b/x\n" with no trailing tab when dates are absent
    head = f"diff --git a/{rel} b/{rel}\nindex 0000000..1111111 100644\n"
    return head + body


# ---------------------------------------------------------------- code

def code_candidates(text: str):
    """(line_index, col_start, col_end, original, replacement) candidates."""
    out = []
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, SyntaxError):
        return out
    for i, t in enumerate(toks):
        (r, c), (r2, c2) = t.start, t.end
        if r != r2:
            continue
        s = t.string
        nxt = toks[i + 1] if i + 1 < len(toks) else None
        prv = toks[i - 1] if i else None
        if t.type == tokenize.NAME and s in ("and", "or", "True", "False"):
            for rep in SWAPS[s]:
                out.append((r - 1, c, c2, s, rep))
        elif t.type == tokenize.NAME and s == "is":
            if nxt is not None and nxt.string == "not":
                out.append((r - 1, c, nxt.end[1], "is not", "is"))
                if PROFILE["name"] == "sample":
                    out.append((r - 1, c, nxt.end[1], "is not", "!="))
            else:
                for rep in SWAPS[s]:
                    out.append((r - 1, c, c2, s, rep))
        elif t.type == tokenize.NAME and s == "in" and prv is not None and prv.string not in ("for", "not"):
            if prv.type in (tokenize.NAME, tokenize.OP, tokenize.STRING, tokenize.NUMBER):
                out.append((r - 1, c, c2, s, "not in"))
        elif t.type == tokenize.NAME and s == "not" and nxt is not None and nxt.string != "in" \
                and (prv is None or prv.string != "is"):
            out.append((r - 1, c, nxt.start[1], "not ", ""))
        elif t.type == tokenize.NAME and s in ("if", "elif", "while") and nxt is not None \
                and nxt.string != "not" and nxt.start[0] == r:
            out.append((r - 1, nxt.start[1], nxt.start[1], "", "not "))
        elif t.type == tokenize.OP and s in ("==", "!=", "<", "<=", ">", ">="):
            for rep in SWAPS[s]:
                out.append((r - 1, c, c2, s, rep))
        elif t.type == tokenize.OP and s in ("+", "-") and prv is not None and \
                prv.type in (tokenize.NAME, tokenize.NUMBER) and prv.string not in ("return", "in", "and", "or", "not", "if", "else", "lambda"):
            out.append((r - 1, c, c2, s, SWAPS[s][0]))
        elif t.type == tokenize.NUMBER and re.fullmatch(r"\d+", s):
            n = int(s)
            out.append((r - 1, c, c2, s, str(n + 1)))
            if n > 0:
                out.append((r - 1, c, c2, s, str(n - 1)))
        elif t.type == tokenize.STRING and re.fullmatch(r"[\"'][A-Za-z_ ]{2,30}[\"']", s):
            out.append((r - 1, c, c2, s, s[:-1] + s[-2] + s[-1]))
    return out


def mutate_code(src: Path, rel: str, rng: random.Random):
    text = (src / rel).read_text()
    cands = code_candidates(text)
    if PROFILE["name"] == "sample":
        # identity -> equality swaps are frequent in the sample (and usually equivalent)
        cands = cands + [cnd for cnd in cands if cnd[3] in ("is", "is not") and cnd[4] in ("==", "!=")]
    # string-literal tweaks are rare in the sample: ~5% of code mutants
    is_str = lambda cnd: cnd[3][:1] in "\"'"
    if rng.random() < 0.05:
        cands = [cnd for cnd in cands if is_str(cnd)] or cands
    else:
        cands = [cnd for cnd in cands if not is_str(cnd)]
    rng.shuffle(cands)
    lines = text.splitlines(keepends=True)
    for li, c1, c2, orig, rep in cands[:50]:
        line = lines[li]
        new_line = line[:c1] + rep + line[c2:]
        if new_line == line:
            continue
        new = "".join(lines[:li] + [new_line] + lines[li + 1:])
        try:
            ast.parse(new)
        except SyntaxError:
            continue
        return new
    return None


# ---------------------------------------------------------------- comment

def mutate_comment(src: Path, rel: str, rng: random.Random):
    text = (src / rel).read_text()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    lines = text.splitlines(keepends=True)
    # statement start lines that are inside a def/class body, or top-level imports
    spots = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for st in node.body:
                spots.append(st.lineno - 1)
    for st in tree.body:
        if isinstance(st, (ast.Import, ast.ImportFrom)):
            spots.append(st.lineno - 1)
    if not spots:
        return None
    li = rng.choice(spots)
    target = lines[li]
    indent = target[: len(target) - len(target.lstrip())]
    new_lines = lines[:li] + [f"{indent}# {rng.choice(COMMENTS)}\n"] + lines[li:]
    new = "".join(new_lines)
    try:
        ast.parse(new)
    except SyntaxError:
        return None
    return new


# ---------------------------------------------------------------- fixture

def mutate_fixture(src: Path, rel: str, rng: random.Random):
    text = (src / rel).read_text()
    lines = text.splitlines(keepends=True)
    idx = [i for i, l in enumerate(lines) if l.strip() and not l.lstrip().startswith(("#", "--"))]
    if not idx:
        return None
    for _ in range(20):
        li = rng.choice(idx)
        line = lines[li].rstrip("\n")
        words = line.split(" ")
        op = rng.choice(["insert", "insert", "split", "drop", "swap"])
        if op == "insert":
            p = rng.randrange(len(words) + 1)
            words = words[:p] + [rng.choice(FIX_TOKENS)] + words[p:]
            new_line = " ".join(words)
        elif op == "split":
            m = [mm for mm in re.finditer(r"\b[a-z_]\w*\.\w+", line)]
            if not m:
                continue
            mm = rng.choice(m)
            dot = line.index(".", mm.start())
            new_line = line[:dot] + rng.choice("abxyz") + " " + line[dot:]
        elif op == "drop":
            if len(words) < 3:
                continue
            p = rng.randrange(len(words))
            new_line = " ".join(words[:p] + words[p + 1:])
        else:
            if len(words) < 3:
                continue
            p = rng.randrange(len(words) - 1)
            words[p], words[p + 1] = words[p + 1], words[p]
            new_line = " ".join(words)
        if new_line != line:
            return "".join(lines[:li] + [new_line + "\n"] + lines[li + 1:])
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--prefix", default="s")
    ap.add_argument("--test-code-frac", type=float, default=0.0,
                    help="fraction of code mutants placed in tests/**/*.py instead of sqlglot/")
    ap.add_argument("--profile", choices=["synth", "sample"], default="synth",
                    help="sample: the sample's kind mix (code 16 / fixture 8 / multi 3 / comment 3), "
                         "mutation points uniform over the code base, more is -> == swaps")
    a = ap.parse_args()
    PROFILE["name"] = a.profile
    rng = random.Random(a.seed)
    random.seed(a.seed)

    py = sorted(str(p.relative_to(a.src)) for p in (a.src / "sqlglot").rglob("*.py"))
    test_py = sorted(str(r) for r in (p.relative_to(a.src) for p in (a.src / "tests").rglob("*.py"))
                     if "fixtures" not in r.parts and r.name not in ("gen_fixtures.py", "leakcheck.py")
                     and "sqlglot" not in r.parts[1:])
    fx_opt = sorted(str(p.relative_to(a.src)) for p in (a.src / "tests/fixtures/optimizer").glob("*.sql"))
    fx_other = sorted(str(p.relative_to(a.src)) for p in (a.src / "tests/fixtures").glob("*.sql"))

    weights = None
    if a.profile == "sample":  # pick files in proportion to their mutation points
        weights = [len(code_candidates((a.src / r).read_text())) + 1 for r in py]

    def one(kind):
        for _ in range(50):
            if kind == "code":
                if rng.random() < a.test_code_frac:
                    rel = rng.choice(test_py)
                elif weights:
                    rel = rng.choices(py, weights)[0]
                else:
                    rel = rng.choice(py)
                new = mutate_code(a.src, rel, rng)
            elif kind == "comment":
                rel = rng.choice(py)
                new = mutate_comment(a.src, rel, rng)
            else:
                rel = rng.choice(fx_opt if rng.random() < 0.85 else fx_other)
                new = mutate_fixture(a.src, rel, rng)
            if new is not None:
                d = udiff(rel, (a.src / rel).read_text(), new)
                if d:
                    return rel, d
        raise RuntimeError(kind)

    (a.out / "commits").mkdir(parents=True, exist_ok=True)
    kinds = {}
    for i in range(a.n):
        mix = [53, 27, 10, 10] if a.profile == "sample" else [50, 25, 10, 15]
        kind = rng.choices(["code", "fixture", "comment", "multi"], mix)[0]
        if kind == "multi":
            parts = [one("comment")]
            for k in rng.sample(["code", "fixture", "comment", "code"], rng.choice([1, 1, 2])):
                parts.append(one(k))
            seen, diffs = set(), []
            for rel, d in parts:
                if rel not in seen:
                    seen.add(rel)
                    diffs.append((rel, d))
            patch = "".join(d for _, d in sorted(diffs))
        else:
            patch = one(kind)[1]
        cid = f"{a.prefix}{i:04d}"
        (a.out / "commits" / f"{cid}.patch").write_text(patch)
        kinds[cid] = kind
    (a.out / "kinds.json").write_text(json.dumps(kinds, indent=1))
    print(f"wrote {a.n} patches to {a.out}")


if __name__ == "__main__":
    main()
