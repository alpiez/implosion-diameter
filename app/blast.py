"""Blast Radius selector: which sqlglot tests can a patch break?

For each /data/commits/<id>.patch, print {"commit": id, "workers": [[...], ...]}.

Everything is decided from maps built offline from the PINNED commit only
(tools/make_maps.sh): per-test line coverage, lines executed at import (by
importing module), class usage, data files read, and collection-time import
closure. No patch, label or "kind of edit" is ever looked up (CLAUDE.md section 5).

Rules, per changed file (details in CHANGELOG.md, Phase 3):
  * .py edits whose changed lines are all blank/comment lines outside any string
    literal, or that only change trailing whitespace/comments, or only touch
    docstrings, are non-semantic: no tests (except tests reading the file as text).
  * Other .py edits: for every affected old line (removed lines; for insertions,
    the nearest executable lines around the insertion point; expanded to whole
    multi-line statement headers):
      - function body line        -> tests that executed it
        (+ if it also ran at import: users of the importing modules, tests whose
         files import them at collection, and subprocess tests for core modules)
      - def/class header line     -> callers (tests executing the body) / class users
      - class body line           -> tests that used the class, a subclass, or an
                                     enclosing class (dialects include their parts)
      - lambda body in a class    -> tests that executed the lambda
      - module-level line         -> users of the module and of its importers
      - test files                -> plus the file's tests / files importing it
  * tests/fixtures and other data files -> tests that opened the file (or listed its
    directory). SQL fixture lines starting with "--" or blank are dropped by the
    fixture loader (tests/helpers.py:11) and select nothing.
  * Anything unparseable or unknown falls back to wider sets, ultimately all tests.
Then the selected tests are packed onto 1..8 workers (LPT), keeping the cheapest k.
"""

import base64
import json
import os
import re
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("BR_DATA", "/data"))
MAPS = Path(os.environ.get("BR_MAPS") or (HERE / "maps" if (HERE / "maps").is_dir()
                                          else HERE.parent / "maps"))
WORKER_MS = 2000
MAX_WORKERS = 8
CONFIG_FILES = {"pyproject.toml", "setup.cfg", "pytest.ini", "tox.ini", "conftest.py",
                "tests/conftest.py"}


def log(*a) -> None:
    print("[select]", *a, file=sys.stderr)


# ------------------------------------------------------------------------- maps

class Maps:
    def __init__(self, d: Path):
        self.meta = json.loads((d / "meta.json").read_text())
        self.blob_path = d / "files.bin"
        self.blob = None
        self.gcache: dict[int, int] = {}
        self.fcache: dict[str, dict | None] = {}
        self.tests = self.meta["tests"]

    @staticmethod
    def dec(s: str) -> int:
        return int.from_bytes(base64.b64decode(s), "little")

    def g(self, i: int) -> int:
        v = self.gcache.get(i)
        if v is None:
            v = self.gcache[i] = self.dec(self.meta["sets"][i])
        return v

    @property
    def all(self) -> int:
        return self.g(self.meta["all"])

    def file(self, f: str):
        if f in self.fcache:
            return self.fcache[f]
        ent = self.meta["files"].get(f)
        p = None
        if ent is not None:
            if self.blob is None:
                self.blob = self.blob_path.read_bytes()
            p = json.loads(zlib.decompress(self.blob[ent[0]:ent[0] + ent[1]]))
            p["_c"] = {}
        self.fcache[f] = p
        return p

    def s(self, p: dict, i) -> int:
        if i is None:
            return 0
        c = p["_c"]
        v = c.get(i)
        if v is None:
            v = c[i] = self.dec(p["sets"][i])
        return v

    def data_users(self, path: str, include_dirs: bool = True) -> int:
        bits = 0
        df = self.meta["data_files"]
        si = df.get(path)
        if si is not None:
            bits |= self.g(si)
        if include_dirs:
            parent = path.rsplit("/", 1)[0] + "/" if "/" in path else ""
            si = df.get(parent)
            if si is not None:
                bits |= self.g(si)
        return bits


# ------------------------------------------------------------------------- diff parsing

class Block:
    """A run of removed and/or added lines inside one hunk."""

    __slots__ = ("old_start", "removed", "added", "before", "after")

    def __init__(self, old_start: int, before: str | None):
        self.old_start = old_start  # old line number of the first removed line / insertion point
        self.removed: list[tuple[int, str]] = []
        self.added: list[str] = []
        self.before = before  # text of the old line just before the block (if known)
        self.after = None


class FileChange:
    __slots__ = ("old", "new", "blocks", "binary", "hunks")

    def __init__(self):
        self.old: str | None = None
        self.new: str | None = None
        self.blocks: list[Block] = []
        self.binary = False
        # raw hunks for applying the patch (phase 4): [old_start, old_count, [(tag, text)],
        # old_no_eol, new_no_eol, new_count]
        self.hunks: list[list] = []

    @property
    def path(self) -> str | None:
        return self.old or self.new


def _unquote(p: str) -> str:
    p = p.strip()
    if p.startswith('"') and p.endswith('"'):
        p = p[1:-1].encode("latin-1", "backslashreplace").decode("unicode_escape").encode(
            "latin-1", "ignore").decode("utf-8", "replace")
    return p


def _strip_prefix(p: str) -> str | None:
    p = _unquote(p.split("\t")[0])
    if p == "/dev/null":
        return None
    if p.startswith(("a/", "b/")):
        return p[2:]
    return p


HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_patch(text: str) -> list[FileChange]:
    files: list[FileChange] = []
    cur: FileChange | None = None
    lines = text.replace("\r\n", "\n").split("\n")
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if line.startswith("diff --git "):
            cur = FileChange()
            files.append(cur)
            m = re.match(r'^diff --git (".*?"|\S+) (".*?"|\S+)$', line)
            if m:
                cur.old, cur.new = _strip_prefix(m.group(1)), _strip_prefix(m.group(2))
            i += 1
            continue
        if line.startswith("--- ") and i + 1 < n and lines[i + 1].startswith("+++ "):
            if cur is None or cur.blocks:
                cur = FileChange()
                files.append(cur)
            cur.old = _strip_prefix(line[4:])
            cur.new = _strip_prefix(lines[i + 1][4:])
            i += 2
            continue
        if cur is not None and line.startswith(("new file mode", "deleted file mode")):
            if line.startswith("new"):
                cur.old = None
            else:
                cur.new = None
            i += 1
            continue
        if cur is not None and line.startswith("rename from "):
            cur.old = _unquote(line[len("rename from "):])
            i += 1
            continue
        if cur is not None and line.startswith("rename to "):
            cur.new = _unquote(line[len("rename to "):])
            i += 1
            continue
        if cur is not None and (line.startswith("Binary files") or line.startswith("GIT binary patch")):
            cur.binary = True
            i += 1
            continue
        m = HUNK.match(line)
        if m and cur is not None:
            old_ln = int(m.group(1))
            old_cnt = int(m.group(2)) if m.group(2) is not None else 1
            new_cnt = int(m.group(4)) if m.group(4) is not None else 1
            hunk = [old_ln, old_cnt, [], False, False, new_cnt]
            cur.hunks.append(hunk)
            if old_cnt == 0:
                old_ln += 1  # "-k,0" means: insert after old line k
            i += 1
            o_left, n_left = old_cnt, new_cnt
            block = None
            prev_text = None
            while i < n and (o_left > 0 or n_left > 0):
                hl = lines[i]
                if hl.startswith("\\"):
                    if hunk[2]:
                        t = hunk[2][-1][0]
                        hunk[3] = hunk[3] or t in " -"
                        hunk[4] = hunk[4] or t in " +"
                    i += 1
                    continue
                tag, body = (hl[:1], hl[1:]) if hl else (" ", "")
                hunk[2].append((tag if tag in "+-" else " ", body))
                if tag == "-":
                    if block is None:
                        block = Block(old_ln, prev_text)
                        cur.blocks.append(block)
                    block.removed.append((old_ln, body))
                    old_ln += 1
                    o_left -= 1
                elif tag == "+":
                    if block is None:
                        block = Block(old_ln, prev_text)
                        cur.blocks.append(block)
                    block.added.append(body)
                    n_left -= 1
                else:
                    if block is not None:
                        block.after = body
                    block = None
                    prev_text = body
                    old_ln += 1
                    o_left -= 1
                    n_left -= 1
                i += 1
            # a trailing "\ No newline" line right after the hunk's last line
            if i < n and lines[i].startswith("\\") and hunk[2]:
                t = hunk[2][-1][0]
                hunk[3] = hunk[3] or t in " -"
                hunk[4] = hunk[4] or t in " +"
                i += 1
            continue
        i += 1
    return files


# ------------------------------------------------------------------------- python helpers

COMMENT_LINE = re.compile(r"^\s*(#.*)?$")
CODING = re.compile(r"^[ \t\f]*#.*?coding[:=]")


def _code_part(line: str) -> str | None:
    """The line without its trailing comment and trailing whitespace; None if unsure."""
    out = []
    q = None
    i = 0
    while i < len(line):
        ch = line[i]
        if q:
            if ch == "\\":
                out.append(line[i:i + 2])
                i += 2
                continue
            if line.startswith(q, i):
                out.append(q)
                i += len(q)
                q = None
                continue
            out.append(ch)
        else:
            if ch == "#":
                break
            if ch in "'\"":
                q = line[i:i + 3] if line[i:i + 3] in ("'''", '"""') else ch
                out.append(q)
                i += len(q)
                continue
            out.append(ch)
        i += 1
    if q:
        return None  # unterminated string on this line: can't tell
    return "".join(out).rstrip()


def _in_string(strings: list, line: int) -> bool:
    for s, e, _ in strings:
        if s <= line <= e:
            return True
    return False


def _boundary_in_string(strings: list, k: int) -> bool:
    """Is the boundary between old lines k-1 and k inside a multi-line string?"""
    for s, e, _ in strings:
        if s <= k - 1 and e >= k:
            return True
    return False


def _docstring_only(strings: list, block: Block) -> bool:
    """All removed lines strictly inside one docstring, insertion too."""
    lines = [ln for ln, _ in block.removed] or [block.old_start]
    for s, e, doc in strings:
        if doc and all(s < ln < e for ln in lines) and (block.removed or s < block.old_start <= e):
            return True
    return False


def non_semantic(p: dict, block: Block) -> bool:
    strings = p.get("strings", [])
    texts = [t for _, t in block.removed] + block.added
    if block.before is not None and block.before.rstrip().endswith("\\"):
        return False  # a line inserted after a continuation would change the statement
    if all(COMMENT_LINE.match(t) for t in texts):
        if any(_in_string(strings, ln) for ln, _ in block.removed):
            return _docstring_only(strings, block)
        if block.added and _boundary_in_string(strings, block.old_start):
            return _docstring_only(strings, block)
        if block.old_start <= 2 and any(CODING.match(t) for t in texts):
            return False
        return True
    if _docstring_only(strings, block):
        return True
    if len(block.removed) == len(block.added):
        for (ln, old), new in zip(block.removed, block.added):
            if _in_string(strings, ln):
                return False
            a, b = _code_part(old), _code_part(new)
            if a is None or b is None or a != b:
                return False
        return True
    return False


def scopes(struct: list, line: int) -> list:
    """Enclosing def/class/lambda entries of a line, outermost first."""
    out = [e for e in struct if e[0] <= line <= e[1]]
    out.sort(key=lambda e: (e[0], -e[1]))
    return out


def changed_columns(block: Block, ln: int):
    """Column range [a, b) of the change on old line ln (for a 1:1 line edit), else None."""
    idx = [x for x, _ in block.removed].index(ln) if any(x == ln for x, _ in block.removed) else -1
    if idx < 0 or len(block.removed) != len(block.added):
        return None
    old, new = block.removed[idx][1], block.added[idx]
    p = 0
    while p < min(len(old), len(new)) and old[p] == new[p]:
        p += 1
    s = 0
    while s < min(len(old), len(new)) - p and old[-1 - s] == new[-1 - s]:
        s += 1
    return p, len(old) - s


# ------------------------------------------------------------------------- selection

ASSIGN_START = re.compile(r"^\s*([A-Z][A-Z0-9_]*)\s*(?::[^=]*)?=(?!=)")


def _entry_key(line: str) -> str | None:
    """Text before the first ':' at bracket depth 0 outside strings (a dict entry's key)."""
    depth = 0
    quote = None
    i = 0
    while i < len(line):
        ch = line[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if line.startswith(quote, i):
                i += len(quote)
                quote = None
                continue
        elif ch in "\"'":
            quote = line[i:i + 3] if line[i:i + 3] in ('"""', "'''") else ch
            i += len(quote)
            continue
        elif ch == "#":
            return None
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == ":" and depth == 0:
            return line[:i].strip()
        i += 1
    return None


class Selector:
    def __init__(self, maps: Maps):
        self.m = maps
        self.core = set(maps.meta.get("core_modules", []))
        self.noncov = 0                           # tests selected NOT via runtime coverage
        self.covlines: list[tuple[str, int]] = []  # (file, line) whose coverage was used
        self.use_attrs = True                     # phase 5a attribute rule (needs an import probe)
        self.attr_files: list[str] = []           # files where the attribute rule was used

    # ---- users of a module: tests that executed it or used a class defined in it
    def mod_users(self, f: str, with_importers: bool = True) -> int:
        p = self.m.file(f)
        bits = 0
        if p:
            bits |= self.m.s(p, p["mod_users"])
            if f.startswith("tests/"):
                bits |= self.m.s(p, p["file_tests"]) | self.m.s(p, p["collect_users"])
        if with_importers:
            for imp in self.m.meta["importers"].get(f, []):
                q = self.m.file(imp)
                if q:
                    bits |= self.m.s(q, q["mod_users"])
                    if imp.startswith("tests/"):
                        bits |= self.m.s(q, q["file_tests"]) | self.m.s(q, q["collect_users"])
        if f in self.core:
            bits |= self.m.g(self.m.meta["subprocess"])
        return bits

    def collect_users(self, f: str) -> int:
        p = self.m.file(f)
        return self.m.s(p, p["collect_users"]) if p else 0

    def class_users(self, p: dict, f: str, qual: str) -> int:
        """Users of class `qual` in file f, and of every enclosing class."""
        bits = 0
        parts = qual.split(".")
        for i in range(len(parts), 0, -1):
            q = ".".join(parts[:i])
            si = p["class_users"].get(q)
            if si is not None:
                bits |= self.m.s(p, si)
        return bits

    def body_cov(self, p: dict, start: int, end: int) -> int:
        bits = 0
        cov = p["cov"]
        for ln in range(start, end + 1):
            si = cov.get(str(ln))
            if si is not None:
                bits |= self.m.s(p, si)
        return bits

    def affected_lines(self, p: dict, block: Block) -> set[int]:
        lines = {ln for ln, _ in block.removed}
        if block.added and not block.removed:
            known = set(map(int, p["cov"])) | set(map(int, p["imp"]))
            k = block.old_start
            for step, start in ((-1, k - 1), (1, k)):
                ln = start
                for _ in range(40):
                    if ln < 1:
                        break
                    if ln in known:
                        lines.add(ln)
                        break
                    ln += step
            lines.add(k - 1)
            lines.add(k)
        for s0, s1 in p.get("stmts", []):
            if any(s0 <= ln <= s1 for ln in list(lines)):
                lines |= set(range(s0, s1 + 1))
        return {ln for ln in lines if ln >= 1}

    def python_line(self, f: str, p: dict, ln: int, block: Block) -> int:
        """Tests for one affected line. Records in self.covlines the (file, line) pairs whose
        runtime coverage was used, and ORs every other source into self.noncov (phase 4:
        split tests can be narrowed to units only when selected through coverage alone)."""
        bits = 0
        nc = 0
        cov = p["cov"].get(str(ln))
        if cov is not None:
            bits |= self.m.s(p, cov)
            self.covlines.append((f, ln))
        mods = p["imp"].get(str(ln))
        chain = scopes(p["struct"], ln)
        inner = chain[-1] if chain else None
        # lambda: if the change sits entirely inside the lambda's body it is runtime-only
        if inner is not None and inner[2] == "l":
            cols = changed_columns(block, ln)
            s, e, _, _, c0, c1, _ = inner
            inside = s < ln < e or (
                cols is not None and (ln > s or cols[0] >= c0) and (ln < e or cols[1] <= c1))
            if inside:
                return bits
            chain = [c for c in chain if c[2] != "l"]
            inner = chain[-1] if chain else None
        if inner is not None and inner[2] in "fc" and ln < inner[6]:
            # header line (decorators, signature, default values, bases)
            if inner[2] == "f":
                bits |= self.body_cov(p, inner[6], inner[1])
                self.covlines.extend((f, x) for x in range(inner[6], inner[1] + 1))
            else:
                nc |= self.class_users(p, f, inner[3]) or self.mod_users(f)
                nc |= self.body_cov(p, inner[0], inner[1])
            chain = chain[:-1]
            inner = chain[-1] if chain else None
            if (inner is None or inner[2] == "c") and f.startswith("tests/"):
                nc |= self.m.s(p, p["file_tests"])  # test methods are collected by name
            self.noncov |= nc
            return bits | nc
        if inner is not None and inner[2] == "f":
            if mods:  # a function body that ran at import time (metaclass, module-level call)
                for mf in mods:
                    nc |= self.mod_users(mf) | self.collect_users(mf)
            self.noncov |= nc
            return bits | nc
        if inner is not None and inner[2] == "c":
            cu = self.class_users(p, f, inner[3])
            if f.startswith("tests/"):
                cu |= self.m.s(p, p["file_tests"]) | self.m.s(p, p["collect_users"])
            if not cu:
                cu = self.mod_users(f)
            self.noncov |= cu
            return bits | cu
        # module level
        nc |= self.mod_users(f)
        self.noncov |= nc
        return bits | nc

    def attr_block(self, f: str, p: dict, block: Block) -> int | None:
        """Phase 5a: an edit confined to one UPPER_CASE class attribute (maps "attrs").
        Key level when only values of existing table entries change; attribute level for
        other edits inside the statement. None = not applicable (use the line rules)."""
        attrs = p.get("attrs")
        if not attrs:
            return None
        lines = [ln for ln, _ in block.removed] or [block.old_start - 1, block.old_start]
        ents = [attrs.get(str(ln)) for ln in lines]
        if any(e is None for e in ents) or len({e[3] for e in ents}) != 1:
            return None
        s0 = ents[0][3]
        for st in p["struct"]:  # edits inside a lambda body are runtime: line rules
            if st[2] == "l" and any(st[0] <= ln <= st[1] for ln in lines):
                return None
        removed = [t for _, t in block.removed]
        added = block.added
        if not removed and any(ASSIGN_START.match(t) for t in added):
            return None  # a new attribute
        if s0 in lines and block.removed:  # the target line itself: the name must not change
            i = lines.index(s0)
            m_old = ASSIGN_START.match(removed[i])
            m_new = ASSIGN_START.match(added[i]) if i < len(added) else None
            if not m_old or not m_new or m_old.group(1) != m_new.group(1):
                return None
        bits = 0
        cov = p["cov"]
        for ln in lines:
            si = cov.get(str(ln))
            if si is not None:
                bits |= self.m.s(p, si)
        key_ok = bool(removed) and len(removed) == len(added) and all(e[0] >= 0 for e in ents)
        if key_ok:
            for e, old, new in zip(ents, removed, added):
                if e[2] and _entry_key(old) != _entry_key(new):
                    key_ok = False
                    break
        if key_ok:
            for e in ents:
                bits |= self.m.s(p, e[0])
            return bits
        if ents[0][1] < 0:
            return None
        return bits | self.m.s(p, ents[0][1])

    def python_file(self, fc: FileChange, why: list) -> int:
        f = fc.old
        p = self.m.file(f)
        bits = self.m.data_users(f, include_dirs=False)  # tests reading the source as text
        self.noncov |= bits
        if p is None:
            why.append(f"{f}: not in maps -> all")
            self.noncov |= self.m.all
            return self.m.all
        if fc.new is None or fc.new != fc.old:  # deleted or renamed away
            why.append(f"{f}: deleted/renamed -> module users + collection")
            r = bits | self.mod_users(f) | self.collect_users(f) | self.body_cov(
                p, 1, max([e[1] for e in p["struct"]] + [1]) + 1000)
            self.noncov |= r
            return r
        for block in fc.blocks:
            if non_semantic(p, block):
                continue
            ab = self.attr_block(f, p, block) if self.use_attrs else None
            if ab is not None:
                bits |= ab
                self.noncov |= ab
                if f not in self.attr_files:
                    self.attr_files.append(f)
                why.append(f"{f}@{block.old_start}: class attribute -> {bin(ab).count('1')}")
                continue
            for ln in sorted(self.affected_lines(p, block)):
                b = self.python_line(f, p, ln, block)
                bits |= b
            why.append(f"{f}@{block.old_start}: semantic")
        return bits

    def data_file(self, fc: FileChange, why: list) -> int:
        f = fc.path
        if f in CONFIG_FILES or f.endswith("/conftest.py"):
            why.append(f"{f}: config -> all")
            return self.m.all
        if f.startswith("tests/fixtures/") and f.endswith(".sql") and fc.old and fc.new == fc.old:
            texts = [t for b in fc.blocks for t in [x for _, x in b.removed] + b.added]
            if texts and all((not t) or t.startswith("--") for t in texts):
                why.append(f"{f}: only '--' comment / blank lines (dropped by the loader)")
                return 0
        bits = 0
        for path in {fc.old, fc.new} - {None}:
            bits |= self.m.data_users(path)
        why.append(f"{f}: data -> {bin(bits).count('1')}")
        return bits

    def select(self, text: str, why: list) -> int:
        self.noncov = 0
        self.covlines = []
        self.attr_files = []
        return self._select(text, why)

    def _select(self, text: str, why: list) -> int:
        try:
            files = parse_patch(text)
        except Exception as e:  # noqa: BLE001
            why.append(f"unparseable patch ({e!r}) -> all")
            self.noncov = self.m.all
            return self.m.all
        if not files:
            why.append("no file changes found -> all" if text.strip() else "empty patch")
            self.noncov = self.m.all if text.strip() else 0
            return self.noncov
        bits = 0
        for fc in files:
            path = fc.path
            if path is None:
                continue
            if fc.binary:
                b = self.m.data_users(path) if not path.endswith(".py") else self.m.all
            elif path.endswith(".py") and (path.startswith(("sqlglot/", "tests/"))):
                if fc.old is None:  # new module
                    b = self.m.data_users(fc.new)
                    why.append(f"{fc.new}: new file -> {bin(b).count('1')} (dir listings)")
                elif fc.new is not None and fc.new != fc.old:  # rename
                    b = self.m.data_users(fc.new)
                    bits |= self.python_file(fc, why)
                else:
                    bits |= self.python_file(fc, why)  # tracks noncov itself
                    continue
            elif path.endswith(".py") and path.split("/")[-1] == "conftest.py":
                b = self.m.all
            else:
                b = self.data_file(fc, why)
            self.noncov |= b
            bits |= b
        return bits


# ------------------------------------------------------------------------- packing

def pack(tests: list[tuple[int, int]]) -> list[list[int]]:
    """tests: (number, ms). Best of k = 1..8 with LPT; cost = max load + 2 s * k."""
    if not tests:
        return []
    order = sorted(tests, key=lambda t: (-t[1], t[0]))
    best = None
    for k in range(1, min(MAX_WORKERS, len(order)) + 1):
        loads = [0] * k
        bins: list[list[int]] = [[] for _ in range(k)]
        for num, d in order:
            i = loads.index(min(loads))
            loads[i] += d
            bins[i].append(num)
        cost = max(loads) + WORKER_MS * k
        if best is None or cost < best[0]:
            best = (cost, bins)
    return [sorted(b) for b in best[1] if b]


def duration_ms(seconds) -> int:
    import math
    try:
        return math.floor(float(seconds) * 1000 + 0.5)
    except (TypeError, ValueError):
        return 0


# ------------------------------------------------------------------------- main

def main() -> None:
    tests = json.loads((DATA / "tests.json").read_text())["tests"]
    num_by_id = {}
    dur = {}
    for t in tests:
        num_by_id[t["id"]] = t["number"]
        dur[t["number"]] = duration_ms(t.get("seconds", 0))
    all_numbers = sorted(dur)
    patches = sorted((DATA / "commits").glob("*.patch"))
    try:
        maps = Maps(MAPS)
        sel = Selector(maps)
        bit_num = [num_by_id.get(tid) for tid in maps.tests]
        unknown = sorted(set(dur) - {n for n in bit_num if n is not None})
        if unknown:
            log(f"{len(unknown)} tests not in maps: always selected")
    except Exception as e:  # noqa: BLE001
        log(f"maps unavailable ({e!r}): selecting everything")
        maps = sel = None
        bit_num, unknown = [], all_numbers

    out = []
    for patch in patches:
        commit = patch.stem
        why: list[str] = []
        try:
            if sel is None:
                nums = all_numbers
            else:
                text = patch.read_text(encoding="utf-8", errors="replace")
                bits = sel.select(text, why)
                nums = set(unknown) if bits else set()
                if bits:
                    x = bits
                    while x:
                        low = x & -x
                        i = low.bit_length() - 1
                        if i < len(bit_num) and bit_num[i] is not None:
                            nums.add(bit_num[i])
                        x ^= low
                nums = sorted(nums)
            workers = pack([(n, dur[n]) for n in nums])
        except Exception as e:  # noqa: BLE001
            why.append(f"error {e!r} -> all")
            workers = pack([(n, dur[n]) for n in all_numbers])
        out.append(json.dumps({"commit": commit, "workers": workers}, separators=(",", ":")))
        log(f"{commit}: {sum(map(len, workers))} tests on {len(workers)} workers | " + "; ".join(why)[:300])
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))


if __name__ == "__main__":
    main()
