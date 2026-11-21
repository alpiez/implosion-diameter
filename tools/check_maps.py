"""Check map quality against labelled patches (a naive preview of the phase-3 rules).

For every patch: parse the hunks, take the changed OLD lines (and, for pure
insertions, the old lines around the insertion point), then build a candidate
set from the maps only:

  runtime     cov[line]
  import      for lines executed at import: mod_users of each importing module, plus
              mod_users of the files that import that module (one level), plus
              subprocess tests when the module is a core module
  class body  class_users of the enclosing class (lines inside a class, outside defs)
  data file   data_files[path] (fixtures etc.)
  test file   cov + file_tests when the edit is outside any def
  unknown     everything

and report, per kind, how many commits are fully covered (recall) and the
selected cost against the oracle. Comment-only and whitespace-only edits are NOT
filtered here, so this measures map recall, not selector precision.

  python3 tools/check_maps.py --maps maps --data work/synth --answers work/synth/answers.json
"""

import argparse
import base64
import json
import re
import sys
import zlib
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval import best_pack_cost, ms  # noqa: E402


class Maps:
    def __init__(self, d: Path):
        self.meta = json.loads((d / "meta.json").read_text())
        self.blob = (d / "files.bin").read_bytes()
        self.gsets = [self.dec(s) for s in self.meta["sets"]]
        self.cache = {}

    @staticmethod
    def dec(s: str) -> int:
        return int.from_bytes(base64.b64decode(s), "little")

    def file(self, f: str):
        if f not in self.cache:
            ent = self.meta["files"].get(f)
            if ent is None:
                self.cache[f] = None
            else:
                p = json.loads(zlib.decompress(self.blob[ent[0]:ent[0] + ent[1]]))
                p["_sets"] = [self.dec(s) for s in p["sets"]]
                self.cache[f] = p
        return self.cache[f]

    def S(self, p, i) -> int:
        return p["_sets"][i]


def parse_patch(text: str):
    """{path: {"old": set(changed old lines), "ins": set(old line before which insertion)}}"""
    files = {}
    cur = None
    old_ln = new_ln = 0
    for line in text.splitlines():
        if line.startswith("diff --git"):
            cur = None
        elif line.startswith("--- "):
            path = line[4:].split("\t")[0]
            src = path[2:] if path.startswith("a/") else None
        elif line.startswith("+++ "):
            path = line[4:].split("\t")[0]
            dst = path[2:] if path.startswith("b/") else None
            name = src or dst
            cur = files.setdefault(name, {"old": set(), "ins": set(), "new_file": src is None})
        elif line.startswith("@@"):
            m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            old_ln, new_ln = int(m.group(1)), int(m.group(3))
        elif cur is not None and line.startswith("-"):
            cur["old"].add(old_ln)
            old_ln += 1
        elif cur is not None and line.startswith("+"):
            cur["ins"].add(old_ln)  # inserted before old line old_ln
            new_ln += 1
        elif cur is not None and line.startswith(" "):
            old_ln += 1
            new_ln += 1
    return files


def enclosing(struct, ln):
    """innermost (kind, qualname) of def/class containing line ln (lambdas ignored)."""
    best = None
    for s, e, kind, q, *_ in struct:
        if kind in "cf" and s <= ln <= e and (best is None or s >= best[0]):
            best = (s, kind, q)
    return best


COLLECT = True


def candidates(maps: Maps, patch: dict, why: list) -> int:
    allbits = maps.gsets[maps.meta["all"]]
    sel = 0
    for f, ch in patch.items():
        if f is None:
            continue
        if not f.endswith(".py"):
            hits = 0
            for path, si in maps.meta["data_files"].items():
                if path == f or (path.endswith("/") and f.startswith(path)):
                    hits |= maps.gsets[si]
            why.append(f"data {f}: {bin(hits).count('1')}")
            if not hits and f.startswith(("sqlglot/", "tests/")):
                why.append(f"  unknown data file -> all")
                hits = allbits
            sel |= hits
            continue
        p = maps.file(f)
        if p is None:
            why.append(f"no map for {f} -> {'nothing (new file)' if ch['new_file'] else 'all'}")
            if not ch["new_file"]:
                sel |= allbits
            continue
        lines = set(ch["old"]) | set(ch["ins"]) | {l - 1 for l in ch["ins"]}
        # a line inside a multi-line statement header stands for the whole header
        for s0, s1 in p.get("stmts", []):
            if any(s0 <= l <= s1 for l in list(lines)):
                lines |= set(range(s0, s1 + 1))
        for ln in sorted(lines):
            bits = 0
            c = p["cov"].get(str(ln))
            if c is not None:
                bits |= maps.S(p, c)
            enc = enclosing(p["struct"], ln)
            mods = p["imp"].get(str(ln))
            if mods:
                for m in mods:
                    mp = maps.file(m)
                    if mp:
                        bits |= maps.S(mp, mp["mod_users"])
                        if m.startswith("tests/"):
                            bits |= maps.S(mp, mp["file_tests"])
                    for imp_by in maps.meta["importers"].get(m, []):
                        ip = maps.file(imp_by)
                        if ip:
                            bits |= maps.S(ip, ip["mod_users"])
                            if imp_by.startswith("tests/"):
                                bits |= maps.S(ip, ip["file_tests"])
                    if COLLECT and mp:
                        bits |= maps.S(mp, mp["collect_users"])
                    if m in maps.meta["core_modules"]:
                        bits |= maps.gsets[maps.meta["subprocess"]]
                if enc and enc[1] == "c":
                    bits |= maps.S(p, p["class_users"].get(enc[2], p["any"]))
            if f.startswith("tests/") and (enc is None or enc[1] == "c"):
                bits |= maps.S(p, p["file_tests"])
            why.append(f"{f}:{ln} enc={enc and enc[1:]} cov={c is not None} imp={len(mods or [])} "
                       f"-> {bin(bits).count('1')}")
            sel |= bits
    return sel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", type=Path, default=Path("maps"))
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--answers", type=Path, required=True)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--no-collect", action="store_true", help="ignore collect_users")
    a = ap.parse_args()
    global COLLECT
    COLLECT = not a.no_collect
    maps = Maps(a.maps)
    tests = json.loads((a.data / "tests.json").read_text())["tests"]
    num_by_id = {t["id"]: t["number"] for t in tests}
    dur = {t["number"]: ms(t["seconds"]) for t in tests}
    bit_to_num = [num_by_id.get(t) for t in maps.meta["tests"]]
    answers = json.loads(a.answers.read_text())["commits"]
    stats = defaultdict(lambda: [0, 0, 0, 0, 0])  # n, ok, cost, oracle, missed tests
    for c in sorted(answers):
        p = a.data / "commits" / f"{c}.patch"
        why = []
        bits = candidates(maps, parse_patch(p.read_text()), why)
        sel = {bit_to_num[i] for i in range(len(bit_to_num)) if bits >> i & 1}
        fail = set(answers[c]["failing"])
        missed = fail - sel
        kind = answers[c].get("kind", "?")
        st = stats[kind]
        st[0] += 1
        st[1] += not missed
        st[2] += best_pack_cost([dur[n] for n in sel])
        st[3] += best_pack_cost([dur[n] for n in fail])
        st[4] += len(missed)
        if missed or a.verbose:
            print(f"{c} {kind:8} fail={len(fail)} sel={len(sel)} "
                  f"{'MISSED ' + str(sorted(missed)[:8]) if missed else 'ok'}")
            if missed:
                for w in why:
                    print("     ", w)
    tot = [sum(v[i] for v in stats.values()) for i in range(5)]
    for k, v in sorted(stats.items()) + [("TOTAL", tot)]:
        print(f"{k:8} n={v[0]:4} covered={v[1]:4} ({100 * v[1] / v[0]:5.1f}%) "
              f"cost={v[2] / 1000:9.1f}s oracle={v[3] / 1000:7.1f}s missed_tests={v[4]}")


if __name__ == "__main__":
    main()
