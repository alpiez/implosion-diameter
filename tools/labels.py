"""Convert oracle output (JSON lines with test ids) to answers.json format.

  python3 tools/labels.py labels.jsonl tests.json [--kinds kinds.json] > answers.json
"""
import argparse
import json
import sys

ap = argparse.ArgumentParser()
ap.add_argument("labels")
ap.add_argument("tests")
ap.add_argument("--kinds", help="optional {commit: kind} json")
a = ap.parse_args()
num = {t["id"]: t["number"] for t in json.load(open(a.tests))["tests"]}
kinds = json.load(open(a.kinds)) if a.kinds else {}
out = {}
bad = 0
for line in open(a.labels):
    r = json.loads(line)
    if r.get("error"):
        bad += 1
        print(f"skip {r['commit']}: {r['error'][:200]}", file=sys.stderr)
        continue
    out[r["commit"]] = {"failing": sorted(num[i] for i in r["failing"]),
                        "kind": kinds.get(r["commit"], "unknown")}
json.dump({"commits": out}, sys.stdout, indent=1, sort_keys=True)
print(f"{len(out)} commits, {bad} skipped", file=sys.stderr)
