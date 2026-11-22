# Dev tools (not part of the submitted image)

All test execution happens in the `br-dev` image (`dev/Dockerfile`), which mirrors the
labelling environment: the pinned python:3.13-slim image, `requirements.lock`, and sqlglot @ eb11aec at `/opt/sqlglot`.

```sh
docker build -f dev/Dockerfile -t br-dev .
```

| Tool | What it does |
|---|---|
| `run_oracle.sh DIR TESTS [args]` | Docker wrapper for `oracle.py`. Labels every `DIR/*.patch` and prints JSON lines. |
| `oracle.py` | Inside br-dev, for each patch: copy the checkout, `git apply`, run the full suite with pytest (`--timeout`, xdist, `--continue-on-collection-errors`), and report the failing test ids. A collection error marks the whole file as failing. |
| `oracle_plugin.py` | pytest plugin that records any failed report per nodeid (including subtests, setup and teardown), collection errors, and failure text. |
| `labels.py` | Oracle JSON lines to `answers.json` format (test numbers). |
| `eval.py` | Scores a selector on labelled patches: `--image` (scorer flags, CPU measured inside the container via `cpuwrap.py`), `--cmd` (local, reads `$BR_DATA`), or `--output`. |
| `mutate.py` | Synthetic patches in the sample's styles (`--profile sample`: the sample's kind mix, mutation points chosen uniformly across the code base). **Local evaluation only, never shipped or used as a lookup table.** |
| `edge.py` | Hand-made edge-case patches (evaluation only). |
| `make_maps.sh` | **Regenerates `maps/` from the pinned commit only**: runs `trace_map.py`, then `import_closure.py`, then `build_maps.py`. About 1 min, byte-identical output. |
| `trace_map.py` | Inside br-dev: a fork-per-test tracer (`sys.monitoring`). Records each test's lines, classes used, files opened and subprocess use, plus per-module import contexts, the import graph and the class table. |
| `import_closure.py` | Inside br-dev: one fresh interpreter per test file, recording which project modules that file imports at collection time. |
| `attr_maps.py` | Phase 5a, called by `build_maps.py`: per-line readers of UPPER_CASE class attributes and table entries, with import-time consumption checks (class hooks, import-time reads, class-body derivations). |
| `build_maps.py` | Compacts the raw trace into `maps/meta.json` and `maps/files.bin` (format in its docstring). |
| `check_maps.py` | Map-recall check: replays labelled patches against the maps with naive rules. Evaluation only. |
| `measure_vcpu.py` | **Regenerates `maps/vcpu.json`** (phase 4): clean-tree CPU of every test and every split-test unit, measured through `app/verify.py` inside the submission image, plus overhead per test file. Pinned commit only. |
| `preload_list.py` | **Regenerates `maps/preload.json`**: the standard-library modules sqlglot and its tests import (fresh interpreter, submission image). The verifier imports them once before forking. Pinned commit only. |
| `verify_eval.py` | Phase 4 safety check: verifies ALL static candidates of labelled commits inside the image and reports `unsafe` (labelled failing but passed in the verifier) and `extra`. Evaluation only. |
| `sim_verify.py` | Replays the runtime verification plan (`runner.Planner`) on `verify_eval.py` results, with real outcomes. Used to tune the cost model. Phase 5c: `--ev-cap/--ev-k/--ev-mode/--ev-cover/--ev-always/--report` replay the evidence rule. Evaluation only. |

Typical loop:

```sh
python3 tools/mutate.py --src work/sqlglot --out work/synth --n 300 --seed 1
tools/run_oracle.sh work/synth/commits sample/tests.json --jobs 4 --workers 4 > work/synth/labels.jsonl
python3 tools/labels.py work/synth/labels.jsonl sample/tests.json --kinds work/synth/kinds.json > work/synth/answers.json
cp sample/tests.json work/synth/
python3 tools/eval.py --data work/synth --answers work/synth/answers.json --image blast-radius-entry -v
# or without docker (the selector reads $BR_DATA and finds ../maps itself):
python3 tools/eval.py --data work/synth --answers work/synth/answers.json --cmd "python3 app/main.py"
```

Phase 4 loop (verification):

```sh
docker build -t br-sub .                       # the submission image (has /opt/sqlglot + .pyc)
tools/make_maps.sh                             # also traces split-test units -> maps/units.bin
docker run --rm -v "$PWD":/w:ro -v "$PWD/work":/out br-sub \
    python /w/tools/measure_vcpu.py --tests /w/sample/tests.json --out /out/vcpu.json
cp work/vcpu.json maps/vcpu.json
docker run --rm -v "$PWD":/w:ro -v "$PWD/maps":/maps br-sub python /w/tools/preload_list.py --out /maps/preload.json
docker run --rm -v "$PWD":/w:ro -v "$PWD/work":/out br-sub python /w/tools/verify_eval.py \
    --data /w/work/synth --answers /w/work/synth/answers.json --out /out/synth_verify.jsonl --parallel 10
python3 tools/sim_verify.py --data work/synth --verify work/synth_verify.jsonl -v
docker build -t blast-radius-entry . && python3 tools/eval.py --data work/holdout \
    --answers work/holdout/answers.json --image blast-radius-entry --env BR_BUDGET_CAP=2000
```

Measure CPU in the submission image, not br-dev: br-dev has no `.pyc`, so imports are 3–4× slower.

Evaluation sets in `work/`: `synth` (300, seed 1, used for tuning), `edge` (12), `holdout` (200, seed 2, `--test-code-frac 0.1`, **never tuned on**), and from phase 5b `calib` (300, `--profile sample --seed 3`, tuning) and `confirm` (295, `--profile sample --seed 4`, **never tuned on**: only for the 5c ship gate).

`work/` is gitignored. `work/sqlglot` is a copy of the pinned checkout, taken from the image.
