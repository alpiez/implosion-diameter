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
| `mutate.py` | Synthetic patches in the sample's styles. **Local evaluation only, never shipped or used as a lookup table.** |

Typical loop:

```sh
python3 tools/mutate.py --src work/sqlglot --out work/synth --n 300 --seed 1
tools/run_oracle.sh work/synth/commits sample/tests.json --jobs 4 --workers 4 > work/synth/labels.jsonl
python3 tools/labels.py work/synth/labels.jsonl sample/tests.json --kinds work/synth/kinds.json > work/synth/answers.json
cp sample/tests.json work/synth/
python3 tools/eval.py --data work/synth --answers work/synth/answers.json --image blast-radius-entry -v
```

`work/` is gitignored. `work/sqlglot` is a copy of the pinned checkout, taken from the image.
