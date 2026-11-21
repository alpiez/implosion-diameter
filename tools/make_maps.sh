#!/bin/sh
# Regenerate maps/ from the pinned sqlglot commit (eb11aec) only.
#   tools/make_maps.sh [tests.json]      (default: sample/tests.json)
# Steps, all inside br-dev (dev/Dockerfile):
#   1. trace_map.py      fork-per-test tracing of the clean suite  -> work/trace/
#   2. import_closure.py fresh-process import closure per test file -> work/trace/closure.json
#   3. build_maps.py     compact maps                               -> maps/
# No patch, label or synthetic data is read by any of these steps.
set -eu
here=$(cd "$(dirname "$0")/.." && pwd)
tests=${1:-$here/sample/tests.json}
mkdir -p "$here/work" "$here/maps"
chmod 777 "$here/work" "$here/maps"
cp "$tests" "$here/work/tests.json"
rm -rf "$here/work/trace"
run() { docker run --rm -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e PYTHON_CPU_COUNT=1 \
  -v "$here/tools:/tools:ro" -v "$here/work:/out" -v "$here/maps:/maps" br-dev "$@"; }
run python /tools/trace_map.py --tests /out/tests.json --out /out/trace --jobs 12
run python /tools/import_closure.py --tests /out/tests.json --out /out/trace/closure.json
run python /tools/build_maps.py --trace /out/trace --out /maps
