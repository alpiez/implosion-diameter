#!/bin/sh
# Label patches with the oracle inside br-dev.
#   tools/run_oracle.sh <patch-dir> <tests.json> [oracle args...] > labels.jsonl
set -eu
pdir=$(cd "$1" && pwd); tests=$(cd "$(dirname "$2")" && pwd)/$(basename "$2"); shift 2
here=$(cd "$(dirname "$0")" && pwd)
exec docker run --rm -i \
  -v "$here:/tools:ro" -v "$pdir:/patches:ro" -v "$tests:/data/tests.json:ro" \
  br-dev sh -c 'python /tools/oracle.py --tests /data/tests.json "$@" /patches/*.patch' oracle "$@"
