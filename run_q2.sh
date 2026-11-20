#!/usr/bin/env bash
# Full Question-2 pipeline: hardware-aware NAS search -> train best architecture from scratch.
# Usage:  bash run_q2.sh            (PY env var can override the python interpreter)
set -eo pipefail
PY=${PY:-python}
cd "$(dirname "$0")"
mkdir -p logs
$PY -u -W ignore -m q2_nas.search --trials 50 --epochs 8 2>&1 | tee logs/q2_search.log
$PY -u -W ignore -m q2_nas.train_best --epochs 60 2>&1 | tee logs/q2_train_best.log
