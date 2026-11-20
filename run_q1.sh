#!/usr/bin/env bash
# Full Question-1 pipeline: teacher -> 3 baselines -> 3 distilled students.
# Usage:  bash run_q1.sh            (PY env var can override the python interpreter)
set -eo pipefail
PY=${PY:-python}
cd "$(dirname "$0")"
mkdir -p logs
$PY -u -W ignore -m q1_kd.train_teacher --epochs 70 2>&1 | tee logs/q1_teacher.log
$PY -u -W ignore -m q1_kd.train_baseline --epochs 60 2>&1 | tee logs/q1_baselines.log
$PY -u -W ignore -m q1_kd.train_kd --epochs 60 2>&1 | tee logs/q1_kd.log
