#!/usr/bin/env bash
# All four arms on identical folds: RQ1, the RQ2 heart-rate ablation, and the
# no-context arm that tests whether the sequence model earns its place.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ARGS="--epochs 8 --train-stride 10 --context-len 20 --class-weight-power ${WEIGHT_POWER:-0.5}"
SUFFIX="${RUN_SUFFIX:-}"

for variant in accel_only accel_hr hr_only; do
  echo "=== ${variant}${SUFFIX} ==="
  .venv/bin/python scripts/train.py --variant "$variant" --run-id "${variant}${SUFFIX}" $ARGS
done

echo "=== accel_only_nocontext${SUFFIX} ==="
.venv/bin/python scripts/train.py --variant accel_only --context-model none \
  --run-id "accel_only_nocontext${SUFFIX}" $ARGS
