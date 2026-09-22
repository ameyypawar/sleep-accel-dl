#!/usr/bin/env bash
# All three input variants on identical folds -- RQ1 plus the RQ2 ablation.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ARGS="--epochs 8 --train-stride 10 --context-len 20"

for variant in accel_only accel_hr hr_only; do
  echo "=== $variant ==="
  .venv/bin/python scripts/train.py --variant "$variant" --run-id "$variant" $ARGS
done

# The no-context arm: does the sequence model actually do the work?
echo "=== accel_only, no context ==="
.venv/bin/python scripts/train.py --variant accel_only --context-model none \
  --run-id accel_only_nocontext $ARGS
