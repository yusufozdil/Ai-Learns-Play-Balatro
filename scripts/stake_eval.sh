#!/usr/bin/env bash
# Deterministic evaluation of one checkpoint on all 8 stakes (Red Deck) on a given seed set.
# usage: scripts/stake_eval.sh CKPT PREFIX [SET=holdout]   -> runs/eval/PREFIX_<stake>_<set>.json
set -euo pipefail
cd "$(dirname "$0")/.."
SET=${3:-holdout}
for s in white red green black blue purple orange gold; do
  uv run --no-sync python -u scripts/evaluate.py --checkpoint "$1" --name "$2_$s" --modes det --stake "$s" \
     --set "$SET" --no-baselines
done
