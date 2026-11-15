#!/usr/bin/env bash
# Phase 10F experiments: 24M-step runs of the e2e_long recipe (same BC init, seed 0) with and without the new reward terms,
# then raw test evaluation. Same total steps everywhere so the LR schedule matches.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_ALLOC_CONF=expandable_segments:True
UVR="uv run --no-sync python -u"
BASE="--init runs/bc/model.pt --lr 1e-4 --kl-coef 0.3 --kl-anneal-updates 120 --critic-warmup 2 --reorder-cost -0.05 \
      --total-steps 24e6 --eval-every 8 --eval-games 512"
run() {  # name, extra flags
  $UVR scripts/train_ppo.py --name "$1" $BASE ${2:-} > runs/$1.log 2>&1
  $UVR scripts/evaluate.py --checkpoint runs/$1/latest.pt --name "$1_latest_raw"
}
run r_base
run r_margin "--margin 1.0 --margin-cap 4"
run r_margin_zp "--margin 1.0 --margin-cap 4 --zero-play -0.3"
echo CHAIN_DONE
