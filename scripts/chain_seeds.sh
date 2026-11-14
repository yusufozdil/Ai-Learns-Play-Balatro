#!/usr/bin/env bash
# Phase 10D chain: head-freeze evaluations, then two 100M seed repeats of the e2e_long recipe (+ loop-fix fine-tune), then test evals.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_ALLOC_CONF=expandable_segments:True
UVR="uv run --no-sync python -u"
REF=runs/e2e_long_fix/latest.pt

# 1) head freeze: one phase handed to the teacher, the model plays the rest (raw deterministic + sampled)
for r in play+shop+pack blind+shop+pack blind+play+pack blind+play+shop; do
  n=freeze_no_$( (echo blind; echo play; echo shop; echo pack) | grep -vxF -f <(tr + '\n' <<<"$r") )
  $UVR scripts/evaluate.py --checkpoint $REF --route $r --name $n
done

# 2) seed repeats (same recipe as e2e_long / e2e_long_fix, only --seed differs)
for s in 1 2; do
  $UVR scripts/train_ppo.py --name e2e_s$s --seed $s --init runs/bc/model.pt --lr 1e-4 --kl-coef 0.3 \
      --kl-anneal-updates 120 --critic-warmup 2 --reorder-cost -0.05 --total-steps 100e6 \
      --eval-every 8 --eval-games 512 > runs/e2e_s$s.log 2>&1
  $UVR scripts/train_ppo.py --name e2e_s${s}_fix --seed $((s+10)) --init runs/e2e_s$s/latest.pt --lr 5e-5 \
      --kl-coef 0 --critic-warmup 0 --reorder-cost -0.25 --total-steps 6e6 \
      --eval-every 8 --eval-games 512 > runs/e2e_s${s}_fix.log 2>&1
  for c in e2e_s$s/latest e2e_s${s}_fix/latest; do
    $UVR scripts/evaluate.py --checkpoint runs/$c.pt --name ${c//\//_}_raw
  done
done
echo CHAIN_DONE
