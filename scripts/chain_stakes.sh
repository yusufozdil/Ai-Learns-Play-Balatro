#!/usr/bin/env bash
# Stake ladder: fine-tune the reference policy (runs/e2e_long_fix/latest.pt) on harder stakes in three stages,
# each starting from the previous stage's latest.pt, with the Phase 10F rewards (margin + zero_play).
# After every stage: raw deterministic test (1000 games, Red Deck) on all 8 stakes, to track both the gain on the
# target stakes and forgetting on the easy ones. Reference numbers: notes.md section 0.0 (stake table).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_ALLOC_CONF=expandable_segments:True
UVR="uv run --no-sync python -u"
COMMON="--lr 5e-5 --kl-coef 0 --critic-warmup 2 --reorder-cost -0.25 --margin 1.0 --margin-cap 4 --zero-play -0.3 \
        --eval-every 8 --eval-games 512"
DECKS="red,blue,yellow,green,black"
STAKES_ALL="white red green black blue purple orange gold"

stage() {  # name, init, steps, decks, stakes, eval_stake
  $UVR scripts/train_ppo.py --name "$1" --init "$2" --total-steps "$3" --decks "$4" --stakes "$5" \
       --eval-deck red --eval-stake "$6" $COMMON > runs/$1.log 2>&1
  for s in $STAKES_ALL; do
    $UVR scripts/evaluate.py --checkpoint runs/$1/latest.pt --name "$1_latest_raw_red_$s" \
         --modes det --stake "$s" --no-baselines
  done
}

stage st1 runs/e2e_long_fix/latest.pt 24e6 red white,red,green green
stage st2 runs/st1/latest.pt 30e6 "$DECKS" white,red,green,black,blue black
stage st3 runs/st2/latest.pt 40e6 "$DECKS" blue,purple,orange,gold orange
echo CHAIN_DONE
