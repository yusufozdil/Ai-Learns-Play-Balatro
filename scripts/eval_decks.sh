#!/usr/bin/env bash
# Deterministic evaluation of a checkpoint on all 15 decks (White stake, no baselines).
#   scripts/eval_decks.sh CKPT NAME_PREFIX [SET]      (SET: dev (default) | holdout | val)
# Writes runs/eval/<NAME_PREFIX>_<deck>_white[_<set>].json, then prints the table.
set -euo pipefail
CKPT=${1:?checkpoint}
PREFIX=${2:?name prefix}
SET=${3:-dev}
cd "$(dirname "$0")/.."
DECKS="red blue yellow green black magic nebula ghost abandoned checkered zodiac painted anaglyph plasma erratic"
for d in $DECKS; do
  uv run --no-sync python scripts/evaluate.py --checkpoint "$CKPT" --modes det --no-baselines \
    --deck "$d" --stake white --set "$SET" --name "${PREFIX}_${d}_white"
done
SUF=""
if [ "$SET" != "dev" ]; then SUF="_$SET"; fi
uv run --no-sync python scripts/summarize_evals.py "runs/eval/${PREFIX}_*_white${SUF}.json"
