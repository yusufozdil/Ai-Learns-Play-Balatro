#!/usr/bin/env bash
# Freeze a git ref into a detached worktree with its own venv + engine build.
#   scripts/freeze_checkout.sh <git-ref> [dir]
# Default dir: $FREEZE_BASE/<shortsha>  (FREEZE_BASE=/home/yuse/Desktop/balatroRL-runs).
# runs/ inside the frozen checkout is a symlink to the main checkout's runs/ so all
# outputs land in one place. Idempotent. Prints the checkout path on stdout (last line).
set -euo pipefail
export PATH="$HOME/.cargo/bin:$PATH"

REF="${1:?usage: freeze_checkout.sh <git-ref> [dir]}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Main checkout = parent of the shared .git dir (works from inside any worktree).
COMMON="$(git -C "$HERE" rev-parse --path-format=absolute --git-common-dir)"
MAIN="${MAIN_CHECKOUT:-$(dirname "$COMMON")}"
BASE="${FREEZE_BASE:-/home/yuse/Desktop/balatroRL-runs}"

SHA="$(git -C "$MAIN" rev-parse --short=7 "$REF^{commit}")"
DIR="${2:-$BASE/$SHA}"

if [ ! -e "$DIR/.git" ]; then
  mkdir -p "$(dirname "$DIR")"
  git -C "$MAIN" worktree add --detach "$DIR" "$SHA" >&2
fi

# runs/ -> main runs/
mkdir -p "$MAIN/runs"
if [ -L "$DIR/runs" ]; then
  :
elif [ -e "$DIR/runs" ]; then
  # untracked leftovers: merge then replace with symlink
  cp -an "$DIR/runs/." "$MAIN/runs/" 2>/dev/null || true
  rm -rf "$DIR/runs"
  ln -s "$MAIN/runs" "$DIR/runs"
else
  ln -s "$MAIN/runs" "$DIR/runs"
fi

cd "$DIR"
if [ ! -f .freeze_ready ] || [ "$(cat .freeze_ready)" != "$SHA" ]; then
  uv sync --no-install-project --inexact --group train -q >&2
  uv run --no-sync maturin develop --release --quiet >&2
  echo "$SHA" > .freeze_ready
fi

# Verify the engine resolves inside this checkout.
uv run --no-sync python -c "
import balatro_rl, os, sys
p = os.path.realpath(balatro_rl.__file__)
root = os.path.realpath('$DIR')
sys.exit(0 if p.startswith(root + os.sep) else 'balatro_rl resolves outside checkout: ' + p)
" >&2

echo "$DIR"
