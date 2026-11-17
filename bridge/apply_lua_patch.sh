#!/usr/bin/env bash
# Apply the council-of-jokers bridge patch to the installed BalatroBot mod.
#
#   bridge/apply_lua_patch.sh [--revert]
#
# Source of truth: pristine BalatroBot 1.5.2 in $BB_STAGING (default ~/balatro-bridge/staging/
# balatrobot-1.5.2) + bridge/lua/balatrobot-1.5.2.patch + the new files in bridge/lua/. The
# currently installed files are backed up once to <mod>/.bridge_backup/<timestamp>/ before they are
# overwritten. Idempotent: re-running reinstalls pristine + patch. Restart the game afterwards.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
STAGING="${BB_STAGING:-$HOME/balatro-bridge/staging/balatrobot-1.5.2}"
MODS="${BB_MODS:-$HOME/.steam/steam/steamapps/compatdata/2379780/pfx/drive_c/users/steamuser/AppData/Roaming/Balatro/Mods}"
MOD="$MODS/balatrobot"
[ -d "$MOD" ] || { echo "installed mod not found: $MOD" >&2; exit 1; }
[ -d "$STAGING/src/lua" ] || { echo "pristine BalatroBot not found: $STAGING" >&2; exit 1; }

BACKUP="$MOD/.bridge_backup"
FILES="balatrobot.lua src/lua/utils/gamestate.lua src/lua/endpoints/sell.lua"

if [ "${1:-}" = "--revert" ]; then
  for f in $FILES; do cp -v "$STAGING/$f" "$MOD/$f"; done
  rm -fv "$MOD/src/lua/utils/bridge_ext.lua" "$MOD/src/lua/endpoints/reroll_boss.lua" \
    "$MOD/src/lua/endpoints/continue_endless.lua"
  echo "reverted to pristine BalatroBot 1.5.2 files"
  exit 0
fi

if [ ! -d "$BACKUP" ]; then
  mkdir -p "$BACKUP/orig"
  for f in $FILES; do
    mkdir -p "$BACKUP/orig/$(dirname "$f")"
    cp "$MOD/$f" "$BACKUP/orig/$f"
  done
  echo "backed up installed originals to $BACKUP/orig"
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
for f in $FILES; do
  mkdir -p "$TMP/$(dirname "$f")"
  cp "$STAGING/$f" "$TMP/$f"
done
(cd "$TMP" && patch -p1 --quiet < "$HERE/lua/balatrobot-1.5.2.patch")
for f in $FILES; do cp -v "$TMP/$f" "$MOD/$f"; done
cp -v "$HERE/lua/bridge_ext.lua" "$MOD/src/lua/utils/bridge_ext.lua"
cp -v "$HERE/lua/endpoints/reroll_boss.lua" "$MOD/src/lua/endpoints/reroll_boss.lua"
cp -v "$HERE/lua/endpoints/continue_endless.lua" "$MOD/src/lua/endpoints/continue_endless.lua"
echo "bridge patch version $(grep -o 'VERSION = [0-9]*' "$HERE/lua/bridge_ext.lua") installed; restart the game"
