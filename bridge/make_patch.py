"""Regenerate `bridge/lua/balatrobot-1.5.2.patch` from a pristine BalatroBot 1.5.2 tree.

Usage: python bridge/make_patch.py ~/balatro-bridge/staging/balatrobot-1.5.2

The patch changes three stock files minimally (everything else lives in new files):
  * balatrobot.lua: load `src/lua/utils/bridge_ext.lua` and register the `reroll_boss` endpoint,
  * src/lua/utils/gamestate.lua: add `raw` to every card and call `augment` on the gamestate,
  * src/lua/endpoints/sell.lua: allow `sell` while a booster pack is open.
Normally you do NOT need this script: `bridge/apply_lua_patch.sh` applies the checked-in patch.
"""

from __future__ import annotations

import difflib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def edit_balatrobot(text: str) -> str:
    text = text.replace(
        '  -- Use consumable endpoint\n  "src/lua/endpoints/use.lua",\n',
        '  -- Use consumable endpoint\n  "src/lua/endpoints/use.lua",\n'
        "  -- council-of-jokers bridge patch: boss reroll\n"
        '  "src/lua/endpoints/reroll_boss.lua",\n'
        '  "src/lua/endpoints/continue_endless.lua",\n',
    )
    text = text.replace(
        "-- Load core modules\n",
        "-- council-of-jokers bridge patch: raw state extension (never fatal)\n"
        "do\n"
        "  local ok, mod = pcall(function()\n"
        '    return assert(SMODS.load_file("src/lua/utils/bridge_ext.lua"))()\n'
        "  end)\n"
        "  if ok then\n"
        "    BB_BRIDGE_EXT = mod\n"
        "  else\n"
        "    sendErrorMessage('bridge_ext load failed: ' .. tostring(mod), 'BB.BRIDGE')\n"
        "  end\n"
        "end\n"
        "-- Load core modules\n",
        1,
    )
    return text


def edit_gamestate(text: str) -> str:
    text = text.replace(
        "  return {\n    id = card.sort_id or 0,\n",
        "  local bridge_raw = nil\n"
        "  if BB_BRIDGE_EXT then\n"
        "    local ok, r = pcall(BB_BRIDGE_EXT.card_raw, card)\n"
        "    if ok then bridge_raw = r end\n"
        "  end\n"
        "  return {\n    raw = bridge_raw,\n    id = card.sort_id or 0,\n",
        1,
    )
    text = text.replace(
        "    state_data.pack = extract_area(G.pack_cards)\n  end\n\n  return state_data\n",
        "    state_data.pack = extract_area(G.pack_cards)\n  end\n\n"
        "  if BB_BRIDGE_EXT then\n"
        "    local ok, err = pcall(BB_BRIDGE_EXT.augment, state_data)\n"
        "    if not ok then state_data.bridge_error = tostring(err) end\n"
        "  end\n\n  return state_data\n",
        1,
    )
    return text


def edit_sell(text: str) -> str:
    """Selling is legal while a booster pack is open (e.g. to make room for a Buffoon pick)."""
    text = text.replace(
        "requires_state = { G.STATES.SELECTING_HAND, G.STATES.SHOP },",
        "requires_state = { G.STATES.SELECTING_HAND, G.STATES.SHOP,"
        " G.STATES.SMODS_BOOSTER_OPENED },",
        1,
    )
    text = text.replace(
        "local valid_state = (G.STATE == G.STATES.SHOP or G.STATE == G.STATES.SELECTING_HAND)",
        "local valid_state = (G.STATE == G.STATES.SHOP or G.STATE == G.STATES.SELECTING_HAND"
        " or G.STATE == G.STATES.SMODS_BOOSTER_OPENED)",
        1,
    )
    return text


def main() -> None:
    src = Path(sys.argv[1]).expanduser()
    out = []
    for rel, fn in (
        ("balatrobot.lua", edit_balatrobot),
        ("src/lua/utils/gamestate.lua", edit_gamestate),
        ("src/lua/endpoints/sell.lua", edit_sell),
    ):
        old = (src / rel).read_text()
        new = fn(old)
        assert new != old, rel
        out.extend(
            difflib.unified_diff(
                old.splitlines(keepends=True),
                new.splitlines(keepends=True),
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
            )
        )
    (HERE / "lua" / "balatrobot-1.5.2.patch").write_text("".join(out))
    print("wrote", HERE / "lua" / "balatrobot-1.5.2.patch")


if __name__ == "__main__":
    main()
