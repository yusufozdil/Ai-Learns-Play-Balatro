"""Regenerate `bridge/lua/balatrobot-1.5.2.patch` from a pristine BalatroBot 1.5.2 tree.

Usage: python bridge/make_patch.py ~/balatro-bridge/staging/balatrobot-1.5.2

The patch changes five stock files minimally (everything else lives in new files):
  * balatrobot.lua: load `src/lua/utils/bridge_ext.lua` and register the `reroll_boss` endpoint,
  * src/lua/utils/gamestate.lua: add `raw` to every card and call `augment` on the gamestate,
  * src/lua/endpoints/sell.lua: allow `sell` while a booster pack is open,
  * src/lua/endpoints/buy.lua: the game's room rule (Negative cards) and `use` = buy and use,
  * src/lua/endpoints/pack.lua: the same room rule for a joker picked from a pack.
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


def edit_buy(text: str) -> str:
    """`buy` follows the real game's room rule and learns `use` (buy and use at once).

    Stock `buy` refuses a joker / consumable when the area is full, but the game itself has
    room for a Negative card (`G.FUNCS.check_for_buy_space`: `card.ability.card_limit` is +1) and
    lets "Buy and use" proceed with full consumable slots (`buy_from_shop` skips the space check
    for `id == 'buy_and_use'`).
    """
    text = text.replace(
        """    voucher = {
      type = "integer",
      required = false,
      description = "0-based index of voucher to buy",
    },
""",
        """    voucher = {
      type = "integer",
      required = false,
      description = "0-based index of voucher to buy",
    },
    use = {
      type = "boolean",
      required = false,
      description = "bridge patch: buy a consumable and use it at once (card must be given)",
    },
""",
        1,
    )
    text = text.replace(
        """    -- Ensure there is space in joker area
    if card.set == "JOKER" then
      if gamestate.jokers.count >= gamestate.jokers.limit then""",
        """    -- bridge patch: the game's own room rule (Negative cards bring their own slot)
    local shop_obj = args.card and G.shop_jokers and G.shop_jokers.cards[pos] or nil
    local function bridge_has_room(area)
      if not shop_obj or not shop_obj.ability then
        return nil
      end
      return #area.cards + 1 + (shop_obj.ability.extra_slots_used or 0)
        <= area.config.card_limit + (shop_obj.ability.card_limit or 0)
    end
    local is_consumable = card.set == "PLANET" or card.set == "SPECTRAL" or card.set == "TAROT"
    if args.use then
      if not (args.card and is_consumable) then
        send_response({
          message = "Invalid arguments. `use` needs a consumable `card`",
          name = BB_ERROR_NAMES.BAD_REQUEST,
        })
        return
      end
      if not shop_obj or not shop_obj:can_use_consumeable() then
        send_response({
          message = "Consumable '" .. tostring(card.label) .. "' cannot be used at this time",
          name = BB_ERROR_NAMES.NOT_ALLOWED,
        })
        return
      end
    end

    -- Ensure there is space in joker area
    if card.set == "JOKER" then
      local room = bridge_has_room(G.jokers)
      if room == nil then
        room = gamestate.jokers.count < gamestate.jokers.limit
      end
      if not room then""",
        1,
    )
    text = text.replace(
        """    if card.set == "PLANET" or card.set == "SPECTRAL" or card.set == "TAROT" then
      if gamestate.consumables.count >= gamestate.consumables.limit then""",
        """    if is_consumable and not args.use then
      local room = bridge_has_room(G.consumeables)
      if room == nil then
        room = gamestate.consumables.count < gamestate.consumables.limit
      end
      if not room then""",
        1,
    )
    text = text.replace(
        """    if args.voucher or args.pack then
      G.FUNCS.use_card(btn)
    else""",
        """    if args.voucher or args.pack then
      G.FUNCS.use_card(btn)
    elseif args.use then
      -- bridge patch: the shop's "Buy and use" button (buy_from_shop runs use_card at the end)
      G.FUNCS.buy_from_shop({ config = { ref_table = shop_obj, id = "buy_and_use" } })
    else""",
        1,
    )
    text = text.replace(
        """          local shop_decreased = (shop_count == initial_shop_count - 1)
          local dest_increased = (dest_count == initial_dest_count + 1)
          local money_deducted = (G.GAME.dollars == initial_money - card.cost.buy)
          if shop_decreased and dest_increased and money_deducted and G.STATE == G.STATES.SHOP then
            done = true
          end
        elseif args.voucher then""",
        """          local shop_decreased = (shop_count == initial_shop_count - 1)
          local dest_increased = (dest_count == initial_dest_count + 1)
          local money_deducted = (G.GAME.dollars == initial_money - card.cost.buy)
          if args.use then
            -- bridge patch: the card is used, not kept, and its effect may change the money
            -- (The Hermit, Temperance): no money test, wait until the use animation is over
            local settled = not G.CONTROLLER.locks.use and not (G.GAME.STOP_USE and G.GAME.STOP_USE > 0)
            if shop_decreased and settled and G.STATE == G.STATES.SHOP then
              done = true
            end
          elseif shop_decreased and dest_increased and money_deducted and G.STATE == G.STATES.SHOP then
            done = true
          end
        elseif args.voucher then""",
        1,
    )
    return text


def edit_pack(text: str) -> str:
    """Picking a Negative joker out of a pack needs no free slot (it brings its own)."""
    text = text.replace(
        """        if joker_count >= joker_limit then""",
        """        local negative_bonus = (card.ability and card.ability.card_limit) or 0 -- bridge patch
        if joker_count + 1 + ((card.ability and card.ability.extra_slots_used) or 0)
            > joker_limit + negative_bonus then""",
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
        ("src/lua/endpoints/buy.lua", edit_buy),
        ("src/lua/endpoints/pack.lua", edit_pack),
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
