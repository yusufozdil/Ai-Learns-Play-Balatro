---Bridge extension for BalatroBot (council-of-jokers, patch version 3).
---Exposes raw game information that the stock gamestate only has as effect text:
---  * per card `raw` (the `card.ability` table, seal, edition, base, flags),
---  * `bridge` (selected G.GAME fields: current_round targets, round_resets, tags, blind, ...),
---  * `discard` (the discard pile) and `playing_cards` (every playing card of the run).
---Everything is wrapped in pcall by the caller; an error never breaks the stock gamestate.
local M = { VERSION = 3 }

local MAX_DEPTH = 3

---Convert a Lua value to something the BalatroBot JSON encoder handles. Functions, userdata and
---cyclic references are dropped; non-finite numbers become strings.
local function plain(v, depth, seen)
  local t = type(v)
  if t == "number" then
    if v ~= v then
      return "nan"
    elseif v == math.huge then
      return "inf"
    elseif v == -math.huge then
      return "-inf"
    end
    return v
  elseif t == "string" or t == "boolean" then
    return v
  elseif t == "table" then
    depth = depth or 0
    if depth >= MAX_DEPTH then
      return nil
    end
    seen = seen or {}
    if seen[v] then
      return nil
    end
    seen[v] = true
    local out = {}
    local n = 0
    local is_array = true
    for k, _ in pairs(v) do
      n = n + 1
      if type(k) ~= "number" then
        is_array = false
      end
    end
    if is_array and n > 0 then
      for i = 1, #v do
        out[i] = plain(v[i], depth + 1, seen)
      end
    else
      for k, x in pairs(v) do
        if type(k) == "string" or type(k) == "number" then
          local pv = plain(x, depth + 1, seen)
          if pv ~= nil then
            out[tostring(k)] = pv
          end
        end
      end
    end
    seen[v] = nil
    return out
  end
  return nil
end
M.plain = plain

local function center_key(card)
  if card.config and card.config.center then
    return card.config.center.key
  end
  return nil
end

---Raw per-card information (joker, consumable, playing card, booster, voucher).
function M.card_raw(card)
  local raw = {
    center_key = center_key(card),
    sort_id = card.sort_id,
    debuff = card.debuff and true or false,
    facing = card.facing,
    sell_cost = card.sell_cost,
    cost = card.cost,
    seal = card.seal,
    edition = plain(card.edition, 1),
    ability = plain(card.ability, 0),
  }
  if card.base then
    raw.base = {
      nominal = card.base.nominal,
      suit_nominal = card.base.suit_nominal,
      id = card.base.id,
      value = card.base.value,
      suit = card.base.suit,
    }
  end
  return raw
end

local function cards_of(area)
  local out = {}
  if area and area.cards then
    for i, c in ipairs(area.cards) do
      out[i] = c
    end
  end
  return out
end

local function compact_card(card, where)
  local ab = card.ability or {}
  return {
    id = card.sort_id,
    where = where,
    suit = card.base and card.base.suit,
    rank = card.base and card.base.value,
    enhancement = ab.effect ~= "Base" and ab.name or nil,
    center_key = center_key(card),
    seal = card.seal,
    edition = plain(card.edition, 1),
    perma_bonus = ab.perma_bonus,
    debuff = card.debuff and true or false,
    hidden = (card.facing == "back") or nil,
    played_this_ante = ab.played_this_ante,
  }
end

local function tags_raw()
  local out = {}
  if G.GAME and G.GAME.tags then
    for i, tag in ipairs(G.GAME.tags) do
      out[i] = { key = tag.key, name = tag.name, ability = plain(tag.ability, 1) }
    end
  end
  return out
end

local function blind_raw()
  local b = G.GAME and G.GAME.blind
  if not b then
    return nil
  end
  return {
    name = b.name,
    key = b.config and b.config.blind and b.config.blind.key,
    disabled = b.disabled and true or false,
    triggered = b.triggered and true or false,
    chips = b.chips,
    mult = b.mult,
    boss = b.boss and true or false,
    prepped = b.prepped and true or false,
    hands = plain(b.hands, 1),
    only_hand = b.only_hand,
    block_play = b.block_play and true or false,
    hands_sub = b.hands_sub,
    discards_sub = b.discards_sub,
  }
end

local function used_voucher_keys()
  local out = {}
  if G.GAME and G.GAME.used_vouchers then
    for k, v in pairs(G.GAME.used_vouchers) do
      if v then
        out[#out + 1] = k
      end
    end
    table.sort(out)
  end
  return out
end

---Selected G.GAME fields.
function M.game_raw()
  local g = G.GAME
  if not g then
    return nil
  end
  local booster = nil
  if SMODS and SMODS.OPENED_BOOSTER and G.pack_cards and not G.pack_cards.REMOVED then
    local b = SMODS.OPENED_BOOSTER
    booster = {
      key = center_key(b),
      extra = b.ability and b.ability.extra,
      choose = b.config and b.config.center and b.config.center.config and b.config.center.config.choose,
    }
  end
  return {
    patch_version = M.VERSION,
    round = g.round,
    dollars = g.dollars,
    hands_played = g.hands_played,
    skips = g.skips,
    unused_discards = g.unused_discards,
    chips = g.chips,
    interest_cap = g.interest_cap,
    interest_amount = g.interest_amount,
    discount_percent = g.discount_percent,
    inflation = g.inflation,
    bankrupt_at = g.bankrupt_at,
    last_hand_played = g.last_hand_played,
    last_tarot_planet = g.last_tarot_planet,
    pack_choices = g.pack_choices,
    pack_size = g.pack_size,
    starting_deck_size = g.starting_deck_size,
    probabilities = plain(g.probabilities, 1),
    pool_flags = plain(g.pool_flags, 1),
    modifiers = plain(g.modifiers, 1),
    starting_params = plain(g.starting_params, 1),
    round_resets = plain(g.round_resets, 1),
    current_round = plain(g.current_round, 0),
    orbital_choices = plain(g.orbital_choices, 0),
    consumeable_usage_total = plain(g.consumeable_usage_total, 1),
    consumeable_usage = plain(g.consumeable_usage, 1),
    bosses_used = plain(g.bosses_used, 1),
    used_vouchers = used_voucher_keys(),
    tags = tags_raw(),
    blind = blind_raw(),
    booster = booster,
    state_complete = G.STATE_COMPLETE and true or false,
    controller_locked = G.CONTROLLER and G.CONTROLLER.locked and true or false,
    hand_limit = G.hand and G.hand.config and G.hand.config.card_limit,
  }
end

---Add the extension fields to a stock gamestate table (in place).
function M.augment(state)
  state.bridge = M.game_raw()
  local discard = {}
  for i, c in ipairs(cards_of(G.discard)) do
    discard[i] = compact_card(c, "discard")
  end
  state.discard = discard
  local all = {}
  if G.playing_cards then
    for i, c in ipairs(G.playing_cards) do
      all[i] = compact_card(c, "all")
    end
  end
  state.playing_cards = all
end

return M
