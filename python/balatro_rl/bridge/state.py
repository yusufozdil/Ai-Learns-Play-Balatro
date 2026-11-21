"""BalatroBot gamestate (with the bridge Lua patch) -> engine state JSON -> ``Game``.

``translate(gs)`` returns a :class:`Translation` whose ``state`` is a dict in the format of
``Game.export_state()``; ``build_game`` turns it into a ``balatro_rl._engine.Game`` whose
observation and masks are what the policy sees. Hidden information is filled arbitrarily (the
draw pile keeps the order BalatroBot reports; RNG streams come from a fixed template): the
observation never contains it (``docs/design/rl_interface.md`` section 9).

Mapping tables and the list of things that cannot be mapped: ``docs/design/bridge.md``.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from functools import lru_cache

from balatro_rl import _engine

from . import engine_tables as et
from . import tables as T


def strip_game_text(gs):
    """Copy of a gamestate without the game's localized description prose.

    Drops every ``effect`` (card / blind description text) except the ``ability.effect`` type key,
    and every ``tag_effect``. Names, keys, ids and numbers are kept; the translation never reads
    the removed fields. Used for everything the bridge writes to disk (fixtures, state dumps).
    """

    def walk(o, parent=None):
        if isinstance(o, dict):
            return {
                k: walk(v, k)
                for k, v in o.items()
                if k != "tag_effect" and not (k == "effect" and parent != "ability")
            }
        if isinstance(o, list):
            return [walk(v, parent) for v in o]
        return o

    return walk(gs)


PHASE_OF_STATE = {
    "BLIND_SELECT": "BlindSelect",
    "SELECTING_HAND": "Play",
    "SHOP": "Shop",
    "SMODS_BOOSTER_OPENED": "PackOpen",
    "GAME_OVER": "GameOver",
}
#: BalatroBot states that are transitions: wait / cash out, never translate.
TRANSIENT_STATES = (
    "ROUND_EVAL",
    "HAND_PLAYED",
    "DRAW_TO_HAND",
    "NEW_ROUND",
    "TAROT_PACK",
    "PLANET_PACK",
    "SPECTRAL_PACK",
    "STANDARD_PACK",
    "BUFFOON_PACK",
    "PLAY_TAROT",
)
BLIND_KINDS = ("Small", "Big", "Boss")
MAX_SELECT = 5


class TranslateError(ValueError):
    """The gamestate cannot be translated (unknown key, missing bridge patch, ...)."""


@dataclass
class Translation:
    """Result of :func:`translate`."""

    state: dict
    phase: str
    #: engine hand position -> BalatroBot hand index (the engine sorts the hand)
    hand_real_index: list[int] = field(default_factory=list)
    #: ``True`` if the BalatroBot hand order differs from the engine's sorted order
    hand_needs_rearrange: bool = False
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- templates


@lru_cache(maxsize=64)
def _template_json(deck: str, stake: str, endless: bool = False) -> str:
    g = _engine.Game(1, deck=deck.lower(), stake=stake.lower(), endless=endless)
    return g.export_state()


def template(deck: str, stake: str, endless: bool = False) -> dict:
    return json.loads(_template_json(deck, stake, endless))


# --------------------------------------------------------------------------- cards


def _enum(table: dict, key, what: str):
    try:
        return table[key]
    except KeyError as e:
        raise TranslateError(f"unmapped {what}: {key!r}") from e


def _edition_of(raw: dict) -> str:
    ed = raw.get("edition")
    if not isinstance(ed, dict) or not ed:
        return "Base"
    t = ed.get("type")
    if t is None:
        for k in T.EDITION_TYPE_TO_ID:
            if ed.get(k):
                t = k
    return T.EDITION_TYPE_TO_ID.get(t, "Base") if t else "Base"


def _card_fields(c: dict) -> dict:
    """Common playing-card fields from a BalatroBot card (with ``raw``) or a compact entry."""
    raw = c.get("raw")
    if raw is not None:
        base = raw.get("base") or {}
        rank, suit = base.get("value"), base.get("suit")
        ck = raw.get("center_key") or "c_base"
        ed = _edition_of(raw)
        seal = raw.get("seal")
        ab = raw.get("ability") or {}
        perma = ab.get("perma_bonus", 0) or 0
        debuff = bool(raw.get("debuff"))
        face_down = raw.get("facing") == "back"
        played_ante = bool(ab.get("played_this_ante"))
        uid = raw.get("sort_id", c.get("id", 0))
    else:  # compact entry (discard pile / playing_cards)
        rank, suit = c.get("rank"), c.get("suit")
        ck = c.get("center_key") or "c_base"
        ed = _edition_of(c)
        seal = c.get("seal")
        perma = c.get("perma_bonus", 0) or 0
        debuff = bool(c.get("debuff"))
        face_down = bool(c.get("hidden"))
        played_ante = bool(c.get("played_this_ante"))
        uid = c.get("id", 0)
    return {
        "uid": int(uid),
        "rank": _enum(T.RANK_VALUE_TO_ID, str(rank), "rank"),
        "suit": suit if suit in T.SUITS else _enum({}, suit, "suit"),
        "enhancement": T.ENHANCEMENT_KEY_TO_ID.get(ck, "None"),
        "edition": ed,
        "seal": T.SEAL_TO_ID.get(seal, "None") if seal else "None",
        "perma_bonus": int(perma),
        "debuffed": debuff,
        "face_down": face_down,
        "played_ante": played_ante,
    }


def playing_card(c: dict, *, in_round: bool = True, hidden_ok: bool = True) -> dict:
    d = _card_fields(c)
    if not in_round:
        d["debuffed"] = False
        d["face_down"] = False
    return d


def _engine_sort_key(card: dict):
    stone = card["enhancement"] == "Stone"
    return (stone, -T_RANK_NUM[card["rank"]], T.SUITS.index(card["suit"]), card["uid"])


T_RANK_NUM = {v: k for k, v in T.RANK_NUM_TO_ID.items()}


# --------------------------------------------------------------------------- jokers


def _suit_lane(suit: str | None) -> int:
    return 0 if not suit else T.SUITS.index(suit) + 1


def _hand_lane(name: str | None) -> int:
    return 0 if not name else T.HAND_IDS.index(T.HAND_NAME_TO_ID[name]) + 1


def joker_aux(jid: str, ab: dict, cr: dict) -> int:
    """Packed target lanes (suit 0, rank 8, hand 16, free byte 24) of ``JokerInstance::aux``."""
    suit = rank = hand = byte = 0
    if jid == "AncientJoker":
        suit = _suit_lane((cr.get("ancient_card") or {}).get("suit"))
    elif jid == "Castle":
        suit = _suit_lane((cr.get("castle_card") or {}).get("suit"))
    elif jid == "TheIdol":
        c = cr.get("idol_card") or {}
        suit = _suit_lane(c.get("suit"))
        rank = int(c.get("id") or 0)
    elif jid == "MailInRebate":
        rank = int((cr.get("mail_card") or {}).get("id") or 0)
    elif jid == "ToDoList":
        hand = _hand_lane(ab.get("to_do_poker_hand"))
    elif jid == "LoyaltyCard":
        byte = 1
    return suit | (rank << 8) | (hand << 16) | (byte << 24)


def _num(v, default=0.0) -> float:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else float(default)


def _snap(v: float, base: float, step: float) -> float:
    """Rebuild ``base`` + n x ``step`` by repeated addition, as the game accumulates it.

    BalatroBot serialises numbers with 14 significant digits, so ``1.2000000000000002`` (three
    Constellation planets) arrives as ``1.2``; the 1-chip differences this causes in the final
    score are removed by replaying the accumulation in double precision.
    """
    n = round((v - base) / step)
    if n < 0 or abs(base + n * step - v) > 1e-6:
        return v
    out = base
    for _ in range(n):
        out += step
    return out


def joker_state(
    jid: str, ab: dict, default_value: float, default_counter: int
) -> tuple[float, int]:
    """(``value``, ``counter``) of an engine joker from the real ``card.ability`` table."""
    ex = ab.get("extra")
    exd = ex if isinstance(ex, dict) else {}
    value, counter = default_value, default_counter
    if jid in ("CeremonialDagger", "RideTheBus", "GreenJoker", "RedCard", "FlashCard", "Popcorn",
               "SpareTrousers"):  # fmt: skip
        value = _num(ab.get("mult"), default_value)
    elif jid in (
        "Madness",
        "Hologram",
        "LuckyCat",
        "Campfire",
        "GlassJoker",
        "HitTheRoad",
        "Canio",
    ):
        value = _num(ab.get("x_mult"), default_value)  # steps are exact in binary
    elif jid in ("Constellation", "Vampire"):
        value = _snap(_num(ab.get("x_mult"), default_value), 1.0, 0.1)
    elif jid == "Obelisk":
        value = _snap(_num(ab.get("x_mult"), default_value), 1.0, 0.2)
    elif jid == "Ramen":
        value = _snap(_num(ab.get("x_mult"), default_value), 2.0, -0.01)
    elif jid in ("Runner", "IceCream", "SquareJoker", "Castle", "WeeJoker"):
        value = _num(exd.get("chips"), default_value)
    elif jid == "Rocket":
        value = _num(exd.get("dollars"), default_value)
    elif jid == "TurtleBean":
        value = _num(exd.get("h_size"), default_value)
    elif jid == "Yorick":
        value = _num(ab.get("x_mult"), default_value)
        counter = int(_num(ab.get("yorick_discards"), default_counter))
    elif jid == "Seltzer":
        counter = int(_num(ex if not isinstance(ex, dict) else None, default_counter))
    elif jid == "InvisibleJoker":
        counter = int(_num(ab.get("invis_rounds"), 0))
    elif jid == "LoyaltyCard":
        counter = int(_num(ab.get("hands_played_at_create"), 0))
    return value, counter


def joker_instance(c: dict, uid: int, cr: dict, *, shop: bool = False) -> dict:
    raw = c.get("raw")
    if raw is None:
        raise TranslateError("gamestate has no `raw` card data: apply the bridge Lua patch")
    key = c["key"]
    jid = _enum(T.JOKER_KEY_TO_ID, key, "joker key")
    ab = raw.get("ability") or {}
    facts = et.joker_facts()[jid]
    value, counter = joker_state(jid, ab, facts["value"], facts["counter"])
    perishable = bool(ab.get("perishable"))
    return {
        "uid": uid,
        "id": jid,
        "edition": _edition_of(raw),
        "stickers": {
            "eternal": bool(ab.get("eternal")),
            "perishable": perishable,
            "perish_rounds_left": int(ab.get("perish_tally", 0) or 0) if perishable else 0,
            "rental": bool(ab.get("rental")),
        },
        "debuffed": bool(raw.get("debuff")),
        "sell_value": int((c.get("cost") or {}).get("sell", 1)),
        "value": value,
        "counter": counter,
        "aux": joker_aux(jid, ab, cr),
    }


def consumable_instance(c: dict) -> dict:
    return {
        "id": _enum(T.CONSUMABLE_KEY_TO_ID, c["key"], "consumable key"),
        "negative": _edition_of(c["raw"]) == "Negative" if c.get("raw") else False,
        "sell_value": int((c.get("cost") or {}).get("sell", 1)),
    }


# --------------------------------------------------------------------------- the translation


def _areas(gs: dict, name: str) -> list[dict]:
    a = gs.get(name)
    if not a:
        return []
    cards = a.get("cards") or []
    if isinstance(cards, dict):  # Lua tables with holes can come out as objects
        cards = [cards[k] for k in sorted(cards, key=lambda x: int(x))]
    return cards


def _hand_name_to_id(name: str | None) -> str | None:
    return T.HAND_NAME_TO_ID.get(name) if name else None


def _list(v) -> list:
    """Empty Lua tables are encoded as ``[]`` even when they are maps."""
    return v if isinstance(v, list) else []


def _dict(v) -> dict:
    return v if isinstance(v, dict) else {}


def _blind_kind_and_status(gs: dict, phase: str) -> str:
    blinds = gs["blinds"]
    want = ("CURRENT",) if phase == "Play" else ("SELECT",)
    for kind in BLIND_KINDS:
        if blinds[kind.lower()]["status"] in want:
            return kind
    if phase == "Play":  # should not happen; fall back to the select flag
        for kind in BLIND_KINDS:
            if blinds[kind.lower()]["status"] == "SELECT":
                return kind
    return "Small"


def translate(
    gs: dict, *, pack_origin: str = "Shop", seed: int = 1, endless: bool = False
) -> Translation:
    """Translate a BalatroBot gamestate (bridge patch version 1) into an engine state."""
    st = gs["state"]
    if st not in PHASE_OF_STATE:
        raise TranslateError(f"state {st!r} is not translatable")
    br = gs.get("bridge")
    if not br:
        raise TranslateError("gamestate has no `bridge` block: apply the bridge Lua patch")
    phase = PHASE_OF_STATE[st]
    warnings: list[str] = []
    deck = T.DECK_TO_ID[gs["deck"]]
    stake = T.STAKE_TO_ID[gs["stake"]]
    s = template(deck, stake, endless)
    cr = _dict(br.get("current_round"))
    rr = _dict(br.get("round_resets"))

    s["phase"] = phase
    s["outcome"] = "Loss" if phase == "GameOver" else None
    s["won"] = bool(gs.get("won"))
    s["ante"] = int(gs["ante_num"])
    s["round"] = int(gs["round_num"])
    blind = _blind_kind_and_status(gs, phase)
    s["blind"] = blind
    binfo = gs["blinds"][blind.lower()]
    bl = _dict(br.get("blind"))
    target = bl.get("chips") if phase == "Play" and bl.get("chips") else binfo["score"]
    s["target"] = float(target)

    # --- tags / boss -------------------------------------------------------------------
    s["blind_tags"] = [
        T.TAG_NAME_TO_ID.get(gs["blinds"][k]["tag_name"], "Boss") for k in ("small", "big")
    ]
    orb = br.get("orbital_choices")
    orb_ante = orb[s["ante"] - 1] if isinstance(orb, list) and 0 < s["ante"] <= len(orb) else {}
    s["blind_tag_hands"] = [
        _hand_name_to_id(_dict(orb_ante).get(k)) if t == "Orbital" else None
        for k, t in zip(("Small", "Big"), s["blind_tags"], strict=True)
    ]
    boss_key = _dict(rr.get("blind_choices")).get("Boss")
    s["boss"] = T.BOSS_KEY_TO_ID.get(boss_key) if boss_key else None
    if boss_key and s["boss"] is None:
        raise TranslateError(f"unmapped boss {boss_key!r}")
    s["boss_disabled"] = bool(bl.get("disabled"))
    used = _dict(br.get("bosses_used"))
    s["boss_seen"] = [int(used.get(k, 0) or 0) for k, _ in T.BOSSES]
    bs = s["boss_state"]
    boss_id = s["boss"]
    if boss_id == "TheOx":
        bs["hand"] = _hand_name_to_id(cr.get("most_played_poker_hand"))
    if boss_id == "TheMouth" and phase == "Play":
        # the first hand type of the round locks the others (they score 0); the real game
        # keeps it in `G.GAME.blind.only_hand`
        bs["hand"] = _hand_name_to_id(bl.get("only_hand"))
    bs["removed"] = int(bl.get("hands_sub") or 0) + int(bl.get("discards_sub") or 0)
    bs["draws"] = 1 + int(cr.get("hands_played", 0) or 0) + int(cr.get("discards_used", 0) or 0)
    bs["triggered"] = bool(bl.get("triggered"))

    # --- economy / round ---------------------------------------------------------------
    s["money"] = int(gs["money"])
    s["interest_cap"] = int(round(_num(br.get("interest_cap"), 25) / 5))
    rd = _dict(gs.get("round"))
    s["hands_left"] = int(rd.get("hands_left", 0) or 0)
    s["discards_left"] = int(rd.get("discards_left", 0) or 0)
    s["hands_played_round"] = int(rd.get("hands_played", 0) or 0)
    s["discards_used_round"] = int(rd.get("discards_used", 0) or 0)
    s["round_score"] = float(rd.get("chips", 0) or 0)
    s["last_hand"] = _hand_name_to_id(br.get("last_hand_played"))

    # --- jokers ------------------------------------------------------------------------
    jokers_area = gs["jokers"]
    jcards = _areas(gs, "jokers")
    jokers = [joker_instance(c, i, cr) for i, c in enumerate(jcards)]
    s["jokers"] = jokers
    s["next_joker_uid"] = len(jokers) + 1000
    s["mods"] = et.fold_joker_mods(jokers)
    negatives = sum(1 for j in jokers if j["edition"] == "Negative")
    s["joker_slots"] = int(jokers_area["limit"]) - negatives
    cons_area = gs["consumables"]
    ccards = _areas(gs, "consumables")
    s["consumables"] = [consumable_instance(c) for c in ccards]
    s["consumable_slots"] = int(cons_area["limit"]) - sum(c["negative"] for c in s["consumables"])
    jokers_hidden = any((c.get("raw") or {}).get("facing") == "back" for c in jcards) and bool(
        boss_id == "AmberAcorn" and phase == "Play" and not s["boss_disabled"]
    )
    bs["jokers_hidden"] = jokers_hidden
    if boss_id == "CrimsonHeart" and phase == "Play" and not s["boss_disabled"]:
        for j in jokers:
            if j["debuffed"] and not j["stickers"]["perishable"]:
                bs["joker_uid"] = j["uid"]

    # --- consumable use history --------------------------------------------------------
    usage = _dict(br.get("consumeable_usage"))
    counts = [0] * len(T.CONSUMABLES)
    uniq_planets = 0
    for i, (key, _) in enumerate(T.CONSUMABLES):
        u = usage.get(key)
        n = int(_dict(u).get("count", 0) or 0) if u else 0
        counts[i] = n
        if i < 12 and n > 0:
            uniq_planets += 1
    s["consumable_use_counts"] = counts
    ltp = br.get("last_tarot_planet")
    s["last_tarot_planet"] = T.CONSUMABLE_KEY_TO_ID.get(ltp) if ltp else None
    tot = _dict(br.get("consumeable_usage_total"))
    st_run = s["stats"]["run"]
    st_run["blinds_skipped"] = int(br.get("skips", 0) or 0)
    st_run["tarots_used"] = int(tot.get("tarot", 0) or 0)
    st_run["planets_used"] = int(tot.get("planet", 0) or 0)
    st_run["unique_planets_used"] = uniq_planets
    st_run["bosses_defeated"] = max(0, s["ante"] - 1)

    # --- playing cards -----------------------------------------------------------------
    in_round = phase == "Play"
    hand_cards = _areas(gs, "hand")
    if in_round and boss_id == "CeruleanBell" and not s["boss_disabled"]:
        for c in hand_cards:  # the force-selected card (Cerulean Bell)
            raw = c.get("raw") or {}
            if (raw.get("ability") or {}).get("forced_selection"):
                bs["card_uid"] = int(raw.get("sort_id", c.get("id", 0)))
    hand = [playing_card(c, in_round=in_round) for c in hand_cards]
    draw = [playing_card(c, in_round=False) for c in _areas(gs, "cards")]
    discard = [playing_card(c, in_round=False) for c in _list(gs.get("discard"))]
    all_cards = _list(gs.get("playing_cards"))
    order = sorted(range(len(hand)), key=lambda i: _engine_sort_key(hand[i]))
    s["deck"] = {
        "draw_pile": draw,
        "hand": [hand[i] for i in order],
        "played": [],
        "discard_pile": discard,
        "next_uid": max(
            [c["id"] for c in all_cards] + [c["uid"] for c in hand + draw + discard] + [0]
        )
        + 1,
    }
    n_cards = len(hand) + len(draw) + len(discard)
    if all_cards and n_cards != len(all_cards):
        # cards neither in hand/draw/discard (played cards mid-animation): count them as discarded
        seen = {c["uid"] for c in hand + draw + discard}
        for c in all_cards:
            if c["id"] not in seen:
                s["deck"]["discard_pile"].append(playing_card(c, in_round=False))
        n_cards = len(all_cards)
    start = int(br.get("starting_deck_size") or s["starting_deck_size"])
    s["starting_deck_size"] = start
    s["stats"]["cards_added"] = max(0, n_cards - start)
    s["stats"]["cards_destroyed"] = max(0, start - n_cards)

    # --- hand levels and play counts ---------------------------------------------------
    hands = gs["hands"]
    s["hand_levels"] = [int(hands[name]["level"]) for name, _ in T.HAND_TYPES]
    s["hand_plays_run"] = [int(hands[name]["played"]) for name, _ in T.HAND_TYPES]
    s["hand_plays_round"] = [int(hands[name]["played_this_round"]) for name, _ in T.HAND_TYPES]
    s["stats"]["hands_played"] = sum(s["hand_plays_run"])
    s["stats"]["unused_discards"] = int(br.get("unused_discards", 0) or 0)
    s["stats"]["rounds_won"] = max(0, s["round"] - (1 if phase == "Play" else 0))
    s["stats"]["shops_entered"] = 1 if phase in ("Shop", "PackOpen") else 0

    # --- base values -------------------------------------------------------------------
    s["base_hands"] = int(rr.get("hands", s["base_hands"]))
    s["base_discards"] = int(rr.get("discards", s["base_discards"]))
    temp = int(rr.get("temp_handsize", 0) or 0) if phase == "Play" else 0
    s["round_hand_size"] = temp
    boss_hand_size = -1 if (boss_id == "TheManacle" and in_round and not s["boss_disabled"]) else 0
    limit = br.get("hand_limit") or (gs.get("hand") or {}).get("limit") or 8
    s["base_hand_size"] = max(1, int(limit) - s["mods"]["hand_size"] - temp - boss_hand_size)

    # --- vouchers / shop ---------------------------------------------------------------
    owned = sorted(
        (k for k in _list(br.get("used_vouchers"))),
        key=lambda k: list(T.VOUCHER_KEY_TO_ID).index(k),
    )
    s["vouchers"] = [_enum(T.VOUCHER_KEY_TO_ID, k, "voucher key") for k in owned]
    s["shop_mods"] = et.fold_shop_mods(s["vouchers"])
    av = _dict(cr.get("voucher"))
    ante_voucher_key = av.get("1")
    ante_voucher = T.VOUCHER_KEY_TO_ID.get(ante_voucher_key) if ante_voucher_key else None
    s["ante_voucher"] = None if ante_voucher in s["vouchers"] else ante_voucher
    s["tags"] = [
        T.TAG_KEY_TO_ID[t["key"]] for t in _list(br.get("tags")) if t.get("key") in T.TAG_KEY_TO_ID
    ]
    s["pending_packs"] = []
    s["boss_rerolls_ante"] = 1 if rr.get("boss_rerolled") else 0
    pool = _dict(br.get("pool_flags"))
    s["run_flags"] = ["GrosMichelExtinct"] if pool.get("gros_michel_extinct") else []

    shop_cards = []
    if gs.get("shop") is not None or gs.get("packs") is not None:
        for c in _areas(gs, "shop"):
            free = int((c.get("cost") or {}).get("buy", 1)) == 0
            cset = c["set"]
            if cset == "JOKER":
                kind = {"Joker": joker_instance(c, 2000 + len(shop_cards), cr, shop=True)}
            elif cset in ("TAROT", "PLANET", "SPECTRAL"):
                kind = {"Consumable": _enum(T.CONSUMABLE_KEY_TO_ID, c["key"], "consumable key")}
            else:
                kind = {"PlayingCard": playing_card(c, in_round=False)}
            shop_cards.append({"kind": kind, "free": free})
        shop_packs = [
            {
                "spec": T.pack_spec(c["key"]),
                "free": int((c.get("cost") or {}).get("buy", 1)) == 0,
            }
            for c in _areas(gs, "packs")
        ]
        shop_vouchers = [
            {
                "id": _enum(T.VOUCHER_KEY_TO_ID, c["key"], "voucher key"),
                "from_ante": c["key"] == ante_voucher_key,
            }
            for c in _areas(gs, "vouchers")
        ]
        base_cost = int(rr.get("reroll_cost", 5) or 5)
        inc = int(cr.get("reroll_cost_increase", 0) or 0)
        free_left = int(cr.get("free_rerolls", 0) or 0)
        s["shop"] = {
            "cards": shop_cards,
            "base_slots": int((gs.get("shop") or {}).get("limit") or len(shop_cards)),
            "packs": shop_packs,
            "vouchers": shop_vouchers,
            "rerolls": inc,
            "free_rerolls_used": max(0, s["mods"]["free_rerolls"] - free_left),
            "reroll_base": base_cost,
        }
        # keep the engine's tag-joker split (cards beyond base_slots are tag extras)
        s["shop"]["base_slots"] = min(s["shop"]["base_slots"], len(shop_cards))

    # --- open pack ---------------------------------------------------------------------
    if phase == "PackOpen":
        booster = _dict(br.get("booster"))
        key = booster.get("key")
        if not key:
            raise TranslateError("pack open without a booster key")
        items = []
        for c in _areas(gs, "pack"):
            cset = c["set"]
            if cset == "JOKER":
                items.append({"Joker": joker_instance(c, 3000 + len(items), cr)})
            elif cset in ("TAROT", "PLANET", "SPECTRAL"):
                items.append(
                    {"Consumable": _enum(T.CONSUMABLE_KEY_TO_ID, c["key"], "consumable key")}
                )
            else:
                items.append({"Card": playing_card(c, in_round=False)})
        s["pack"] = {
            "spec": T.pack_spec(key),
            "items": items,
            "picks_left": int(br.get("pack_choices") or 1),
            "origin": pack_origin,
        }
    else:
        s["pack"] = None

    s["rng"] = template(deck, stake, endless)["rng"]
    return Translation(
        state=s,
        phase=phase,
        hand_real_index=list(order),
        hand_needs_rearrange=order != list(range(len(order))),
        warnings=warnings,
    )


def build_game(tr: Translation, *, reseed: int | None = None):
    """``Game.from_state`` of a translation (raises ``ValueError`` if invariants fail)."""
    return _engine.Game.from_state(json.dumps(tr.state), reseed=reseed)


def clone_state(tr: Translation) -> dict:
    return copy.deepcopy(tr.state)
