"""Facts about the engine that the translator needs, derived from the engine itself.

The engine state JSON carries two *derived caches* that ``Game.from_state`` verifies
(``mods`` = fold of the jokers' passive modifiers, ``shop_mods`` = fold of the owned vouchers'
passives). Instead of re-typing those rules, they are measured on the engine:

* per-joker defaults (``value``, ``counter``) and passive modifiers: a fresh run with that
  ``start_joker`` is exported,
* per-voucher passive modifiers: a shop with the voucher in a slot is built through
  ``from_state`` and the voucher is bought.

Folding mirrors ``hooks::collect_modifiers`` (jokers) and ``ShopMods::merge`` (vouchers); both
are checked against the engine over thousands of states in ``tests_py/test_bridge.py``.
"""

from __future__ import annotations

import copy
import json
from functools import lru_cache

from balatro_rl import _engine

from .tables import JOKER_IDS, VOUCHER_IDS

KIND_BUY_VOUCHER = 12

_MODS_DEFAULT = {
    "rules": {
        "four_fingers": False,
        "shortcut": False,
        "smeared": False,
        "pareidolia": False,
        "splash": False,
    },
    "prob_mult": 1,
    "hand_size": 0,
    "hands_per_round": 0,
    "discards_per_round": 0,
    "joker_slots": 0,
    "consumable_slots": 0,
    "debt_limit": 0,
    "free_rerolls": 0,
    "free_planets": False,
    "allow_duplicates": False,
    "interest_per_5": 0,
}

_SHOP_MODS_NONE = {
    "card_slots": 0,
    "discount": "None",
    "edition_mult": 1,
    "reroll_base": None,
    "tarot_weight": None,
    "planet_weight": None,
    "playing_cards": False,
    "illusion": False,
    "telescope": False,
    "omen_globe": False,
    "observatory": False,
    "boss_reroll": "None",
}
_DISCOUNT_ORDER = {"None": 0, "Clearance": 1, "Liquidation": 2}
_BOSS_REROLL_ORDER = {"None": 0, "OncePerAnte": 1, "Unlimited": 2}


def mods_default() -> dict:
    return copy.deepcopy(_MODS_DEFAULT)


def shop_mods_none() -> dict:
    return copy.deepcopy(_SHOP_MODS_NONE)


@lru_cache(maxsize=1)
def joker_facts() -> dict[str, dict]:
    """``{JokerId: {"value", "counter", "mods"}}`` of a freshly created joker."""
    out = {}
    for jid in JOKER_IDS:
        s = json.loads(_engine.Game(1, start_joker=jid).export_state())
        j = s["jokers"][0]
        out[jid] = {"value": j["value"], "counter": j["counter"], "mods": s["mods"]}
    return out


def fold_joker_mods(jokers: list[dict]) -> dict:
    """``collect_modifiers`` over engine joker JSON (debuffed jokers are skipped)."""
    facts = joker_facts()
    m = mods_default()
    for j in jokers:
        if j["debuffed"]:
            continue
        jm = facts[j["id"]]["mods"]
        if j["id"] == "TurtleBean":
            jm = dict(jm, hand_size=int(max(j["value"], 0.0)))
        for k, v in jm.items():
            if k == "rules":
                for rk, rv in v.items():
                    m["rules"][rk] = m["rules"][rk] or rv
            elif k == "prob_mult":
                m[k] *= v
            elif isinstance(v, bool):
                m[k] = m[k] or v
            else:
                m[k] += v - _MODS_DEFAULT[k]
    return m


def _shop_template() -> dict:
    g = _engine.Game(3)
    for _ in range(400):
        s = json.loads(g.export_state())
        if s["phase"] == "Shop" and not s["vouchers"]:
            return s
        kind, a0, a1, mask = g.teacher_parts()
        g.step_parts(kind, a0, a1, mask)
    raise RuntimeError("no shop state found")


@lru_cache(maxsize=1)
def voucher_facts() -> dict[str, dict]:
    """``{VoucherId: shop_mods contribution}`` measured by buying each voucher in the engine."""
    tmpl = _shop_template()
    out: dict[str, dict] = {}
    for idx, vid in enumerate(VOUCHER_IDS):
        s = copy.deepcopy(tmpl)
        s["money"] = 1000
        s["shop"]["vouchers"] = [{"id": vid, "from_ante": False}]
        s["ante_voucher"] = None
        if idx % 2 == 1:  # upgrades need their base voucher owned (requires())
            base = VOUCHER_IDS[idx - 1]
            s["vouchers"] = [base]
            s["shop_mods"] = out[base]
        g = _engine.Game.from_state(json.dumps(s))
        g.step_parts(KIND_BUY_VOUCHER, 0, 0, 0)
        after = json.loads(g.export_state())
        assert vid in after["vouchers"], vid
        out[vid] = after["shop_mods"]
    return out


def fold_shop_mods(vouchers: list[str]) -> dict:
    """``fold_shop_mods`` over owned voucher ids (engine ``ShopMods::merge`` semantics)."""
    # The measured value of a voucher is the fold of everything owned when it was bought, so a
    # plain per-voucher merge needs the single-voucher contribution: measure on a clean slate.
    m = shop_mods_none()
    for v in vouchers:
        o = _single_contribution(v)
        m["card_slots"] += o["card_slots"]
        if _DISCOUNT_ORDER[o["discount"]] > _DISCOUNT_ORDER[m["discount"]]:
            m["discount"] = o["discount"]
        m["edition_mult"] = max(m["edition_mult"], o["edition_mult"])
        a, b = m["reroll_base"], o["reroll_base"]
        m["reroll_base"] = b if a is None else a if b is None else min(a, b)
        for k in ("tarot_weight", "planet_weight"):
            a, b = m[k], o[k]
            m[k] = b if a is None else a if b is None else max(a, b)
        for k in ("playing_cards", "illusion", "telescope", "omen_globe", "observatory"):
            m[k] = m[k] or o[k]
        if _BOSS_REROLL_ORDER[o["boss_reroll"]] > _BOSS_REROLL_ORDER[m["boss_reroll"]]:
            m["boss_reroll"] = o["boss_reroll"]
    return m


@lru_cache(maxsize=64)
def _single_contribution(vid: str) -> dict:
    """A voucher's own passive (measured with only the voucher owned, bypassing ``requires``).

    Upgrades are measured by diffing against their base; the diff is exact because every
    passive either adds (``card_slots``) or overrides with a better value.
    """
    full = voucher_facts()[vid]
    idx = VOUCHER_IDS.index(vid)
    if idx % 2 == 0:
        return full
    base = voucher_facts()[VOUCHER_IDS[idx - 1]]
    own = dict(full)
    own["card_slots"] = full["card_slots"] - base["card_slots"]
    return own
