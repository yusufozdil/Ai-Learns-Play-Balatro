"""Human-readable descriptions of structured actions and states (from the exported state JSON)."""

from __future__ import annotations

import json

from balatro_rl.select import KIND_NAMES


def _item(k) -> tuple[str, str]:
    """(category, name) of a shop card / pack item ``kind`` field."""
    if isinstance(k, dict):
        for key, v in k.items():
            if key == "Joker":
                ed = v.get("edition", "Base")
                return ("Joker", v["id"] + ("" if ed == "Base" else f" [{ed}]"))
            return (key, v if isinstance(v, str) else json.dumps(v)[:60])
    return ("?", str(k))


def _safe(lst, i):
    return lst[i] if 0 <= i < len(lst) else None


def action_item(s: dict, a) -> tuple[str, str] | None:
    """The thing an action is about, as ``(category, name)``; ``None`` for item-less actions."""
    kind, a0 = int(a[0]), int(a[1])
    name = KIND_NAMES[kind]
    sh = s.get("shop") or {}
    if name in ("Buy", "BuyAndUse"):
        c = _safe(sh.get("cards", []), a0)
        return _item(c["kind"]) if c else ("?", f"slot {a0}")
    if name == "BuyVoucher":
        v = _safe(sh.get("vouchers", []), a0)
        return ("Voucher", v["id"]) if v else ("Voucher", f"slot {a0}")
    if name == "OpenPack":
        p = _safe(sh.get("packs", []), a0)
        return ("Pack", f"{p['spec']['kind']}-{p['spec']['size']}") if p else ("Pack", f"slot {a0}")
    if name == "SellJoker" or name == "MoveJoker":
        j = _safe(s["jokers"], a0)
        return ("Joker", j["id"]) if j else ("Joker", f"slot {a0}")
    if name in ("UseConsumable", "SellConsumable"):
        c = _safe(s["consumables"], a0)
        return ("Cons", c["id"] if isinstance(c, dict) else str(c)) if c is not None else None
    if name == "PickPackItem":
        p = s.get("pack") or {}
        it = _safe(p.get("items", []), a0)
        return _item(it) if it is not None else ("?", f"item {a0}")
    return None


def describe_action(s: dict, a) -> str:
    """e.g. ``Buy Joker Hologram``, ``Reroll``, ``EndShop``, ``SkipBlind (tag Speed)``."""
    kind, a0, a1, cards = (int(x) for x in a)
    name = KIND_NAMES[kind]
    it = action_item(s, a)
    if name == "SkipBlind":
        tags = s.get("blind_tags") or []
        t = tags[0 if s.get("blind") == "Small" else 1] if tags else "?"
        return f"SkipBlind (tag {t})"
    if name == "MoveJoker":
        j2 = _safe(s["jokers"], a1)
        return f"MoveJoker {it[1]} -> slot {a1}" + (f" (swap with {j2['id']})" if j2 else "")
    if name == "SelectBlind":
        return f"SelectBlind ({s.get('blind')} blind, target {s.get('target', 0):.0f})"
    if name == "EndShop":
        return "EndShop (leave the shop)"
    if it is None:
        return name
    if it[0] == "Pack":
        return f"{name} {it[1]}"
    extra = ""
    if cards:
        pos = [b for b in range(32) if cards >> b & 1]
        hand = s["deck"]["hand"]
        extra = " on " + ",".join(
            f"{hand[b]['rank']}{hand[b]['suit'][0]}" if b < len(hand) else str(b) for b in pos
        )
    return f"{name} {it[0]} {it[1]}{extra}"


def describe_state(s: dict) -> str:
    jk = ", ".join(j["id"] for j in s["jokers"]) or "-"
    return (
        f"ante {s['ante']} {s['blind']} ({s['phase']}), ${s['money']}, jokers [{jk}], "
        f"vouchers {s.get('vouchers')}"
    )
