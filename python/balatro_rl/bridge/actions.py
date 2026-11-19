"""Engine action ``[kind, arg0, arg1, card_mask]`` -> BalatroBot API call sequence.

The engine hand is rank-sorted (``Deck::sort_hand``), BalatroBot's is in the order the game
keeps it. ``Translation.hand_real_index[p]`` is the BalatroBot index of engine hand position
``p``. Actions that depend on the order of the selected cards (Play: left-to-right scoring order;
Death, Strength... : leftmost/rightmost targets) first send ``rearrange hand`` so that the real
hand is in engine order, after which engine position == BalatroBot index.

Table of kinds -> calls (docs/design/bridge.md section 5):

======================  =============================================================
engine kind             BalatroBot calls
======================  =============================================================
0 SelectBlind           ``select``
1 SkipBlind             ``skip``
2 Play                  [``rearrange hand``] ``play cards``
3 Discard               ``discard cards``
4 SellJoker             ``sell joker``
5 EndShop               ``next_round``
6 UseConsumable         [``rearrange hand``] ``use consumable [cards]``
7 SellConsumable        ``sell consumable``
8 MoveJoker             ``rearrange jokers`` (remove ``from``, insert at ``to``)
9 Buy                   ``buy card``
10 BuyAndUse            ``buy card`` then ``use consumable`` (the new last slot); with full
                        consumable slots ``buy card use=true`` (bridge patch v5, the game's
                        own "Buy and use" button needs no free slot)
11 OpenPack             ``buy pack``
12 BuyVoucher           ``buy voucher``
13 Reroll               ``reroll``
14 PickPackItem         [``rearrange hand``] ``pack card [targets]``
15 SkipPack             ``pack skip``
16 RerollBoss           ``reroll_boss`` (bridge Lua patch)
======================  =============================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .state import Translation

KIND_NAMES = (
    "SelectBlind",
    "SkipBlind",
    "Play",
    "Discard",
    "SellJoker",
    "EndShop",
    "UseConsumable",
    "SellConsumable",
    "MoveJoker",
    "Buy",
    "BuyAndUse",
    "OpenPack",
    "BuyVoucher",
    "Reroll",
    "PickPackItem",
    "SkipPack",
    "RerollBoss",
)
(
    SELECT_BLIND,
    SKIP_BLIND,
    PLAY,
    DISCARD,
    SELL_JOKER,
    END_SHOP,
    USE_CONSUMABLE,
    SELL_CONSUMABLE,
    MOVE_JOKER,
    BUY,
    BUY_AND_USE,
    OPEN_PACK,
    BUY_VOUCHER,
    REROLL,
    PICK_PACK_ITEM,
    SKIP_PACK,
    REROLL_BOSS,
) = range(17)


@dataclass
class Call:
    """One API call. ``tag`` marks calls whose parameters depend on the previous response."""

    method: str
    params: dict = field(default_factory=dict)
    tag: str = ""

    def describe(self) -> str:
        return f"{self.method}({self.params})" if self.params else self.method


def mask_positions(mask: int) -> list[int]:
    return [i for i in range(32) if (mask >> i) & 1]


def joker_move_permutation(n: int, frm: int, to: int) -> list[int]:
    """``rearrange jokers`` argument: new slot ``i`` holds old slot ``perm[i]``."""
    order = list(range(n))
    order.pop(frm)
    order.insert(to, frm)
    return order


def decode_action(action, n_hand: int | None = None) -> dict:
    """Human-readable form of an engine action (for the log)."""
    kind, a0, a1, mask = (int(x) for x in action)
    d = {"kind": KIND_NAMES[kind] if 0 <= kind < len(KIND_NAMES) else kind, "arg0": a0}
    if kind == MOVE_JOKER:
        d["arg1"] = a1
    if mask:
        d["cards"] = mask_positions(mask)
    return d


def plan_action(action, tr: Translation) -> list[Call]:
    """API calls realising ``action`` on the state ``tr`` was translated from."""
    kind, a0, a1, mask = (int(x) for x in action)
    pos = mask_positions(mask)
    calls: list[Call] = []

    def hand_rearrange() -> None:
        if tr.hand_needs_rearrange:
            calls.append(Call("rearrange", {"hand": list(tr.hand_real_index)}))

    if kind == SELECT_BLIND:
        calls.append(Call("select"))
    elif kind == SKIP_BLIND:
        calls.append(Call("skip"))
    elif kind in (PLAY, DISCARD):
        if kind == PLAY:
            hand_rearrange()
            idx = pos
        else:
            idx = [tr.hand_real_index[p] for p in pos]
        calls.append(Call("play" if kind == PLAY else "discard", {"cards": idx}))
    elif kind == SELL_JOKER:
        calls.append(Call("sell", {"joker": a0}))
    elif kind == END_SHOP:
        calls.append(Call("next_round"))
    elif kind == USE_CONSUMABLE:
        if pos:
            hand_rearrange()
            calls.append(Call("use", {"consumable": a0, "cards": pos}))
        else:
            calls.append(Call("use", {"consumable": a0}))
    elif kind == SELL_CONSUMABLE:
        calls.append(Call("sell", {"consumable": a0}))
    elif kind == MOVE_JOKER:
        n = len(tr.state["jokers"])
        calls.append(Call("rearrange", {"jokers": joker_move_permutation(n, a0, a1)}))
    elif kind == BUY:
        calls.append(Call("buy", {"card": a0}))
    elif kind == BUY_AND_USE:
        st = tr.state
        held = st.get("consumables") or []
        capacity = st.get("consumable_slots", 2) + sum(1 for c in held if c["negative"])
        if len(held) >= capacity:
            # stock `buy` refuses a consumable with full slots; the game's own button does not
            calls.append(Call("buy", {"card": a0, "use": True}))
        else:
            calls.append(Call("buy", {"card": a0}))
            calls.append(Call("use", {}, tag="use_new_consumable"))
    elif kind == OPEN_PACK:
        calls.append(Call("buy", {"pack": a0}))
    elif kind == BUY_VOUCHER:
        calls.append(Call("buy", {"voucher": a0}))
    elif kind == REROLL:
        calls.append(Call("reroll"))
    elif kind == PICK_PACK_ITEM:
        params: dict = {"card": a0}
        if pos:
            hand_rearrange()
            params["targets"] = pos
        calls.append(Call("pack", params))
    elif kind == SKIP_PACK:
        calls.append(Call("pack", {"skip": True}))
    elif kind == REROLL_BOSS:
        calls.append(Call("reroll_boss"))
    else:
        raise ValueError(f"unknown action kind {kind}")
    return calls
