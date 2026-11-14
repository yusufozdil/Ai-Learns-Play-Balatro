"""Reference implementation (pure Python/numpy) of the action-building side of the interface:
the sequential card-selection rule, a mask-conforming action sampler, and a mask check.

The model's autoregressive action head (Phase 9) follows the same steps with learned logits
instead of uniform choices: kind -> arg0 -> (arg1) -> cards one at a time + STOP. The rule is
specified in ``docs/design/rl_interface.md`` section 6 and is tested against the core's
``CardSelect::next_candidates`` / ``can_finish`` (``tests_py/test_select.py``).
"""

from __future__ import annotations

import numpy as np

from balatro_rl import _engine

_CAPS = _engine.layout()["caps"]
KINDS: int = _CAPS["kinds"]
CONS_CAP: int = _CAPS["cons"]
PACK_ITEM_CAP: int = _CAPS["pack_items"]
HAND_CAP: int = _CAPS["hand"]
NSEL: int = _CAPS["nsel"]

KIND_NAMES: tuple[str, ...] = (
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
KIND_PLAY, KIND_DISCARD, KIND_USE_CONSUMABLE, KIND_MOVE_JOKER, KIND_PICK_PACK_ITEM = 2, 3, 6, 8, 14


def sel_row(kind: int, arg0: int) -> int | None:
    """Row of the card-selection table of ``(kind, arg0)``; ``None`` if it never takes cards."""
    if kind == KIND_PLAY:
        return 0
    if kind == KIND_DISCARD:
        return 1
    if kind == KIND_USE_CONSUMABLE and arg0 < CONS_CAP:
        return 2 + arg0
    if kind == KIND_PICK_PACK_ITEM and arg0 < PACK_ITEM_CAP:
        return 2 + CONS_CAP + arg0
    return None


def next_candidates(candidates: int, required: int, mn: int, mx: int, partial: int) -> int:
    """Hand positions (bit set) that may be added to ``partial`` so that a legal completion
    still exists. ``mn`` is not needed (a selection is never stuck below it) but kept for symmetry
    with ``can_finish``."""
    k = partial.bit_count()
    if k >= mx:
        return 0
    pool = candidates & ~partial
    missing = required & ~partial
    if missing and mx - k == missing.bit_count():
        return missing
    return pool


def can_finish(candidates: int, required: int, mn: int, mx: int, partial: int) -> bool:
    """``partial`` is a legal complete selection (the model may emit STOP)."""
    k = partial.bit_count()
    return partial & ~candidates == 0 and partial & required == required and mn <= k <= mx


def select_cards(candidates: int, required: int, mn: int, mx: int, rng: np.random.Generator) -> int:
    """Sequential uniform selection: at each step uniform among ``{STOP if allowed} + next``."""
    partial = 0
    while True:
        nxt = next_candidates(candidates, required, mn, mx, partial)
        stop = can_finish(candidates, required, mn, mx, partial)
        opts = [i for i in range(HAND_CAP) if nxt >> i & 1]
        n = len(opts) + int(stop)
        if n == 0:  # cannot happen: the rule never dead-ends
            raise RuntimeError("card selection dead-ended")
        r = int(rng.integers(n))
        if stop and r == 0:
            return partial
        partial |= 1 << opts[r - int(stop)]


def sample_action(masks: dict[str, np.ndarray], i: int, rng: np.random.Generator) -> list[int]:
    """A uniformly random mask-conforming action for env ``i`` of a batch of ``masks``."""
    kinds = np.flatnonzero(masks["kind_mask"][i])
    kind = int(rng.choice(kinds))
    arg0 = int(rng.choice(np.flatnonzero(masks["arg0_mask"][i, kind])))
    arg1 = 0
    if kind == KIND_MOVE_JOKER:
        dest = [int(d) for d in np.flatnonzero(masks["arg1_mask"][i]) if d != arg0]
        arg1 = int(rng.choice(dest))
    cards = 0
    row = sel_row(kind, arg0)
    if row is not None and masks["sel_max"][i, row] > 0:
        cards = select_cards(
            int(masks["sel_candidates"][i, row]),
            int(masks["sel_required"][i, row]),
            int(masks["sel_min"][i, row]),
            int(masks["sel_max"][i, row]),
            rng,
        )
    return [kind, arg0, arg1, cards]


def sample_actions(masks: dict[str, np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """``sample_action`` for every env (a Python loop: a reference, not a fast path)."""
    n = masks["kind_mask"].shape[0]
    return np.array([sample_action(masks, i, rng) for i in range(n)], dtype=np.int64)


def conforms(masks: dict[str, np.ndarray], i: int, action) -> bool:
    """Whether ``action`` of env ``i`` is expressible through the masks (kind, arg0, arg1 and a
    card set that the sequential rule can build)."""
    kind, arg0, arg1, cards = (int(x) for x in action)
    if not 0 <= kind < KINDS or not masks["kind_mask"][i, kind]:
        return False
    if not masks["arg0_mask"][i, kind, arg0 if 0 <= arg0 < masks["arg0_mask"].shape[2] else 0]:
        return False
    if kind == KIND_MOVE_JOKER and (arg1 == arg0 or not masks["arg1_mask"][i, arg1]):
        return False
    row = sel_row(kind, arg0)
    mx = int(masks["sel_max"][i, row]) if row is not None else 0
    if mx == 0:
        return cards == 0
    cand = int(masks["sel_candidates"][i, row])
    req = int(masks["sel_required"][i, row])
    mn = int(masks["sel_min"][i, row])
    return can_finish(cand, req, mn, mx, cards)


def unpack_bits(mask: np.ndarray, nbits: int = HAND_CAP) -> np.ndarray:
    """Expand a ``uint32`` bit-set array ``(...)`` to a bool array ``(..., nbits)``
    (bit ``i`` -> index ``i``)."""
    m = np.asarray(mask, dtype=np.uint64)[..., None]
    return (m >> np.arange(nbits, dtype=np.uint64) & np.uint64(1)).astype(bool)
