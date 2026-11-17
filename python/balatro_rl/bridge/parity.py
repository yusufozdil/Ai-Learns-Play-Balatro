"""Score parity: what the engine predicts for a Play on a translated real state.

``predict_play`` plays the same cards on copies of the translated state and reads the engine's
``round_score`` delta. The copies differ only in the RNG streams that scoring can consume
(Lucky / Glass rolls, Misprint, Space Joker, ...): if they all agree the prediction is
*deterministic* and must match the real game exactly; otherwise the hand is flagged
*stochastic* and compared as a range (``min <= real <= max``) instead.
"""

from __future__ import annotations

import json

from balatro_rl import _engine

from . import tables as T

#: streams a scoring step can consume (everything except the draw-pile shuffle)
SCORING_STREAMS = ["jokers", "cards", "consumables", "misc", "shop", "packs", "bosses", "tags"]
KIND_PLAY = 2

#: jokers whose score contribution is random in the engine (static hint for the report)
STOCHASTIC_JOKERS = {
    "Misprint",
    "Bloodstone",
    "SpaceJoker",
    "LuckyCat",  # needs a Lucky card trigger
    "BusinessCard",
    "EightBall",
    "Seance",
    "Hallucination",
}


def _mix(i: int) -> int:
    """Well-spread u64 seed for variant ``i``. Consecutive small seeds give correlated
    streams in ``GameRng::new`` (observed: 256 consecutive seeds never produced the extreme
    Misprint rolls), so the variants use a golden-ratio stride instead."""
    return ((i + 1) * 0x9E3779B97F4A7C15) % (1 << 64)


def static_flags(state: dict, mask: int) -> list[str]:
    """Reasons (from the visible state) why the played hand's score may be random."""
    hand = state["deck"]["hand"]
    played = [hand[i] for i in range(len(hand)) if (mask >> i) & 1]
    flags = []
    if any(c["enhancement"] == "Lucky" for c in played):
        flags.append("lucky_card")
    if any(c["enhancement"] == "Glass" for c in played):
        flags.append("glass_card")  # breaking never changes the score of the hand itself
    for j in state["jokers"]:
        if j["id"] in STOCHASTIC_JOKERS and not j["debuffed"]:
            flags.append(f"joker:{j['id']}")
    return flags


def predict_play(state: dict, mask: int, variants: int = 4, wide: int = 256) -> dict:
    """Engine prediction for ``Play(mask)`` on ``state`` (engine state dict)."""
    text = json.dumps(state)
    before = float(state["round_score"])
    gains: list[float] = []
    hand_types: list[str | None] = []
    invalid = False
    i = 0
    while i < variants:
        if i == 4 and len(set(gains)) > 1:
            variants = wide  # random hand: sample the support more densely
        i += 1
        g = _engine.Game.from_state(text, reseed=_mix(i), streams=SCORING_STREAMS)
        _, _, inv = g.step_parts(KIND_PLAY, 0, 0, mask)
        if inv:
            invalid = True
            break
        after = json.loads(g.export_state())
        gains.append(float(after["round_score"]) - before)
        hand_types.append(after["last_hand"])
    if invalid or not gains:
        return {"invalid": True}
    return {
        "invalid": False,
        "pred": gains[0],
        "min": min(gains),
        "max": max(gains),
        "deterministic": len(set(gains)) == 1,
        "hand_type": hand_types[0],
        "flags": static_flags(state, mask),
    }


def real_hand_type(before: dict, after: dict) -> str | None:
    """Engine HandType id of the hand whose ``played`` counter grew between two gamestates."""
    for name, eid in T.HAND_TYPES:
        if after["hands"][name]["played"] > before["hands"][name]["played"]:
            return eid
    return None


def compare(pred: dict, real_gain: float) -> dict:
    """Parity verdict for one hand."""
    if pred.get("invalid"):
        return {"status": "pred_invalid"}
    exact = abs(pred["pred"] - real_gain) < 0.5
    if pred["deterministic"]:
        status = "exact" if exact else "mismatch"
    else:
        inside = pred["min"] - 0.5 <= real_gain <= pred["max"] + 0.5
        status = "exact" if exact else ("in_range" if inside else "stochastic_miss")
    return {
        "status": status,
        "diff": real_gain - pred["pred"],
        "ratio": (real_gain / pred["pred"]) if pred["pred"] else None,
    }
