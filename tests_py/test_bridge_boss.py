"""Per-round boss state of the real game as the bridge hands it to the engine.

The fixtures are real gamestates (``scripts/bridge_play.py --dump-states``) of the 100-game run
(``runs/bridge/final200_real100``), seeds RLR064 (The Eye), RLR098 (The Mouth), RLR024 (The Needle).
They check that the policy sees exactly what the simulator shows it at the same point:

* The Eye: the per-round hand-type counters (the engine has no separate "used hand types" set;
  rejected repeats count as played, as in the base game) and a 0 score for a repeated type,
* The Mouth: ``boss_state.hand`` = ``blind.only_hand`` once the first hand was played, empty before,
* The Needle: ``hands_left`` 1 and ``removed`` = the hands taken away, before and after discards.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
from balatro_rl.bridge import parity
from balatro_rl.bridge.state import build_game, translate

FIX = Path(__file__).parent / "fixtures" / "bridge_boss"
PAIR, HIGH = 1, 0  # engine hand type indices


def load(name: str) -> dict:
    with gzip.open(FIX / name, "rt") as f:
        return json.load(f)


def _mask(state: dict, ranks: list[str]) -> int:
    """Hand-position mask of the first card of each requested rank (engine order)."""
    hand = state["deck"]["hand"]
    used: set[int] = set()
    for r in ranks:
        i = next(i for i, c in enumerate(hand) if c["rank"] == r and i not in used)
        used.add(i)
    return sum(1 << i for i in used)


def test_eye_counts_the_hand_types_played_this_round():
    gs = load("eye_pair_used.json.gz")
    assert gs["bridge"]["blind"]["name"] == "The Eye"
    assert gs["bridge"]["blind"]["hands"]["Pair"] is True  # the real game's per-blind record
    tr = translate(gs)
    s = tr.state
    assert s["boss"] == "TheEye" and s["blind"] == "Boss" and not s["boss_disabled"]
    assert s["hand_plays_round"][PAIR] == 1 and sum(s["hand_plays_round"]) == 1
    assert s["hands_played_round"] == 1 and s["hands_left"] == 3
    # the policy's observation carries the counter (a different counter is a different input)
    g = build_game(tr)
    obs = g.observation()
    alt = json.loads(json.dumps(s))
    alt["hand_plays_round"] = [0] * 12
    from balatro_rl import _engine

    obs0 = _engine.Game.from_state(json.dumps(alt)).observation()
    assert not np.array_equal(obs["global_f"], obs0["global_f"])
    assert obs["global_i"][2] == obs0["global_i"][2] > 0  # same boss id


def test_eye_repeated_hand_type_scores_zero_and_a_new_one_scores():
    tr = translate(load("eye_pair_used.json.gz"))
    s = tr.state
    # J J 4 is a Pair again: 0 (the real game: RLR064 decision 123 scored 0, prediction 0)
    pair = parity.predict_play(s, _mask(s, ["Jack", "Jack", "Four"]))
    assert pair["hand_type"] == "Pair" and pair["pred"] == 0.0
    # a High Card has not been played this round: it scores
    high = parity.predict_play(s, _mask(s, ["Ace"]))
    assert high["hand_type"] == "HighCard" and high["pred"] > 0


def test_mouth_locks_the_first_hand_type_after_it_was_played():
    tr0 = translate(load("mouth_round_start.json.gz"))
    assert tr0.state["boss"] == "TheMouth" and tr0.state["blind"] == "Boss"
    assert tr0.state["boss_state"]["hand"] is None  # `only_hand` is false before the first hand
    assert build_game(tr0).observation()["global_i"][4] == 0

    gs = load("mouth_highcard_locked.json.gz")
    assert gs["bridge"]["blind"]["only_hand"] == "High Card"
    tr = translate(gs)
    s = tr.state
    assert s["boss_state"]["hand"] == "HighCard"
    obs = build_game(tr).observation()
    assert obs["global_i"][4] == HIGH + 1  # boss_hand lane of the observation
    assert s["hand_plays_round"][HIGH] == 1
    pair = parity.predict_play(s, _mask(s, ["Two", "Two"]))
    assert pair["hand_type"] == "Pair" and pair["pred"] == 0.0
    # another High Card is the locked type: it scores
    again = parity.predict_play(s, _mask(s, ["Ace"]))
    assert again["hand_type"] == "HighCard" and again["pred"] > 0


def test_needle_leaves_one_hand_before_and_after_discards():
    gs = load("needle_after_discards.json.gz")
    br = gs["bridge"]
    assert br["blind"]["name"] == "The Needle" and br["blind"]["hands_sub"] == 3
    tr = translate(gs)
    s = tr.state
    assert s["boss"] == "TheNeedle" and s["blind"] == "Boss"
    assert s["hands_left"] == 1 and s["boss_state"]["removed"] == 3
    assert s["base_hands"] == 4  # the real round_resets: the Needle only changes the round's hands
    assert s["discards_left"] == 2 and s["discards_used_round"] == 2
    assert s["target"] == float(br["blind"]["chips"])
    obs = build_game(tr).observation()
    assert abs(float(obs["global_f"][7]) - 1 / 8) < 1e-6  # hands_left / 8


def test_the_fool_is_dead_while_c_fool_is_the_last_tarot_planet():
    """Real game (measured through the bridge): after The Fool is used `last_tarot_planet` is
    `c_fool` and the next Fool is refused ("cannot be used at this time") until another Tarot or
    Planet is used. The old engine kept the previous card and offered the Fool anyway (9 rejects in
    the 100-game run, RLR010)."""
    from balatro_rl import _engine

    gs = load("needle_after_discards.json.gz")  # any Play state works; the Fool needs no target
    gs["bridge"]["last_tarot_planet"] = "c_fool"
    gs["consumables"] = {
        "count": 1,
        "limit": 2,
        "cards": [{"key": "c_fool", "set": "TAROT", "raw": {"edition": None}, "cost": {"sell": 1}}],
    }
    tr = translate(gs)
    assert tr.state["last_tarot_planet"] == "TheFool"
    fool_kind = 6  # UseConsumable
    masks = build_game(tr).masks()
    assert not masks["arg0_mask"][fool_kind][0]
    # with a real Planet as the last card the same Fool is usable
    gs["bridge"]["last_tarot_planet"] = "c_pluto"
    tr = translate(gs)
    assert tr.state["last_tarot_planet"] == "Pluto"
    assert _engine.Game.from_state(json.dumps(tr.state)).masks()["arg0_mask"][fool_kind][0]
