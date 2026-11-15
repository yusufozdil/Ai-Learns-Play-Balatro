"""Deck and stake selection through the Python constructors (Phase 5E)."""

import json

import pytest
from balatro_rl import BalatroVecEnv, Game

DECKS = [
    "red",
    "blue",
    "yellow",
    "green",
    "black",
    "magic",
    "nebula",
    "ghost",
    "abandoned",
    "checkered",
    "zodiac",
    "painted",
    "anaglyph",
    "plasma",
    "erratic",
]
STAKES = ["white", "red", "green", "black", "blue", "purple", "orange", "gold"]


def test_defaults_are_red_white():
    a = json.loads(Game(seed=3).state_json())
    b = json.loads(Game(seed=3, deck="red", stake="white").state_json())
    assert a == b
    assert a["config"]["deck"] == "Red" and a["config"]["stake"] == "White"


@pytest.mark.parametrize("deck", DECKS)
def test_every_deck_builds(deck):
    s = json.loads(Game(seed=1, deck=deck).state_json())
    assert s["config"]["deck"].lower() == deck


@pytest.mark.parametrize("stake", STAKES)
def test_every_stake_builds(stake):
    s = json.loads(Game(seed=1, stake=stake).state_json())
    assert s["config"]["stake"].lower() == stake


def test_names_are_case_insensitive_and_unknown_rejected():
    Game(seed=1, deck="Plasma", stake="GOLD")
    with pytest.raises(ValueError, match="unknown deck"):
        Game(seed=1, deck="rainbow")
    with pytest.raises(ValueError, match="unknown stake"):
        Game(seed=1, stake="platinum")
    with pytest.raises(ValueError):
        BalatroVecEnv(2, deck="nope")


def test_effects_reach_the_state():
    plasma = json.loads(Game(seed=1, deck="plasma").state_json())
    assert plasma["target"] == 600.0  # 300 x 2
    yellow = json.loads(Game(seed=1, deck="yellow").state_json())
    assert yellow["money"] == 14
    blue_stake = json.loads(Game(seed=1, stake="blue").state_json())
    assert blue_stake["base_discards"] == 3


def test_vec_env_runs_a_non_default_combination():
    env = BalatroVecEnv(4, seed=2, num_threads=1, deck="ghost", stake="gold")
    env.reset()
    assert env.num_envs == 4
