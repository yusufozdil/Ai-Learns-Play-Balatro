"""Build-power meter binding: deterministic in (state, salt), does not touch the game."""

from __future__ import annotations

import pytest
from balatro_rl import _engine


def _game_at_blind_select(seed: int = 5, rounds: int = 3):
    """Play the teacher until `rounds` blinds are behind us and a BlindSelect is open."""
    g = _engine.Game(seed)
    seen = 0
    for _ in range(2000):
        if g.is_terminal():
            break
        s = g.state_json()
        if '"phase":"BlindSelect"' in s:
            if seen >= rounds:
                return g
            seen += 1
        g.step_parts(*g.teacher_parts())
    pytest.fail("no BlindSelect reached")


@pytest.mark.parametrize("boss_aware", [False, True])
def test_power_deterministic_and_pure(boss_aware: bool) -> None:
    g = _game_at_blind_select()
    before = g.state_json()
    a = _engine.power(g, k=8, boss_aware=boss_aware)
    b = _engine.power(g, k=8, boss_aware=boss_aware)
    c = _engine.power(before, k=8, boss_aware=boss_aware)  # JSON in: same state, same answer
    assert a == b == c
    assert g.state_json() == before
    assert -3.0 <= a["phi"] <= 3.0
    assert 0.0 <= a["frac_beat"] <= 1.0
    assert a["k"] == 8
    assert a["target"] > 0
    # Another salt is another Monte-Carlo sample (same state).
    assert _engine.power(g, k=8, boss_aware=boss_aware, salt=1) != a


def test_power_follows_the_state_not_the_game_rng() -> None:
    g1, g2 = _game_at_blind_select(5), _game_at_blind_select(5)
    assert _engine.power(g1, 4) == _engine.power(g2, 4)
    # The ante's actual boss matters (v2) -- the target differs from the plain boss (v1) at
    # least for Wall / Violet Vessel / Needle; here just check both run and are finite.
    for flag in (False, True):
        assert (
            _engine.power(g1, 4, boss_aware=flag)["phi"]
            == _engine.power(g2, 4, boss_aware=flag)["phi"]
        )
