"""The sequential card-selection rule: Python reference vs brute force, and vs the core."""

import itertools
import random

import numpy as np
import pytest
from balatro_rl import Game
from balatro_rl.select import (
    CONS_CAP,
    can_finish,
    next_candidates,
    sel_row,
    select_cards,
)


def bits(positions):
    return sum(1 << p for p in positions)


def valid_sets(cand, req, mn, mx):
    pos = [i for i in range(32) if cand >> i & 1]
    out = set()
    for k in range(mn, min(mx, len(pos)) + 1):
        for c in itertools.combinations(pos, k):
            m = bits(c)
            if m & req == req:
                out.add(m)
    return out


def reachable_stops(cand, req, mn, mx):
    """All selections the sequential rule can end on (DFS over next_candidates; also asserts
    that it never dead-ends)."""
    seen, stops, stack = set(), set(), [0]
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        nxt = next_candidates(cand, req, mn, mx, p)
        stop = can_finish(cand, req, mn, mx, p)
        assert nxt or stop, f"dead end at {p:b} for {cand:b}/{req:b}/{mn}/{mx}"
        if stop:
            stops.add(p)
        for i in range(32):
            if nxt >> i & 1:
                stack.append(p | 1 << i)
    return stops


def random_rule(rnd):
    n = rnd.randint(1, 9)
    pos = rnd.sample(range(12), n)
    cand = bits(pos)
    mx = rnd.randint(1, min(5, n))
    mn = rnd.randint(1, mx)
    req = bits(rnd.sample(pos, rnd.randint(0, min(1, mx)))) if rnd.random() < 0.4 else 0
    return cand, req, mn, mx


def test_rule_matches_brute_force_and_never_dead_ends():
    rnd = random.Random(1)
    for _ in range(300):
        cand, req, mn, mx = random_rule(rnd)
        assert reachable_stops(cand, req, mn, mx) == valid_sets(cand, req, mn, mx)


def test_rule_with_several_required_cards():
    # Not produced today (Cerulean forces one card) but the rule must stay exact.
    rnd = random.Random(2)
    for _ in range(200):
        n = rnd.randint(2, 8)
        pos = rnd.sample(range(10), n)
        cand = bits(pos)
        mx = rnd.randint(2, min(5, n))
        mn = rnd.randint(1, mx)
        req = bits(rnd.sample(pos, rnd.randint(2, mx)))
        assert reachable_stops(cand, req, mn, mx) == valid_sets(cand, req, mn, mx)


def test_select_cards_always_returns_a_legal_set():
    rng = np.random.default_rng(0)
    rnd = random.Random(3)
    for _ in range(200):
        cand, req, mn, mx = random_rule(rnd)
        s = select_cards(cand, req, mn, mx, rng)
        assert s in valid_sets(cand, req, mn, mx)


def _walk_games():
    """States of real games: random mix of teacher and random actions."""
    for seed in range(14):
        g = Game(seed=seed, max_episode_steps=10**9)
        mix = random.Random(seed)
        for _ in range(250):
            if g.is_terminal():
                break
            yield g
            parts = g.teacher_parts() if mix.random() < 0.75 else g.random_parts()
            g.step_parts(*parts)


def _check_state(g, rnd, seen_rows):
    """Compare the Python rule with the core on the current state of ``g``; returns #checks."""
    mm = g.masks()
    checked = 0
    for row in range(mm["sel_max"].shape[0]):
        mx = int(mm["sel_max"][row])
        if mx == 0:
            continue
        if row == 0:
            kind, arg0 = 2, 0
        elif row == 1:
            kind, arg0 = 3, 0
        elif row < 2 + CONS_CAP:
            kind, arg0 = 6, row - 2
        else:
            kind, arg0 = 14, row - 2 - CONS_CAP
        assert sel_row(kind, arg0) == row
        cand = int(mm["sel_candidates"][row])
        req = int(mm["sel_required"][row])
        mn = int(mm["sel_min"][row])
        assert cand and mn >= 1 and mx >= mn
        cpos = [i for i in range(32) if cand >> i & 1]
        partials = {0, cand, req}
        for _ in range(12):
            k = rnd.randint(0, min(len(cpos), 6))
            partials.add(bits(rnd.sample(cpos, k)))
        for p in partials:
            ctx = (kind, arg0, bin(cand), bin(req), mn, mx, bin(p))
            assert next_candidates(cand, req, mn, mx, p) == g.select_next_candidates(
                kind, arg0, p
            ), ctx
            assert can_finish(cand, req, mn, mx, p) == g.select_can_finish(kind, arg0, p), ctx
            checked += 1
        # A sequential walk: every prefix agrees.
        p = 0
        while True:
            nxt = next_candidates(cand, req, mn, mx, p)
            assert nxt == g.select_next_candidates(kind, arg0, p)
            assert can_finish(cand, req, mn, mx, p) == g.select_can_finish(kind, arg0, p)
            if not nxt:
                break
            p |= 1 << rnd.choice([i for i in range(32) if nxt >> i & 1])
        seen_rows.add((kind, bool(req)))
    return checked


def test_python_rule_equals_core_on_real_states():
    rnd = random.Random(7)
    seen_rows = set()
    checked = 0
    for g in _walk_games():
        checked += _check_state(g, rnd, seen_rows)
        # Cerulean Bell's mandatory card on some Play states (the walk continues from there).
        if (
            g.masks()["kind_mask"][2]
            and rnd.random() < 0.2
            and g.debug_force_card(rnd.randrange(4))
        ):
            checked += _check_state(g, rnd, seen_rows)
    assert checked > 2000
    assert (2, False) in seen_rows and (3, False) in seen_rows
    assert (2, True) in seen_rows, "Cerulean Bell's required card was never exercised"
    assert any(k in (6, 14) for k, _ in seen_rows), "no targeted consumable / pack row seen"


def test_core_rejects_rows_without_cards():
    g = Game(seed=0)
    with pytest.raises(ValueError):
        g.select_next_candidates(0, 0, 0)  # SelectBlind takes no cards
    with pytest.raises(ValueError):
        g.select_can_finish(2, 0, 0)  # Play in BlindSelect: no hand
