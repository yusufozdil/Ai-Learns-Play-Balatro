"""Branching API (I0): Game.clone / export_state / from_state / reseed, VecEnv export_state /
import_state / set_start_jokers / drain_finished_ex."""

import json

import numpy as np
import pytest
from balatro_rl import BalatroVecEnv, Game

STREAMS = ["deck", "shop", "packs", "bosses", "tags", "jokers", "cards", "consumables", "misc"]


def view(g):
    return {**g.observation(), **g.masks()}


def same_view(a, b):
    return all(np.array_equal(a[k], b[k]) for k in a) and a.keys() == b.keys()


def advance(g, steps, teacher=True):
    """Play `steps` steps (stops at the end of the run) with the heuristic teacher (reaches deep
    states: shops, packs, later antes) or the random sampler; returns the actions."""
    acts = []
    for _ in range(steps):
        if g.is_terminal():
            break
        a = g.teacher_parts() if teacher else g.random_parts()
        g.step_parts(*a)
        acts.append(a)
    return acts


def play_until(g, pred, limit=400):
    for _ in range(limit):
        if pred(g):
            return True
        advance(g, 1)
    return False


def phase(g):
    return json.loads(g.state_json())["phase"]


def trajectory(g, actions):
    out = []
    for a in actions:
        if g.is_terminal():
            break
        r = g.step_parts(*a)
        out.append((r, view(g)))
    return out


def same_traj(a, b):
    return len(a) == len(b) and all(x[0] == y[0] and same_view(x[1], y[1]) for x, y in zip(a, b))


# ---------------------------------------------------------------- Game.clone / export / reseed


@pytest.mark.parametrize("seed", [1, 5, 11])
def test_clone_continues_identically(seed):
    g = Game(seed=seed)
    advance(g, 37)
    c = g.clone()
    assert same_view(view(g), view(c))
    ref = []
    # The same actions on both (the teacher of the original decides; it is not cloned).
    for _ in range(250):
        if g.is_terminal():
            break
        a = g.teacher_parts()
        assert g.step_parts(*a) == c.step_parts(*a)
        assert same_view(view(g), view(c))
        ref.append(a)
    assert len(ref) > 50


def test_clone_is_independent():
    g = Game(seed=2)
    advance(g, 20)
    before = g.export_state()
    c = g.clone()
    advance(c, 60)
    assert g.export_state() == before
    advance(g, 5)
    assert c.export_state() != g.export_state()


def test_export_from_state_round_trip():
    g = Game(seed=4)
    advance(g, 45)
    s = g.export_state()
    assert s == g.state_json()
    h = Game.from_state(s)
    assert h.export_state() == s
    acts = advance(g, 120)
    h2 = Game.from_state(s)
    trajectory(h2, acts)
    assert h2.export_state() == g.export_state()


def test_from_state_rejects_bad_input():
    with pytest.raises(ValueError):
        Game.from_state("{not json")
    g = Game(seed=1)
    with pytest.raises(ValueError):
        g.reseed(1, streams=["nope"])
    # a finished run cannot be resumed
    for _ in range(100_000):
        if g.is_terminal():
            break
        g.step_parts(*g.random_parts())
    with pytest.raises(ValueError):
        Game.from_state(g.export_state())


BRANCH_PHASES = ["BlindSelect", "Play", "Shop"]


@pytest.mark.parametrize("target", BRANCH_PHASES)
def test_reseed_keeps_the_view_and_changes_the_future(target):
    g = Game(seed=6)
    assert play_until(g, lambda x: phase(x) == target and json.loads(x.state_json())["round"] >= 2)
    base = view(g)
    c = g.clone()
    c.reseed(777)
    assert same_view(base, view(c)), "reseed must not change anything visible"
    assert c.export_state() != g.export_state()
    # Hidden information differs: same actions, different future.
    g2 = g.clone()
    acts = advance(g2, 200)
    t_orig = trajectory(g.clone(), acts)
    t_same = trajectory(g.clone(), acts)
    t_new = trajectory(c, acts)
    assert same_traj(t_orig, t_same)
    assert not same_traj(t_orig, t_new)


def test_reseed_is_deterministic_and_seed_dependent():
    g = Game(seed=8)
    advance(g, 30)
    a, b, c = g.clone(), g.clone(), g.clone()
    a.reseed(5)
    b.reseed(5)
    c.reseed(6)
    assert a.export_state() == b.export_state()
    assert a.export_state() != c.export_state()


def test_reseed_with_face_down_cards_keeps_the_view():
    g = Game(seed=9)
    assert play_until(g, lambda x: phase(x) == "Play")
    g.debug_face_down(0b10101)
    base = view(g)
    c = g.clone()
    c.reseed(31)
    assert same_view(base, view(c))
    # hidden identities were re-sampled
    hg = json.loads(g.state_json())["deck"]["hand"]
    hc = json.loads(c.state_json())["deck"]["hand"]
    assert hg != hc
    assert len(hg) == len(hc)


def hand_obs(g):
    o = g.observation()
    return o["hand_i"].copy(), o["hand_f"].copy()


def test_partial_reseed_streams():
    """Reseeding only the shop stream leaves the next hand alone; reseeding the deck stream
    changes it."""
    g = Game(seed=12)
    assert g.legal_actions()[0] == "SelectBlind"
    shop_only, deck_only, plain = g.clone(), g.clone(), g.clone()
    shop_only.reseed(99, streams=["shop", "packs", "bosses", "tags", "jokers", "cards"])
    deck_only.reseed(99, streams=["deck"])
    for x in (shop_only, deck_only, plain):
        x.step_parts(0, 0, 0, 0)  # SelectBlind: the round starts, a hand is drawn
    h0 = hand_obs(plain)
    hs = hand_obs(shop_only)
    hd = hand_obs(deck_only)
    assert all(np.array_equal(p, q) for p, q in zip(h0, hs))
    assert not all(np.array_equal(p, q) for p, q in zip(h0, hd))
    # Everything but the deck stream identical to the plain clone's state right after reseed.
    a, b = plain.clone(), plain.clone()
    b.reseed(3, streams=["misc"])
    assert same_view(view(a), view(b))


def test_reseed_all_equals_listing_every_stream():
    g = Game(seed=13)
    advance(g, 25)
    a, b = g.clone(), g.clone()
    a.reseed(42)
    b.reseed(42, streams=STREAMS)
    assert a.export_state() == b.export_state()


def test_game_start_joker():
    g = Game(seed=1, start_joker="Greedy Joker")
    plain = Game(seed=1)
    assert g.observation()["joker_valid"][0] == 1
    assert plain.observation()["joker_valid"][0] == 0
    with pytest.raises(ValueError):
        Game(seed=1, start_joker="no such joker")


# --------------------------------------------------------------------------- VecEnv branching


def drive(env, steps):
    for _ in range(steps):
        env.step(env.random_actions())


def test_vec_export_import_round_trip_matches_source_and_game():
    src = BalatroVecEnv(4, seed=3, num_threads=3, max_episode_steps=5000)
    src.reset()
    for _ in range(45):
        src.step(src.teacher_actions())
    i = 2
    state = src.export_state(i)
    g = Game.from_state(state)
    dst = BalatroVecEnv(3, seed=99, num_threads=2, max_episode_steps=5000)
    dst.reset()
    dst.import_state(1, state)
    for k in src.obs:
        assert np.array_equal(src.obs[k][i], dst.obs[k][1]), k
    assert dst.reward[1] == 0 and dst.done[1] == 0 and dst.invalid[1] == 0
    n = 0
    while not g.is_terminal() and n < 400:
        a = g.teacher_parts()
        r, done, inv = g.step_parts(*a)
        sa = src.random_actions().copy()  # advances the sampler; rows overwritten below
        sa[i] = a
        src.step(sa)
        da = dst.random_actions().copy()
        da[1] = a
        dst.step(da)
        assert src.reward[i] == dst.reward[1] == pytest.approx(r)
        assert src.done[i] == dst.done[1] == done
        if done:
            break
        gv = view(g)
        for k in gv:
            full = {**dst.obs, **dst.masks}[k][1]
            assert np.array_equal(full, gv[k]), k
            assert np.array_equal({**src.obs, **src.masks}[k][i], gv[k]), k
        n += 1
    assert n > 50


def test_import_bookkeeping_and_finished_flags():
    src = BalatroVecEnv(1, seed=3, num_threads=1)
    src.reset()
    drive(src, 30)
    state = src.export_state(0)
    env = BalatroVecEnv(2, seed=5, num_threads=1, max_episode_steps=7, deck="red", stake="white")
    env.reset()
    env.import_state(0, state)
    while not env.done[0]:
        env.step(env.random_actions())
    fin = env.drain_finished_ex()
    mine = [f for f in fin if f["env"] == 0]
    assert len(mine) == 1
    f = mine[0]
    assert f["imported"] is True
    assert f["deck"] == "red" and f["stake"] == "white"
    assert f["steps"] <= 7 and f["code"] in (1, 2, 3)
    assert f["start_joker"] is None
    # After the auto-reset the next episode is a normal one.
    env.step(env.random_actions())
    while not env.done[0]:
        env.step(env.random_actions())
    f2 = [f for f in env.drain_finished_ex() if f["env"] == 0][-1]
    assert f2["imported"] is False
    assert set(f) >= {"env", "ante", "code", "steps", "return", "blinds_beaten", "invalid"}


def test_import_does_not_disturb_other_envs_or_future_seeds():
    """Env 1's trajectory is independent of an import into env 0; the imported env's next
    episode after the import ends is the one the replaced episode would have reached."""

    def run(do_import):
        env = BalatroVecEnv(2, seed=21, num_threads=2, max_episode_steps=4)
        env.reset()
        donor = BalatroVecEnv(1, seed=8, num_threads=1)
        donor.reset()
        drive(donor, 12)
        st = donor.export_state(0)
        log = []
        for t in range(24):
            if do_import and t == 2:
                env.import_state(0, st)
            a = env.random_actions().copy()
            env.step(a)
            log.append((env.obs["global_f"][1].copy(), env.reward[1], env.done[1]))
        return log

    a, b = run(False), run(True)
    for x, y in zip(a, b):
        assert np.array_equal(x[0], y[0]) and x[1:] == y[1:]


def test_import_across_thread_counts_is_identical():
    donor = BalatroVecEnv(1, seed=8, num_threads=1)
    donor.reset()
    drive(donor, 40)
    st = donor.export_state(0)

    def run(threads):
        env = BalatroVecEnv(9, seed=2, num_threads=threads, max_episode_steps=300)
        env.reset()
        drive(env, 5)
        env.import_state(7, st, reseed=11)
        env.import_state(3, st)
        out = []
        for _ in range(60):
            env.step(env.random_actions())
            out.append((env.obs["hand_f"].copy(), env.reward.copy(), env.done.copy()))
        return out

    a, b = run(1), run(5)
    for x, y in zip(a, b):
        assert all(np.array_equal(p, q) for p, q in zip(x, y))


def test_import_reseed_diverges_but_view_matches():
    env = BalatroVecEnv(1, seed=3, num_threads=1)
    env.reset()
    drive(env, 40)
    st = env.export_state(0)
    a = BalatroVecEnv(2, seed=0, num_threads=1)
    a.reset()
    a.import_state(0, st)
    a.import_state(1, st, reseed=5)
    for k in a.obs:
        assert np.array_equal(a.obs[k][0], a.obs[k][1]), k
    for k in a.masks:
        assert np.array_equal(a.masks[k][0], a.masks[k][1]), k
    differ = False
    for _ in range(200):
        row = env.random_actions()[0].copy()
        acts = np.stack([row, row])
        a.step(acts)
        if not np.array_equal(a.obs["hand_f"][0], a.obs["hand_f"][1]):
            differ = True
            break
    assert differ


def test_import_errors():
    env = BalatroVecEnv(2, seed=1, num_threads=1)
    env.reset()
    st = env.export_state(0)
    with pytest.raises(ValueError):
        env.import_state(5, st)
    with pytest.raises(ValueError):
        env.import_state(0, "{}")
    with pytest.raises(ValueError):
        env.export_state(9)


def test_start_joker_in_first_observation_and_after_auto_reset():
    env = BalatroVecEnv(3, seed=4, num_threads=2, max_episode_steps=3)
    env.reset()
    assert not env.obs["joker_valid"][:, 0].any()
    env.set_start_jokers([None, "Greedy Joker", "Blueprint"])
    # running episodes untouched
    assert not env.obs["joker_valid"][:, 0].any()
    env.reset()
    assert list(env.obs["joker_valid"][:, 0]) == [0, 1, 1]
    ids = env.obs["joker_i"][:, 0, 0]
    assert ids[1] != ids[2] and ids[1] != 0
    base = BalatroVecEnv(3, seed=4, num_threads=1, max_episode_steps=3)
    base.reset()
    # same seeds: env 0 identical to the baseline, the others differ only in the joker rows
    assert np.array_equal(base.obs["hand_f"], env.obs["hand_f"])
    saw_done = False
    for _ in range(12):
        env.step(env.random_actions())
        if env.done[1]:
            saw_done = True
            assert env.obs["joker_valid"][1, 0] == 1  # the next episode has it too
    assert saw_done
    fin = env.drain_finished_ex()
    assert any(f["start_joker"] == "Greedy Joker" for f in fin if f["env"] == 1)
    assert all(f["start_joker"] is None for f in fin if f["env"] == 0)
    env.set_start_jokers(None)
    env.reset()
    assert not env.obs["joker_valid"][:, 0].any()
    with pytest.raises(ValueError):
        env.set_start_jokers(["Greedy Joker"])
    with pytest.raises(ValueError):
        env.set_start_jokers("no such joker")


def test_default_unchanged_without_branching_calls():
    """No branching call -> the original trajectory (same as an env that never heard of it)."""
    a = BalatroVecEnv(2, seed=5, num_threads=1)
    b = BalatroVecEnv(2, seed=5, num_threads=1)
    a.reset()
    b.reset()
    b.set_start_jokers(None)
    for _ in range(40):
        acts = a.random_actions().copy()
        a.step(acts)
        b.step(acts)
        for k in a.obs:
            assert np.array_equal(a.obs[k], b.obs[k])
        assert np.array_equal(a.reward, b.reward)


def test_finished_ex_reports_deck_and_stake():
    env = BalatroVecEnv(
        6, seed=1, num_threads=2, max_episode_steps=2, decks=["red", "plasma"], stakes=["white", "gold"]
    )
    env.reset()
    drive(env, 30)
    fin = env.drain_finished_ex()
    assert fin and all(f["imported"] is False for f in fin)
    assert {(f["deck"], f["stake"]) for f in fin} <= {
        (d, s) for d in ("red", "plasma") for s in ("white", "gold")
    }
    assert len({(f["deck"], f["stake"]) for f in fin}) > 1
    assert env.drain_finished() == []  # shared queue


@pytest.mark.parametrize(
    "kw", [{}, {"deck": "plasma", "stake": "gold"}, {"deck": "ghost", "endless": True}]
)
def test_round_trip_and_reseed_in_every_phase(kw):
    """Export -> from_state is exact and reseed keeps the view at every step of whole runs
    (market, packs with open boosters, boss rounds, endless antes)."""
    phases = set()
    for seed in range(3):
        g = Game(seed=seed, **kw)
        for step in range(3000):
            if g.is_terminal():
                break
            s = g.export_state()
            if step % 7 == 0:
                phases.add(phase(g))
                assert Game.from_state(s).export_state() == s
                c = Game.from_state(s, reseed=step)
                assert same_view(view(g), view(c)), (seed, step, phase(g))
            g.step_parts(*g.teacher_parts())
    assert {"Play", "Shop", "BlindSelect"} <= phases


def test_start_joker_label_is_per_episode_and_import_rejects_endless_mismatch():
    env = BalatroVecEnv(1, seed=4, num_threads=1, max_episode_steps=6)
    env.reset()
    env.set_start_jokers("Greedy Joker")  # no reset: the running episode has no joker
    while not env.done[0]:
        env.step(env.random_actions())
    first = env.drain_finished_ex()[0]
    assert first["start_joker"] is None
    env.step(env.random_actions())
    while not env.done[0]:
        env.step(env.random_actions())
    assert env.drain_finished_ex()[0]["start_joker"] == "Greedy Joker"
    # an endless state cannot enter a non-endless env, and the env stays intact on failure
    st = Game(seed=1, endless=True).export_state()
    before = env.export_state(0)
    with pytest.raises(ValueError):
        env.import_state(0, st)
    assert env.export_state(0) == before
