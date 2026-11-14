import threading
import time

import numpy as np
import pytest
from balatro_rl import LAYOUT, BalatroVecEnv, Game, _engine
from balatro_rl.env import allocate_buffers

OBS = LAYOUT["obs_names"]
MASKS = LAYOUT["mask_names"]


def snap(env):
    """Everything observable after a call, copied."""
    return {
        **{k: v.copy() for k, v in env.obs.items()},
        **{k: v.copy() for k, v in env.masks.items()},
        "reward": env.reward.copy(),
        "done": env.done.copy(),
        "invalid": env.invalid.copy(),
    }


def rollout(num_envs, seed, threads, steps, max_episode_steps=300, teacher=False):
    """Random (or teacher) rollout; returns every observable, stacked over time."""
    env = BalatroVecEnv(
        num_envs, seed=seed, num_threads=threads, max_episode_steps=max_episode_steps
    )
    env.reset()
    log = [snap(env)]
    actions = []
    for _ in range(steps):
        a = (env.teacher_actions() if teacher else env.random_actions()).copy()
        actions.append(a)
        env.step(a)
        log.append(snap(env))
    stacked = {k: np.stack([s[k] for s in log]) for k in log[0]}
    stacked["actions"] = np.stack(actions)
    return stacked, env.drain_finished()


def assert_same(a, b):
    assert a[1] == b[1]
    assert a[0].keys() == b[0].keys()
    for k in a[0]:
        assert np.array_equal(a[0][k], b[0][k]), k


def test_shapes_and_dtypes_follow_the_layout():
    env = BalatroVecEnv(8, seed=1, num_threads=2)
    obs, masks = env.reset()
    assert set(obs) == set(OBS) and set(masks) == set(MASKS)
    spec = {name: (np.dtype(dt), shape) for name, dt, shape in LAYOUT["buffers"]}
    for name, arr in {**obs, **masks}.items():
        dt, shape = spec[name]
        assert arr.dtype == dt and arr.shape == (8, *shape), name
    assert env.reward.dtype == np.float32 and env.reward.shape == (8,)
    assert env.done.dtype == np.uint8 and env.invalid.dtype == np.uint8
    # Fresh run: BlindSelect phase, ante 1, only SelectBlind/SkipBlind.
    names = LAYOUT["feature_names"]
    assert (obs["global_i"][:, names["global_i"].index("phase")] == 0).all()
    assert (obs["global_f"][:, names["global_f"].index("ante")] == 1 / 8).all()
    expected = np.zeros(17, bool)
    expected[[0, 1]] = True
    assert (masks["kind_mask"] == expected).all()
    out = env.step(env.random_actions())
    assert len(out) == 5
    assert all(np.isfinite(v).all() for v in env.obs.values() if v.dtype == np.float32)


def test_raw_engine_buffer_errors():
    env = _engine.VecEnv(2, 0, 1)
    bufs = allocate_buffers(2)
    env.reset(bufs)
    bad = dict(bufs)
    del bad["hand_f"]
    with pytest.raises(KeyError):
        env.reset(bad)
    bad = dict(bufs, hand_f=bufs["hand_f"].astype(np.float64))  # dtype is not converted silently
    with pytest.raises(ValueError, match="dtype"):
        env.reset(bad)
    bad = dict(bufs, hand_f=np.zeros((3, *bufs["hand_f"].shape[1:]), np.float32))
    with pytest.raises(ValueError, match="shape"):
        env.reset(bad)
    swapped = np.zeros(bufs["global_f"].shape[::-1], np.float32).T  # Fortran order
    with pytest.raises(ValueError, match="C-contiguous"):
        env.reset(dict(bufs, global_f=swapped))
    with pytest.raises(ValueError, match="shape"):
        env.step(np.zeros((2, 3), np.int64), bufs)
    with pytest.raises(TypeError):
        env.step(np.zeros((2, 4), np.int32), bufs)
    with pytest.raises(ValueError):
        _engine.VecEnv(0)
    with pytest.raises(ValueError, match="unknown reward term"):
        _engine.VecEnv(1, reward={"nope": 1.0})


@pytest.mark.parametrize("threads", [2, 3, 12])
@pytest.mark.parametrize("teacher", [False, True])
def test_determinism_independent_of_thread_count(threads, teacher):
    steps = 40 if teacher else 400
    a = rollout(24, seed=123, threads=1, steps=steps, teacher=teacher)
    b = rollout(24, seed=123, threads=threads, steps=steps, teacher=teacher)
    assert_same(a, b)
    if not teacher:
        assert (a[0]["done"] != 0).any()  # the rollout actually finished episodes


def test_same_seed_repeats_and_different_seed_differs():
    a = rollout(16, seed=5, threads=2, steps=200)
    b = rollout(16, seed=5, threads=2, steps=200)
    c = rollout(16, seed=6, threads=2, steps=200)
    assert_same(a, b)
    assert not np.array_equal(a[0]["global_f"], c[0]["global_f"])


def test_envs_differ_from_each_other():
    env = BalatroVecEnv(32, seed=9, num_threads=1)
    env.reset()
    for _ in range(40):
        env.step(env.random_actions())
    assert len({tuple(row) for row in env.obs["global_f"]}) > 1


def test_reset_restarts_deterministically():
    env = BalatroVecEnv(8, seed=3, num_threads=2)

    def run():
        env.reset()
        first = snap(env)
        for _ in range(100):
            env.step(env.random_actions())
        return first, snap(env)

    f1, l1 = run()
    f2, l2 = run()
    for k in f1:
        assert np.array_equal(f1[k], f2[k]) and np.array_equal(l1[k], l2[k]), k
    env.reset(seed=4)
    for _ in range(100):
        env.step(env.random_actions())
    assert not np.array_equal(env.obs["global_f"], l1["global_f"])


def test_auto_reset_and_episode_stats():
    env = BalatroVecEnv(8, seed=2, num_threads=2, max_episode_steps=2000)
    env.reset()
    saw_done = False
    for _ in range(3000):
        _, masks, _, done, _ = env.step(env.random_actions())
        assert masks["kind_mask"].any(axis=1).all(), "auto-reset must leave a playable state"
        saw_done |= bool(done.any())
        if saw_done and len(env.drain_finished()) >= 8:
            break
    assert saw_done
    env.step(env.random_actions())


def test_done_obs_is_next_episode_and_stats_match():
    env = BalatroVecEnv(4, seed=11, num_threads=1, max_episode_steps=2000)
    env.reset()
    names = LAYOUT["feature_names"]
    ante_col = names["global_f"].index("ante")
    phase_col = names["global_i"].index("phase")
    found = None
    for _ in range(5000):
        obs, _, _, done, _ = env.step(env.random_actions())
        idx = np.flatnonzero(done)
        if idx.size:
            i = idx[0]
            found = (i, obs["global_f"][i, ante_col], obs["global_i"][i, phase_col], int(done[i]))
            break
    assert found is not None
    i, ante, phase, code = found
    assert code in (1, 2)
    assert ante == 1 / 8 and phase == 0  # a new episode: BlindSelect, ante 1
    stats = env.drain_finished()
    assert stats and stats[0][0] == i and stats[0][2] == code and stats[0][3] > 0
    assert len(stats[0]) == 7 and stats[0][6] == 0  # no invalid actions in this rollout


def test_truncation():
    env = BalatroVecEnv(4, seed=1, num_threads=1, max_episode_steps=3)
    env.reset()
    for _ in range(2):
        env.step(env.random_actions())
        assert (env.done == 0).all()
    env.step(env.random_actions())
    assert (env.done == _engine.DONE_TRUNCATED).all()
    assert all(f[2] == _engine.DONE_TRUNCATED and f[3] == 3 for f in env.drain_finished())


def test_invalid_actions_are_noops_with_a_flag_and_penalty():
    env = BalatroVecEnv(6, seed=1, num_threads=2, reward={"invalid": -0.5})
    env.reset()
    before = snap(env)
    acts = env.random_actions().copy()
    # Fresh run (BlindSelect): Play is illegal, and so are malformed kinds/operands/masks.
    acts[0] = [2, 0, 0, 1]
    acts[1] = [99, 0, 0, 0]
    acts[2] = [-1, 0, 0, 0]
    acts[3] = [4, 256, 0, 0]
    acts[4] = [2, 0, 0, 1 << 40]
    acts[5] = [4, 0, 0, 0]  # SellJoker with no jokers
    env.step(acts)
    assert (env.invalid == 1).all()
    assert np.allclose(env.reward, -0.5) and (env.done == 0).all()
    for k in OBS + MASKS:  # state untouched
        assert np.array_equal(env.obs.get(k, env.masks.get(k)), before[k]), k
    # The steps still count: invalid actions end up in the episode stats on truncation.
    env2 = BalatroVecEnv(2, seed=1, num_threads=1, max_episode_steps=3)
    env2.reset()
    bad = np.tile(np.array([[2, 0, 0, 1]], dtype=np.int64), (2, 1))
    for _ in range(3):
        env2.step(bad)
    assert (env2.done == _engine.DONE_TRUNCATED).all()
    assert [f[6] for f in env2.drain_finished()] == [3, 3]
    # A valid step afterwards behaves like a fresh env's first step.
    ref = BalatroVecEnv(6, seed=1, num_threads=1)
    ref.reset()
    ok = np.zeros((6, 4), dtype=np.int64)
    env.step(ok)
    ref.step(ok)
    assert np.array_equal(env.obs["global_f"], ref.obs["global_f"])
    assert (env.invalid == 0).all()


def test_wrong_action_shape_raises():
    env = BalatroVecEnv(4, seed=1, num_threads=1)
    env.reset()
    with pytest.raises(ValueError):
        env.step(np.zeros((3, 4), dtype=np.int64))
    with pytest.raises((ValueError, TypeError)):  # wrong rank
        env.step(np.zeros(4, dtype=np.int64))


def test_set_num_threads_does_not_change_results():
    env = BalatroVecEnv(20, seed=8, num_threads=1)
    ref = BalatroVecEnv(20, seed=8, num_threads=1)
    env.reset()
    ref.reset()
    for k in range(60):
        if k % 15 == 0:
            env.set_num_threads([1, 4, 12, 2][k // 15])
        env.step(env.random_actions())
        ref.step(ref.random_actions())
        assert np.array_equal(env.obs["global_f"], ref.obs["global_f"])
        assert np.array_equal(env.reward, ref.reward)


def test_reward_terms_are_configurable():
    zero = dict(blind=0.0, ante=0.0, win=0.0, loss=0.0, shaping=0.0, invalid=0.0)

    def returns(**terms):
        env = BalatroVecEnv(
            64, seed=21, num_threads=2, max_episode_steps=2000, reward={**zero, **terms}
        )
        env.reset()
        for _ in range(1500):
            env.step(env.random_actions())
        return env.drain_finished()

    # Default-like blind reward: the return of an episode is the number of blinds beaten.
    stats = returns(blind=1.0)
    assert stats
    for _, _, _, _, ret, won, _ in stats:
        assert ret == pytest.approx(won)
    # Loss penalty: -1 per lost episode, nothing else.
    for _, _, code, _, ret, _, _ in returns(loss=-1.0):
        assert ret == pytest.approx(-1.0 if code == _engine.DONE_LOSS else 0.0)
    # Win bonus.
    for _, _, code, _, ret, _, _ in returns(win=10.0, loss=0.0):
        assert ret == pytest.approx(10.0 if code == _engine.DONE_WIN else 0.0)
    # Shaping: telescopes to at most 1 per round started (rounds won + the last round).
    for _, _, _, _, ret, won, _ in returns(shaping=1.0):
        assert -1e-4 <= ret <= won + 1 + 1e-4


def test_default_reward_values():
    env = BalatroVecEnv(1, seed=0, num_threads=1)
    env.reset()
    # Mask-violating action: the default invalid penalty.
    env.step(np.array([[2, 0, 0, 1]], dtype=np.int64))
    assert env.reward[0] == pytest.approx(-0.05) and env.invalid[0] == 1
    # Everything else is 0 for a SelectBlind.
    env.step(np.array([[0, 0, 0, 0]], dtype=np.int64))
    assert env.reward[0] == 0.0 and env.invalid[0] == 0


def test_game_debug_object():
    g = Game(seed=7)
    assert "BlindSelect" in g.state_json()
    acts = g.legal_actions()
    assert acts and all(isinstance(a, str) for a in acts)
    assert g.legal_count() == len(acts) == len(g.legal_parts())
    obs, masks = g.observation(), g.masks()
    assert set(obs) == set(OBS) and set(masks) == set(MASKS)
    spec = {name: shape for name, _, shape in LAYOUT["buffers"]}
    assert all(v.shape == spec[k] for k, v in {**obs, **masks}.items())
    assert not g.is_terminal() and g.outcome() is None
    reward, done, invalid = g.step_parts(0, 0, 0, 0)
    assert (done, invalid, reward) == (0, False, 0.0)
    reward, done, invalid = g.step_parts(99, 0, 0, 0)  # malformed: no-op
    assert invalid and reward == pytest.approx(-0.05)
    for _ in range(100_000):
        if g.is_terminal():
            break
        g.step_parts(*g.random_parts())
    else:
        pytest.skip("random policy did not terminate")
    assert g.outcome() in ("win", "loss")
    assert not g.masks()["kind_mask"].any()
    with pytest.raises(ValueError):
        g.step_parts(0, 0, 0, 0)


def test_game_matches_vec_env_row():
    for seed in (0, 3):
        g = Game(seed=seed)
        env = BalatroVecEnv(1, seed=seed, num_threads=1)
        env.reset()
        for k, v in {**g.observation(), **g.masks()}.items():
            assert np.array_equal(v, {**env.obs, **env.masks}[k][0]), k
        # Same actions, same trajectory (VecEnv env 0 is episode 0 of Game(seed)).
        for _ in range(60):
            a = env.random_actions()[0].copy()
            g.step_parts(*[int(x) for x in a])
            env.step(a[None, :])
            if env.done[0]:
                break
            for k, v in g.observation().items():
                assert np.array_equal(v, env.obs[k][0]), k


def test_gil_is_released_during_step():
    # A background Python thread counts ticks while the main thread sits inside one long
    # `step`. If `step` held the GIL the ticker could not run at all during the call.
    env = BalatroVecEnv(16384, seed=0, num_threads=1)
    env.reset()
    acts = env.random_actions().copy()
    ticks = [0]
    stop = threading.Event()

    def ticker():
        while not stop.is_set():
            ticks[0] += 1

    th = threading.Thread(target=ticker)
    th.start()
    time.sleep(0.05)  # let the ticker get going
    before = ticks[0]
    t0 = time.perf_counter()
    env.step(acts)
    elapsed = time.perf_counter() - t0
    during = ticks[0] - before
    stop.set()
    th.join()
    assert elapsed > 0.01, "step too fast to measure; raise num_envs"
    assert during > 1000, f"ticker made only {during} ticks during a {elapsed * 1e3:.0f} ms step"
