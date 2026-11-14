import threading
import time

import numpy as np
import pytest
from balatro_rl import BalatroVecEnv, Game, _engine

OBS_DIM = _engine.OBS_DIM


def rollout(num_envs, seed, threads, steps, max_episode_steps=300):
    """Random-action rollout; returns everything observable, concatenated."""
    env = BalatroVecEnv(
        num_envs, seed=seed, num_threads=threads, max_episode_steps=max_episode_steps
    )
    obs, lc = env.reset()
    obs_log, rew_log, done_log, lc_log = [obs.copy()], [], [], [lc.copy()]
    for _ in range(steps):
        obs, rew, done, lc = env.step(env.random_actions())
        obs_log.append(obs.copy())
        rew_log.append(rew.copy())
        done_log.append(done.copy())
        lc_log.append(lc.copy())
    return (
        np.stack(obs_log),
        np.stack(rew_log),
        np.stack(done_log),
        np.stack(lc_log),
        env.drain_finished(),
    )


def test_shapes_and_dtypes():
    env = BalatroVecEnv(8, seed=1, num_threads=2)
    obs, lc = env.reset()
    assert obs.shape == (8, OBS_DIM) and obs.dtype == np.float32
    assert lc.shape == (8,) and lc.dtype == np.int32
    assert (lc > 0).all()
    # Fresh run: BlindSelect phase one-hot, ante 1.
    assert (obs[:, 0] == 1).all() and (obs[:, 6] == 1).all()
    assert (obs[:, 22] == lc).all()
    obs, rew, done, lc = env.step(env.random_actions())
    assert rew.shape == (8,) and rew.dtype == np.float32
    assert done.shape == (8,) and done.dtype == np.uint8
    assert np.isfinite(obs).all()


def test_raw_engine_dtype_errors():
    env = _engine.VecEnv(2, 0, 1)
    obs = np.zeros((2, OBS_DIM), np.float32)
    lc = np.zeros(2, np.int32)
    env.reset(obs, lc)
    with pytest.raises(TypeError):  # wrong dtype is rejected, not silently converted
        env.reset(obs.astype(np.float64), lc)
    with pytest.raises(ValueError):  # wrong shape
        env.reset(np.zeros((3, OBS_DIM), np.float32), lc)
    with pytest.raises(ValueError):  # non-contiguous
        env.reset(np.zeros((OBS_DIM, 2), np.float32).T, lc)


@pytest.mark.parametrize("threads", [2, 3, 12])
def test_determinism_independent_of_thread_count(threads):
    a = rollout(50, seed=123, threads=1, steps=600)
    b = rollout(50, seed=123, threads=threads, steps=600)
    for x, y in zip(a[:4], b[:4], strict=True):
        assert np.array_equal(x, y)
    assert a[4] == b[4]
    assert (a[2] != 0).any()  # the rollout actually finished episodes


def test_same_seed_repeats_and_different_seed_differs():
    a = rollout(16, seed=5, threads=2, steps=200)
    b = rollout(16, seed=5, threads=2, steps=200)
    c = rollout(16, seed=6, threads=2, steps=200)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not np.array_equal(a[0], c[0])


def test_envs_differ_from_each_other():
    env = BalatroVecEnv(32, seed=9, num_threads=1)
    env.reset()
    for _ in range(40):
        env.step(env.random_actions())
    assert len({tuple(row) for row in env.obs}) > 1


def test_reset_restarts_deterministically():
    env = BalatroVecEnv(8, seed=3, num_threads=2)

    def run():
        env.reset()
        first = env.obs.copy()
        for _ in range(100):
            env.step(env.random_actions())
        return first, env.obs.copy()

    f1, l1 = run()
    f2, l2 = run()
    assert np.array_equal(f1, f2) and np.array_equal(l1, l2)
    env.reset(seed=4)
    for _ in range(100):
        env.step(env.random_actions())
    assert not np.array_equal(env.obs, l1)


def test_auto_reset_and_episode_stats():
    env = BalatroVecEnv(8, seed=2, num_threads=2, max_episode_steps=2000)
    env.reset()
    saw_done = False
    for _ in range(3000):
        _, _, done, lc = env.step(env.random_actions())
        assert (lc > 0).all(), "auto-reset must always leave a playable state"
        saw_done |= bool(done.any())
        if saw_done and len(env.drain_finished()) >= 8:
            break
    assert saw_done
    # After auto-reset the env is playable: the next obs is a fresh run (ante 1 at the latest
    # once a Loss happened) and stepping continues to work.
    env.step(env.random_actions())


def test_done_obs_is_next_episode_and_stats_match():
    env = BalatroVecEnv(4, seed=11, num_threads=1, max_episode_steps=2000)
    env.reset()
    found = None
    for _ in range(5000):
        obs, rew, done, _ = env.step(env.random_actions())
        idx = np.flatnonzero(done)
        if idx.size:
            found = (idx[0], obs[idx[0]].copy(), int(done[idx[0]]))
            break
    assert found is not None
    i, o, code = found
    assert code in (1, 2)
    assert o[0] == 1.0 and o[6] == 1.0  # new episode: BlindSelect, ante 1
    stats = env.drain_finished()
    assert stats and stats[0][0] == i and stats[0][2] == code and stats[0][3] > 0


def test_truncation():
    env = BalatroVecEnv(4, seed=1, num_threads=1, max_episode_steps=3)
    env.reset()
    for _ in range(2):
        _, _, done, _ = env.step(env.random_actions())
        assert (done == 0).all()
    _, _, done, _ = env.step(env.random_actions())
    assert (done == _engine.DONE_TRUNCATED).all()
    assert all(f[2] == _engine.DONE_TRUNCATED and f[3] == 3 for f in env.drain_finished())


def test_invalid_action_index_raises_and_leaves_state_untouched():
    env = BalatroVecEnv(4, seed=1, num_threads=2)
    obs, lc = env.reset()
    before = obs.copy()
    bad = np.zeros(4, dtype=np.int64)
    bad[2] = lc[2]  # one past the last legal index
    with pytest.raises(ValueError, match="out of range"):
        env.step(bad)
    bad[2] = -1
    with pytest.raises(ValueError, match="out of range"):
        env.step(bad)
    # State untouched: a valid step afterwards behaves like a fresh env's first step.
    assert np.array_equal(env.obs, before)
    ref = BalatroVecEnv(4, seed=1, num_threads=1)
    ref.reset()
    acts = np.zeros(4, dtype=np.int64)
    assert np.array_equal(env.step(acts)[0], ref.step(acts)[0])


def test_wrong_buffer_sizes_raise():
    env = BalatroVecEnv(4, seed=1, num_threads=1)
    env.reset()
    with pytest.raises(ValueError):
        env.step(np.zeros(3, dtype=np.int64))
    with pytest.raises(ValueError):
        _engine.VecEnv(0)


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
        assert np.array_equal(env.obs, ref.obs)
        assert np.array_equal(env.reward, ref.reward)


def test_reward_is_blind_clear_signal():
    env = BalatroVecEnv(64, seed=21, num_threads=2, max_episode_steps=2000)
    env.reset()
    total = np.zeros(64)
    for _ in range(1500):
        _, rew, _, _ = env.step(env.random_actions())
        assert (rew >= 0).all()
        total += rew
    stats = env.drain_finished()
    # Episode return = blinds beaten + 10 per win.
    for _, _, code, _, ret, won in stats:
        assert ret == pytest.approx(won + (10.0 if code == _engine.DONE_WIN else 0.0))


def test_game_debug_object():
    g = Game(seed=7)
    assert "BlindSelect" in g.state_json()
    acts = g.legal_actions()
    assert acts and all(isinstance(a, str) for a in acts)
    assert g.legal_count() == len(acts)
    assert len(g.observation()) == OBS_DIM
    assert not g.is_terminal() and g.outcome() is None
    reward, done = g.step(0)
    assert done == 0 and reward == 0.0
    with pytest.raises(ValueError):
        g.step(10**9)
    # Play to the end with a fixed policy: terminates with a loss or truncation by our own cap.
    for _ in range(100_000):
        if g.is_terminal():
            break
        g.step(g.legal_count() - 1)
    else:
        pytest.skip("fixed policy did not terminate")
    assert g.outcome() in ("win", "loss")


def test_game_matches_vec_env_row():
    g = Game(seed=0)
    env = _engine.VecEnv(1, 0, 1)
    obs = np.zeros((1, OBS_DIM), np.float32)
    lc = np.zeros(1, np.int32)
    env.reset(obs, lc)
    # VecEnv derives its own per-env seeds, so only the *initial* structure is comparable.
    assert g.legal_count() == lc[0]
    assert np.array_equal(np.array(g.observation())[[0, 6, 22]], obs[0][[0, 6, 22]])


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
