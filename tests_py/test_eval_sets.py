"""Named evaluation sets (dev / holdout / val) and the memory knockout (package A1)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from balatro_rl import BalatroVecEnv  # noqa: E402
from balatro_rl.model import ModelConfig, TeamModel  # noqa: E402
from balatro_rl.train import evaluate as ev  # noqa: E402

TINY = {
    "d_model": 32,
    "layers": 1,
    "heads": 2,
    "ff_mult": 4,
    "dropout": 0.0,
    "memory": "gru",
    "mem_dim": 32,
}

M = (1 << 64) - 1


def _splitmix(z):
    z = (z + 0x9E3779B97F4A7C15) & M
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M
    return z ^ (z >> 31)


def _derive(base, env, ep):
    s = _splitmix(base)
    s = _splitmix(s ^ ((env * 0xD6E8FEB86659FD93) & M))
    return _splitmix(s ^ ((ep * 0xA0761D6478BD642F) & M))


def _train_bases():
    return [ev.train_seed(s, r) for s in range(10) for r in range(10)]


def test_set_definitions():
    assert ev.EVAL_SETS["dev"] == (10_000_019, 1000)
    assert ev.EVAL_SETS["holdout"][1] == 10_000 and ev.EVAL_SETS["val"][1] == 512
    seeds = [v[0] for v in ev.EVAL_SETS.values()] + _train_bases()
    assert len(set(seeds)) == len(seeds)


def test_game_seeds_disjoint_pure_hash():
    seen = {}
    for name, (base, games) in ev.EVAL_SETS.items():
        for i in range(games):
            seen.setdefault(_derive(base, i, 0), []).append(name)
    for b in _train_bases():
        for i in range(64):
            for ep in range(3):
                seen.setdefault(_derive(b, i, ep), []).append("train")
    assert all(len(v) == 1 for v in seen.values())


def test_game_initial_states_disjoint_on_engine():
    """The real engine: first observation of every game of every set / training base is unique."""

    def keys(base, n):
        env = BalatroVecEnv(n, seed=base, num_threads=1)
        obs, _ = env.reset()
        names = sorted(obs)
        snaps = []
        for _ in range(4):  # the reset state is seed-independent; a few teacher steps deal cards
            obs, _, _, _, _ = env.step(env.teacher_actions(np.ones(n, dtype=bool)))
            snaps.append(obs)
        return [
            b"".join(np.ascontiguousarray(o[k][i]).tobytes() for o in snaps for k in names)
            for i in range(n)
        ]

    allk = []
    for base, games in ev.EVAL_SETS.values():
        allk += keys(base, games)
    for b in _train_bases():
        allk += keys(b, 32)
    assert len(set(allk)) == len(allk)


def test_memory_modes_and_normal_is_default():
    model = TeamModel(ModelConfig(**TINY))
    kw = {"max_episode_steps": 150, "threads": 1}
    a = ev.evaluate_model(model, 8, True, **kw)
    b = ev.evaluate_model(model, 8, True, memory="normal", **kw)
    assert a.antes == b.antes and a.wins == b.wins and a.mean_steps == b.mean_steps
    for mode in ("zero", "shuffle"):
        r = ev.evaluate_model(model, 8, True, memory=mode, **kw)
        assert r.games == 8 and r.invalid_per_game == 0
        again = ev.evaluate_model(model, 8, True, memory=mode, **kw)
        assert again.antes == r.antes  # deterministic
    with pytest.raises(AssertionError):
        ev.ModelActor(model, True, memory="bogus")


def test_holdout_prefix_matches_named_seed():
    h = ev.run_games("r", ev.random_actor, 8, seed=ev.HOLDOUT_SEED, max_episode_steps=200)
    h2 = ev.run_games("r", ev.random_actor, 4, seed=ev.EVAL_SETS["holdout"][0], max_episode_steps=200)
    assert h2.antes == h.antes[:4]
