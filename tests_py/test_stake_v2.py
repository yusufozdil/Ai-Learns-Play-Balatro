"""Stake ladder v2 (package B2): per-stake advantage statistics / normalisation, the stake
curriculum weights, scenario resets, boss-hall pool resets."""

import json

import numpy as np
import pytest
import torch
from balatro_rl import BalatroVecEnv
from balatro_rl.train import ppo
from balatro_rl.train.ppo import (
    DECK_COL,
    DECK_NAMES,
    STAKE_COL,
    STAKE_NAMES,
    PPOConfig,
    Trainer,
    group_stats,
    normalize_adv_per_group,
)

# ---------------------------------------------------------------- index <-> name mapping


def test_deck_stake_columns_match_names():
    for stake in STAKE_NAMES:
        env = BalatroVecEnv(2, seed=1, num_threads=1, stake=stake)
        obs, _ = env.reset()
        assert (obs["global_i"][:, STAKE_COL] == STAKE_NAMES.index(stake)).all()
    for deck in DECK_NAMES:
        env = BalatroVecEnv(2, seed=1, num_threads=1, deck=deck)
        obs, _ = env.reset()
        assert (obs["global_i"][:, DECK_COL] == DECK_NAMES.index(deck)).all()


# ---------------------------------------------------------------- advantage statistics


def test_group_stats_and_per_config_normalisation():
    g = torch.Generator().manual_seed(0)
    adv = torch.randn(16, 40, generator=g)
    group = torch.randint(0, 3, (16, 40), generator=g, dtype=torch.long)
    ctrl = torch.rand(16, 40, generator=g) > 0.2
    adv[group == 1] *= 10.0  # a high-variance stake
    adv[group == 2] += 5.0
    cnt, mean, std = group_stats(adv, ctrl, group, 4)
    for k in range(3):
        sel = adv[ctrl & (group == k)]
        assert int(cnt[k]) == sel.numel()
        assert mean[k].item() == pytest.approx(sel.mean().item(), rel=1e-5)
        assert std[k].item() == pytest.approx(sel.std().item(), rel=1e-5)
    assert cnt[3] == 0 and mean[3] == 0 and std[3] == 0
    out = normalize_adv_per_group(adv, ctrl, group, 4, min_rows=10)
    for k in range(3):
        sel = out[ctrl & (group == k)]
        assert sel.mean().abs().item() < 1e-4
        assert sel.std().item() == pytest.approx(1.0, abs=1e-3)
    # groups below min_rows fall back to the global statistics
    a = adv[ctrl]
    glob = (adv - a.mean()) / (a.std() + 1e-8)
    out = normalize_adv_per_group(adv, ctrl, group, 4, min_rows=10_000)
    assert torch.allclose(out, glob, atol=1e-5)
    rare = group.clone()
    rare[0, :5] = 3  # at most 5 rows in group 3
    out = normalize_adv_per_group(adv, ctrl, rare, 4, min_rows=50)
    assert torch.allclose(out[rare == 3], glob[rare == 3], atol=1e-5)


# ---------------------------------------------------------------- config weights


def _stakes_of(env, steps=400):
    seen = []
    for _ in range(steps):
        env.step(env.random_actions())
        seen += [f["stake"] for f in env.drain_finished_ex()]
    return seen


def test_config_weights():
    env = BalatroVecEnv(64, seed=3, num_threads=2, max_episode_steps=15, stakes=["white", "black"])
    env.reset()
    pool = env.config_pool()
    assert pool == [("red", "white"), ("red", "black")]
    assert len(set(_stakes_of(env, 40))) == 2
    env.set_config_weights([0.0, 1.0])
    _stakes_of(env, 20)  # episodes that started before the call finish
    assert set(_stakes_of(env, 60)) == {"black"}
    env.set_config_weights([1.0, 3.0])
    seen = _stakes_of(env, 300)
    frac = np.mean([s == "black" for s in seen])
    assert 0.65 < frac < 0.85, frac
    env.set_config_weights(None)
    frac = np.mean([s == "black" for s in _stakes_of(env, 300)])
    assert 0.4 < frac < 0.6, frac
    with pytest.raises(ValueError):
        env.set_config_weights([1.0])
    with pytest.raises(ValueError):
        env.set_config_weights([0.0, 0.0])
    with pytest.raises(ValueError):
        BalatroVecEnv(2, seed=0).set_config_weights([1.0])


def test_weights_deterministic_across_threads():
    def run(threads):
        env = BalatroVecEnv(
            32, seed=9, num_threads=threads, max_episode_steps=12, decks=["red", "blue"],
            stakes=["white", "green", "black"],
        )  # fmt: skip
        env.reset()
        env.set_config_weights([1, 2, 3, 4, 5, 6])
        out = []
        for _ in range(60):
            env.step(env.random_actions())
            out += [(f["env"], f["deck"], f["stake"]) for f in env.drain_finished_ex()]
        return out

    assert run(1) == run(4)


# ---------------------------------------------------------------- scenario resets


def _run_scenarios(threads, prob=1.0, n=24, steps=80):
    env = BalatroVecEnv(
        n, seed=5, num_threads=threads, max_episode_steps=20, stakes=["white", "black"]
    )
    env.set_scenario_prob(prob)
    obs, _ = env.reset()
    fin = []
    first_obs = []
    for _ in range(steps):
        _, _, _, done, _ = env.step(env.random_actions())
        recs = env.drain_finished_ex()
        fin += recs
        for i in np.nonzero(done)[0]:
            first_obs.append((int(i), obs["global_f"][i].copy(), int(obs["global_i"][i, 0])))
    return fin, first_obs, env


def test_scenario_resets_flagged_deterministic_and_at_blind_select():
    fin, first_obs, env = _run_scenarios(2)
    assert fin and not fin[0]["scenario"]  # episode 0 is natural
    later = [f for f in fin if f["env"] == fin[0]["env"]][1:]
    assert later and all(f["scenario"] for f in later)
    assert all(not f["hall"] and not f["imported"] for f in fin)
    assert (env.origins() == 1).all()
    # every scenario episode starts at a Blind select (phase 0) in a mid-run state
    assert all(ph == 0 for _, _, ph in first_obs)
    # pure function of (seed, env, episode): independent of the thread count
    fin2, first2, _ = _run_scenarios(1)
    assert [(f["env"], f["ante"], f["steps"], f["scenario"]) for f in fin] == [
        (f["env"], f["ante"], f["steps"], f["scenario"]) for f in fin2
    ]
    assert all((a[1] == b[1]).all() for a, b in zip(first_obs, first2, strict=True))
    assert BalatroVecEnv(4, seed=1).origins().sum() == 0


def test_scenario_prob_zero_is_the_old_reset():
    def run(set_zero):
        env = BalatroVecEnv(16, seed=7, num_threads=2, max_episode_steps=15)
        if set_zero:
            env.set_scenario_prob(0.0)
            env.set_reset_pool([], 0.5)
        env.reset()
        out = []
        for _ in range(40):
            o, _, r, d, _ = env.step(env.random_actions())
            out.append((r.copy(), d.copy(), o["global_f"].copy()))
        return out

    a, b = run(False), run(True)
    for x, y in zip(a, b, strict=True):
        assert all((u == v).all() for u, v in zip(x, y, strict=True))


# ---------------------------------------------------------------- boss hall


def _hall_env(n=48, seed=11, min_ante=2):
    env = BalatroVecEnv(n, seed=seed, num_threads=2, max_episode_steps=4000)
    obs, masks = env.reset()
    cands = []
    for _ in range(2500):
        env.step(env.teacher_actions())
        cands += env.hall_collect(min_ante)
        if len(cands) >= 3:
            break
    assert cands, "the teacher never reached an ante-2 boss select"
    return env, cands


def test_boss_hall_collect_and_reset_pool():
    env, cands = _hall_env()
    for _i, ante, boss, _stake, js in cands:
        st = json.loads(js)
        assert st["phase"] == "BlindSelect" and st["blind"] == "Boss" and st["ante"] == ante >= 2
        assert boss
    src = [json.loads(js) for *_, js in cands]
    env.set_reset_pool([c[-1] for c in cands], 1.0)
    assert env.reset_pool_len() == len(cands)
    # run until some env restarts from the pool
    hall_envs = set()
    for _ in range(3000):
        env.step(env.teacher_actions())
        for f in env.drain_finished_ex():
            assert not f["scenario"]
        org = env.origins()
        if (org == 2).any():
            hall_envs = set(np.nonzero(org == 2)[0].tolist())
            break
    assert hall_envs
    i = sorted(hall_envs)[0]
    st = json.loads(env.export_state(i))
    assert st["phase"] == "BlindSelect" and st["blind"] == "Boss" and st["ante"] >= 2
    # same visible state as one of the pool sources, a different hidden future
    match = [
        s
        for s in src
        if s["ante"] == st["ante"] and s["boss"] == st["boss"] and s["jokers"] == st["jokers"]
    ]
    assert match, "the restarted env does not equal any pool state visibly"
    assert all(s["rng"] != st["rng"] for s in match), "hidden future (RNG) not re-sampled"
    # restarted envs differ from each other even when they draw the same pool state
    states = {json.dumps(json.loads(env.export_state(j))["rng"]) for j in hall_envs}
    assert len(states) == len(hall_envs)
    # the episode is flagged hall when it finishes
    got = []
    for _ in range(4000):
        env.step(env.teacher_actions())
        got += [f for f in env.drain_finished_ex() if f["hall"]]
        if got:
            break
    assert got and not got[0]["imported"]


def test_reset_pool_validation():
    env = BalatroVecEnv(2, seed=1, num_threads=1)
    with pytest.raises(ValueError):
        env.set_reset_pool(["not json"], 0.5)
    with pytest.raises(ValueError):
        env.set_reset_pool([], 1.5)
    env.set_reset_pool([], 0.5)
    assert env.reset_pool_len() == 0


# ---------------------------------------------------------------- trainer

TINY = dict(
    num_envs=16, rollout_len=16, chunk=8, group=8, epochs=1, max_episode_steps=200,
    eval_every=0, save_every=1000, decks="red", stakes="white,green",
    model={"d_model": 32, "layers": 1, "heads": 4, "ff_mult": 2, "dropout": 0.0,
           "memory": "gru", "mem_dim": 32},
)  # fmt: skip


def _tiny(tmp_path, name, **kw):
    cfg = PPOConfig(
        name=name, out_dir=str(tmp_path / name), total_steps=2 * 16 * 16, **{**TINY, **kw}
    )
    return Trainer(cfg, echo=lambda *_: None, device="cpu")


def test_trainer_logs_per_stake_advantages_and_v2_flags(tmp_path):
    t = _tiny(
        tmp_path, "v2", adv_norm="per_config", adv_norm_min_rows=1, stake_curriculum="auto",
        stake_curriculum_min_eps=1, scenario_prob=0.2, boss_hall=0.2, boss_hall_min_ante=1,
    )  # fmt: skip
    t.train()
    recs = [json.loads(line) for line in (t.out / "log.jsonl").read_text().splitlines()]
    ups = [r for r in recs if r["type"] == "update"]
    assert len(ups) == 2
    for r in ups:
        assert any(k.startswith("adv_std/") for k in r)
        assert any(k.startswith("adv_rows/") for k in r)
        assert any(k.startswith("adv_std_deck/") for k in r)
        assert abs(sum(r["stake_weights"].values()) - 1) < 1e-9
        assert "hall_pool" in r
    assert ppo.Trainer.finished_stats([]) == {"episodes": 0}
    assert sorted(ups[-1]["stake_weights"]) == ["green", "white"]


def test_finished_stats_separates_scenario_and_hall():
    def rec(**kw):
        d = dict(env=0, ante=3, code=1, steps=10, blinds_beaten=2, invalid=0, deck="red",
                 stake="white", imported=False, scenario=False, hall=False)  # fmt: skip
        d["return"] = 1.0
        d.update(kw)
        return d

    fin = [rec(), rec(code=2, stake="green"), rec(scenario=True, ante=7), rec(hall=True, code=2)]
    st = Trainer.finished_stats(fin)
    assert st["episodes"] == 2 and st["ep_win"] == 0.5 and st["ep_ante"] == 3.0
    assert st["ep_win/green"] == 1.0 and st["ep_n/white"] == 1
    assert st["scn_ante"] == 7.0 and st["scn_win"] == 0.0 and st["hall_win"] == 1.0


def test_hall_reset_only_draws_states_of_the_sampled_stake():
    _, cands = _hall_env()
    pool = [c[-1] for c in cands]  # all White
    env = BalatroVecEnv(8, seed=2, num_threads=1, max_episode_steps=10, stakes=["black"])
    env.reset()
    env.set_reset_pool(pool, 1.0)
    for _ in range(80):
        env.step(env.random_actions())
        assert (env.origins() != 2).all()
        assert not any(f["hall"] for f in env.drain_finished_ex())
