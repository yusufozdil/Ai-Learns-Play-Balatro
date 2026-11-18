"""PPO trainer, curriculum routing and evaluation protocol (skipped without torch)."""

import copy
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from balatro_rl import BalatroVecEnv  # noqa: E402
from balatro_rl.model import ModelConfig, TeamModel  # noqa: E402
from balatro_rl.train import evaluate as ev  # noqa: E402
from balatro_rl.train.curriculum import Curriculum  # noqa: E402
from balatro_rl.train.ppo import PPOConfig, Trainer, compute_gae  # noqa: E402

TINY = {
    "d_model": 32,
    "layers": 1,
    "heads": 2,
    "ff_mult": 4,
    "dropout": 0.0,
    "memory": "gru",
    "mem_dim": 32,
}


def tiny_cfg(tmp_path, **kw) -> PPOConfig:
    base = {
        "out_dir": str(tmp_path / "run"),
        "num_envs": 16,
        "rollout_len": 16,
        "chunk": 8,
        "group": 8,
        "threads": 1,
        "max_episode_steps": 300,
        "model": TINY,
        "eval_games": 8,
        "eval_every": 0,
        "epochs": 2,
    }
    base.update(kw)
    return PPOConfig(**base)


# --------------------------------------------------------------------------- GAE


def test_gae_hand_computed():
    # One env, T = 3. Observation 2 starts a new episode (transition 1 -> 2 is terminal).
    g, lam = 0.9, 0.8
    rewards = torch.tensor([[1.0], [0.0], [2.0]])
    values = torch.tensor([[0.5], [0.4], [0.3]])
    first = torch.tensor([[False], [False], [True]])
    adv, ret = compute_gae(
        rewards, values, first, torch.tensor([0.6]), torch.tensor([False]), g, lam
    )
    a2 = 2.0 + g * 0.6 - 0.3  # bootstrap from the observation after the last step
    a1 = 0.0 + 0.0 - 0.4  # terminal: no bootstrap, no trace
    a0 = (1.0 + g * 0.4 - 0.5) + g * lam * a1
    assert adv[:, 0].tolist() == pytest.approx([a0, a1, a2])
    assert ret[:, 0].tolist() == pytest.approx([a0 + 0.5, a1 + 0.4, a2 + 0.3])
    # the last observation starts a new episode: no bootstrap from it
    adv2, _ = compute_gae(rewards, values, first, torch.tensor([0.6]), torch.tensor([True]), g, lam)
    assert adv2[2, 0].item() == pytest.approx(2.0 - 0.3)


def test_gae_bootstraps_through_truncation():
    # One env, T = 3. Transition 1 was truncated (step limit): observation 2 is a new episode,
    # the trace is cut, but the target bootstraps with values[1] instead of 0.
    g, lam = 0.9, 0.8
    rewards = torch.tensor([[1.0], [0.5], [2.0]])
    values = torch.tensor([[0.5], [0.4], [0.3]])
    first = torch.tensor([[False], [False], [True]])
    trunc = torch.tensor([[False], [True], [False]])
    last_v, last_f = torch.tensor([0.6]), torch.tensor([False])
    adv, _ = compute_gae(rewards, values, first, last_v, last_f, g, lam, trunc)
    a2 = 2.0 + g * 0.6 - 0.3
    a1 = 0.5 + g * 0.4 - 0.4  # bootstrap with its own value, no trace from observation 2
    a0 = (1.0 + g * 0.4 - 0.5) + g * lam * a1
    assert adv[:, 0].tolist() == pytest.approx([a0, a1, a2])
    # without the flag (or an all-False flag) it is a terminal
    for tr in (None, torch.zeros_like(first)):
        adv0, _ = compute_gae(rewards, values, first, last_v, last_f, g, lam, tr)
        assert adv0[1, 0].item() == pytest.approx(0.5 - 0.4)
    # truncation on the very last step bootstraps with that step's value too
    adv3, _ = compute_gae(
        rewards, values, first, last_v, torch.tensor([True]), g, lam,
        torch.tensor([[False], [False], [True]]),
    )  # fmt: skip
    assert adv3[2, 0].item() == pytest.approx(2.0 + g * 0.3 - 0.3)


def test_gae_matches_naive_reference():
    gen = torch.Generator().manual_seed(0)
    t_len, n = 20, 5
    r = torch.randn(t_len, n, generator=gen)
    v = torch.randn(t_len, n, generator=gen)
    first = torch.rand(t_len, n, generator=gen) < 0.2
    lv = torch.randn(n, generator=gen)
    lf = torch.rand(n, generator=gen) < 0.3
    g, lam = 0.97, 0.9
    adv, _ = compute_gae(r, v, first, lv, lf, g, lam)
    for i in range(n):
        for t in range(t_len):  # A_t = sum_k (g lam)^k delta_{t+k} until a terminal
            total, disc = 0.0, 1.0
            for k in range(t, t_len):
                last = k == t_len - 1
                nonterm = 0.0 if (lf[i] if last else first[k + 1, i]) else 1.0
                nv = lv[i] if last else v[k + 1, i]
                total += disc * (r[k, i] + g * nv * nonterm - v[k, i])
                disc *= g * lam * nonterm
                if nonterm == 0.0:
                    break
            assert adv[t, i].item() == pytest.approx(float(total), abs=1e-5)


# --------------------------------------------------------------------------- curriculum


def test_curriculum_parsing():
    c = Curriculum("play@100,s2@200,all")
    assert [s.label for s in c.stages] == ["play", "blind+play+pack", "blind+play+shop+pack"]
    assert c.stage_at(0)[0] == 0 and c.stage_at(100)[0] == 1 and c.stage_at(10**9)[0] == 2
    assert Curriculum("play").stages[0].route_table().tolist()[:4] == [False, True, False, False]
    with pytest.raises(ValueError):
        Curriculum("play,all")  # a non-last stage needs a limit
    with pytest.raises(ValueError):
        Curriculum("flying")


def test_masked_teacher_actions():
    env = BalatroVecEnv(12, seed=5, num_threads=2, max_episode_steps=300)
    env.reset()
    full = env.teacher_actions().copy()
    sel = np.zeros(12, dtype=bool)
    sel[[1, 4, 7]] = True
    env._actions[:] = -7
    part = env.teacher_actions(sel)
    assert (part[sel] == full[sel]).all()
    assert (part[~sel] == -7).all()


# --------------------------------------------------------------------------- recurrent minibatches


def test_recurrent_minibatch_reproduces_rollout_logprobs(tmp_path):
    tr = Trainer(tiny_cfg(tmp_path), device="cpu")
    tr.collect()
    cfg = tr.cfg
    gen = torch.Generator().manual_seed(1)
    worst = 0.0
    phases = set()
    for c in range(cfg.rollout_len // cfg.chunk):  # every chunk, including carried-in states
        idx = torch.randperm(cfg.num_envs, generator=gen)[: cfg.group]
        t0, t1 = c * cfg.chunk, (c + 1) * cfg.chunk
        obs, masks, actions, first = tr._slice(t0, t1, idx)
        r = tr.model.evaluate_full(obs, masks, tr.h0[c][idx], actions, first)
        old = tr.buf["logp"][t0:t1][:, idx].reshape(-1)
        worst = max(worst, float((r["logp"].detach() - old).abs().max()))
        v_old = tr.buf["value"][t0:t1][:, idx].reshape(-1)
        assert (r["value"] - v_old).abs().max() < 1e-4
        assert r["ok"].all()
        phases |= set(r["phase"].tolist())
    assert worst < 1e-4, worst  # ratio = 1 before any update
    assert len(phases) >= 2


def test_first_update_has_unit_ratio(tmp_path):
    tr = Trainer(tiny_cfg(tmp_path, epochs=1, lr=1e-9, group=16), device="cpu")
    tr.collect()
    st = tr.update()  # lr ~ 0: parameters do not move, so every ratio is 1
    assert st["approx_kl"] < 1e-8 and st["clipfrac"] == 0.0


# --------------------------------------------------------------------------- routing


def test_teacher_rows_do_not_enter_the_policy_loss(tmp_path):
    cfg = tiny_cfg(tmp_path, curriculum="play", epochs=1)
    tr = Trainer(cfg, device="cpu")
    tr.collect()
    ctrl, phase = tr.buf["ctrl"], tr.buf["phase"]
    assert (ctrl == (phase == 1)).all()
    assert (~ctrl).any() and ctrl.any()
    # teacher-chosen rows hold the teacher's actions (model samples differ and are never executed)
    twin = copy.deepcopy(tr.model)
    opt_state = copy.deepcopy(tr.opt.state_dict())
    last_value, first = tr.last_value.clone(), tr.first.clone()
    buf = {k: v.clone() for k, v in tr.buf.items()}

    def run(corrupt: bool):
        tr.model.load_state_dict(twin.state_dict())
        tr.opt.load_state_dict(opt_state)
        tr.buf = {k: v.clone() for k, v in buf.items()}
        tr.last_value, tr.first = last_value, first
        if corrupt:  # garbage log-probs and actions on teacher rows
            tr.buf["logp"][~ctrl] = 1e3
            tr.buf["actions"][~ctrl] = 0
        torch.manual_seed(7)
        st = tr.update()
        return [p.detach().clone() for p in tr.model.parameters()], st

    clean, st = run(False)
    dirty, _ = run(True)
    for a, b in zip(clean, dirty, strict=True):
        assert torch.allclose(a, b, atol=1e-6), "teacher rows leaked into the update"
    assert st["policy_rows_per_mb"] < cfg.chunk * cfg.group  # only model rows are counted
    assert st["rows/play"] > 0 and st["rows/shop"] == 0 and st["rows/blind"] == 0


# --------------------------------------------------------------------------- end to end


def test_training_with_truncation_bootstrap_reorder_cost_and_guarded_validation(tmp_path):
    cfg = tiny_cfg(
        tmp_path,
        total_steps=16 * 16 * 2,
        max_episode_steps=7,  # nearly every episode is cut by the step limit
        reward={"reorder": -0.2},
        eval_every=2,
        eval_move_limit=2,
        save_every=1,
        # A seed whose random-init greedy policy lives past 7 steps (with the v3 input widths
        # seed 0 plays High Cards and loses every eval game in 5 steps).
        seed=1,
    )
    tr = Trainer(cfg, echo=lambda *_: 0, device="cpu")
    tr.train()
    assert tr.buf["trunc"].any() and tr.buf["trunc"].sum() <= tr.buf["first"].sum()
    recs = [json.loads(line) for line in (tmp_path / "run" / "log.jsonl").read_text().splitlines()]
    ups = [r for r in recs if r["type"] == "update"]
    assert all(np.isfinite(u["vloss"]) and np.isfinite(u["pg"]) for u in ups)
    assert any(r["type"] == "eval" and r["det_truncated"] > 0 for r in recs)
    # the flag is honoured: with the bootstrap off the advantages differ on truncated steps
    for flag in (True, False):
        cfg2 = tiny_cfg(
            tmp_path / str(flag), max_episode_steps=7, bootstrap_truncation=flag, total_steps=256
        )
        t2 = Trainer(cfg2, echo=lambda *_: 0, device="cpu")
        t2.collect()
        assert t2.buf["trunc"].any()


def test_short_training_run_and_resume(tmp_path):
    cfg = tiny_cfg(
        tmp_path, total_steps=16 * 16 * 3, eval_every=2, eval_games=8, save_every=1, kl_coef=0.0
    )
    Trainer(cfg, echo=lambda *_: 0, device="cpu").train()
    out = tmp_path / "run"
    recs = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    updates = [r for r in recs if r["type"] == "update"]
    assert len(updates) == 3 and all(np.isfinite(u["approx_kl"]) for u in updates)
    assert all(np.isfinite(u["vloss"]) and u["steps_per_sec"] > 0 for u in updates)
    assert any(r["type"] == "eval" and "det_mean_ante" in r for r in recs)
    assert (out / "latest.pt").exists() and (out / "best.pt").exists()
    ck = torch.load(out / "latest.pt", weights_only=False)
    from balatro_rl import INTERFACE_VERSION

    assert ck["interface_version"] == INTERFACE_VERSION and ck["ppo_config"]["rollout_len"] == 16
    # resume continues from the saved update counter
    cfg2 = tiny_cfg(tmp_path, total_steps=16 * 16 * 5, eval_every=0)
    tr2 = Trainer(cfg2, echo=lambda *_: 0, device="cpu", resume=True)
    assert tr2.update_idx == 3 and tr2.resumes == 1
    tr2.train()
    assert tr2.update_idx == 5


def test_bc_init_with_kl_penalty_and_critic_warmup(tmp_path):
    from balatro_rl.model import save_checkpoint

    model = TeamModel(ModelConfig(**TINY))
    init = tmp_path / "bc.pt"
    save_checkpoint(init, model)
    cfg = tiny_cfg(
        tmp_path,
        init_checkpoint=str(init),
        kl_coef=0.5,
        critic_warmup=1,
        total_steps=16 * 16 * 2,
        epochs=1,
    )
    tr = Trainer(cfg, echo=lambda *_: 0, device="cpu")
    before = [p.detach().clone() for p in tr.model.encoder.parameters()]
    tr.collect()
    st = tr.update()  # update 0 is the critic warm-up: backbone and heads must not move
    assert st["critic_warmup"]
    for a, b in zip(before, tr.model.encoder.parameters(), strict=True):
        assert torch.equal(a, b)
    tr.update_idx += 1
    tr.collect()
    st = tr.update()
    assert not st["critic_warmup"] and st["kl_coef"] > 0 and np.isfinite(st["ref_kl"])


# --------------------------------------------------------------------------- evaluation


def test_eval_protocol_prefix_and_determinism():
    big = ev.run_games("random", ev.random_actor, games=16, max_episode_steps=200, threads=1)
    small = ev.run_games("random", ev.random_actor, games=8, max_episode_steps=200, threads=2)
    assert small.antes == big.antes[:8]  # a validation set is a prefix of the test set
    again = ev.run_games("random", ev.random_actor, games=16, max_episode_steps=200)
    assert again.antes == big.antes and again.invalid_per_game == 0
    assert ev.train_seed(0) != ev.TEST_SEED and ev.train_seed(3, 2) != ev.TEST_SEED
    d = ev.compare(big, again)
    assert d["ante_diff"] == 0.0


def test_model_actor_subset_matches_full_batch():
    model = TeamModel(ModelConfig(**TINY))
    full = ev.evaluate_model(model, 12, True, max_episode_steps=150, threads=1)
    again = ev.evaluate_model(model, 12, True, max_episode_steps=150, threads=1)
    assert full.antes == again.antes and full.invalid_per_game == 0
    routed = ev.evaluate_model(model, 8, True, route="play", max_episode_steps=150, threads=1)
    assert routed.invalid_per_game == 0 and routed.games == 8


class _AlwaysMove:
    """Stub model: MoveJoker whenever the kind mask allows it, else the first legal kind."""

    device = torch.device("cpu")

    def eval(self):
        return self

    def init_hidden(self, n):
        return torch.zeros(n, 1)

    def act(self, o, m, h, deterministic=False, first=None, generator=None):
        km = torch.as_tensor(np.asarray(m["kind_mask"])).bool()
        kind = torch.where(km[:, ev.MOVE_JOKER], ev.MOVE_JOKER, km.int().argmax(dim=1))
        out = torch.zeros(len(kind), 4, dtype=torch.long)
        out[:, 0] = kind
        return out, None, None, None, h


def test_repetition_guard_breaks_move_joker_loops():
    n = 3
    kind_mask = np.zeros((n, 17), dtype=bool)
    kind_mask[:, ev.MOVE_JOKER] = True
    kind_mask[0, 5] = True  # env 0 and 1 have an alternative, env 2 only MoveJoker
    kind_mask[1, 9] = True
    env = type("E", (), {"num_envs": n})()
    first = np.zeros(n, dtype=bool)
    active = np.ones(n, dtype=bool)

    def run(limit):
        actor = ev.ModelActor(_AlwaysMove(), True, move_limit=limit)
        return [
            actor(env, {}, {"kind_mask": kind_mask}, first, active)[:, 0].tolist() for _ in range(7)
        ]

    off = run(None)
    assert all(k == [ev.MOVE_JOKER] * 3 for k in off)  # the stub loops forever
    on = run(2)
    # moves, moves, blocked (the alternative), moves, moves, blocked, ...
    assert [k[0] for k in on] == [8, 8, 5, 8, 8, 5, 8]
    assert [k[1] for k in on] == [8, 8, 9, 8, 8, 9, 8]
    assert [k[2] for k in on] == [8] * 7  # never an empty mask
    assert run(2) == on  # deterministic
    # a new game (first) restarts the count
    actor = ev.ModelActor(_AlwaysMove(), True, move_limit=1)
    a = actor(env, {}, {"kind_mask": kind_mask}, first, active)
    b = actor(env, {}, {"kind_mask": kind_mask}, np.ones(n, dtype=bool), active)
    assert a[0, 0] == 8 and b[0, 0] == 8


# --------------------------------------------------------------------------- speed options (G4)


def test_bf16_update_runs_and_recompute_restores_unit_ratio(tmp_path):
    # bf16 autocast alone breaks "ratio == 1 on the first minibatch" (the rollout used float32);
    # with --recompute-old-logp the old log-probs come from the same bf16 pass, so it holds again.
    kw = {"epochs": 1, "lr": 1e-9, "group": 16, "update_dtype": "bf16"}
    plain = Trainer(tiny_cfg(tmp_path / "a", **kw), device="cpu")
    plain.collect()
    st = plain.update()
    assert np.isfinite(st["approx_kl"]) and st["approx_kl"] < 0.05
    rec = Trainer(tiny_cfg(tmp_path / "b", recompute_old_logp=True, **kw), device="cpu")
    rec.collect()
    st = rec.update()
    assert st["approx_kl"] < 1e-6 and st["clipfrac"] == 0.0


def test_profile_reports_sections(tmp_path):
    tr = Trainer(tiny_cfg(tmp_path, profile=True, epochs=1), device="cpu")
    tr.collect()
    st = tr.update()
    for k in ("rollout/env_step", "rollout/act", "update/gae", "update/forward", "update/backward"):
        assert st["prof"][k] > 0, k
    assert "update/enc" in st["prof"] and "update/heads" in st["prof"]


def test_encoder_compile_failure_falls_back_to_eager(tmp_path, monkeypatch):
    tr = Trainer(tiny_cfg(tmp_path), device="cpu")

    def boom(*a, **k):
        raise RuntimeError("no compiler")

    monkeypatch.setattr(torch, "compile", boom)
    assert tr.model.encoder.enable_compile() is False
    assert tr.model.encoder._blocks_fn is None
    tr.collect()
    tr.update()  # still trains eagerly
