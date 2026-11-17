"""Interface v2 (deck + stake in the observation) and the exact v1 -> v2 checkpoint migration.

``fixtures/v1_reference.npz`` was recorded with the *pre-v2* code (interface version 1) from
``runs/bc/model.pt`` and ``runs/e2e_long_fix/latest.pt`` on a fixed Red/White teacher rollout
(4 envs, 220 steps, seed 5): per step the teacher-forced ``evaluate_full`` outputs (value, logp,
per-part log-probs, entropies, argmax hits) and the deterministic ``act`` result. A migrated v1
checkpoint must reproduce them. Skipped when torch or the (gitignored) checkpoints are missing.
"""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from balatro_rl import INTERFACE_VERSION, LAYOUT, BalatroVecEnv  # noqa: E402
from balatro_rl.model import ModelConfig, TeamModel, load_checkpoint, save_checkpoint  # noqa: E402
from balatro_rl.model import migrate as mg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests_py" / "fixtures" / "v1_reference.npz"
STEPS, N, SEED = 220, 4, 5
CHECKPOINTS = {"bc": "runs/bc/model.pt", "e2e": "runs/e2e_long_fix/latest.pt"}

NAMES = LAYOUT["feature_names"]
GF = {n: i for i, n in enumerate(NAMES["global_f"])}
GI = {n: i for i, n in enumerate(NAMES["global_i"])}


def reference(model):
    """The recording protocol of the fixture (identical code ran on the v1 interface)."""
    env = BalatroVecEnv(N, seed=SEED, num_threads=1)
    obs, masks = env.reset()
    hidden = model.init_hidden(N)
    first = None
    out: dict[str, list] = {}
    for _ in range(STEPS):
        a = env.teacher_actions().copy()
        at = torch.from_numpy(a)
        with torch.no_grad():
            r = model.evaluate_full(obs, masks, hidden, at, first)
            d_act, d_logp, _, d_val, _ = model.act(obs, masks, hidden, True, first)
        for k, v in r.items():
            if k == "hidden":
                continue
            out.setdefault(k, []).append(
                v.float().numpy() if v.dtype.is_floating_point else v.numpy()
            )
        out.setdefault("det_actions", []).append(d_act.numpy())
        out.setdefault("det_logp", []).append(d_logp.numpy())
        out.setdefault("det_value", []).append(d_val.numpy())
        hidden = r["hidden"]
        obs, masks, _, done, _ = env.step(a)
        first = torch.from_numpy(done != 0)
    return {k: np.stack(v) for k, v in out.items()}


@pytest.mark.parametrize("tag", list(CHECKPOINTS))
def test_migrated_v1_checkpoint_reproduces_v1_outputs(tag):
    path = ROOT / CHECKPOINTS[tag]
    if not path.exists():
        pytest.skip(f"{path} not available")
    old = torch.load(path, map_location="cpu", weights_only=False)
    if old["interface_version"] != 1:
        pytest.skip("checkpoint already migrated")
    model, ck = load_checkpoint(path)
    assert ck["interface_version"] == INTERFACE_VERSION == 2 and ck["migrated_from"] == 1
    model.eval()
    ref = np.load(FIXTURE)
    got = reference(model)
    assert {k for k in ref.files if k.startswith(tag + "/")} == {f"{tag}/{k}" for k in got}
    worst = 0.0
    for k, v in got.items():
        want = ref[f"{tag}/{k}"]
        if v.dtype.kind == "f":
            worst = max(worst, float(np.abs(v - want).max()))
            # 1e-3, not 1e-6: the fixture predates the sim parity fix (xmult as SMODS'
            # additive `mult + mult*(x-1)`), which moves some score-preview features of this
            # rollout by one chip (env 2 holds Obelisk) -> logp shifts of ~2e-4. Actions are
            # still compared exactly below.
            np.testing.assert_allclose(v, want, rtol=0, atol=1e-3, err_msg=k)
        else:
            np.testing.assert_array_equal(v, want, err_msg=k)
    assert set(np.unique(got["phase"])) >= {0, 1, 2, 3}, "fixture must cover every phase"
    print(f"{tag}: worst abs diff {worst:.2e}")
    # Bit-identical on the machine that recorded the fixture (the encoder splits the global
    # projection into the v1 columns and the new ones, see ``encoder.V1_GLOBAL_F``).
    assert worst <= 1e-3


def test_migrated_checkpoint_resaves_as_v2_and_optimizer_state_is_remapped(tmp_path):
    path = ROOT / CHECKPOINTS["e2e"]
    if not path.exists():
        pytest.skip(f"{path} not available")
    old = torch.load(path, map_location="cpu", weights_only=False)
    if old["interface_version"] != 1:
        pytest.skip("checkpoint already migrated")
    model, ck = load_checkpoint(path)
    save_checkpoint(tmp_path / "m.pt", model)
    again = torch.load(tmp_path / "m.pt", map_location="cpu", weights_only=False)
    assert again["interface_version"] == 2 and "migrated_from" not in again
    m2, _ = load_checkpoint(tmp_path / "m.pt")
    for (k, a), (_, b) in zip(model.state_dict().items(), m2.state_dict().items(), strict=True):
        assert torch.equal(a, b), k
    # The Adam moments of the two remapped parameters now have the v2 shapes: a resumed optimizer
    # steps without a shape error and the zero-initialised inputs get gradients.
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    opt.load_state_dict(ck["optimizer"])
    p = model.encoder.global_proj[0].weight
    assert opt.state_dict()["state"][
        list(model.state_dict()).index("encoder.global_proj.0.weight")
    ]["exp_avg"].shape == tuple(p.shape)
    env = BalatroVecEnv(8, seed=1, num_threads=1)
    obs, masks = env.reset()
    hid = model.init_hidden(8)
    act = torch.from_numpy(env.teacher_actions().copy())
    lp, _, v = model.evaluate(obs, masks, hid, act)
    (lp.sum() + v.sum()).backward()
    opt.step()
    new_cols = [GF[n] for n in mg.ADDED_GLOBAL_F]
    assert p.grad[:, new_cols].abs().sum() > 0


def test_synthetic_v1_checkpoint_migrates_exactly(tmp_path):
    """Self-contained version of the exactness check (no runs/ needed): a random v2 model is cut
    down to the v1 layout by name, saved as a v1 checkpoint and migrated back."""
    torch.manual_seed(0)
    cfg = ModelConfig(d_model=32, layers=1, heads=2, mem_dim=32)
    full = TeamModel(cfg).eval()
    sd = {k: v.clone() for k, v in full.state_dict().items()}
    cols = [GF[n] for n in mg.v1_names("global_f")]
    sd["encoder.global_proj.0.weight"] = sd["encoder.global_proj.0.weight"][:, cols].clone()
    from balatro_rl.model import encoder as enc

    keep = sum(mg.v1_table_sizes().values())
    assert list(enc.TABLE_SIZES)[-2:] == ["deck", "stake"], "new tables come last"
    sd["encoder.table.weight"] = sd["encoder.table.weight"][:keep].clone()
    assert sd["encoder.global_proj.0.weight"].shape[1] == 203
    torch.save(
        {"interface_version": 1, "config": cfg.__dict__, "state_dict": sd}, tmp_path / "v1.pt"
    )
    migrated, ck = load_checkpoint(tmp_path / "v1.pt")
    assert ck["migrated_from"] == 1
    # A v1 observation is a v2 one with the new input columns (and the deck/stake ids) at zero
    # *and* zero-weighted: the migrated model gives the same outputs whatever those hold.
    env = BalatroVecEnv(8, seed=2, num_threads=1)
    obs, masks = env.reset()
    for _ in range(40):
        obs, masks, *_ = env.step(env.teacher_actions())
    base = {k: v.copy() for k, v in obs.items()}
    hid = migrated.init_hidden(8)
    act = torch.from_numpy(env.teacher_actions().copy())
    ref = migrated.evaluate_full(base, masks, hid, act)
    noisy = {k: v.copy() for k, v in obs.items()}
    noisy["global_f"][:, [GF[n] for n in mg.ADDED_GLOBAL_F]] = 0.7
    noisy["global_i"][:, GI["deck"]] = 9
    noisy["global_i"][:, GI["stake"]] = 6
    other = migrated.evaluate_full(noisy, masks, hid, act)
    for k in ("logp", "value", "entropy"):
        assert torch.equal(ref[k], other[k]), k
    # ... and trained columns do matter (the model is not blind to the new inputs).
    migrated.encoder.global_proj[0].weight.data[:, GF["stake_idx"]] += 1.0
    migrated.encoder.table.weight.data[-1] += 1.0
    changed = migrated.evaluate_full(noisy, masks, hid, act)
    assert not torch.equal(ref["value"], changed["value"])
    # Unknown versions are refused.
    torch.save(
        {"interface_version": 7, "config": cfg.__dict__, "state_dict": sd}, tmp_path / "x.pt"
    )
    with pytest.raises(AssertionError):
        load_checkpoint(tmp_path / "x.pt")


def test_v1_data_upgrade_matches_the_v2_encoder_on_red_white():
    """Legacy datasets get the v2 columns appended; at the start of a Red/White run the result is
    exactly what the v2 env writes."""
    env = BalatroVecEnv(4, seed=3, num_threads=1)
    obs, _ = env.reset()
    cols = [GF[n] for n in mg.v1_names("global_f")]
    icols = [GI[n] for n in mg.v1_names("global_i")]
    f, i = mg.upgrade_v1_obs_arrays(obs["global_f"][:, cols], obs["global_i"][:, icols])
    np.testing.assert_array_equal(f, obs["global_f"])
    np.testing.assert_array_equal(i, obs["global_i"])


# --------------------------------------------------------------------------- v2 observation values


def obs_of(deck, stake, steps=0):
    env = BalatroVecEnv(2, seed=4, num_threads=1, deck=deck, stake=stake)
    obs, _ = env.reset()
    for _ in range(steps):
        obs, *_ = env.step(env.teacher_actions())
    return obs


@pytest.mark.parametrize(
    ("deck", "stake", "deck_id", "stake_id", "flags", "hands", "discards", "mph", "mpd"),
    [
        ("red", "white", 0, 0, (0, 0, 0, 0, 0, 0), 4, 4, 1, 0),
        ("plasma", "gold", 13, 7, (1, 1, 1, 1, 1, 0), 4, 2, 1, 0),
        ("green", "blue", 3, 4, (1, 1, 0, 0, 0, 1), 4, 2, 2, 1),
        ("blue", "black", 1, 3, (1, 1, 0, 0, 0, 0), 5, 3, 1, 0),
        ("black", "purple", 4, 5, (1, 1, 0, 0, 0, 0), 3, 2, 1, 0),
        ("red", "orange", 0, 6, (1, 1, 1, 0, 0, 0), 4, 3, 1, 0),
    ],
)
def test_run_configuration_columns(
    deck, stake, deck_id, stake_id, flags, hands, discards, mph, mpd
):
    o = obs_of(deck, stake)
    assert (o["global_i"][:, GI["deck"]] == deck_id).all()
    assert (o["global_i"][:, GI["stake"]] == stake_id).all()
    f = o["global_f"][0]
    got = tuple(
        int(f[GF[n]])
        for n in (
            "no_small_reward",
            "eternal_on",
            "perishable_on",
            "rental_on",
            "plasma_blind_x2",
            "no_interest",
        )
    )
    assert got == flags
    assert f[GF["stake_idx"]] == pytest.approx(stake_id / 7)
    assert f[GF["base_hands"]] == pytest.approx(hands / 8)
    assert f[GF["base_discards"]] == pytest.approx(discards / 8)
    assert f[GF["money_per_hand"]] == pytest.approx(mph / 2)
    assert f[GF["money_per_discard"]] == pytest.approx(mpd / 2)


def test_configuration_columns_persist_through_the_run():
    for deck, stake in (("plasma", "gold"), ("red", "white")):
        a, b = obs_of(deck, stake), obs_of(deck, stake, steps=150)
        for n in mg.ADDED_GLOBAL_F:
            if n not in ("base_hands", "base_discards"):  # vouchers may change those
                assert a["global_f"][0, GF[n]] == b["global_f"][0, GF[n]], n
        assert (a["global_i"][:, 8:] == b["global_i"][:, 8:]).all()


def test_joker_perish_column_exists_in_v1_and_v2():
    assert "perish_rounds_left" in NAMES["joker_f"]
    assert LAYOUT["version"] == 2


# --------------------------------------------------------------------------- per-env sampling


def test_sampled_decks_and_stakes():
    kw = {"decks": ["red", "plasma"], "stakes": ["white", "gold", "blue"], "num_threads": 1}
    env = BalatroVecEnv(64, seed=11, **kw)
    obs, _ = env.reset()
    seen = {
        (int(d), int(s))
        for d, s in zip(
            obs["global_i"][:, GI["deck"]], obs["global_i"][:, GI["stake"]], strict=True
        )
    }
    assert seen <= {(d, s) for d in (0, 13) for s in (0, 7, 4)}
    assert len(seen) >= 5  # 64 envs over 6 pairs
    # Deterministic, and independent of the thread count.
    env2 = BalatroVecEnv(64, seed=11, decks=kw["decks"], stakes=kw["stakes"], num_threads=3)
    obs2, _ = env2.reset()
    np.testing.assert_array_equal(obs["global_i"], obs2["global_i"])
    # Episodes after an auto-reset draw again.
    drawn = set(seen)
    for _ in range(300):
        obs, _, _, done, _ = env.step(env.random_actions())
        for d, s in zip(
            obs["global_i"][:, GI["deck"]], obs["global_i"][:, GI["stake"]], strict=True
        ):
            drawn.add((int(d), int(s)))
    assert len(drawn) == 6
    # Only `stakes` given: the deck is the fixed one; nothing given: unchanged behaviour.
    e3 = BalatroVecEnv(32, seed=1, deck="green", stakes=["white", "gold"], num_threads=1)
    o3, _ = e3.reset()
    assert (o3["global_i"][:, GI["deck"]] == 3).all()
    assert set(np.unique(o3["global_i"][:, GI["stake"]])) == {0, 7}
    with pytest.raises(ValueError):
        BalatroVecEnv(2, decks=["nope"])
    with pytest.raises(ValueError):
        BalatroVecEnv(2, stakes=[])


# --------------------------------------------------------------------------- legacy BC data


def test_v1_bc_dataset_loads_with_appended_columns(tmp_path):
    from balatro_rl.train import bc

    bc.collect(tmp_path, num_envs=6, steps=24, seed=3, threads=1, log=lambda *_: 0)
    new = bc.load_gpu(tmp_path, torch.device("cpu"))
    assert new["global_f"].shape[-1] == 214 and new["global_i"].shape[-1] == 10
    # Rewrite the two files the way a v1 collect stored them (v1 columns only).
    fcols = [GF[n] for n in mg.v1_names("global_f")]
    icols = [GI[n] for n in mg.v1_names("global_i")]
    gf = np.load(tmp_path / "global_f.npy")[..., fcols]
    gi = np.load(tmp_path / "global_i.npy")[..., icols]
    assert gf.shape[-1] == 203 and gi.shape[-1] == 8
    np.save(tmp_path / "global_f.npy", gf)
    np.save(tmp_path / "global_i.npy", gi)
    old = bc.load_gpu(tmp_path, torch.device("cpu"))
    assert old["global_f"].shape == new["global_f"].shape
    assert torch.equal(old["global_i"], new["global_i"])  # Red/White: deck = stake = 0
    drift = [GF["base_hands"], GF["base_discards"]]  # vouchers may have changed these
    keep = [i for i in range(214) if i not in drift]
    assert torch.equal(old["global_f"][..., keep], new["global_f"][..., keep])


# --------------------------------------------------------------------------- PPO paths


def _to_v1(src, dst):
    """Rewrite a v2 PPO checkpoint as a v1 one (cut the new columns / table rows and the matching
    Adam moments by name) - the inverse of ``migrate_v1_to_v2``."""
    ck = torch.load(src, map_location="cpu", weights_only=False)
    cols = [GF[n] for n in mg.v1_names("global_f")]
    keep = sum(mg.v1_table_sizes().values())
    cut = {
        "encoder.global_proj.0.weight": lambda t: t[:, cols].clone(),
        "encoder.table.weight": lambda t: t[:keep].clone(),
    }
    names = [n for n, _ in TeamModel(ModelConfig(**ck["config"])).named_parameters()]
    for name, fn in cut.items():
        shape = tuple(ck["state_dict"][name].shape)
        ck["state_dict"][name] = fn(ck["state_dict"][name])
        st = ck["optimizer"]["state"][names.index(name)]
        for k, v in st.items():
            if isinstance(v, torch.Tensor) and tuple(v.shape) == shape:
                st[k] = fn(v)
    ck["interface_version"] = 1
    torch.save(ck, dst)


def test_ppo_resume_and_init_from_v1_checkpoints_and_sampled_training(tmp_path):
    from balatro_rl.train.ppo import Trainer
    from test_ppo import tiny_cfg

    kw = {"decks": "red,plasma,green", "stakes": "white,gold", "eval_deck": "plasma"}
    cfg = tiny_cfg(
        tmp_path, total_steps=16 * 16 * 2, eval_every=1, eval_games=4, save_every=1, **kw
    )
    tr = Trainer(cfg, echo=lambda *_: 0, device="cpu")
    tr.train()
    seen = {
        (int(d), int(s))
        for d, s in zip(
            tr.obs["global_i"][:, GI["deck"]], tr.obs["global_i"][:, GI["stake"]], strict=True
        )
    }
    assert len(seen) > 1 and all(d in (0, 3, 13) and s in (0, 7) for d, s in seen)
    out = tmp_path / "run"
    v1 = tmp_path / "v1run"
    v1.mkdir()
    _to_v1(out / "latest.pt", v1 / "latest.pt")
    # --resume of a v1 run directory (model + Adam state migrated), then one more update.
    cfg2 = tiny_cfg(
        tmp_path, out_dir=str(v1), total_steps=16 * 16 * 3, eval_every=0, save_every=1, **kw
    )
    t2 = Trainer(cfg2, echo=lambda *_: 0, device="cpu", resume=True)
    assert t2.update_idx == 2
    t2.train()
    assert t2.update_idx == 3
    assert torch.load(v1 / "latest.pt", weights_only=False)["interface_version"] == 2
    # --init (with the KL penalty reference) from a v1 checkpoint.
    _to_v1(out / "latest.pt", tmp_path / "init_v1.pt")
    cfg3 = tiny_cfg(
        tmp_path,
        out_dir=str(tmp_path / "run3"),
        init_checkpoint=str(tmp_path / "init_v1.pt"),
        kl_coef=0.5,
        total_steps=16 * 16,
        **kw,
    )
    t3 = Trainer(cfg3, echo=lambda *_: 0, device="cpu")
    t3.train()
    assert t3.update_idx == 1
    # Default configuration keeps the fixed-config path (no sampling lists).
    assert Trainer.env_configs(tiny_cfg(tmp_path)) == {}
