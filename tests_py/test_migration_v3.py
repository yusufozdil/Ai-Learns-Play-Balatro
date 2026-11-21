"""Interface v3 (next-round hands/discards, run counters, per-joker contribution) and the exact
v2 -> v3 checkpoint migration.

``fixtures/v2_reference.npz`` was recorded with the *pre-v3* code (interface version 2) from
``runs/st3/latest.pt`` and ``runs/st3_endless/latest.pt`` on a fixed teacher rollout over a mix of
decks and stakes (6 envs, 240 steps, seed 7): per step the teacher-forced ``evaluate_full``
outputs (value, logp, per-part log-probs, entropies, argmax hits) and the deterministic ``act``
result. A migrated v2 checkpoint must reproduce them (atol 1e-6, actions exact). Run this file as
a script (``python tests_py/test_migration_v3.py``) to re-record the fixture *on a v2 build*.
Skipped when torch or the (gitignored) checkpoints are missing; the checkpoints are looked up in
``runs/`` of this checkout and then of the main checkout of the git repository (worktrees).
"""

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from balatro_rl import LAYOUT, BalatroVecEnv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests_py" / "fixtures" / "v2_reference.npz"
STEPS, N, SEED = 240, 6, 7
DECKS = ["red", "blue", "yellow", "green", "black"]
STAKES = ["white", "red", "green", "black", "blue", "purple", "orange", "gold"]
CHECKPOINTS = {"st3": "runs/st3/latest.pt", "st3_endless": "runs/st3_endless/latest.pt"}


def find_run_file(rel: str) -> Path | None:
    """``rel`` under this checkout, else under the main checkout of the repository."""
    cands = [ROOT / rel]
    try:
        common = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        cands.append(Path(common).parent / rel)
    except (OSError, subprocess.CalledProcessError):
        pass
    return next((p for p in cands if p.exists()), None)


def state_digest(sd: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def reference(model):
    """The recording protocol of the fixture (identical code ran on the v2 interface)."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)  # one fixed GEMM blocking, whatever the machine
    try:
        env = BalatroVecEnv(N, seed=SEED, num_threads=1, decks=DECKS, stakes=STAKES)
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
    finally:
        torch.set_num_threads(threads)
    return {k: np.stack(v) for k, v in out.items()}


NAMES = LAYOUT["feature_names"]
GF = {n: i for i, n in enumerate(NAMES["global_f"])}
JF = {n: i for i, n in enumerate(NAMES["joker_f"])}


def _mg():
    from balatro_rl.model import migrate

    return migrate


@pytest.mark.parametrize("tag", list(CHECKPOINTS))
def test_migrated_v2_checkpoint_reproduces_v2_outputs(tag):
    from balatro_rl.model import load_checkpoint

    mg = _mg()
    path = find_run_file(CHECKPOINTS[tag])
    if path is None or not FIXTURE.exists():
        pytest.skip("checkpoint or fixture not available")
    old = torch.load(path, map_location="cpu", weights_only=False)
    if old["interface_version"] != 2:
        pytest.skip("not a v2 checkpoint")
    ref = np.load(FIXTURE)
    if str(ref[f"{tag}/digest"]) != state_digest(old["state_dict"]):
        pytest.skip(f"{path} changed since the fixture was recorded")
    model, ck = load_checkpoint(path)
    assert ck["interface_version"] == LAYOUT["version"] == 3 and ck["migrated_from"] == 2
    # The new input columns start at exactly zero weight; every old weight is unchanged.
    sd = model.state_dict()
    g = sd["encoder.global_proj.0.weight"]
    j = sd["encoder.joker_proj.weight"]
    assert not g[:, [GF[n] for n in mg.ADDED_GLOBAL_F_V3]].any()
    assert not j[:, [JF[n] for n in mg.ADDED_JOKER_F_V3]].any()
    assert torch.equal(
        g[:, [GF[n] for n in mg.v2_names("global_f")]],
        old["state_dict"]["encoder.global_proj.0.weight"],
    )
    assert torch.equal(
        j[:, [JF[n] for n in mg.v2_names("joker_f")]],
        old["state_dict"]["encoder.joker_proj.weight"],
    )
    for k, v in old["state_dict"].items():
        if k not in ("encoder.global_proj.0.weight", "encoder.joker_proj.weight"):
            assert torch.equal(sd[k], v), k
    model.eval()
    got = reference(model)
    assert {k for k in ref.files if k.startswith(tag + "/")} == {f"{tag}/{k}" for k in got} | {
        f"{tag}/digest"
    }
    worst = 0.0
    for k, v in got.items():
        want = ref[f"{tag}/{k}"]
        if v.dtype.kind == "f":
            worst = max(worst, float(np.abs(v - want).max()))
            np.testing.assert_allclose(v, want, rtol=0, atol=1e-6, err_msg=k)
        else:
            np.testing.assert_array_equal(v, want, err_msg=k)
    assert set(np.unique(got["phase"])) >= {0, 1, 2, 3}, "fixture must cover every phase"
    print(f"{tag}: worst abs diff {worst:.2e}")


def test_synthetic_v2_checkpoint_migrates_exactly(tmp_path):
    """No runs/ needed: a random v3 model cut down to the v2 layout by name, saved as v2, migrated
    back; the new inputs then have no influence whatever they hold (bit-identical outputs)."""
    from balatro_rl.model import ModelConfig, TeamModel, load_checkpoint

    mg = _mg()
    torch.manual_seed(1)
    cfg = ModelConfig(d_model=32, layers=1, heads=2, mem_dim=32)
    sd = {k: v.clone() for k, v in TeamModel(cfg).state_dict().items()}
    gcols = [GF[n] for n in mg.v2_names("global_f")]
    jcols = [JF[n] for n in mg.v2_names("joker_f")]
    sd["encoder.global_proj.0.weight"] = sd["encoder.global_proj.0.weight"][:, gcols].clone()
    sd["encoder.joker_proj.weight"] = sd["encoder.joker_proj.weight"][:, jcols].clone()
    assert sd["encoder.global_proj.0.weight"].shape[1] == 214
    assert sd["encoder.joker_proj.weight"].shape[1] == 13
    torch.save({"interface_version": 2, "config": cfg.__dict__, "state_dict": sd}, tmp_path / "a")
    model, ck = load_checkpoint(tmp_path / "a")
    assert ck["migrated_from"] == 2 and ck["interface_version"] == 3
    env = BalatroVecEnv(8, seed=4, num_threads=1, decks=DECKS, stakes=STAKES)
    env.set_start_jokers(["Banner", "Burglar", "Joker", "Swashbuckler", None, None, None, None])
    obs, masks = env.reset()
    for _ in range(30):
        obs, masks, *_ = env.step(env.teacher_actions())
    assert np.abs(obs["joker_f"][..., [JF[n] for n in mg.ADDED_JOKER_F_V3]]).sum() > 0
    hid = model.init_hidden(8)
    act = torch.from_numpy(env.teacher_actions().copy())
    ref = model.evaluate_full(obs, masks, hid, act)
    noisy = {k: v.copy() for k, v in obs.items()}
    noisy["global_f"][:, [GF[n] for n in mg.ADDED_GLOBAL_F_V3]] = 0.9
    noisy["joker_f"][..., [JF[n] for n in mg.ADDED_JOKER_F_V3]] = -1.7
    zero = {k: v.copy() for k, v in obs.items()}
    zero["global_f"][:, [GF[n] for n in mg.ADDED_GLOBAL_F_V3]] = 0.0
    zero["joker_f"][..., [JF[n] for n in mg.ADDED_JOKER_F_V3]] = 0.0
    for o in (noisy, zero):
        other = model.evaluate_full(o, masks, hid, act)
        for k in ("logp", "value", "entropy"):
            assert torch.equal(ref[k], other[k]), k
    # Trained columns matter.
    model.encoder.joker_proj.weight.data[:, JF["contrib_mult"]] += 1.0
    model.encoder.global_proj[0].weight.data[:, GF["next_discards"]] += 1.0
    changed = model.evaluate_full(noisy, masks, hid, act)
    assert not torch.equal(ref["value"], changed["value"])


def test_v3_columns_in_the_env():
    """End to end through the Python env: start jokers show their contribution; the next-round
    columns and run counters are written."""
    from balatro_rl.model.encoder import V2_GLOBAL_F, V2_JOKER_F

    assert len(GF) == V2_GLOBAL_F + 6 and len(JF) == V2_JOKER_F + 4
    env = BalatroVecEnv(3, seed=9, num_threads=1)
    env.set_start_jokers(["Joker", "Burglar", "Banner"])
    obs, _ = env.reset()
    sl = lambda x: np.float32(np.sign(x) * np.log1p(abs(x)) / 5)  # noqa: E731
    jf = obs["joker_f"][:, 0]
    assert jf[0, JF["contrib_mult"]] == sl(4.0)
    assert not jf[1, [JF[n] for n in _mg().ADDED_JOKER_F_V3]].any()  # Burglar: in next_*
    assert jf[2, JF["contrib_chips"]] == sl(120.0)
    g = obs["global_f"]
    assert g[0, GF["next_hands"]] == 0.5 and g[0, GF["next_discards"]] == 0.5
    assert g[1, GF["next_hands"]] == 7 / 8 and g[1, GF["next_discards"]] == 0.0
    for n in ("tarots_used", "blinds_skipped", "unique_planets_used", "hands_played_run"):
        assert (g[:, GF[n]] == 0).all(), n
    for _ in range(60):
        obs, *_ = env.step(env.teacher_actions())
    assert (obs["global_f"][:, GF["hands_played_run"]] > 0).any()


def record():  # pragma: no cover - run by hand on a v2 build
    from balatro_rl.model import load_checkpoint

    assert LAYOUT["version"] == 2, "record the v2 fixture on a v2 build"
    arrays = {}
    for tag, rel in CHECKPOINTS.items():
        path = find_run_file(rel)
        assert path is not None, rel
        model, ck = load_checkpoint(path)
        assert ck["interface_version"] == 2
        model.eval()
        for k, v in reference(model).items():
            arrays[f"{tag}/{k}"] = v
        arrays[f"{tag}/digest"] = np.array(state_digest(ck["state_dict"]))
    np.savez_compressed(FIXTURE, **arrays)
    print("wrote", FIXTURE, len(arrays))


if __name__ == "__main__":
    sys.exit(record())
