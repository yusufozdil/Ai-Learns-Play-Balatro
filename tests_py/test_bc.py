"""Behaviour-cloning pipeline smoke test (tiny): teacher data -> train -> loss falls, every
teacher action is expressible by the model's action chain."""

import pytest

torch = pytest.importorskip("torch")

from balatro_rl.model import ModelConfig, TeamModel  # noqa: E402
from balatro_rl.train import bc, play  # noqa: E402


def test_bc_pipeline_tiny(tmp_path):
    meta = bc.collect(tmp_path / "data", num_envs=24, steps=48, seed=3, threads=1, log=lambda *_: 0)
    assert meta["samples"] == 24 * 48
    data = bc.load_gpu(tmp_path / "data", torch.device("cpu"))
    torch.manual_seed(0)
    model = TeamModel(ModelConfig(d_model=32, layers=1, heads=2, mem_dim=32))
    cfg = bc.BCConfig(epochs=3, chunk=16, group=8, val_envs=8, warmup=5, lr=2e-3)
    out = tmp_path / "out"
    out.mkdir()
    hist = bc.train(model, data, cfg, out, log=lambda *_: 0)["history"]
    assert hist[-1]["val_nll"] < hist[0]["val_nll"]
    # the teacher's structured actions are all expressible through the masks / canonical order
    assert hist[-1]["val_expressible"] == 1.0
    assert (out / "model.pt").exists()


def test_play_games_baselines():
    r = play.play_games("random", play.random_policy, games=16, max_episode_steps=500)
    t = play.play_games("teacher", play.teacher_policy, games=16, max_episode_steps=500)
    assert t.mean_ante > r.mean_ante
    assert r.invalid_per_game == 0
