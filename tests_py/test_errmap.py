"""Error maps (python/balatro_rl/analysis): branch fairness, P(win) head, analyzer end to end."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from balatro_rl import _engine
from balatro_rl.analysis import errmap, pwin
from balatro_rl.analysis.features import PHASE_BLIND, PHASE_SHOP, PolicyRunner
from balatro_rl.model.team import ModelConfig, TeamModel

REF_CKPT = Path("/home/yuse/Desktop/balatroRL/runs/e2e_long_fix/latest.pt")
REF_HEAD = Path("/home/yuse/Desktop/balatroRL/runs/errmaps/pwin_head.pt")


def _to_shop(seed: int = 5) -> _engine.Game:
    g = _engine.Game(seed)
    for _ in range(2000):
        if json.loads(g.state_json())["phase"] == "Shop":
            return g
        g.step_parts(*g.teacher_parts())
    raise AssertionError("no shop reached")


def test_branches_reseed_and_common_random_numbers():
    g = _to_shop()
    state = g.export_state()
    futures = {}
    for name, seed in (("a1", 11), ("a2", 11), ("b", 12)):
        c = _engine.Game.from_state(state, reseed=seed)
        for _ in range(60):  # the teacher plays on: shop, blind, ... all hidden info is reseeded
            if c.is_terminal():
                break
            c.step_parts(*c.teacher_parts())
        futures[name] = c.export_state()
    assert futures["a1"] == futures["a2"]  # same seed: identical future (common random numbers)
    assert futures["a1"] != futures["b"]  # different seed: a different future
    # without a reseed the clone replays the real future (this is what the analyzer never uses)
    r1, r2 = g.clone(), g.clone()
    for c in (r1, r2):
        for _ in range(60):
            if c.is_terminal():
                break
            c.step_parts(*c.teacher_parts())
    assert r1.export_state() == r2.export_state()


def _tiny_model() -> TeamModel:
    torch.manual_seed(0)
    return TeamModel(ModelConfig(d_model=32, layers=1, heads=2, mem_dim=32)).eval()


def test_pwin_head_roundtrip_and_metrics(tmp_path):
    torch.manual_seed(1)
    head = pwin.PWinHead(24)
    head.temperature.fill_(1.3)
    x = torch.randn(50, 24)
    path = tmp_path / "h.pt"
    pwin.save_head(path, head, note="x")
    h2 = pwin.load_head(path)
    assert torch.allclose(head.eval()(x), h2(x))
    assert float(h2.temperature) == pytest.approx(1.3)
    p = h2(x).numpy().astype(np.float64)
    assert ((p > 0) & (p < 1)).all()
    y = (np.random.default_rng(0).random(500) < 0.3).astype(float)
    score = y + 0.5 * np.random.default_rng(1).standard_normal(500)
    assert pwin.auc(score, y) > 0.8
    assert pwin.auc(np.zeros(500), y) == pytest.approx(0.5)
    assert pwin.reliability(np.full(500, 0.3), y)["ece"] < 0.1


def test_pwin_head_training_learns(tmp_path):
    rng = np.random.default_rng(0)
    n = 6000
    game = np.repeat(np.arange(n // 10), 10)
    z = rng.standard_normal((n, 8)).astype(np.float32)
    win = (rng.random(n) < 1 / (1 + np.exp(-2 * z[:, 0]))).astype(np.int8)
    data = {
        "feats": z.astype(np.float16),
        "value": z[:, 0],
        "phase": np.zeros(n, np.int8),
        "ante": np.ones(n, np.int8),
        "game": game,
        "win": win,
        "det": np.ones(n, np.int8),
    }
    head, info = pwin.train_head(data, "cpu", epochs=4, batch=256, log=lambda *_: None)
    rep = pwin.evaluate_head(head, data, info["split"], "cpu")
    assert rep["overall"]["auc"] > 0.7
    assert rep["overall"]["ece"] < 0.1


def test_analyzer_one_game_tiny_model():
    """End to end on CPU with a random tiny model and head: 1 game, M = 2."""
    model = _tiny_model()
    runner = PolicyRunner(model)
    head = pwin.PWinHead(runner.feat_dim).eval()
    base = 4242
    acts, codes = errmap.record_games(runner, base, 2)
    cfg = errmap.Config(m_screen=2, m_confirm=3, n_confirm=1, slots=16, min_alt_prob=0.0)
    cfg.phases = (PHASE_BLIND, PHASE_SHOP)
    acts_d = dict(enumerate(acts))
    decisions, info = errmap.run_trunk(runner, head, [1], base, acts_d, cfg)
    assert info[1]["outcome"] == codes[1] and info[1]["steps"] == len(acts[1])
    assert decisions[1], "a game has at least one blind-select decision with alternatives"
    d0 = decisions[1][0]
    assert d0.cands[0][0] == tuple(acts[1][d0.step])  # played action first
    assert all(0 <= p <= 1 for _, p in d0.cands)
    pool = errmap.RolloutPool(runner, head, cfg)
    stats = errmap.screen_and_confirm(pool, decisions, cfg)
    assert stats["decisions"] == len(decisions[1])
    d = decisions[1][0]
    assert d.screen.shape == (len(d.cands), 2)
    assert ((d.screen >= 0) & (d.screen <= 1)).all()
    res = errmap.finalize_game(1, decisions[1], info[1], cfg)
    json.dumps(res)  # serialisable
    assert res["decisions"][0]["actions"][0]["desc"]
    from balatro_rl.analysis.report import build_report

    assert "Error-map report" in build_report([res])


@pytest.mark.skipif(
    not (torch.cuda.is_available() and REF_CKPT.exists() and REF_HEAD.exists()),
    reason="needs CUDA, the reference checkpoint and the trained P(win) head",
)
def test_analyzer_reference_model_m2():
    from balatro_rl.model import load_checkpoint

    model, _ = load_checkpoint(REF_CKPT, "cuda")
    runner = PolicyRunner(model)
    head = pwin.load_head(REF_HEAD, "cuda")
    cfg = errmap.Config(m_screen=2, m_confirm=4, n_confirm=1, slots=256)
    base = 10_000_019
    acts, codes = errmap.record_games(runner, base, 3)
    decisions, info = errmap.run_trunk(runner, head, [0], base, dict(enumerate(acts)), cfg)
    assert len(decisions[0]) > 5
    pool = errmap.RolloutPool(runner, head, cfg)
    errmap.screen_and_confirm(pool, decisions, cfg)
    res = errmap.finalize_game(0, decisions[0], info[0], cfg)
    assert res["curve"] and all(0 <= c["pwin"] <= 1 for c in res["curve"])
    assert os.path.exists(REF_HEAD)
