"""Ensemble actor sanity: one model through the vote / route path == plain ModelActor."""

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

CK = Path(
    os.environ.get(
        "ENSEMBLE_TEST_CKPT", "/home/yuse/Desktop/balatroRL/runs/e2e_long_fix/latest.pt"
    )
)
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not CK.exists(), reason="needs CUDA and the reference ckpt"
)


def test_single_model_equals_plain():
    from balatro_rl.model import load_checkpoint
    from balatro_rl.train import evaluate as ev
    from balatro_rl.train.ensemble import EnsembleActor

    model, _ = load_checkpoint(CK, "cuda")
    g = 16
    plain = ev.run_games("plain", ev.ModelActor(model, True, 1), g)
    vote = ev.run_games("vote1", EnsembleActor([model], "vote"), g)
    route = ev.run_games("route", EnsembleActor([model, model], "route", route_models=[0] * 4), g)
    assert plain.antes == vote.antes and plain.wins == vote.wins
    assert plain.antes == route.antes and plain.wins == route.wins


def test_duplicate_models_vote_equals_plain():
    from balatro_rl.model import load_checkpoint
    from balatro_rl.train import evaluate as ev
    from balatro_rl.train.ensemble import EnsembleActor

    model, _ = load_checkpoint(CK, "cuda")
    plain = ev.run_games("plain", ev.ModelActor(model, True, 1), 16)
    vote = ev.run_games("vote2", EnsembleActor([model, model], "vote"), 16)
    assert plain.antes == vote.antes and plain.wins == vote.wins
