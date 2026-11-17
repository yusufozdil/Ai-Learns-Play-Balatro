"""Frozen-policy feature tap: runs the team model and captures the critic's input
(``[encoder summary, GRU memory]``), the representation a P(win) head is trained on."""

from __future__ import annotations

import numpy as np
import torch

from balatro_rl.model.inputs import prepare

PHASE_BLIND, PHASE_PLAY, PHASE_SHOP, PHASE_PACK = 0, 1, 2, 3
BLIND_BOSS = 2


class PolicyRunner:
    """Stateless batched policy step with a forward hook on the critic that captures its input.

    ``run`` returns ``(actions (n,4) int64 tensor, hidden_out, feats (n, d + mem), value)``. The
    caller owns the recurrent state. The model is never modified (hook only)."""

    def __init__(self, model, seed: int = 0):
        self.model = model.eval()
        self.dev = model.device
        self.gen = torch.Generator(device=self.dev).manual_seed(seed)
        self._feat: torch.Tensor | None = None
        model.critic.register_forward_hook(self._hook)

    def _hook(self, _m, inp, _out):
        self._feat = inp[0].detach()

    @property
    def feat_dim(self) -> int:
        return self.model.critic[0].in_features

    def init_hidden(self, n: int) -> torch.Tensor:
        return self.model.init_hidden(n, self.dev)

    @torch.no_grad()
    def run(self, obs, masks, hidden, deterministic: bool = True, prepared=None):
        o, m = prepared if prepared is not None else prepare(obs, masks, self.dev)
        r = self.model._run(o, m, hidden, None, None, deterministic, self.gen)
        return r["actions"], r["hidden"], self._feat, r["value"]

    @torch.no_grad()
    def logp_of(self, prepared, hidden_in, actions):
        """Teacher-forced structured log-prob of ``actions`` (n,4) for prepared rows."""
        o, m = prepared
        r = self.model._run(o, m, hidden_in, None, actions, False, None)
        return r["logp"], r["ok"]


def obs_phase(obs) -> np.ndarray:
    return obs["global_i"][:, 0]


def obs_blind(obs) -> np.ndarray:
    return obs["global_i"][:, 1]


def obs_ante(obs) -> np.ndarray:
    """Ante (1..) from the scaled ``global_f[:, 0]`` (= ante / 8)."""
    return np.rint(obs["global_f"][:, 0] * 8).astype(np.int64)
