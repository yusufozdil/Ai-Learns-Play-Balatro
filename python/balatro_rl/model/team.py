"""The team network: shared token encoder -> strategy memory (GRU) -> per-phase heads + critic.

``docs/design/model.md`` has the architecture diagram, sizes and API contract.

Sequences / memory
------------------
``hidden`` is ``(n_envs, mem_dim)`` (``mem_dim = 0`` for ``memory="none"``: all code paths stay
the same, the tensors are just empty). ``first`` marks observations that start a new episode (the
previous step finished the episode: ``done != 0``) and zeroes the memory *before* the GRU update:

* ``act``: one step for ``n`` envs, ``first`` is ``(n,)`` or ``None``.
* ``evaluate`` / ``evaluate_full``: ``first=None`` is one step per row (``hidden_in`` is the state
  *before* the step, per row). With ``first`` of shape ``(T, n)`` the inputs are the time-major
  flattening ``(T * n, ...)`` of a ``(T, n)`` rollout and ``hidden_in`` is ``(n, mem_dim)`` (the
  state before step 0); the GRU is scanned over ``T`` with resets where ``first`` is set.
"""

from __future__ import annotations

import contextlib
import dataclasses
from dataclasses import dataclass

import torch
import torch.nn as nn

from balatro_rl.env import INTERFACE_VERSION
from balatro_rl.model.encoder import Encoder
from balatro_rl.model.heads import Head
from balatro_rl.model.inputs import is_prepared, prepare
from balatro_rl.model.migrate import migrate_v1_to_v2

HEAD_NAMES = ("blind", "play", "shop", "pack")
# Phase index (BlindSelect 0, Play 1, Shop 2, PackOpen 3, GameOver 4, Won 5) -> head. The two
# terminal phases are never observed in a running env; they fall back to the Blind head.
PHASE_TO_HEAD = (0, 1, 2, 3, 0, 0)


@dataclass
class ModelConfig:
    d_model: int = 128
    layers: int = 4
    heads: int = 4
    ff_mult: int = 4
    dropout: float = 0.0
    memory: str = "gru"  # "gru" | "none"
    mem_dim: int = 128

    def __post_init__(self):
        assert self.memory in ("gru", "none")


class TeamModel(nn.Module):
    # Optional profiling hook (set by the PPO trainer with ``--profile``): ``_prof(name)`` returns a
    # context manager timing a section. ``None`` = no profiling (the default, zero overhead).
    _prof = None

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        d = c.d_model
        self.mem_dim = c.mem_dim if c.memory == "gru" else 0
        self.encoder = Encoder(d, c.layers, c.heads, c.ff_mult, c.dropout)
        self.cell = nn.GRUCell(d, self.mem_dim) if self.mem_dim else None
        self.ctx = nn.Sequential(
            nn.Linear(d + self.mem_dim, d), nn.GELU(), nn.Linear(d, d), nn.LayerNorm(d)
        )
        self.heads = nn.ModuleList([Head(d) for _ in HEAD_NAMES])
        self.critic = nn.Sequential(nn.Linear(d + self.mem_dim, d), nn.GELU(), nn.Linear(d, 1))
        self.register_buffer("phase_head", torch.tensor(PHASE_TO_HEAD), persistent=False)

    # ------------------------------------------------------------------ helpers

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def init_hidden(self, n: int, device=None) -> torch.Tensor:
        return torch.zeros(n, self.mem_dim, device=device or self.device)

    def num_parameters(self) -> dict[str, int]:
        def count(m):
            return sum(p.numel() for p in m.parameters())

        out = {"encoder": count(self.encoder), "memory": count(self.cell) if self.cell else 0}
        out["ctx"] = count(self.ctx)
        for name, head in zip(HEAD_NAMES, self.heads, strict=True):
            out[f"head_{name}"] = count(head)
        out["critic"] = count(self.critic)
        out["total"] = count(self)
        return out

    def _inputs(self, obs, masks):
        if is_prepared(obs):
            return obs, masks
        return prepare(obs, masks, self.device)

    def _memory(self, summary, hidden, first):
        """Returns ``(h (B, mem), hidden_out (n, mem))``."""
        if not self.mem_dim:
            n = hidden.shape[0] if first is None or first.dim() == 2 else first.shape[0]
            return summary.new_zeros(summary.shape[0], 0), summary.new_zeros(n, 0)
        if first is None:  # one step per row
            h = self.cell(summary, hidden)
            return h, h
        if first.dim() == 1:  # one step, n envs
            h = self.cell(summary, hidden * (~first)[:, None].to(hidden.dtype))
            return h, h
        t_len, n = first.shape
        s = summary.view(t_len, n, -1)
        h = hidden
        outs = []
        for t in range(t_len):
            h = self.cell(s[t], h * (~first[t])[:, None].to(h.dtype))
            outs.append(h)
        return torch.stack(outs).reshape(t_len * n, -1), h

    def _heads(self, ctx, ptr, hand, masks, phase, actions, deterministic, generator):
        n = ctx.shape[0]
        head_id = self.phase_head[phase]
        out: dict[str, torch.Tensor] = {}
        for hid, head in enumerate(self.heads):
            idx = (head_id == hid).nonzero().squeeze(1)
            if idx.numel() == 0:
                continue
            sub = {k: v[idx] for k, v in masks.items()}
            res = head(
                ctx[idx],
                ptr[idx],
                hand[idx],
                sub,
                None if actions is None else actions[idx],
                deterministic,
                generator,
            )
            for k, v in res.items():
                if k not in out:
                    out[k] = torch.zeros((n, *v.shape[1:]), dtype=v.dtype, device=v.device)
                out[k] = out[k].index_copy(0, idx, v)
        return out

    def _sec(self, name: str):
        return self._prof(name) if self._prof is not None else contextlib.nullcontext()

    def _run(self, obs, masks, hidden, first, actions, deterministic, generator):
        obs, masks = self._inputs(obs, masks)
        with self._sec("enc"):
            summary, hand, ptr = self.encoder(obs)
        with self._sec("mem"):
            h, hidden_out = self._memory(summary, hidden, first)
            ctx = self.ctx(torch.cat([summary, h], dim=-1))
        # The critic always runs in float32, also under bf16 autocast (a no-op otherwise): value
        # targets are small differences of returns, bf16 rounding would swamp them.
        with torch.autocast(summary.device.type, enabled=False):
            value = self.critic(torch.cat([summary, h], dim=-1).float()).squeeze(-1)
        phase = obs["global_i"][:, 0]
        with self._sec("heads"):
            res = self._heads(ctx, ptr, hand, masks, phase, actions, deterministic, generator)
        res["value"] = value
        res["hidden"] = hidden_out
        res["phase"] = phase
        return res

    # ------------------------------------------------------------------ public API

    @torch.no_grad()
    def act(self, obs, masks, hidden, deterministic=False, first=None, generator=None):
        """Sample (or argmax) one structured action per env.

        Returns ``(actions int64 (n, 4), logprob (n,), entropy (n,), value (n,), new_hidden)``;
        ``actions = [kind, arg0, arg1, card_mask]`` ready for ``env.step`` (mask-conforming by
        construction). ``logprob`` is the sum of the kind, arg0, arg1 and card-step terms."""
        r = self._run(obs, masks, hidden, first, None, deterministic, generator)
        return r["actions"], r["logp"], r["entropy"], r["value"], r["hidden"]

    def evaluate(self, obs, masks, hidden_in, actions, first=None):
        """Teacher-forced ``(logprob, entropy, value)`` of ``actions`` (see the module docstring
        for the ``(T, n)`` sequence form). ``evaluate(act(...))`` reproduces the logprob."""
        r = self.evaluate_full(obs, masks, hidden_in, actions, first)
        return r["logp"], r["entropy"], r["value"]

    def evaluate_full(self, obs, masks, hidden_in, actions, first=None) -> dict[str, torch.Tensor]:
        """``evaluate`` plus per-part log-probabilities / argmax hits (``lp_*``, ``hit_*``,
        ``has_*``, ``ok`` = the action is expressible by the masks), ``hidden`` (the state after
        the last step) and ``phase``."""
        return self._run(obs, masks, hidden_in, first, actions, False, None)

    def value(self, obs, masks, hidden, first=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Critic only (bootstrap value); returns ``(value, new_hidden)``."""
        obs, masks = self._inputs(obs, masks)
        with torch.no_grad():
            summary, _, _ = self.encoder(obs)
            h, hidden_out = self._memory(summary, hidden, first)
            v = self.critic(torch.cat([summary, h], dim=-1)).squeeze(-1)
        return v, hidden_out


def save_checkpoint(path, model: TeamModel, **extra) -> None:
    torch.save(
        {
            "interface_version": INTERFACE_VERSION,
            "config": dataclasses.asdict(model.config),
            "state_dict": model.state_dict(),
            **extra,
        },
        path,
    )


def load_checkpoint(path, device="cpu") -> tuple[TeamModel, dict]:
    """Load a checkpoint. A version-1 checkpoint is migrated exactly (``model.migrate``: the new
    inputs start with zero weights); anything else than the current version is refused. The
    returned dict is the (migrated) checkpoint, ``ck["interface_version"]`` is the current one and
    ``ck["migrated_from"]`` is set when a migration happened."""
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck["interface_version"] == 1 and INTERFACE_VERSION == 2:
        ck = migrate_v1_to_v2(ck)
    assert ck["interface_version"] == INTERFACE_VERSION, "checkpoint is for another interface"
    model = TeamModel(ModelConfig(**ck["config"])).to(device)
    model.load_state_dict(ck["state_dict"])
    return model, ck
