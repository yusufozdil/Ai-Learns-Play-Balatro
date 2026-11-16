"""Multi-model actors for evaluation: D4 (ensemble vote) and D10 (cross-play head swap).

``EnsembleActor`` has the ``act(env, obs, masks, first, active) -> int64 (n, 4)`` contract of
``evaluate.ModelActor``. Every model keeps its OWN recurrent state for every env and is stepped
on every env step (also when it is not the one choosing), so each memory stays consistent with
the trajectory actually played.

* ``mode="vote"``: each model proposes its deterministic action; every candidate is scored by
  the mean over models of that model's log-probability of the full structured action (kind +
  arg0 + arg1 + card steps, the PPO ``evaluate`` quantity; ``vote="mean_prob"`` averages
  probabilities instead). Highest score wins, ties go to the lowest model index. A candidate a
  model cannot express (mask violation) scores ``-inf`` for that model.
* ``mode="route"``: ``route_models`` maps each phase to the model that plays it.
"""

from __future__ import annotations

import numpy as np

from balatro_rl.model.team import PHASE_TO_HEAD

PHASES = ("blind", "play", "shop", "pack")


def parse_route_models(spec: str | None, n_models: int) -> list[int]:
    """``"shop=1,blind=0"`` -> model index per head phase (blind, play, shop, pack); unnamed
    phases default to model 0."""
    out = [0] * len(PHASES)
    for part in (spec or "").replace(" ", "").split(","):
        if not part:
            continue
        name, _, idx = part.partition("=")
        if name not in PHASES or not idx.isdigit() or int(idx) >= n_models:
            raise ValueError(f"bad route-models entry {part!r} ({n_models} models)")
        out[PHASES.index(name)] = int(idx)
    return out


class EnsembleActor:
    def __init__(
        self,
        models,
        mode: str = "vote",
        vote: str = "mean_logp",
        route_models: list[int] | None = None,
        seed: int = 0,
    ):
        import torch

        assert mode in ("vote", "route") and vote in ("mean_logp", "mean_prob")
        self.torch = torch
        self.models = [m.eval() for m in models]
        self.mode, self.vote = mode, vote
        rm = route_models if route_models is not None else [0] * len(PHASES)
        # phase id (0..5) -> model index, through the head mapping (terminal phases -> blind)
        self.route = np.array([rm[h] for h in PHASE_TO_HEAD], dtype=np.int64)
        self.gens = [torch.Generator(device=m.device).manual_seed(seed) for m in self.models]
        self.hidden = None

    def __call__(self, env, obs, masks, first, active):
        torch = self.torch
        n = env.num_envs
        dev = self.models[0].device
        if self.hidden is None:
            self.hidden = [m.init_hidden(n) for m in self.models]
        out = np.zeros((n, 4), dtype=np.int64)
        idx = np.flatnonzero(active)
        if len(idx) == 0:
            return out
        sub = len(idx) < n
        if sub:
            o = {k: v[idx] for k, v in obs.items()}
            m = {k: v[idx] for k, v in masks.items()}
            idx_t = torch.from_numpy(idx).to(dev)
            f = torch.from_numpy(first[idx]).to(dev)
        else:
            o, m = obs, masks
            f = torch.from_numpy(first).to(dev)

        acts, lps, new_h, hin = [], [], [], []
        for k, model in enumerate(self.models):
            h = self.hidden[k][idx_t] if sub else self.hidden[k]
            a, lp, _, _, h2 = model.act(o, m, h, True, first=f, generator=self.gens[k])
            acts.append(a)
            lps.append(lp)
            new_h.append(h2)
            hin.append(h)

        if self.mode == "route":
            phase = o["global_i"][:, 0]
            pick = torch.from_numpy(self.route[phase]).to(a.device)
            stack = torch.stack(acts)  # (M, n, 4)
            chosen = stack[pick, torch.arange(len(pick), device=a.device)]
        elif len(self.models) == 1:
            chosen = acts[0]
        else:
            chosen = self._vote(o, m, f, acts, lps, hin)

        for k in range(len(self.models)):
            if sub:
                self.hidden[k][idx_t] = new_h[k]
            else:
                self.hidden[k] = new_h[k]
        out[idx] = chosen.cpu().numpy()
        return out

    def _vote(self, o, m, f, acts, lps, hin):
        torch = self.torch
        n_m = len(self.models)
        # score[k, j, row]: model k's log-prob of candidate j (own candidate: from act())
        score = []
        for k, model in enumerate(self.models):
            row = []
            for j in range(n_m):
                if j == k:
                    row.append(lps[k])
                else:
                    with torch.no_grad():
                        r = model.evaluate_full(o, m, hin[k], acts[j], first=f)
                    row.append(torch.where(r["ok"], r["logp"], torch.full_like(r["logp"], -1e30)))
            score.append(torch.stack(row))
        s = torch.stack(score).float()  # (M models, M candidates, n)
        if self.vote == "mean_prob":
            s = torch.where(s <= -1e29, torch.zeros_like(s), s.exp())
        total = s.mean(0)  # (M candidates, n)
        best = total.argmax(0)  # first maximum = lowest index
        stack = torch.stack(acts)
        return stack[best, torch.arange(best.numel(), device=best.device)]
