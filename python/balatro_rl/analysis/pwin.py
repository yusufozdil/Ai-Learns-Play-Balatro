"""Calibrated P(win) head on frozen policy features: data collection, training, calibration.

The head is a small MLP on the critic's input (``[summary, GRU memory]``, 256 values for the
reference); the policy checkpoint is never touched. Data: decisions of the reference policy
(deterministic and sampled) with the final outcome of the game as the label.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from balatro_rl import BalatroVecEnv
from balatro_rl.analysis.features import (
    PHASE_BLIND,
    PHASE_SHOP,
    PolicyRunner,
    obs_ante,
    obs_phase,
)

HEAD_VERSION = 1


class PWinHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.in_dim, self.hidden = in_dim, hidden
        self.register_buffer("mu", torch.zeros(in_dim))
        self.register_buffer("sd", torch.ones(in_dim))
        self.register_buffer("temperature", torch.ones(()))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x.float() - self.mu) / self.sd).squeeze(-1)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calibrated P(win) (temperature-scaled sigmoid)."""
        return torch.sigmoid(self.logits(x) / self.temperature)


def save_head(path, head: PWinHead, **meta) -> None:
    torch.save(
        {
            "version": HEAD_VERSION,
            "in_dim": head.in_dim,
            "hidden": head.hidden,
            "state_dict": head.state_dict(),
            "meta": meta,
        },
        path,
    )


def load_head(path, device="cpu") -> PWinHead:
    ck = torch.load(path, map_location=device, weights_only=False)
    assert ck["version"] == HEAD_VERSION
    head = PWinHead(ck["in_dim"], ck["hidden"])
    head.load_state_dict(ck["state_dict"])
    return head.to(device).eval()


# --------------------------------------------------------------------------- data collection


def collect_shard(
    model,
    out: Path,
    games: int,
    seed: int,
    deterministic: bool,
    keep_other: float = 0.1,
    actor_seed: int = 0,
    max_episode_steps: int = 3000,
    game_offset: int = 0,
    env_kwargs: dict | None = None,
) -> dict:
    """Play the first episode of ``games`` envs (base ``seed``) with the policy and write one npz
    shard: ``feats`` (float16), ``value`` (critic), ``phase``, ``ante``, ``game`` (unique id),
    ``win`` (final outcome), ``det``. Every BlindSelect/Shop decision is kept, other phases with
    probability ``keep_other``. ``env_kwargs``: extra ``BalatroVecEnv`` arguments (a ``decks`` /
    ``stakes`` mix for a head of a stake-trained policy)."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(actor_seed + 12345)
    runner = PolicyRunner(model, seed=actor_seed)
    env = BalatroVecEnv(
        games, seed=seed, max_episode_steps=max_episode_steps, **(env_kwargs or {})
    )
    obs, masks = env.reset()
    hidden = runner.init_hidden(games)
    active = np.ones(games, bool)
    finished = np.zeros(games, bool)
    win = np.zeros(games, np.int8)
    f_feats, f_val, f_phase, f_ante, f_game = [], [], [], [], []
    while active.any():
        idx = np.flatnonzero(active)
        sub = len(idx) < games
        if sub:
            o = {k: v[idx] for k, v in obs.items()}
            m = {k: v[idx] for k, v in masks.items()}
            idx_t = torch.from_numpy(idx).to(runner.dev)
            h = hidden[idx_t]
        else:
            o, m, h = obs, masks, hidden
        act, h2, feats, value = runner.run(o, m, h, deterministic)
        if sub:
            hidden[idx_t] = h2
        else:
            hidden = h2
        ph = obs_phase(o)
        keep = (ph == PHASE_BLIND) | (ph == PHASE_SHOP) | (rng.random(len(idx)) < keep_other)
        if keep.any():
            kt = torch.from_numpy(np.flatnonzero(keep)).to(runner.dev)
            f_feats.append(feats[kt].half().cpu().numpy())
            f_val.append(value[kt].float().cpu().numpy())
            f_phase.append(ph[keep].astype(np.int8))
            f_ante.append(obs_ante(o)[keep].astype(np.int8))
            f_game.append(idx[keep].astype(np.int64))
        full = np.zeros((games, 4), np.int64)
        full[idx] = act.cpu().numpy()
        obs, masks, _, _done, _ = env.step(full)
        for rec in env.drain_finished():
            e = rec[0]
            if not finished[e]:
                finished[e] = True
                active[e] = False
                win[e] = int(rec[2] == 2)
    game = np.concatenate(f_game)
    np.savez(
        out,
        feats=np.concatenate(f_feats),
        value=np.concatenate(f_val),
        phase=np.concatenate(f_phase),
        ante=np.concatenate(f_ante),
        game=game + game_offset,
        win=win[game],
        det=np.full(len(game), int(deterministic), np.int8),
    )
    return {
        "rows": int(len(game)),
        "games": games,
        "win_rate": float(win.mean()),
        "seconds": time.perf_counter() - t0,
    }


# --------------------------------------------------------------------------- metrics


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """ROC AUC (rank statistic, ties get their average rank)."""
    y = y.astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    _, start, counts = np.unique(s, return_index=True, return_counts=True)
    avg = start + (counts + 1) / 2.0  # average 1-based rank of each tie group
    ranks_sorted = np.repeat(avg, counts)
    ranks = np.empty(len(s))
    ranks[order] = ranks_sorted
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10) -> dict:
    b = np.clip((p * bins).astype(int), 0, bins - 1)
    rows, ece = [], 0.0
    for k in range(bins):
        mk = b == k
        n = int(mk.sum())
        if n == 0:
            rows.append({"bin": k, "n": 0, "p": None, "y": None})
            continue
        pm, ym = float(p[mk].mean()), float(y[mk].mean())
        ece += n / len(p) * abs(pm - ym)
        rows.append({"bin": k, "n": n, "p": pm, "y": ym})
    return {"ece": float(ece), "bins": rows}


def metrics(p: np.ndarray, y: np.ndarray) -> dict:
    base = float(y.mean())
    return {
        "n": int(len(y)),
        "win_rate": base,
        "logloss": logloss(p, y),
        "logloss_const": logloss(np.full_like(p, base), y),
        "auc": auc(p, y),
        "ece": reliability(p, y)["ece"],
        "brier": float(((p - y) ** 2).mean()),
    }


# --------------------------------------------------------------------------- training


def load_shards(paths) -> dict[str, np.ndarray]:
    parts = [np.load(p) for p in paths]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}


def split_games(game: np.ndarray, frac=(0.8, 0.1, 0.1)) -> np.ndarray:
    """0 train / 1 val (early stopping + temperature) / 2 test (calibration report), by game id
    (a hash, so all decisions of a game stay together)."""
    h = (game.astype(np.uint64) + np.uint64(1)) * np.uint64(0x9E3779B97F4A7C15)
    h ^= h >> np.uint64(29)
    h *= np.uint64(0xBF58476D1CE4E5B9)
    u = (h >> np.uint64(11)).astype(np.float64) / float(1 << 53)
    return np.where(u < frac[0], 0, np.where(u < frac[0] + frac[1], 1, 2))


def train_head(
    data: dict[str, np.ndarray],
    device: str = "cuda",
    epochs: int = 6,
    batch: int = 4096,
    lr: float = 3e-4,
    seed: int = 0,
    log=print,
) -> tuple[PWinHead, dict]:
    torch.manual_seed(seed)
    split = split_games(data["game"])
    x = torch.from_numpy(data["feats"]).to(device)  # float16 on device
    y = torch.from_numpy(data["win"].astype(np.float32)).to(device)
    tr = torch.from_numpy(np.flatnonzero(split == 0)).to(device)
    va = torch.from_numpy(np.flatnonzero(split == 1)).to(device)
    head = PWinHead(x.shape[1]).to(device)
    xs = x[tr[:200_000]].float()
    head.mu.copy_(xs.mean(0))
    head.sd.copy_(xs.std(0).clamp_min(1e-3))
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-2)
    steps = epochs * max(1, len(tr) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    bce = nn.BCEWithLogitsLoss()
    best, best_sd = 1e9, None
    for ep in range(epochs):
        head.train()
        perm = tr[torch.randperm(len(tr), device=device)]
        for i in range(0, len(perm) - batch + 1, batch):
            b = perm[i : i + batch]
            loss = bce(head.logits(x[b]), y[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        head.eval()
        with torch.no_grad():
            lv = bce(head.logits(x[va]), y[va]).item()
        log(f"epoch {ep + 1}/{epochs} val bce {lv:.4f}")
        if lv < best:
            best, best_sd = lv, {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_sd)
    head.eval()
    # temperature scaling on the validation games (one scalar, LBFGS on log T)
    with torch.no_grad():
        zv = head.logits(x[va])
    logt = torch.zeros((), device=device, requires_grad=True)
    o2 = torch.optim.LBFGS([logt], lr=0.5, max_iter=50)

    def closure():
        o2.zero_grad()
        ls = bce(zv / logt.exp(), y[va])
        ls.backward()
        return ls

    o2.step(closure)
    head.temperature.fill_(float(logt.detach().exp()))
    return head, {"val_bce": best, "temperature": float(head.temperature), "split": split}


@torch.no_grad()
def predict(head: PWinHead, feats: np.ndarray, device="cuda", batch=65536) -> np.ndarray:
    head = head.to(device).eval()
    out = []
    for i in range(0, len(feats), batch):
        out.append(head(torch.from_numpy(feats[i : i + batch]).to(device)).cpu().numpy())
    return np.concatenate(out)


def calibrated_critic(data, split) -> np.ndarray:
    """Baseline: logistic regression of the win on the raw critic value and a per-ante intercept
    (fitted on the validation games); returns its P(win) on the test rows."""
    ante = data["ante"].astype(np.int64).clip(0, 9)
    x = np.concatenate(
        [data["value"][:, None] / 20.0, np.eye(10, dtype=np.float32)[ante]], axis=1
    ).astype(np.float32)
    xt = torch.from_numpy(x)
    y = torch.from_numpy(data["win"].astype(np.float32))
    va = torch.from_numpy(np.flatnonzero(split == 1))
    w = torch.zeros(x.shape[1], requires_grad=True)
    opt = torch.optim.LBFGS([w], lr=0.5, max_iter=200)

    def closure():
        opt.zero_grad()
        ls = nn.functional.binary_cross_entropy_with_logits(xt[va] @ w, y[va]) + 1e-4 * (w**2).sum()
        ls.backward()
        return ls

    opt.step(closure)
    with torch.no_grad():
        return torch.sigmoid(xt[torch.from_numpy(np.flatnonzero(split == 2))] @ w).numpy()


def evaluate_head(head: PWinHead, data, split, device="cuda") -> dict:
    """Metrics on the test games: overall, by ante, by phase, by mode; raw critic as baseline."""
    te = split == 2
    y = data["win"][te].astype(np.float64)
    p = predict(head, data["feats"][te], device).astype(np.float64)
    v = data["value"][te].astype(np.float64)
    ante, ph, det = data["ante"][te], data["phase"][te], data["det"][te]
    out = {
        "overall": metrics(p, y),
        "critic_auc": auc(v, y),
        "calibrated_critic": metrics(calibrated_critic(data, split).astype(np.float64), y),
        "reliability": reliability(p, y),
        "by_ante": {},
        "by_phase": {},
        "by_mode": {},
    }

    def add(store, key, k):
        if k.sum() > 50:
            m = metrics(p[k], y[k])
            m["critic_auc"] = auc(v[k], y[k])
            store[key] = m

    for a in sorted(np.unique(ante)):
        add(out["by_ante"], int(a), ante == a)
    for q, name in ((0, "BlindSelect"), (1, "Play"), (2, "Shop"), (3, "Pack")):
        add(out["by_phase"], name, ph == q)
    for d, name in ((1, "det"), (0, "sampled")):
        add(out["by_mode"], name, det == d)
    return out
