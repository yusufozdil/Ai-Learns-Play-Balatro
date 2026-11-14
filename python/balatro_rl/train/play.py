"""Policy evaluation in the env: play one full game per env (first episode of every env, so the
statistics are not biased towards short games) for random / teacher / model policies."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from balatro_rl import BalatroVecEnv

EVAL_SEED = 10_000_019


@dataclass
class GameStats:
    name: str
    games: int
    mean_ante: float
    std_ante: float
    win_rate: float
    truncated: float
    mean_blinds: float
    invalid_per_game: float
    seconds: float
    ante_hist: dict[int, int]

    def line(self) -> str:
        return (
            f"{self.name:<22} games={self.games} ante={self.mean_ante:.2f}+-{self.std_ante:.2f} "
            f"win={100 * self.win_rate:.1f}% blinds={self.mean_blinds:.1f} "
            f"truncated={100 * self.truncated:.1f}% invalid/game={self.invalid_per_game:.2f} "
            f"({self.seconds:.0f}s)"
        )


def play_games(
    name: str,
    action_fn,
    games: int = 1024,
    seed: int = EVAL_SEED,
    max_episode_steps: int = 3000,
    threads: int = 0,
    on_step=None,
) -> GameStats:
    """``action_fn(env, obs, masks, first) -> int64 (n, 4)`` (``first`` = bool (n,) marking envs
    whose last step finished an episode, i.e. fresh episodes). Runs until every env finished its
    first episode."""
    t0 = time.perf_counter()
    env = BalatroVecEnv(games, seed=seed, num_threads=threads, max_episode_steps=max_episode_steps)
    obs, masks = env.reset()
    first = np.ones(games, dtype=bool)
    finished: dict[int, tuple] = {}
    while len(finished) < games:
        actions = action_fn(env, obs, masks, first)
        obs, masks, _, done, _ = env.step(actions)
        first = done != 0
        for rec in env.drain_finished():
            finished.setdefault(rec[0], rec)
    recs = [finished[i] for i in range(games)]
    ante = np.array([r[1] for r in recs], dtype=np.float64)
    code = np.array([r[2] for r in recs])
    hist = {int(a): int((ante == a).sum()) for a in np.unique(ante)}
    return GameStats(
        name=name,
        games=games,
        mean_ante=float(ante.mean()),
        std_ante=float(ante.std()),
        win_rate=float((code == 2).mean()),
        truncated=float((code == 3).mean()),
        mean_blinds=float(np.mean([r[5] for r in recs])),
        invalid_per_game=float(np.mean([r[6] for r in recs])),
        seconds=time.perf_counter() - t0,
        ante_hist=hist,
    )


def random_policy(env, obs, masks, first):
    return env.random_actions()


def teacher_policy(env, obs, masks, first):
    return env.teacher_actions()


class ModelPolicy:
    """Wraps a ``TeamModel`` (hidden state kept per env, reset on episode start)."""

    def __init__(self, model, deterministic: bool, seed: int = 0):
        import torch

        self.model = model.eval()
        self.deterministic = deterministic
        self.hidden = None
        self.gen = torch.Generator(device=model.device).manual_seed(seed)
        self._torch = torch

    def __call__(self, env, obs, masks, first):
        torch = self._torch
        if self.hidden is None:
            self.hidden = self.model.init_hidden(env.num_envs)
        f = torch.from_numpy(first).to(self.model.device)
        a, _, _, _, self.hidden = self.model.act(
            obs, masks, self.hidden, self.deterministic, first=f, generator=self.gen
        )
        return a.cpu().numpy()
