"""Evaluation protocol: a fixed set of games, deterministic and sampled play, paired comparisons.

* **Test set.** ``TEST_SEED`` with ``games = 1000`` envs: game ``i`` is the first episode of env
  ``i`` (its seed is a pure function of ``(TEST_SEED, i, 0)``, independent of the number of envs
  and of the thread count). Training uses a different base seed (``train_seed`` asserts it), so
  the test games are never trained on. A smaller validation subset (the first ``k`` games) is
  literally a prefix of the test set; every policy plays the *same* games, which makes paired
  comparisons possible (``compare``).
* **Policies** are ``act(env, obs, masks, first, active) -> int64 (n, 4)`` callables;
  ``ModelActor`` (model deterministic/sampled, optionally routed: the model plays only some
  phases and the heuristic teacher the rest), ``teacher_actor``, ``random_actor``. Only envs that
  have not yet finished their first game are *active*: the model is run on that subset, so the
  long tail of slow games is cheap.
* **Metrics** (``EvalResult``): mean ante (+ std, standard error), win rate, ante histogram,
  invalid-action rate, episode length, blinds beaten, truncation rate, plus the per-game arrays.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from balatro_rl import BalatroVecEnv
from balatro_rl.select import KIND_MOVE_JOKER as MOVE_JOKER
from balatro_rl.train.curriculum import Curriculum

TEST_SEED = 10_000_019
TEST_GAMES = 1000
MAX_EPISODE_STEPS = 3000


def train_seed(seed: int, resume: int = 0) -> int:
    """Base env seed for training runs; never equal to ``TEST_SEED``."""
    s = 20_000_003 + 1_000_003 * int(seed) + 7_919 * int(resume)
    assert s != TEST_SEED
    return s


@dataclass
class EvalResult:
    name: str
    games: int
    mean_ante: float
    std_ante: float
    sem_ante: float
    win_rate: float
    truncated: float
    mean_blinds: float
    mean_steps: float
    invalid_per_game: float
    invalid_rate: float  # invalid actions per env step
    seconds: float
    ante_hist: dict[int, int]
    antes: list[int] = field(default_factory=list, repr=False)  # per game (index = env)
    wins: list[int] = field(default_factory=list, repr=False)

    def line(self) -> str:
        return (
            f"{self.name:<24} games={self.games} ante={self.mean_ante:.2f}+-{self.sem_ante:.2f} "
            f"win={100 * self.win_rate:.1f}% blinds={self.mean_blinds:.1f} "
            f"steps={self.mean_steps:.0f} "
            f"trunc={100 * self.truncated:.1f}% invalid/step={self.invalid_rate:.4f} "
            f"({self.seconds:.0f}s)"
        )

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("antes")
        d.pop("wins")
        return d


def compare(a: EvalResult, b: EvalResult) -> dict:
    """Paired difference ``a - b`` over the shared game seeds (same index = same game)."""
    n = min(len(a.antes), len(b.antes))
    da = np.array(a.antes[:n], dtype=float) - np.array(b.antes[:n], dtype=float)
    dw = np.array(a.wins[:n], dtype=float) - np.array(b.wins[:n], dtype=float)
    return {
        "games": n,
        "ante_diff": float(da.mean()),
        "ante_diff_sem": float(da.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
        "win_diff": float(dw.mean()),
        "win_diff_sem": float(dw.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
    }


# --------------------------------------------------------------------------- actors


def random_actor(env, obs, masks, first, active):
    return env.random_actions()


def teacher_actor(env, obs, masks, first, active):
    return env.teacher_actions(active)


class ModelActor:
    """Model policy (deterministic or sampled). ``route``: ``None`` = the model plays every
    phase, else a :class:`Curriculum` stage spec / route table; the teacher plays the others."""

    def __init__(
        self,
        model,
        deterministic: bool,
        seed: int = 0,
        route: str | None = None,
        move_limit: int | None = None,
    ):
        import torch

        self.torch = torch
        self.model = model.eval()
        self.deterministic = deterministic
        self.gen = torch.Generator(device=model.device).manual_seed(seed)
        self.hidden = None
        self.route = None if route in (None, "all") else Curriculum(route).stages[0].route_table()
        self.zeros = None
        # Repetition guard (evaluation only): after ``move_limit`` consecutive MoveJoker actions of
        # a game, MoveJoker is masked for the next decision (the count restarts on any other
        # action or a new game). Deterministic; ``None`` = off (the model is judged as it is).
        self.move_limit = move_limit
        self.moves = None

    def __call__(self, env, obs, masks, first, active):
        torch = self.torch
        n = env.num_envs
        dev = self.model.device
        if self.hidden is None:
            self.hidden = self.model.init_hidden(n)
            self.zeros = np.zeros((n, 4), dtype=np.int64)
            self.moves = np.zeros(n, dtype=np.int64)
        idx = np.flatnonzero(active)
        out = self.zeros.copy()
        if len(idx) == 0:
            return out
        sub = len(idx) < n
        if sub:
            o = {k: v[idx] for k, v in obs.items()}
            m = {k: v[idx] for k, v in masks.items()}
            idx_t = torch.from_numpy(idx).to(dev)
            h = self.hidden[idx_t]
            f = torch.from_numpy(first[idx]).to(dev)
        else:
            o, m, h = obs, masks, self.hidden
            f = torch.from_numpy(first).to(dev)
        if self.move_limit is not None:
            self.moves[first] = 0
            block = self.moves[idx] >= self.move_limit
            if block.any():
                km = m["kind_mask"].copy()
                other = km.copy()
                other[:, MOVE_JOKER] = False
                block &= other.any(axis=1)  # never leave a state without any legal kind
                km[block, MOVE_JOKER] = False
                m = {**m, "kind_mask": km}
        a, _, _, _, h2 = self.model.act(o, m, h, self.deterministic, first=f, generator=self.gen)
        if sub:
            self.hidden[idx_t] = h2
        else:
            self.hidden = h2
        out[idx] = a.cpu().numpy()
        if self.move_limit is not None:
            self.moves[idx] = np.where(out[idx, 0] == MOVE_JOKER, self.moves[idx] + 1, 0)
        if self.route is not None:
            use_teacher = active & ~self.route[obs["global_i"][:, 0]]
            if use_teacher.any():
                t = env.teacher_actions(use_teacher)
                out[use_teacher] = t[use_teacher]
        return out


# --------------------------------------------------------------------------- the game loop


def run_games(
    name: str,
    act,
    games: int = TEST_GAMES,
    seed: int = TEST_SEED,
    max_episode_steps: int = MAX_EPISODE_STEPS,
    threads: int = 0,
    deck: str = "red",
    stake: str = "white",
) -> EvalResult:
    """Play the first episode of each of ``games`` envs and summarise (see the module docs).
    ``deck`` / ``stake``: the run configuration of every game (default Red / White, the
    configuration all existing baselines and validation numbers refer to)."""
    t0 = time.perf_counter()
    env = BalatroVecEnv(
        games,
        seed=seed,
        num_threads=threads,
        max_episode_steps=max_episode_steps,
        deck=deck,
        stake=stake,
    )
    obs, masks = env.reset()
    first = np.ones(games, dtype=bool)
    active = np.ones(games, dtype=bool)
    finished: dict[int, tuple] = {}
    while len(finished) < games:
        actions = act(env, obs, masks, first, active)
        obs, masks, _, done, _ = env.step(actions)
        first = done != 0
        for rec in env.drain_finished():
            if rec[0] not in finished:
                finished[rec[0]] = rec
                active[rec[0]] = False
    recs = [finished[i] for i in range(games)]
    ante = np.array([r[1] for r in recs], dtype=np.float64)
    code = np.array([r[2] for r in recs])
    steps = np.array([r[3] for r in recs], dtype=np.float64)
    invalid = np.array([r[6] for r in recs], dtype=np.float64)
    return EvalResult(
        name=name,
        games=games,
        mean_ante=float(ante.mean()),
        std_ante=float(ante.std()),
        sem_ante=float(ante.std(ddof=1) / np.sqrt(games)) if games > 1 else float("nan"),
        win_rate=float((code == 2).mean()),
        truncated=float((code == 3).mean()),
        mean_blinds=float(np.mean([r[5] for r in recs])),
        mean_steps=float(steps.mean()),
        invalid_per_game=float(invalid.mean()),
        invalid_rate=float(invalid.sum() / max(1.0, steps.sum())),
        seconds=time.perf_counter() - t0,
        ante_hist={int(a): int((ante == a).sum()) for a in np.unique(ante)},
        antes=[int(a) for a in ante],
        wins=[int(c == 2) for c in code],
    )


def evaluate_model(
    model,
    games: int = TEST_GAMES,
    deterministic: bool = True,
    route: str | None = None,
    seed: int = TEST_SEED,
    actor_seed: int = 1,
    name: str | None = None,
    move_limit: int | None = None,
    **kw,
) -> EvalResult:
    actor = ModelActor(model, deterministic, actor_seed, route, move_limit)
    name = name or f"model_{'det' if deterministic else 'sample'}" + (f"[{route}]" if route else "")
    return run_games(name, actor, games, seed, **kw)
