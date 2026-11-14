"""Thin Python wrapper around the Rust ``VecEnv`` (Gymnasium-like, no gymnasium dependency).

TEMPORARY interface (Phase 7B). Phase 8 replaces the observation (``OBS_DIM`` summary vector)
and the action space (index into the *current* legal-action list) with the real encoding and a
fixed-size action codec + mask; Phase 10 replaces the placeholder reward. The buffer plumbing
(preallocated numpy arrays filled in place by Rust) is meant to stay.
"""

from __future__ import annotations

import numpy as np

from balatro_rl import _engine

OBS_DIM: int = _engine.OBS_DIM
DONE_RUNNING: int = _engine.DONE_RUNNING
DONE_LOSS: int = _engine.DONE_LOSS
DONE_WIN: int = _engine.DONE_WIN
DONE_TRUNCATED: int = _engine.DONE_TRUNCATED


class BalatroVecEnv:
    """Batched environment owning its output buffers.

    ``reset()`` and ``step(actions)`` return *views of internal buffers* that are overwritten by
    the next call; copy them if you need to keep them. Finished envs auto-reset: after a step with
    ``done[i] != 0``, ``obs[i]`` is the first observation of the next episode.

    ``done`` codes: 0 running, 1 lost, 2 won, 3 truncated (``max_episode_steps``).
    ``legal_count[i]`` is the number of valid action indices for env ``i`` (actions are indices
    into the env's current legal-action list; invalid indices raise ``ValueError``).
    """

    def __init__(
        self,
        num_envs: int,
        seed: int = 0,
        num_threads: int = 0,
        max_episode_steps: int = 10_000,
        deck: str = "red",
        stake: str = "white",
        endless: bool = False,
    ) -> None:
        self._env = _engine.VecEnv(
            num_envs, seed, num_threads, max_episode_steps, deck, stake, endless
        )
        n = num_envs
        self.num_envs = n
        self.obs = np.zeros((n, OBS_DIM), dtype=np.float32)
        self.reward = np.zeros(n, dtype=np.float32)
        self.done = np.zeros(n, dtype=np.uint8)
        self.legal_count = np.zeros(n, dtype=np.int32)
        self._actions = np.zeros(n, dtype=np.int64)

    @property
    def num_threads(self) -> int:
        return self._env.num_threads

    def set_num_threads(self, num_threads: int) -> None:
        self._env.set_num_threads(num_threads)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Reset every env to episode 0; returns ``(obs, legal_count)`` buffer views."""
        self._env.reset(self.obs, self.legal_count, seed)
        return self.obs, self.legal_count

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Step with ``actions`` (int64 ``(n,)``); returns ``(obs, reward, done, legal_count)``."""
        actions = np.ascontiguousarray(actions, dtype=np.int64)
        self._env.step(actions, self.obs, self.reward, self.done, self.legal_count)
        return self.obs, self.reward, self.done, self.legal_count

    def random_actions(self) -> np.ndarray:
        """Uniform random legal index per env (buffer view; no Python loop)."""
        self._env.random_actions(self._actions)
        return self._actions

    def drain_finished(self) -> list[tuple[int, int, int, int, float, int]]:
        """Episodes finished since the last call: ``(env, ante, done_code, steps, return,
        blinds_beaten)``."""
        return self._env.drain_finished()
