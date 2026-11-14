"""Thin Python wrapper around the Rust ``VecEnv`` (Gymnasium-like, no gymnasium dependency).

The interface (token observation, structured action ``[kind, arg0, arg1, card_mask]``, masks,
reward) is specified in ``docs/design/rl_interface.md``; ``_engine.layout()`` returns the buffer
names, dtypes, shapes, feature names and vocabulary sizes. This wrapper owns every buffer; Rust
fills them in place (GIL released), so ``reset`` / ``step`` return *views* that the next call
overwrites. Copy what you need to keep.
"""

from __future__ import annotations

import numpy as np

from balatro_rl import _engine

INTERFACE_VERSION: int = _engine.INTERFACE_VERSION
DONE_RUNNING: int = _engine.DONE_RUNNING
DONE_LOSS: int = _engine.DONE_LOSS
DONE_WIN: int = _engine.DONE_WIN
DONE_TRUNCATED: int = _engine.DONE_TRUNCATED

LAYOUT: dict = _engine.layout()
"""Capacities, buffer specs, feature names, vocabularies and pointer targets."""


def allocate_buffers(num_envs: int) -> dict[str, np.ndarray]:
    """Zeroed caller-owned arrays for ``num_envs`` envs (every buffer of the interface)."""
    return {
        name: np.zeros((num_envs, *shape), dtype=np.dtype(dtype))
        for name, dtype, shape in LAYOUT["buffers"]
    }


class BalatroVecEnv:
    """Batched environment owning its buffers.

    * ``reset()`` returns ``(obs, masks)``; ``step(actions)`` returns
      ``(obs, masks, reward, done, invalid)``. ``obs`` and ``masks`` are ``dict[str, ndarray]``
      (see ``LAYOUT["obs_names"]`` / ``LAYOUT["mask_names"]``), all views of internal buffers.
    * ``actions`` is int64 ``(num_envs, 4)`` = ``[kind, arg0, arg1, card_mask]``; build it from
      the masks (see ``balatro_rl.select``). A malformed or illegal action is a no-op:
      ``invalid[i] = 1`` and the reward is ``reward["invalid"]``.
    * Finished envs auto-reset: after a step with ``done[i] != 0``, ``obs[*][i]`` and
      ``masks[*][i]`` are the first state of the next episode and ``reward[i]`` belongs to the
      finishing transition. ``done``: 0 running, 1 lost, 2 won, 3 truncated.
    * ``reward`` (constructor) overrides terms of the default reward: ``blind``, ``ante``,
      ``win``, ``loss``, ``shaping``, ``invalid``, ``reorder``.
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
        reward: dict[str, float] | None = None,
    ) -> None:
        self._env = _engine.VecEnv(
            num_envs, seed, num_threads, max_episode_steps, deck, stake, endless, reward
        )
        self.num_envs = num_envs
        self._buffers = allocate_buffers(num_envs)
        self.obs = {k: self._buffers[k] for k in LAYOUT["obs_names"]}
        self.masks = {k: self._buffers[k] for k in LAYOUT["mask_names"]}
        self.reward = self._buffers["reward"]
        self.done = self._buffers["done"]
        self.invalid = self._buffers["invalid"]
        self._actions = np.zeros((num_envs, 4), dtype=np.int64)

    @property
    def num_threads(self) -> int:
        return self._env.num_threads

    def set_num_threads(self, num_threads: int) -> None:
        self._env.set_num_threads(num_threads)

    def reset(self, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Reset every env to episode 0; returns ``(obs, masks)`` buffer views."""
        self._env.reset(self._buffers, seed)
        return self.obs, self.masks

    def step(
        self, actions: np.ndarray
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        """Step with ``actions`` (int64 ``(n, 4)``); returns
        ``(obs, masks, reward, done, invalid)``."""
        actions = np.ascontiguousarray(actions, dtype=np.int64)
        self._env.step(actions, self._buffers)
        return self.obs, self.masks, self.reward, self.done, self.invalid

    def random_actions(self) -> np.ndarray:
        """Uniformly random mask-conforming action per env (buffer view; no Python loop)."""
        self._env.random_actions(self._actions)
        return self._actions

    def teacher_actions(self, select: np.ndarray | None = None) -> np.ndarray:
        """The action of the fair-information heuristic team agent in each env's current state
        (buffer view; parallel, GIL released). With ``select`` (bool ``(n,)``) only those envs
        are computed (a curriculum asks the teacher about a fraction of the envs); the other
        rows of the returned buffer are stale and must not be used."""
        if select is not None:
            select = np.ascontiguousarray(select, dtype=bool)
        self._env.teacher_actions(self._actions, select)
        return self._actions

    def drain_finished(self) -> list[tuple[int, int, int, int, float, int, int]]:
        """Episodes finished since the last call: ``(env, ante, done_code, steps, return,
        blinds_beaten, invalid_actions)``."""
        return self._env.drain_finished()
