"""Observation / mask preparation: numpy (env buffers, stored datasets) -> torch tensors.

Conventions (``docs/design/rl_interface.md``): categorical ``*_i`` -> int64, continuous ``*_f``
-> float32, ``*_valid`` and the bool masks -> bool, ``uint32`` bit-set masks -> int64 (bit ``i`` =
hand position ``i``). Stored datasets may keep float16 / uint8 / int32-viewed masks; everything is
converted here. Always copies (the env buffers are overwritten by the next step).
"""

from __future__ import annotations

import numpy as np
import torch

from balatro_rl.env import LAYOUT

OBS_NAMES: tuple[str, ...] = tuple(LAYOUT["obs_names"])
MASK_NAMES: tuple[str, ...] = tuple(LAYOUT["mask_names"])
_BITSET = ("sel_candidates", "sel_required")


def _to(x, device, dtype=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x.to(device=device)
    else:
        t = torch.tensor(np.asarray(x), device=device)  # copies
    return t if dtype is None else t.to(dtype)


def prepare_obs(obs: dict, device) -> dict[str, torch.Tensor]:
    out = {}
    for name in OBS_NAMES:
        if name.endswith("_valid"):
            out[name] = _to(obs[name], device, torch.bool)
        elif name.endswith("_i"):
            out[name] = _to(obs[name], device, torch.long)
        else:
            out[name] = _to(obs[name], device, torch.float32)
    return out


def prepare_masks(masks: dict, device) -> dict[str, torch.Tensor]:
    out = {}
    for name in MASK_NAMES:
        x = masks[name]
        if name in _BITSET:
            if isinstance(x, np.ndarray):
                if x.dtype == np.int32:  # int32 view of a uint32 set
                    x = x.view(np.uint32)
                x = x.astype(np.int64)  # uint32 -> int64
                out[name] = _to(x, device)
            else:  # int32 view of a uint32 set (stored datasets) or already int64
                t = _to(x, device, torch.long)
                out[name] = t & 0xFFFFFFFF if x.dtype == torch.int32 else t
        elif name in ("sel_min", "sel_max"):
            out[name] = _to(x, device, torch.long)
        else:
            out[name] = _to(x, device, torch.bool)
    return out


def is_prepared(obs: dict) -> bool:
    return isinstance(obs[OBS_NAMES[0]], torch.Tensor) and obs["global_i"].dtype == torch.long


def prepare(obs: dict, masks: dict, device) -> tuple[dict, dict]:
    return prepare_obs(obs, device), prepare_masks(masks, device)
