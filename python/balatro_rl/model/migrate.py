"""Checkpoint / data migration between interface versions (``docs/design/rl_interface.md``).

**v1 -> v2** (Phase 10G) only *appended* inputs: 11 ``global_f`` scalars (stake flags, deck
economy, base hands/discards), the two ``global_i`` columns ``deck`` / ``stake`` and their embedding
tables. Everything is matched **by name**: the v1 name lists below are the v2 lists minus the added
names, so a reordered v2 layout would still migrate correctly. The new input columns of the first
``global_proj`` Linear get zero weights and the new embedding rows are zero, hence a migrated model
computes *exactly* the function of the v1 model (the new inputs have no influence until trained).
The Adam moments of a resumable checkpoint are padded the same way (zero moments).
"""

from __future__ import annotations

import numpy as np
import torch

from balatro_rl.env import LAYOUT
from balatro_rl.model import encoder as enc

SUPPORTED_VERSIONS = (1, 2)

# Names added by interface version 2.
ADDED_GLOBAL_F = (
    "stake_idx",
    "no_small_reward",
    "eternal_on",
    "perishable_on",
    "rental_on",
    "plasma_blind_x2",
    "base_hands",
    "base_discards",
    "no_interest",
    "money_per_hand",
    "money_per_discard",
)
ADDED_GLOBAL_I = ("deck", "stake")

_GLOBAL_PROJ_W = "encoder.global_proj.0.weight"
_TABLE_W = "encoder.table.weight"


def v1_names(group: str) -> list[str]:
    """Column names of ``group`` in interface version 1 (v2 names minus the added ones)."""
    names = list(LAYOUT["feature_names"][group])
    drop = {"global_f": ADDED_GLOBAL_F, "global_i": ADDED_GLOBAL_I}.get(group, ())
    return [n for n in names if n not in drop]


def v1_table_sizes() -> dict[str, int]:
    """Embedding tables of version 1 (name -> vocabulary size), in their v1 order."""
    return {t: n for t, n in enc.TABLE_SIZES.items() if t not in enc.NEW_TABLES_V2}


def _remap_global_proj(w: torch.Tensor) -> torch.Tensor:
    old, new = v1_names("global_f"), list(LAYOUT["feature_names"]["global_f"])
    assert w.shape[-1] == len(old), f"global_proj input is {w.shape[-1]} wide, v1 has {len(old)}"
    out = w.new_zeros(*w.shape[:-1], len(new))
    out[..., [new.index(n) for n in old]] = w
    return out


def _remap_table(w: torch.Tensor) -> torch.Tensor:
    old_sizes = v1_table_sizes()
    assert w.shape[0] == sum(old_sizes.values()), "embedding table is not a v1 table"
    old_off, new_off = enc.table_offsets(old_sizes), enc.table_offsets()
    out = w.new_zeros(sum(enc.TABLE_SIZES.values()), *w.shape[1:])
    for name, size in old_sizes.items():
        out[new_off[name] : new_off[name] + size] = w[old_off[name] : old_off[name] + size]
    return out


_REMAP = {_GLOBAL_PROJ_W: _remap_global_proj, _TABLE_W: _remap_table}


def migrate_v1_to_v2(ck: dict) -> dict:
    """In-place conversion of a loaded v1 checkpoint dict to interface version 2 (also returned).

    ``state_dict`` and, when present, the Adam moments in ``optimizer`` are remapped; the other
    entries are untouched. ``ck["migrated_from"] = 1`` records it (dropped by the next save)."""
    assert ck["interface_version"] == 1
    sd = ck["state_dict"] = dict(ck["state_dict"])
    shapes = {}
    for name, fn in _REMAP.items():
        shapes[name] = tuple(sd[name].shape)
        sd[name] = fn(sd[name])
    opt = ck.get("optimizer")
    if opt is not None:
        # Adam keeps per-parameter state keyed by the parameter's index in ``model.parameters()``;
        # v2 adds no parameter, so the index of a name is its position in a fresh model's list.
        from balatro_rl.model.team import ModelConfig, TeamModel

        model = TeamModel(ModelConfig(**ck["config"]))
        order = [n for n, _ in model.named_parameters()]
        for name, fn in _REMAP.items():
            idx = order.index(name)
            st = opt["state"].get(idx)
            if st is None:
                continue
            for k, v in st.items():
                if isinstance(v, torch.Tensor) and tuple(v.shape) == shapes[name]:
                    st[k] = fn(v)
    ck["interface_version"] = 2
    ck["migrated_from"] = 1
    return ck


# --------------------------------------------------------------------------- stored data


def v1_data_width() -> int:
    return len(v1_names("global_f"))


def upgrade_v1_global_f(a: np.ndarray) -> np.ndarray:
    """Append the v2 ``global_f`` columns to v1 data (any leading shape; returns a new array).

    A v1 dataset has no record of its run configuration; it was collected on Red/White, so the
    appended columns hold the Red/White values at the start of a run (4 base hands / 4 base
    discards, $1 per unused hand, everything else 0). ``base_hands`` / ``base_discards`` drift from
    that when a voucher changes them; that part is not recoverable."""
    names = list(LAYOUT["feature_names"]["global_f"])
    out = np.zeros((*a.shape[:-1], len(names)), dtype=a.dtype)
    out[..., [names.index(n) for n in v1_names("global_f")]] = a
    out[..., names.index("base_hands")] = 4 / 8
    out[..., names.index("base_discards")] = 4 / 8
    out[..., names.index("money_per_hand")] = 1 / 2
    return out


def upgrade_v1_global_i(a: np.ndarray) -> np.ndarray:
    """Append the v2 ``global_i`` columns (``deck`` = Red, ``stake`` = White = 0)."""
    cols = list(LAYOUT["feature_names"]["global_i"])
    out = np.zeros((*a.shape[:-1], len(cols)), dtype=a.dtype)
    out[..., [cols.index(n) for n in v1_names("global_i")]] = a
    return out


def upgrade_v1_obs_arrays(global_f: np.ndarray, global_i: np.ndarray):
    """``(upgrade_v1_global_f(global_f), upgrade_v1_global_i(global_i))``."""
    return upgrade_v1_global_f(global_f), upgrade_v1_global_i(global_i)
