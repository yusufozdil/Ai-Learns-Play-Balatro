"""Checkpoint / data migration between interface versions (``docs/design/rl_interface.md`` §12).

Every version so far only *appended* inputs:

* **v1 -> v2** (Phase 10G): 11 ``global_f`` scalars (stake flags, deck economy, base
  hands/discards), the two ``global_i`` columns ``deck`` / ``stake`` and their embedding tables.
* **v2 -> v3**: 6 ``global_f`` scalars (next round's hands / discards, run counters) and the 4
  ``joker_f`` contribution columns.

Everything is matched **by name**: the names of version ``k`` are the current lists minus the
names added after ``k``, so a reordered layout would still migrate correctly. The new input
columns of the first ``global_proj`` Linear (and of ``joker_proj``) get zero weights and the new
embedding rows are zero, hence a migrated model computes *exactly* the function of the old model
(the new inputs have no influence until trained). The Adam moments of a resumable checkpoint are
padded the same way (zero moments). A v1 checkpoint migrates in two steps (v1 -> v2 -> v3).
"""

from __future__ import annotations

import numpy as np
import torch

from balatro_rl.env import INTERFACE_VERSION, LAYOUT
from balatro_rl.model import encoder as enc

SUPPORTED_VERSIONS = (1, 2, 3)

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

# Names added by interface version 3.
ADDED_GLOBAL_F_V3 = (
    "next_hands",
    "next_discards",
    "tarots_used",
    "blinds_skipped",
    "unique_planets_used",
    "hands_played_run",
)
ADDED_JOKER_F_V3 = ("contrib_chips", "contrib_mult", "contrib_xmult", "contrib_money")

# version -> {group: names added by that version}
ADDED: dict[int, dict[str, tuple[str, ...]]] = {
    2: {"global_f": ADDED_GLOBAL_F, "global_i": ADDED_GLOBAL_I},
    3: {"global_f": ADDED_GLOBAL_F_V3, "joker_f": ADDED_JOKER_F_V3},
}

_GLOBAL_PROJ_W = "encoder.global_proj.0.weight"
_JOKER_PROJ_W = "encoder.joker_proj.weight"
_TABLE_W = "encoder.table.weight"


def names_at(version: int, group: str) -> list[str]:
    """Column names of ``group`` in interface ``version`` (current names minus later additions)."""
    drop = set()
    for v, groups in ADDED.items():
        if v > version:
            drop.update(groups.get(group, ()))
    return [n for n in LAYOUT["feature_names"][group] if n not in drop]


def v1_names(group: str) -> list[str]:
    """Column names of ``group`` in interface version 1."""
    return names_at(1, group)


def v2_names(group: str) -> list[str]:
    """Column names of ``group`` in interface version 2."""
    return names_at(2, group)


def v1_table_sizes() -> dict[str, int]:
    """Embedding tables of version 1 (name -> vocabulary size), in their v1 order."""
    return {t: n for t, n in enc.TABLE_SIZES.items() if t not in enc.NEW_TABLES_V2}


def _remap_cols(w: torch.Tensor, old: list[str], new: list[str]) -> torch.Tensor:
    assert w.shape[-1] == len(old), f"input is {w.shape[-1]} wide, expected {len(old)}"
    out = w.new_zeros(*w.shape[:-1], len(new))
    out[..., [new.index(n) for n in old]] = w
    return out


def _cols(group: str, src: int, dst: int):
    return lambda w: _remap_cols(w, names_at(src, group), names_at(dst, group))


def _remap_table(w: torch.Tensor) -> torch.Tensor:
    old_sizes = v1_table_sizes()
    assert w.shape[0] == sum(old_sizes.values()), "embedding table is not a v1 table"
    old_off, new_off = enc.table_offsets(old_sizes), enc.table_offsets()
    out = w.new_zeros(sum(enc.TABLE_SIZES.values()), *w.shape[1:])
    for name, size in old_sizes.items():
        out[new_off[name] : new_off[name] + size] = w[old_off[name] : old_off[name] + size]
    return out


# Per step ``k -> k + 1``: parameter name -> remap function.
_STEPS = {
    1: {_GLOBAL_PROJ_W: _cols("global_f", 1, 2), _TABLE_W: _remap_table},
    2: {_GLOBAL_PROJ_W: _cols("global_f", 2, 3), _JOKER_PROJ_W: _cols("joker_f", 2, 3)},
}


def _migrate_step(ck: dict, src: int) -> dict:
    remap = _STEPS[src]
    sd = ck["state_dict"] = dict(ck["state_dict"])
    shapes = {}
    for name, fn in remap.items():
        shapes[name] = tuple(sd[name].shape)
        sd[name] = fn(sd[name])
    opt = ck.get("optimizer")
    if opt is not None:
        # Adam keeps per-parameter state keyed by the parameter's index in ``model.parameters()``;
        # no version adds a parameter, so the index of a name is its position in a fresh model.
        from balatro_rl.model.team import ModelConfig, TeamModel

        model = TeamModel(ModelConfig(**ck["config"]))
        order = [n for n, _ in model.named_parameters()]
        for name, fn in remap.items():
            st = opt["state"].get(order.index(name))
            if st is None:
                continue
            for k, v in st.items():
                if isinstance(v, torch.Tensor) and tuple(v.shape) == shapes[name]:
                    st[k] = fn(v)
    ck["interface_version"] = src + 1
    ck.setdefault("migrated_from", src)
    return ck


def migrate_v1_to_v2(ck: dict) -> dict:
    """In-place conversion of a loaded v1 checkpoint dict to interface version 2 (also returned).

    ``state_dict`` and, when present, the Adam moments in ``optimizer`` are remapped; the other
    entries are untouched. ``ck["migrated_from"] = 1`` records it (dropped by the next save)."""
    assert ck["interface_version"] == 1
    return _migrate_step(ck, 1)


def migrate_v2_to_v3(ck: dict) -> dict:
    """In-place conversion of a v2 checkpoint dict to interface version 3 (see
    :func:`migrate_v1_to_v2`; ``migrated_from`` keeps the oldest version)."""
    assert ck["interface_version"] == 2
    return _migrate_step(ck, 2)


def migrate(ck: dict) -> dict:
    """Bring a checkpoint dict of any supported version to the current one (in place)."""
    v = ck["interface_version"]
    assert v in SUPPORTED_VERSIONS, f"unsupported interface version {v}"
    while ck["interface_version"] < INTERFACE_VERSION:
        _migrate_step(ck, ck["interface_version"])
    return ck


# --------------------------------------------------------------------------- stored data


def v1_data_width() -> int:
    return len(v1_names("global_f"))


def _upgrade(a: np.ndarray, group: str, old: list[str]) -> np.ndarray:
    names = list(LAYOUT["feature_names"][group])
    out = np.zeros((*a.shape[:-1], len(names)), dtype=a.dtype)
    out[..., [names.index(n) for n in old]] = a
    return out


def upgrade_v1_global_f(a: np.ndarray) -> np.ndarray:
    """Append the columns of later versions to v1 or v2 ``global_f`` data (any leading shape;
    returns a new array of the current width).

    v2 columns of v1 data: a v1 dataset has no record of its run configuration; it was collected
    on Red/White, so they hold the Red/White values at the start of a run (4 base hands / 4 base
    discards, $1 per unused hand, everything else 0). ``base_hands`` / ``base_discards`` drift
    from that when a voucher changes them; that part is not recoverable.

    v3 columns: ``next_hands`` / ``next_discards`` are set to ``base_hands`` / ``base_discards``
    (exact without hand / discard jokers and Needle / Water / Burglar), the run counters to 0
    (not recoverable)."""
    names = list(LAYOUT["feature_names"]["global_f"])
    if a.shape[-1] == len(v1_names("global_f")):
        out = _upgrade(a, "global_f", v1_names("global_f"))
        out[..., names.index("base_hands")] = 4 / 8
        out[..., names.index("base_discards")] = 4 / 8
        out[..., names.index("money_per_hand")] = 1 / 2
    else:
        assert a.shape[-1] == len(v2_names("global_f")), f"unknown width {a.shape[-1]}"
        out = _upgrade(a, "global_f", v2_names("global_f"))
    out[..., names.index("next_hands")] = out[..., names.index("base_hands")]
    out[..., names.index("next_discards")] = out[..., names.index("base_discards")]
    return out


def upgrade_v1_global_i(a: np.ndarray) -> np.ndarray:
    """Append the v2 ``global_i`` columns (``deck`` = Red, ``stake`` = White = 0)."""
    return _upgrade(a, "global_i", v1_names("global_i"))


def upgrade_v2_joker_f(a: np.ndarray) -> np.ndarray:
    """Append the v3 contribution columns to v1 / v2 ``joker_f`` data as zeros (no contribution;
    not recoverable from the stored observation)."""
    return _upgrade(a, "joker_f", v2_names("joker_f"))


def upgrade_v1_obs_arrays(global_f: np.ndarray, global_i: np.ndarray):
    """``(upgrade_v1_global_f(global_f), upgrade_v1_global_i(global_i))``."""
    return upgrade_v1_global_f(global_f), upgrade_v1_global_i(global_i)


# Stored-data upgrades by buffer name: ``(accepted legacy widths, upgrade function)``.
LEGACY_DATA = {
    "global_f": (
        (len(v1_names("global_f")), len(v2_names("global_f"))),
        upgrade_v1_global_f,
    ),
    "global_i": ((len(v1_names("global_i")),), upgrade_v1_global_i),
    "joker_f": ((len(v2_names("joker_f")),), upgrade_v2_joker_f),
}
