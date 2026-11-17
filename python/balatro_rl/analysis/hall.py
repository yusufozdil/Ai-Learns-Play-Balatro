"""Error hall: pool files of confirmed mistake states (``docs/design/error_hall.md``).

A pool file is JSON lines, one *confirmed mistake* per line: ``state`` (the ``export_state`` JSON
string of the state BEFORE the mistaken decision) plus metadata (category, phase, ante, stake,
deck, dP(win) of the better action, played / better action, game, step, ...). Written by
``scripts/errhall_mine.py``, read by the trainer (``--error-hall-pool``).

No label ever enters training: the pool is only a set of *start states*.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from balatro_rl.analysis.report import _cat

POOL_VERSION = 1
MASK64 = (1 << 64) - 1
REQUIRED = ("state", "category", "deck", "stake", "dq")


def category_of(played: str, better: str) -> str:
    """Compact mistake category from the two action descriptions, e.g. ``SellJoker->SellJoker``
    (sold the wrong joker), ``SelectBlind->RerollBoss``, ``BuyJoker->SellJoker``."""
    return f"{_cat(played)}->{_cat(better)}".replace(" ", "")


def pool_record(state: str, mistake: dict, **meta) -> dict:
    """One pool line from the exported ``state`` JSON string and a ``finalize_game`` mistake."""
    cfg = json.loads(state)["config"]
    return {
        "v": POOL_VERSION,
        "state": state,
        "category": category_of(mistake["played"], mistake["better"]),
        "phase": mistake["phase"],
        "ante": int(mistake["ante"]),
        "deck": str(cfg["deck"]).lower(),
        "stake": str(cfg["stake"]).lower(),
        "dq": float(mistake["delta"]),
        "se": float(mistake["se"]),
        "q_played": float(mistake["q_played"]),
        "q_better": float(mistake["q_better"]),
        "played": mistake["played"],
        "better": mistake["better"],
        "played_action": mistake.get("played_action"),
        "better_action": mistake.get("better_action"),
        "played_kind": mistake["played_kind"],
        "better_kind": mistake["better_kind"],
        "step": int(mistake["step"]),
        "pwin_before": float(mistake["pwin_before"]),
        **meta,
    }


def write_pool(path: Path | str, records: Iterable[dict], append: bool = False) -> int:
    """Write (or append) pool lines; returns the number of lines written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a" if append else "w") as f:
        for r in records:
            missing = [k for k in REQUIRED if k not in r]
            if missing:
                raise ValueError(f"pool record without {missing}")
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
            n += 1
    return n


def in_holdout(rec: dict, frac: float) -> bool:
    """Deterministic split BY GAME: all mistakes of a game are held out together (a hash of
    ``seed_base`` and ``game``), so a held-out state never shares its run with a training state."""
    if frac <= 0:
        return False
    z = (int(rec.get("seed_base", 0)) * 0x9E3779B97F4A7C15 + int(rec.get("game", 0))) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    z ^= z >> 31
    return (z >> 11) * (1.0 / (1 << 53)) < frac


def read_pool(
    paths: str | Path | Iterable[str | Path],
    min_dq: float = 0.0,
    holdout: float = 0.0,
    part: str = "train",
) -> list[dict]:
    """Records of one or several pool files (``paths`` may be a comma-separated string); records
    with ``dq < min_dq`` are dropped. ``holdout`` = fraction of the GAMES kept out of training:
    ``part="train"`` returns the rest, ``"holdout"`` the held-out games (for the probe), ``"all"``
    everything."""
    assert part in ("train", "holdout", "all")
    if isinstance(paths, str):
        paths = [p for p in (x.strip() for x in paths.split(",")) if p]
    elif isinstance(paths, Path):
        paths = [paths]
    out: list[dict] = []
    for p in paths:
        with Path(p).open() as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("v") != POOL_VERSION:
                    raise ValueError(f"{p}:{ln}: pool version {r.get('v')} != {POOL_VERSION}")
                missing = [k for k in REQUIRED if k not in r]
                if missing:
                    raise ValueError(f"{p}:{ln}: pool record without {missing}")
                if r["dq"] < min_dq:
                    continue
                if part != "all" and in_holdout(r, holdout) != (part == "holdout"):
                    continue
                out.append(r)
    return out


def summarize(records: list[dict]) -> dict:
    """Counts per category / stake / deck / ante and the mean dP(win)."""
    return {
        "n": len(records),
        "mean_dq": sum(r["dq"] for r in records) / max(len(records), 1),
        "category": dict(Counter(r["category"] for r in records).most_common()),
        "stake": dict(Counter(r["stake"] for r in records).most_common()),
        "deck": dict(Counter(r["deck"] for r in records).most_common()),
        "ante": dict(sorted(Counter(r["ante"] for r in records).items())),
    }
