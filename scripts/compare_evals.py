"""Compare raw eval JSONs (from ``scripts/evaluate.py``): win rate / mean ante with standard
errors per file, and the paired difference (same game index = same game) of every later file
against the first.

    uv run --no-sync python scripts/compare_evals.py A.json B.json [C.json ...] [--mode det]

Files must come from the same game set (same ``--set`` / ``--seed-base``, same deck/stake); a
warning is printed otherwise. Games are paired by index over the shorter prefix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from balatro_rl.train import evaluate as ev


def load(path: Path, mode: str) -> tuple[ev.EvalResult, dict]:
    d = json.loads(path.read_text())
    meta = {k: d.get(k) for k in ("set", "seed_base", "deck", "stake")}
    r = d[mode]
    r["ante_hist"] = {int(k): v for k, v in r["ante_hist"].items()}
    return ev.EvalResult(**r), meta


def sem(x: np.ndarray) -> float:
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--mode", default="det", help="det or sample (default det)")
    args = ap.parse_args()
    rs = [load(f, args.mode) for f in args.files]
    metas = {json.dumps(m) for _, m in rs}
    if len(metas) > 1:
        print("WARNING: files differ in set/seed/deck/stake:", *sorted(metas), sep="\n  ")
    print(f"{'file':<40} {'games':>6} {'win% +- SE':>14} {'ante +- SE':>14}")
    for f, (r, _) in zip(args.files, rs, strict=False):
        w = np.array(r.wins, dtype=float)
        a = np.array(r.antes, dtype=float)
        print(
            f"{f.stem[:40]:<40} {len(w):>6} {100 * w.mean():>8.2f} +- {100 * sem(w):<4.2f} "
            f"{a.mean():>8.3f} +- {sem(a):<5.3f}"
        )
    base = rs[0][0]
    for f, (r, _) in zip(args.files[1:], rs[1:], strict=False):
        c = ev.compare(r, base)
        print(
            f"paired {f.stem} - {args.files[0].stem}: n={c['games']} "
            f"d_win={100 * c['win_diff']:+.2f} +- {100 * c['win_diff_sem']:.2f} pts "
            f"(z={c['win_diff'] / c['win_diff_sem']:+.1f})  "
            f"d_ante={c['ante_diff']:+.3f} +- {c['ante_diff_sem']:.3f} "
            f"(z={c['ante_diff'] / c['ante_diff_sem']:+.1f})"
        )


if __name__ == "__main__":
    main()
