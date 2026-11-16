"""Table of raw eval JSONs: name / games / mean ante / win% +- SE / trunc%.

    uv run --no-sync python scripts/summarize_evals.py 'runs/eval/decks_*.json' [--mode det]
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pattern", nargs="+", help="glob(s) of eval JSON files")
    ap.add_argument("--mode", default="det")
    args = ap.parse_args()
    files = sorted({f for p in args.pattern for f in glob.glob(p)})
    print(f"{'name':<44} {'games':>6} {'ante':>7} {'win% +- SE':>14} {'trunc%':>7}")
    for f in files:
        d = json.loads(Path(f).read_text())
        if args.mode not in d:
            continue
        r = d[args.mode]
        w = np.array(r["wins"], dtype=float)
        n = len(w)
        se = 100 * w.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
        print(
            f"{Path(f).stem[:44]:<44} {n:>6} {np.mean(r['antes']):>7.3f} "
            f"{100 * w.mean():>8.2f} +- {se:<4.2f} {100 * r['truncated']:>7.2f}"
        )


if __name__ == "__main__":
    main()
