"""Error rate per category of one or several error-map directories (``errmap_analyze.py`` outputs),
side by side: the measurement of the error hall (``docs/design/error_hall.md``).

    uv run --no-sync python scripts/errhall_compare.py runs/errmaps/st3_ctrl runs/errmaps/st3_hall \
        [--names control,hall] [--top 15]

Per directory: games analysed, analysed decisions, confirmed mistakes, mistakes per game and per
1000 analysed decisions; then per category (``played->better``, the pool's ``category``:
SellJoker->SellJoker = sold the wrong joker, SelectBlind -> RerollBoss, ...) the count and the
rate per 1000 analysed decisions. Use the same analysis settings and the same game set for every
directory; a mistake is relative to the analysed policy itself (Q of "this policy plays on").
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from balatro_rl.analysis.hall import category_of
from balatro_rl.analysis.report import load_games


def summarize(d: Path) -> dict:
    games = load_games(d)
    dec = sum(len(g["decisions"]) for g in games)
    cats = Counter(category_of(m["played"], m["better"]) for g in games for m in g["mistakes"])
    return {
        "games": len(games),
        "decisions": dec,
        "mistakes": sum(cats.values()),
        "cats": cats,
        "wins": sum(g["outcome"] == "Win" for g in games),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--names", default=None, help="comma list of column names")
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()
    names = a.names.split(",") if a.names else [d.name for d in a.dirs]
    assert len(names) == len(a.dirs)
    sums = [summarize(d) for d in a.dirs]
    print("| | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for label, f in (
        ("games analysed", lambda s: f"{s['games']}"),
        ("analysed decisions", lambda s: f"{s['decisions']}"),
        ("confirmed mistakes", lambda s: f"{s['mistakes']}"),
        ("mistakes / game", lambda s: f"{s['mistakes'] / max(s['games'], 1):.3f}"),
        (
            "mistakes / 1000 decisions",
            lambda s: f"{1000 * s['mistakes'] / max(s['decisions'], 1):.2f}",
        ),
    ):
        print(f"| {label} | " + " | ".join(f(s) for s in sums) + " |")
    allc = Counter()
    for s in sums:
        allc.update(s["cats"])
    print()
    print("Per category: count (per 1000 analysed decisions)")
    print()
    print("| category | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for cat, _ in allc.most_common(a.top):
        row = [
            f"{s['cats'][cat]} ({1000 * s['cats'][cat] / max(s['decisions'], 1):.2f})" for s in sums
        ]
        print(f"| {cat} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
