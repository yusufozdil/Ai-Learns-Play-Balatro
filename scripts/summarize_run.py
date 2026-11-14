"""Print the learning curve of a PPO run from ``runs/<name>/log.jsonl`` as a table.

uv run --no-sync python scripts/summarize_run.py runs/e2e [--points 20]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--points", type=int, default=20)
    args = ap.parse_args()
    recs = [json.loads(x) for x in (args.run / "log.jsonl").read_text().splitlines()]
    ups = [r for r in recs if r["type"] == "update"]
    evs = [r for r in recs if r["type"] == "eval"]
    k = max(1, len(ups) // args.points)
    print("train (mean over windows of", k, "updates; episodes finished inside the window)")
    print("update   Msteps  st/s  ep_ante  ep_win%  blinds  reward/step  kl     ev    ent")
    for i in range(0, len(ups), k):
        w = ups[i : i + k]

        def m(key, w=w):
            return sum(x.get(key, float("nan")) for x in w) / len(w)

        print(
            f"{w[-1]['update']:>6} {w[-1]['global_step'] / 1e6:7.2f} {m('steps_per_sec'):5.0f} "
            f"{m('ep_ante'):8.2f} {100 * m('ep_win'):8.1f} {m('ep_blinds'):7.2f} "
            f"{m('mean_reward_step'):11.4f} {m('approx_kl'):.4f} {m('explained_var'):5.2f} "
            f"{m('ent'):.2f}"
        )
    print("\nvalidation (deterministic / sampled; prefix of the test set)")
    print("update   Msteps  det_ante  det_win%  det_blinds | sam_ante  sam_win%")
    for e in evs:
        print(
            f"{e['update']:>6} {e['global_step'] / 1e6:7.2f} {e['det_mean_ante']:9.2f} "
            f"{100 * e['det_win_rate']:9.1f} {e['det_mean_blinds']:11.1f} | "
            f"{e.get('sam_mean_ante', float('nan')):8.2f} "
            f"{100 * e.get('sam_win_rate', float('nan')):8.1f}"
        )


if __name__ == "__main__":
    main()
