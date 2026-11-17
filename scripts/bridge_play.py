"""Play real Balatro (via BalatroBot) with a trained checkpoint and log score parity.

    uv run --no-sync python scripts/bridge_play.py --checkpoint runs/st2_v2_s1/latest.pt \
        --games 3 --deck RED --stake WHITE --seed-prefix BRG --log runs/bridge/run1.jsonl

The game must already be running with the bridge-patched BalatroBot
(``bridge/apply_lua_patch.sh`` then ``bridge/restart_game.sh``); see ``docs/design/bridge.md``.
Seed of game ``i`` = ``<seed-prefix><i:03d>`` (upper-case alphanumeric, at most 8 characters).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from balatro_rl.bridge.client import DEFAULT_URL, Client  # noqa: E402
from balatro_rl.bridge.runner import ModelPolicy, RandomPolicy, Runner, seed_for  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--checkpoint", help="model checkpoint (.pt); omit with --policy random")
    ap.add_argument("--policy", choices=["model", "random"], default="model")
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--first-game", type=int, default=0, help="index of the first game (seed)")
    ap.add_argument("--deck", default="RED")
    ap.add_argument("--stake", default="WHITE")
    ap.add_argument("--seed-prefix", default="BRG")
    ap.add_argument("--log", required=True, help="JSONL log path (appended)")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--timeout", type=float, default=90.0, help="per API call, seconds")
    ap.add_argument("--no-predict", action="store_true", help="skip the parity predictions")
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--endless",
        action="store_true",
        help="EXPERIMENTAL: after the ante-8 win keep playing in endless mode (default off)",
    )
    ap.add_argument(
        "--endless-max-ante", type=int, default=12, help="stop after this ante is beaten"
    )
    ap.add_argument(
        "--fixture-limit", type=int, default=3, help="fixtures kept per phase/pack kind"
    )
    ap.add_argument("--fixture-dir", help="save a few gzipped gamestates per phase here")
    args = ap.parse_args()

    client = Client(args.url, timeout=args.timeout)
    if not client.wait_healthy(20):
        print("BalatroBot does not answer on", args.url, file=sys.stderr)
        return 2
    if args.policy == "model":
        if not args.checkpoint:
            ap.error("--checkpoint is required for --policy model")
        policy = ModelPolicy(args.checkpoint, args.device)
    else:
        policy = RandomPolicy()
    runner = Runner(
        client,
        policy,
        args.log,
        predict=not args.no_predict,
        fixture_dir=args.fixture_dir,
        fixture_limit=args.fixture_limit,
        endless=args.endless,
        endless_max_ante=args.endless_max_ante,
    )
    results = []
    try:
        for i in range(args.first_game, args.first_game + args.games):
            results.append(
                runner.play_game(i, args.deck, args.stake, seed_for(args.seed_prefix, i))
            )
            if results[-1].outcome in ("hang", "error") and not client.wait_healthy(10):
                print("game process is gone; restart with bridge/restart_game.sh", file=sys.stderr)
                break
    finally:
        runner.close()
    wins = sum(r.outcome == "win" for r in results)
    print(json.dumps({"games": len(results), "wins": wins,
                      "outcomes": [r.outcome for r in results]}))  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
