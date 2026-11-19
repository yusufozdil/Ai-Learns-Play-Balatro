"""Play real Balatro (via BalatroBot) with a trained checkpoint and log score parity.

    uv run --no-sync python scripts/bridge_play.py --checkpoint runs/st2_v2_s1/latest.pt \
        --games 3 --deck RED --stake WHITE --seed-prefix BRG --log runs/bridge/run1.jsonl

The game must already be running with the bridge-patched BalatroBot
(``bridge/apply_lua_patch.sh`` then ``bridge/restart_game.sh``); see ``docs/design/bridge.md``.
Seed of game ``i`` = ``<seed-prefix><i:03d>`` (upper-case alphanumeric, at most 8 characters).

Unattended runs (docs/design/bridge.md section 12):

* After a game that ended in ``hang`` / ``error`` (or an endless continuation that crashed or
  hung) the game process is restarted with ``bridge/restart_game.sh`` (3 attempts, a ``restart``
  event goes to the log) and the run goes on with the NEXT seed. A seed is never played twice in
  one run, so a lost game cannot be re-rolled into a win.
* ``--resume`` skips every seed that already has a ``game_end`` record in ``--log``: kill the
  run any time and start the identical command line again.
* ``--restart-every N`` restarts the game between games every N games (memory hygiene).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from balatro_rl.bridge.client import DEFAULT_URL, Client  # noqa: E402
from balatro_rl.bridge.recovery import finished_seeds, restart_game  # noqa: E402
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
        "--resume",
        action="store_true",
        help="skip seeds that already have a game_end record in --log (continue a killed run)",
    )
    ap.add_argument("--no-restart", action="store_true", help="never restart the game process")
    ap.add_argument("--restart-attempts", type=int, default=3)
    ap.add_argument("--restart-every", type=int, default=0, help="restart every N games (0=off)")
    ap.add_argument(
        "--max-failures",
        type=int,
        default=5,
        help="stop after this many consecutive games lost to hang/error (0 = never stop)",
    )
    ap.add_argument(
        "--fixture-limit", type=int, default=3, help="fixtures kept per phase/pack kind"
    )
    ap.add_argument("--fixture-dir", help="save a few gzipped gamestates per phase here")
    ap.add_argument(
        "--dump-states", help="save EVERY gamestate the policy decides on here (gzip, per game)"
    )
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
        state_dump_dir=args.dump_states,
    )
    results = []
    done = finished_seeds(args.log) if args.resume else set()
    played = 0
    failures = 0
    rc = 0

    def restart(reason: str) -> bool:
        if args.no_restart:
            return client.wait_healthy(10)
        info = restart_game(client, attempts=args.restart_attempts)
        runner.log("restart", reason=reason, **info)
        print(f"[restart] {reason}: {info}", flush=True)
        return bool(info["ok"])

    try:
        for i in range(args.first_game, args.first_game + args.games):
            seed = seed_for(args.seed_prefix, i)
            if seed in done:
                print(f"[game {i}] {seed} already in the log, skipped", flush=True)
                continue
            if not client.health() and not restart("unhealthy before game"):
                print("game process is gone and could not be restarted", file=sys.stderr)
                rc = 3
                break
            r = runner.play_game(i, args.deck, args.stake, seed)
            results.append(r)
            played += 1
            lost_to_game = r.outcome in ("hang", "error") or r.endless_end in ("hang", "crash")
            failures = failures + 1 if r.outcome in ("hang", "error") else 0
            if lost_to_game or not client.health():
                if not restart(f"game {i} {r.outcome} {r.endless_end or ''}".strip()):
                    print("game process is gone and could not be restarted", file=sys.stderr)
                    rc = 3
                    break
            elif (
                args.restart_every
                and played % args.restart_every == 0
                and not restart(f"periodic after {played} games")
            ):
                rc = 3
                break
            if args.max_failures and failures >= args.max_failures:
                print(f"{failures} consecutive failed games, stopping", file=sys.stderr)
                rc = 4
                break
    finally:
        runner.close()
    wins = sum(r.outcome == "win" for r in results)
    print(json.dumps({"games": len(results), "wins": wins,
                      "outcomes": [r.outcome for r in results]}))  # fmt: skip
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
