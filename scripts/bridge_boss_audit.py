"""Audit the boss state of recorded real gamestates (``bridge_play.py --dump-states DIR``).

    uv run --no-sync python scripts/bridge_boss_audit.py 'runs/bridge/bossfix_states_rlr64/*' \
        --boss TheEye
    uv run --no-sync python scripts/bridge_boss_audit.py 'DIR/*' --payout DIR_LOG.jsonl --game 64

For every ``SELECTING_HAND`` state the real boss fields (``bridge.blind``) are printed next to what
``state.translate`` hands to the engine (``boss_state``, per-round hand counters, hands left) and
the observation lanes the policy reads (``boss`` id, ``boss_hand``). ``--payout`` replays every
winning Play of the log on the translated state and compares the engine's money after the round
with the real money of the next shop (docs/design/bridge.md, "Audit of the 100-game run").
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from balatro_rl.bridge.state import build_game, translate  # noqa: E402


def load(path: str) -> dict:
    with gzip.open(path, "rt") as f:
        return json.load(f)


def decision_no(path: str) -> int:
    return int(os.path.basename(path).split("_d")[1][:4])


def audit_states(paths: list[str], bosses: set[str]) -> None:
    for p in paths:
        gs = load(p)
        if gs["state"] != "SELECTING_HAND":
            continue
        tr = translate(gs)
        s = tr.state
        if bosses and s["boss"] not in bosses:
            continue
        bl = gs["bridge"]["blind"]
        obs = build_game(tr).observation()
        plays = {i: n for i, n in enumerate(s["hand_plays_round"]) if n}
        print(
            f"{os.path.basename(p)} {s['blind']}/{s['boss']} disabled={s['boss_disabled']} "
            f"real(only_hand={bl.get('only_hand')}, hands_sub={bl.get('hands_sub')}, "
            f"discards_sub={bl.get('discards_sub')}, hands={bl.get('hands')}) "
            f"engine(hands_left={s['hands_left']}, discards_left={s['discards_left']}, "
            f"plays_round={plays}, boss_state={s['boss_state']}) "
            f"obs(boss={int(obs['global_i'][2])}, boss_hand={int(obs['global_i'][4])})"
        )


def audit_payout(paths: list[str], log: str, game: int) -> None:
    decisions = {}
    with open(log) as f:
        for line in f:
            d = json.loads(line)
            if d["ev"] == "decision" and d.get("game") == game:
                decisions[d["step"]] = d
    by_no = {decision_no(p): p for p in paths if "roundeval" not in p}
    bad = total = 0
    for n in sorted(by_no):
        d = decisions.get(n)
        if not d or d["action"]["kind"] != "Play" or n + 1 not in by_no:
            continue
        before = load(by_no[n])
        nxt = load(by_no[n + 1])
        if nxt["state"] != "SHOP":
            continue
        tr = translate(before)
        g = build_game(tr)
        g.step_parts(2, 0, 0, sum(1 << i for i in d["action"]["cards"]))
        engine = json.loads(g.export_state())["money"]
        total += 1
        bad += engine != nxt["money"]
        print(f"decision {n} {tr.state['blind']}: engine {engine} real {nxt['money']}"
              f"{'  <-- DIFF' if engine != nxt['money'] else ''}")  # fmt: skip
    print(f"{bad} of {total} cash-outs differ")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("pattern", help="glob of gzipped gamestates, quoted")
    ap.add_argument("--boss", help="comma-separated engine boss ids (TheEye,TheMouth,...)")
    ap.add_argument("--payout", help="JSONL log of the same game: compare engine and real money")
    ap.add_argument("--game", type=int, default=0)
    args = ap.parse_args()
    paths = sorted(glob.glob(args.pattern))
    if args.payout:
        audit_payout(paths, args.payout, args.game)
    else:
        audit_states(paths, set(args.boss.split(",")) if args.boss else set())


if __name__ == "__main__":
    main()
