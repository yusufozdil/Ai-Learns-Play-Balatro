"""Reproduce a parity mismatch offline from a ``<log>.mismatch/g*_s*.json.gz`` dump.

    uv run --no-sync python scripts/bridge_replay_mismatch.py \
        runs/bridge/main3.mismatch/g20_s88.json.gz

Prints the real state facts (hand, held cards, jokers with their raw ability), the translated
engine state facts and the engine's predictions for the recorded Play.
"""

from __future__ import annotations

import gzip
import json
import sys

from balatro_rl.bridge import parity
from balatro_rl.bridge.state import build_game, translate


def main() -> None:
    with gzip.open(sys.argv[1], "rt") as f:
        d = json.load(f)
    gs, mask = d["gamestate"], d["mask"]
    tr = translate(gs)
    s = tr.state
    hand = s["deck"]["hand"]
    print(
        "boss", s["boss"], "blind", s["blind"], "disabled", s["boss_disabled"], "money", s["money"]
    )
    print("round score", s["round_score"], "target", s["target"], "hands left", s["hands_left"])
    print("hand levels", s["hand_levels"])
    for i, c in enumerate(hand):
        mark = "PLAY" if (mask >> i) & 1 else "held"
        print(
            f"  [{i}] {mark} {c['rank']} {c['suit']} {c['enhancement']} {c['edition']} "
            f"{c['seal']} perma={c['perma_bonus']} debuffed={c['debuffed']} "
            f"fd={c['face_down']}"
        )
    for j, jc in zip(s["jokers"], gs["jokers"]["cards"], strict=True):
        print("  joker", j["id"], j["edition"], "value", j["value"], "counter", j["counter"])
        print("     raw ability:", json.dumps(jc["raw"]["ability"], sort_keys=True)[:400])
    g = build_game(tr)
    print("engine legal play:", mask in [p[3] for p in g.legal_parts() if p[0] == 2])
    pred = parity.predict_play(s, mask, variants=4, wide=256)
    print("prediction:", {k: v for k, v in pred.items() if k != "flags"}, pred.get("flags"))
    print("real chips after is not in the dump; see the log's `play` event")


if __name__ == "__main__":
    main()
