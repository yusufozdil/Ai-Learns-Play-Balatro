"""Targeted score parity per joker: `add` one joker to a fresh real run, play one hand, compare.

    uv run --no-sync python scripts/bridge_joker_sweep.py --log runs/bridge/sweep.jsonl \
        [--edition NEGATIVE] [--only j_ramen,j_hit_the_road]

For every joker of the game (BalatroBot ``add`` endpoint) a fresh Red/White run is started, the
joker is added, the Small blind is selected, and a 5-card hand is played (first positions of the
engine's sorted hand, then a 2nd hand). The engine's prediction for the translated state is compared
with the real chips (``parity.compare``). Not a replacement for the full-game parity (jokers
interact), but it covers every joker's own scoring rule, including the ones a policy rarely buys.
"""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from balatro_rl.bridge import actions as A  # noqa: E402
from balatro_rl.bridge import parity  # noqa: E402
from balatro_rl.bridge import tables as T  # noqa: E402
from balatro_rl.bridge.client import ApiError, BridgeError, Client  # noqa: E402
from balatro_rl.bridge.state import build_game, translate  # noqa: E402


def play_once(c: Client, gs: dict, positions: int) -> dict | None:
    tr = translate(gs)
    game = build_game(tr)
    legal = [p for p in game.legal_parts() if p[0] == A.PLAY]
    if not legal:
        return None
    want = [p for p in legal if bin(p[3]).count("1") == positions]
    act = (want or legal)[0]
    mask = act[3]
    pred = parity.predict_play(tr.state, mask)
    before = gs
    after = c.call("rearrange", hand=tr.hand_real_index) if tr.hand_needs_rearrange else gs
    pos = A.mask_positions(mask)
    after = c.call("play", cards=pos)
    real = float((after.get("round") or {}).get("chips", 0)) - float(
        (before.get("round") or {}).get("chips", 0)
    )
    return {
        "pred": pred,
        "real": real,
        "verdict": parity.compare(pred, real),
        "after": after,
        "tr": tr,
        "real_hand": parity.real_hand_type(before, after),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--edition", default=None, help="FOIL / HOLO / POLYCHROME / NEGATIVE")
    ap.add_argument("--only", default=None, help="comma separated j_* keys")
    ap.add_argument("--seed-prefix", default="SWP")
    a = ap.parse_args()
    c = Client(timeout=60)
    keys = [k for k, _ in T.JOKERS]
    if a.only:
        keys = [k for k in keys if k in a.only.split(",")]
    bad = 0
    with open(a.log, "a") as log:
        for i, key in enumerate(keys):
            rec = {"key": key, "id": T.JOKER_KEY_TO_ID[key], "edition": a.edition}
            try:
                if c.gamestate()["state"] != "MENU":
                    c.call("menu")
                gs = c.call("start", deck="RED", stake="WHITE", seed=f"{a.seed_prefix}{i:03d}")
                params = {"key": key}
                if a.edition:
                    params["edition"] = a.edition
                gs = c.call("select")
                gs = c.call("add", **params)  # `add` needs SELECTING_HAND or SHOP
                out = []
                for n_cards in (5, 4):
                    if gs["state"] != "SELECTING_HAND":
                        break
                    r = play_once(c, gs, n_cards)
                    if r is None:
                        break
                    gs = r["after"]
                    rec_play = {
                        "pred": r["pred"],
                        "real": r["real"],
                        "verdict": r["verdict"],
                        "hand": r["real_hand"],
                        "cards": [
                            f"{c_['rank']}{c_['suit'][0]}" for c_ in r["tr"].state["deck"]["hand"]
                        ],
                    }
                    out.append(rec_play)
                    bad += r["verdict"]["status"] not in ("exact", "in_range")
                    if gs["state"] != "SELECTING_HAND":
                        break
                rec["plays"] = out
            except (ApiError, BridgeError, KeyError, ValueError) as e:
                rec["error"] = f"{type(e).__name__}: {e}"
            log.write(json.dumps(rec) + "\n")
            log.flush()
            status = [p["verdict"]["status"] for p in rec.get("plays", [])]
            print(f"{key:<22} {status or rec.get('error')}", flush=True)
            time.sleep(0.2)
    print("non-exact plays:", bad)


if __name__ == "__main__":
    main()
