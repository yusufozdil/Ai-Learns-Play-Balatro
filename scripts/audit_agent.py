"""Exploit audit of a policy: record its games on the test set, replay them in the Rust ``Game``
and log every action with the context an auditor needs (bought items, money, scores, ...).

    uv run --no-sync python scripts/audit_agent.py --mode det --checkpoint runs/a_e2e/best.pt \
        --games 1000 --out /tmp/audit_det.pkl
    uv run --no-sync python scripts/audit_agent.py --mode teacher --out /tmp/audit_teacher.pkl

``--mode`` is ``det`` / ``sample`` (model) or ``teacher``. Output pickle:
``{"acts": per-game action tuples, "res": drain_finished records, "games": per-game
{"ev": [event dicts], "fin": final-state summary, "money_curve": [...]}}``. Game ``i`` is replayed
with ``Game(TEST_SEED, env_index=i)``, which is the first episode of env ``i`` of the evaluation.
"""

from __future__ import annotations

import argparse
import json
import pickle
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from balatro_rl import BalatroVecEnv, _engine
from balatro_rl.train import evaluate as ev

KINDS = [
    "SelectBlind", "SkipBlind", "Play", "Discard", "SellJoker", "EndShop", "UseConsumable",
    "SellConsumable", "MoveJoker", "Buy", "BuyAndUse", "OpenPack", "BuyVoucher", "Reroll",
    "PickPackItem", "SkipPack", "RerollBoss",
]  # fmt: skip


def collect(mode: str, games: int, checkpoint: Path | None) -> tuple[list, list]:
    """Play the first episode of ``games`` test envs; returns (actions per game, records)."""
    env = BalatroVecEnv(games, seed=ev.TEST_SEED, max_episode_steps=ev.MAX_EPISODE_STEPS)
    if mode == "teacher":
        act = ev.teacher_actor
    else:
        import torch
        from balatro_rl.model import load_checkpoint

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _ = load_checkpoint(checkpoint, dev)
        act = ev.ModelActor(model, mode == "det", seed=7)
    obs, masks = env.reset()
    first = np.ones(games, bool)
    active = np.ones(games, bool)
    acts: list[list[tuple[int, ...]]] = [[] for _ in range(games)]
    fin: dict[int, tuple] = {}
    while len(fin) < games:
        a = act(env, obs, masks, first, active)
        for i in np.flatnonzero(active):
            acts[i].append(tuple(int(x) for x in a[i]))
        obs, masks, _, done, _ = env.step(a)
        first = done != 0
        for rec in env.drain_finished():
            if rec[0] not in fin:
                fin[rec[0]] = rec
                active[rec[0]] = False
    return acts, [fin[i] for i in range(games)]


def _card(c: dict) -> str:
    enh = "" if c["enhancement"] == "None" else "+" + c["enhancement"]
    return f"{c['rank']}{c['suit'][0]}{enh}"


def _item(k) -> tuple[str, str]:
    if isinstance(k, dict):
        for key, v in k.items():
            if key == "Joker":
                return ("Joker", v["id"])
            return (key, v if isinstance(v, str) else json.dumps(v)[:60])
    return ("?", str(k))


def replay(args: tuple[int, list]) -> dict:
    """Replay one recorded game and log one event per action (pre-state context + money)."""
    i, acts = args
    g = _engine.Game(ev.TEST_SEED, env_index=i)
    events = []
    money_curve = []
    s = json.loads(g.state_json())
    for a in acts:
        kind = KINDS[a[0]]
        pre = s
        e = {"k": kind, "a": a, "ante": pre["ante"], "ph": pre["phase"], "money": pre["money"]}
        if kind in ("Buy", "BuyAndUse"):
            c = pre["shop"]["cards"][a[1]]
            if c is not None:
                e["item"] = _item(c["kind"])
                e["free"] = c.get("free")
                joker = c["kind"].get("Joker") if isinstance(c["kind"], dict) else None
                if joker:
                    e["ed"] = joker["edition"]
        elif kind == "BuyVoucher":
            e["item"] = ("Voucher", pre["shop"]["vouchers"][a[1]])
        elif kind == "OpenPack":
            p = pre["shop"]["packs"][a[1]]
            e["item"] = ("Pack", f"{p['spec']['kind']}-{p['spec']['size']}")
            e["free"] = p.get("free")
        elif kind == "PickPackItem":
            e["item"] = _item(pre["pack"]["items"][a[1]])
            e["pack"] = pre["pack"]["spec"]["kind"]
        elif kind in ("UseConsumable", "SellConsumable"):
            cc = pre["consumables"][a[1]]
            e["item"] = ("Cons", cc["id"] if isinstance(cc, dict) else str(cc))
        elif kind == "SellJoker":
            j = pre["jokers"][a[1]]
            e["item"] = ("Joker", j["id"])
            e["sellv"] = j["sell_value"]
        elif kind in ("Play", "Discard"):
            e["cards"] = [_card(pre["deck"]["hand"][b]) for b in range(32) if a[3] >> b & 1]
        elif kind == "SkipBlind":
            e["tag"] = pre["blind_tags"][0 if pre["blind"] == "Small" else 1]
        r = g.step_parts(*a)
        s = json.loads(g.state_json())
        e["inv"] = r[2]
        e["money_after"] = s["money"]
        if kind == "Play":
            same = s["round"] == pre["round"]
            e["hand"] = s["last_hand"]
            e["score"] = s["round_score"] - (pre["round_score"] if same else 0)
            e["same_round"] = same
            e["ph_after"] = s["phase"]
        if s["phase"] == "Shop" and pre["phase"] != "Shop":
            money_curve.append((s["ante"], s["blind"], s["money"], len(s["jokers"])))
        events.append(e)
    deck = s["deck"]
    fin = {
        "ante": s["ante"],
        "outcome": s["outcome"],
        "round": s["round"],
        "money": s["money"],
        "jokers": [(j["id"], j["edition"], j["sell_value"]) for j in s["jokers"]],
        "vouchers": s["vouchers"],
        "hand_levels": s["hand_levels"],
        "deck_size": sum(len(deck[k]) for k in ("draw_pile", "hand", "discard_pile", "played")),
        "best_hand": s["stats"]["best_hand"],
    }
    return {"ev": events, "fin": fin, "money_curve": money_curve}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["det", "sample", "teacher"], required=True)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--games", type=int, default=ev.TEST_GAMES)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=8)
    args = ap.parse_args()
    acts, res = collect(args.mode, args.games, args.checkpoint)
    with Pool(args.procs) as p:
        games = p.map(replay, list(enumerate(acts)), chunksize=4)
    args.out.write_bytes(pickle.dumps({"acts": acts, "res": res, "games": games}))
    wins = np.mean([r[2] == 2 for r in res])
    print(f"{args.mode}: ante {np.mean([r[1] for r in res]):.2f} win {100 * wins:.1f}%")


if __name__ == "__main__":
    main()
