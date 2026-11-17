"""P(win) head on the frozen reference policy (Part 1 of the error maps, docs/design/error_maps.md).

    # 1. collect: shards of decisions (det and sampled alternate), 5000 games each
    flock runs/.gpu.lock uv run --no-sync python scripts/errmap_pwin.py collect \
        --checkpoint runs/e2e_long_fix/latest.pt --out runs/errmaps/pwin_data --shards 0-3
    # 2. train + calibrate (writes runs/errmaps/pwin_head.pt and pwin_head_report.json)
    flock runs/.gpu.lock uv run --no-sync python scripts/errmap_pwin.py train \
        --data runs/errmaps/pwin_data --out runs/errmaps/pwin_head.pt

Shard ``s`` plays base seed ``333_000_001 + 1009 * s`` (never the dev / holdout / val bases);
even shards are deterministic play, odd shards sampled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from balatro_rl.analysis import pwin
from balatro_rl.model import load_checkpoint
from balatro_rl.train import evaluate as ev

BASE = 333_000_001


def shard_seed(s: int) -> int:
    seed = BASE + 1009 * s
    assert seed not in (ev.TEST_SEED, ev.HOLDOUT_SEED, ev.VAL_SEED)
    return seed


def parse_range(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def cmd_collect(a) -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(a.checkpoint, dev)
    a.out.mkdir(parents=True, exist_ok=True)
    for s in parse_range(a.shards):
        path = a.out / f"shard_{s:03d}.npz"
        if path.exists() and not a.force:
            print(f"skip {path}")
            continue
        info = pwin.collect_shard(
            model,
            path,
            a.games,
            shard_seed(s),
            deterministic=(s % 2 == 0),
            keep_other=a.keep_other,
            actor_seed=s,
            game_offset=s * 1_000_000,
        )
        print(f"shard {s} ({'det' if s % 2 == 0 else 'sample'}): {info}", flush=True)


def cmd_train(a) -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    paths = sorted(a.data.glob("shard_*.npz"))
    data = pwin.load_shards(paths)
    print(f"{len(paths)} shards, {len(data['win'])} rows, {len(np.unique(data['game']))} games")
    head, info = pwin.train_head(data, dev, epochs=a.epochs, seed=a.seed)
    rep = pwin.evaluate_head(head, data, info["split"], dev)
    rep["train"] = {k: v for k, v in info.items() if k != "split"}
    rep["rows"] = int(len(data["win"]))
    rep["games"] = int(len(np.unique(data["game"])))
    rep["test_games"] = int(len(np.unique(data["game"][info["split"] == 2])))
    pwin.save_head(
        a.out,
        head.cpu(),
        checkpoint=str(a.checkpoint),
        metrics=rep["overall"],
        critic_auc=rep["critic_auc"],
        rows=rep["rows"],
    )
    a.out.with_name(a.out.stem + "_report.json").write_text(json.dumps(rep, indent=1))
    o = rep["overall"]
    print(
        f"test: n={o['n']} logloss={o['logloss']:.4f} (const {o['logloss_const']:.4f}) "
        f"AUC={o['auc']:.4f} (critic V: {rep['critic_auc']:.4f}) ECE={o['ece']:.4f} "
        f"T={info['temperature']:.3f}"
    )
    cc = rep["calibrated_critic"]
    print(
        f"baseline calibrated critic (V + per-ante intercept): logloss={cc['logloss']:.4f} "
        f"AUC={cc['auc']:.4f} ECE={cc['ece']:.4f}"
    )
    print("by ante:")
    for k, m in rep["by_ante"].items():
        print(
            f"  ante {k}: n={m['n']} win={m['win_rate']:.3f} ll={m['logloss']:.4f} "
            f"auc={m['auc']:.3f} (V {m['critic_auc']:.3f}) ece={m['ece']:.4f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--checkpoint", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--shards", default="0-3")
    c.add_argument("--games", type=int, default=5000)
    c.add_argument("--keep-other", type=float, default=0.1)
    c.add_argument("--force", action="store_true")
    t = sub.add_parser("train")
    t.add_argument("--checkpoint", type=Path, default=Path("/home/yuse/Desktop/balatroRL/runs/e2e_long_fix/latest.pt"))
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--epochs", type=int, default=6)
    t.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"collect": cmd_collect, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
