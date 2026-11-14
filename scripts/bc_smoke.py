"""Behaviour-cloning smoke test: collect teacher data, train the team model on it, then play
games with the BC policy (deterministic and sampled) vs random and the teacher.

Usage (from the repo root; data and checkpoints go to ``runs/bc*`` which is gitignored)::

    uv run --no-sync python scripts/bc_smoke.py --stage all
    uv run --no-sync python scripts/bc_smoke.py --stage collect --steps 1000 --envs 1024
    uv run --no-sync python scripts/bc_smoke.py --stage train --epochs 12
    uv run --no-sync python scripts/bc_smoke.py --stage eval --games 1024
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from balatro_rl.model import ModelConfig, TeamModel, load_checkpoint
from balatro_rl.train import bc, play

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["collect", "train", "eval", "all"], default="all")
    ap.add_argument("--data", type=Path, default=ROOT / "runs" / "bc_data")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "bc")
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--lr", type=float, default=7e-4)
    ap.add_argument("--value-coef", type=float, default=0.1)
    ap.add_argument("--max-minutes", type=float, default=40.0)
    ap.add_argument("--memory", choices=["gru", "none"], default="gru")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--games", type=int, default=1024)
    ap.add_argument("--no-baselines", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    args.out.mkdir(parents=True, exist_ok=True)

    if args.stage in ("collect", "all") and not (args.data / "meta.json").exists():
        print(f"collecting {args.envs * args.steps} samples -> {args.data}")
        meta = bc.collect(args.data, args.envs, args.steps, args.seed, args.noise)
        print(json.dumps(meta, indent=2))

    if args.stage in ("train", "all"):
        torch.manual_seed(0)
        model = TeamModel(
            ModelConfig(
                d_model=args.d_model, layers=args.layers, memory=args.memory, mem_dim=args.d_model
            )
        ).to(dev)
        print("parameters:", model.num_parameters())
        t0 = time.perf_counter()
        data = bc.load_gpu(args.data, dev)
        print(
            f"data on {dev}: {time.perf_counter() - t0:.0f}s, "
            f"{sum(v.numel() * v.element_size() for v in data.values()) / 1e9:.2f} GB"
        )
        cfg = bc.BCConfig(
            epochs=args.epochs,
            chunk=args.chunk,
            group=args.group,
            lr=args.lr,
            value_coef=args.value_coef,
            max_minutes=args.max_minutes,
        )
        result = bc.train(model, data, cfg, args.out)
        del data
        torch.cuda.empty_cache()
        print("final:", json.dumps(result["final"], indent=2))

    if args.stage in ("eval", "all"):
        model, _ = load_checkpoint(args.out / "model.pt", dev)
        results = []
        if not args.no_baselines:
            results.append(play.play_games("random", play.random_policy, args.games))
            results.append(play.play_games("teacher", play.teacher_policy, args.games))
        results.append(
            play.play_games("bc_deterministic", play.ModelPolicy(model, True), args.games)
        )
        results.append(play.play_games("bc_sampled", play.ModelPolicy(model, False, 1), args.games))
        for r in results:
            print(r.line())
        (args.out / "games.json").write_text(json.dumps([r.__dict__ for r in results], indent=2))


if __name__ == "__main__":
    main()
