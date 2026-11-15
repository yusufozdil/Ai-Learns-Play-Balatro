"""Recurrent PPO training (see docs/design/training.md for the long-run commands).

Examples (from the repo root; runs go to ``runs/<name>/``, gitignored)::

    # end-to-end PPO started from the BC checkpoint, with a decaying KL penalty to the BC policy
    uv run --no-sync python scripts/train_ppo.py --name e2e --init runs/bc/model.pt \\
        --kl-coef 0.5 --critic-warmup 5 --total-steps 30e6

    # from scratch, curriculum stage 1 (the model plays PLAY, the teacher everything else)
    uv run --no-sync python scripts/train_ppo.py --name s1 --curriculum play --total-steps 5e6

    # continue an interrupted run (same --name / --out-dir)
    uv run --no-sync python scripts/train_ppo.py --name e2e --resume ...

Every PPOConfig field is a flag (underscores -> dashes); ``--help`` lists them.
"""

from __future__ import annotations

import argparse
import dataclasses
import types
import typing
from pathlib import Path

from balatro_rl.train.ppo import PPOConfig, Trainer

ROOT = Path(__file__).resolve().parents[1]


def _is_optional(tp) -> tuple[bool, type]:
    if typing.get_origin(tp) in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        return True, args[0]
    return False, tp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    defaults = PPOConfig()
    for f in dataclasses.fields(PPOConfig):
        if f.name in ("model", "out_dir", "reward"):
            continue
        opt, tp = _is_optional(f.type if not isinstance(f.type, str) else eval(f.type))
        flag = "--" + f.name.replace("_", "-")
        default = getattr(defaults, f.name)
        if tp is bool:
            ap.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
        else:
            ap.add_argument(flag, type=float if tp is float else tp, default=default)
    ap.add_argument("--init", dest="init_checkpoint", help="alias of --init-checkpoint")
    ap.add_argument("--out-dir", default=None, help="default: runs/<name>")
    ap.add_argument("--resume", action="store_true", help="continue from <out-dir>/latest.pt")
    ap.add_argument("--shaping", type=float, default=0.0, help="env reward shaping coefficient")
    ap.add_argument("--loss-reward", type=float, default=0.0, help="env reward on losing a run")
    ap.add_argument(
        "--reorder-cost",
        type=float,
        default=0.0,
        help="env reward of every MoveJoker action (negative = cost; breaks swap loops)",
    )
    ap.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="env reward weight of ln(min(score/target, margin-cap)) per beaten blind (0 = off)",
    )
    ap.add_argument("--margin-cap", type=float, default=4.0, help="cap of score/target in --margin")
    ap.add_argument(
        "--zero-play",
        type=float,
        default=0.0,
        help="env reward of a hand rejected by a boss (Psychic/Eye/Mouth; negative = cost)",
    )
    ap.add_argument("--memory", choices=["gru", "none"], default="gru")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    args = ap.parse_args()

    kw = {
        f.name: getattr(args, f.name)
        for f in dataclasses.fields(PPOConfig)
        if hasattr(args, f.name) and f.name not in ("model", "out_dir")
    }
    for k in (
        "num_envs",
        "rollout_len",
        "epochs",
        "chunk",
        "group",
        "kl_anneal_updates",
        "critic_warmup",
        "eval_every",
        "eval_games",
        "save_every",
        "seed",
        "max_episode_steps",
        "threads",
        "endless_max_ante",
    ):
        kw[k] = int(kw[k])
    kw["total_steps"] = float(kw["total_steps"])
    if args.init_checkpoint:
        kw["init_checkpoint"] = args.init_checkpoint
    kw["out_dir"] = args.out_dir or str(ROOT / "runs" / args.name)
    kw["model"] = {
        "d_model": args.d_model,
        "layers": args.layers,
        "heads": 4,
        "ff_mult": 4,
        "dropout": 0.0,
        "memory": args.memory,
        "mem_dim": args.d_model,
    }
    kw["reward"] = {}
    if args.shaping:
        kw["reward"]["shaping"] = args.shaping
    if args.loss_reward:
        kw["reward"]["loss"] = args.loss_reward
    if args.reorder_cost:
        kw["reward"]["reorder"] = args.reorder_cost
    if args.margin:
        kw["reward"]["margin"] = args.margin
        kw["reward"]["margin_cap"] = args.margin_cap
    if args.zero_play:
        kw["reward"]["zero_play"] = args.zero_play
    cfg = PPOConfig(**kw)
    Trainer(cfg, resume=args.resume).train()


if __name__ == "__main__":
    main()
