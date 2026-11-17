"""Throughput benchmark of the PPO loop: N real updates of a given flag set, no evaluation, no
checkpoints. Takes every ``train_ppo.py`` flag (same defaults: 1024 envs x 128 steps) plus:

    --updates N      measured updates (default 4)
    --warmup W       untimed updates first (compile / allocator warm-up; default 1)
    --bench-out F    append the JSON result line to this file

Prints s/update, rollout s, update s, steps/s and peak VRAM. Example (the e2e recipe's setting;
every GPU process of this repo goes through the shared lock)::

    flock runs/.gpu.lock uv run --no-sync python scripts/bench_update.py \\
        --init runs/bc/model.pt --kl-coef 0.3 --update-dtype bf16 --compile --updates 4

Flags worth sweeping: ``--update-dtype bf16``, ``--recompute-old-logp``, ``--compile``,
``--epochs 1``, ``--num-envs 2048``, ``--profile`` (adds per-section times, slower: syncs).
"""

from __future__ import annotations

import json
import tempfile
import time

import numpy as np
import torch
from balatro_rl.train.ppo import Trainer
from train_ppo import build_parser, make_config


def main() -> None:
    ap = build_parser(__doc__)
    ap.add_argument("--updates", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--bench-out", default=None)
    args = ap.parse_args()
    with tempfile.TemporaryDirectory(prefix="bench_ppo_") as tmp:
        args.out_dir = args.out_dir or tmp
        args.eval_every = 0
        cfg = make_config(args)
        tr = Trainer(cfg)
        rows = cfg.num_envs * cfg.rollout_len
        recs = []
        for i in range(args.warmup + args.updates):
            t0 = time.perf_counter()
            col = tr.collect()
            st = tr.update()
            tr.global_step += rows
            tr.update_idx += 1
            total = time.perf_counter() - t0
            tag = "warmup" if i < args.warmup else "measured"
            print(
                f"[{tag} {i}] total {total:.2f}s rollout {col['rollout_seconds']:.2f}s "
                f"update {st['update_seconds']:.2f}s kl={st['approx_kl']:.4f} "
                f"clip={st['clipfrac']:.3f} vl={st['vloss']:.4f}",
                flush=True,
            )
            if i >= args.warmup:
                rec = {
                    "total": total,
                    "rollout": col["rollout_seconds"],
                    "update": st["update_seconds"],
                    "approx_kl": st["approx_kl"],
                    "clipfrac": st["clipfrac"],
                }
                if "prof" in st:
                    rec["prof"] = st["prof"]
                recs.append(rec)
    mean = {k: float(np.mean([r[k] for r in recs])) for k in ("total", "rollout", "update")}
    out = {
        "flags": {
            k: getattr(cfg, k)
            for k in (
                "num_envs",
                "rollout_len",
                "epochs",
                "chunk",
                "group",
                "update_dtype",
                "recompute_old_logp",
                "compile",
                "kl_coef",
            )
        },
        "s_per_update": mean["total"],
        "rollout_s": mean["rollout"],
        "update_s": mean["update"],
        "steps_per_s": rows / mean["total"],
        "compile_s": tr.compile_seconds,
        "peak_vram_alloc_gb": torch.cuda.max_memory_allocated() / 2**30,
        "peak_vram_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
        "approx_kl": float(np.mean([r["approx_kl"] for r in recs])),
        "clipfrac": float(np.mean([r["clipfrac"] for r in recs])),
    }
    if "prof" in recs[0]:
        keys = recs[0]["prof"].keys()
        out["prof"] = {k: float(np.mean([r["prof"].get(k, 0.0) for r in recs])) for k in keys}
    print(json.dumps(out))
    if args.bench_out:
        with open(args.bench_out, "a") as f:
            f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
