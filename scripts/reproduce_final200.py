"""Reproduce the training chain of the shipped model ``final200`` from scratch.

The chain (``configs/final200/*.json``, one file per stage, in order)::

    bc -> e2e_long -> e2e_long_fix -> st1 -> st2_v2_s1 -> st3 -> st3_endless -> obs_v2 -> final200

``bc`` is behaviour cloning of the heuristic agent (``scripts/bc_smoke.py``); every other stage is
``scripts/train_ppo.py`` initialised from the previous stage's ``latest.pt``. Each config is the
recorded ``config.json`` of the original run; this driver turns every key into the matching CLI
flag, so the recipe lives in the JSON files, not here.

    uv run --no-sync python scripts/reproduce_final200.py --dry-run      # print the commands
    uv run --no-sync python scripts/reproduce_final200.py                # everything, ~17 GPU hours
    uv run --no-sync python scripts/reproduce_final200.py --smoke        # tiny end-to-end test
    uv run --no-sync python scripts/reproduce_final200.py --from st3     # start at a later stage

Outputs go to ``<out-root>/<stage>/`` (default ``runs/``). The driver is resumable: a finished
stage is skipped, an interrupted PPO stage continues from its ``latest.pt`` (``--resume``).
A rerun is statistically similar to the original, not bit-identical (see docs/design/training.md).
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "final200"

# PPO config keys with no CLI flag any more, and the value they had (= the only value the
# current code behaves like). A recorded value that differs is an error, never silently dropped.
REMOVED_KEYS = {
    "error_hall": 0.0,
    "error_hall_pool": "",
    "error_hall_min_dq": 0.0,
    "error_hall_holdout": 0.0,
}
# keys whose recorded null is the default of the current flag (flag omitted)
NULLABLE_KEYS = {"target_kl", "eval_move_limit", "init_checkpoint"}
# reward dict key -> flag
REWARD_FLAGS = {
    "shaping": "--shaping",
    "loss": "--loss-reward",
    "reorder": "--reorder-cost",
    "margin": "--margin",
    "margin_cap": "--margin-cap",
    "zero_play": "--zero-play",
}
SKIP_KEYS = {"name", "out_dir", "init_checkpoint", "model", "reward"}

# --smoke: a few seconds per stage, still exercises rollout, update, eval, checkpoint, init.
SMOKE_PPO = {
    "num_envs": 64,
    "rollout_len": 32,
    "chunk": 32,
    "group": 64,
    "eval_every": 2,
    "eval_games": 8,
    "save_every": 1,
}
SMOKE_UPDATES = 2
SMOKE_BC = {"envs": 192, "steps": 64, "epochs": 1, "chunk": 32, "group": 32, "games": 8}


def load_stages() -> list[dict]:
    stages = [json.loads(p.read_text()) for p in sorted(CONFIG_DIR.glob("*.json"))]
    assert stages and stages[0]["name"] == "bc", "configs/final200 must start with bc"
    return stages


def rebase(path: str, out_root: Path) -> Path:
    """Config paths are repo-relative ``runs/...``; map the ``runs`` part to ``out_root``."""
    p = Path(path)
    assert p.parts[0] == "runs", path
    return out_root.joinpath(*p.parts[1:])


def ppo_command(cfg: dict, out_root: Path, smoke: bool) -> list[str]:
    cfg = dict(cfg)
    name = cfg["name"]
    if smoke:
        cfg.update(SMOKE_PPO)
        cfg["total_steps"] = float(SMOKE_UPDATES * SMOKE_PPO["num_envs"] * SMOKE_PPO["rollout_len"])
    cmd = ["scripts/train_ppo.py", "--name", name, "--out-dir", str(out_root / name), "--resume"]
    if cfg.get("init_checkpoint"):
        cmd += ["--init", str(rebase(cfg["init_checkpoint"], out_root))]
    for k, v in cfg.items():
        if k in SKIP_KEYS:
            continue
        if k in REMOVED_KEYS:
            if v != REMOVED_KEYS[k]:
                raise SystemExit(
                    f"{name}: key {k}={v!r} has no flag any more (was {REMOVED_KEYS[k]!r})"
                )
            continue
        flag = "--" + k.replace("_", "-")
        if v is None:
            if k not in NULLABLE_KEYS:
                raise SystemExit(f"{name}: key {k} is null but its flag has no null default")
        elif isinstance(v, bool):
            cmd.append(flag if v else "--no-" + flag[2:])
        else:
            cmd += [flag, str(v)]
    for k, v in cfg.get("reward", {}).items():
        if k not in REWARD_FLAGS:
            raise SystemExit(f"{name}: unknown reward term {k!r}")
        cmd += [REWARD_FLAGS[k], str(v)]
    m = cfg["model"]
    if m["heads"] != 4 or m["ff_mult"] != 4 or m["dropout"] != 0.0 or m["mem_dim"] != m["d_model"]:
        raise SystemExit(f"{name}: model block {m} not expressible with the CLI flags")
    cmd += ["--memory", m["memory"], "--d-model", str(m["d_model"]), "--layers", str(m["layers"])]
    return cmd


def bc_command(cfg: dict, out_root: Path, smoke: bool) -> list[str]:
    cfg = dict(cfg)
    if smoke:
        cfg.update(SMOKE_BC)
    cmd = [
        "scripts/bc_smoke.py",
        "--stage",
        "all",
        "--data",
        str(rebase(cfg["data_dir"], out_root)),
        "--out",
        str(rebase(cfg["out_dir"], out_root)),
    ]
    for k in (
        "envs",
        "steps",
        "seed",
        "noise",
        "epochs",
        "chunk",
        "group",
        "lr",
        "value_coef",
        "max_minutes",
        "memory",
        "d_model",
        "layers",
        "games",
    ):
        cmd += ["--" + k.replace("_", "-"), str(cfg[k])]
    if smoke:
        cmd.append("--no-baselines")
    return cmd


def is_done(cfg: dict, out_root: Path, smoke: bool) -> bool:
    out = ROOT / out_root / cfg["name"]
    if cfg["name"] == "bc":
        return (out / "model.pt").exists() and (out / "games.json").exists()
    latest = out / "latest.pt"
    if not latest.exists():
        return False
    import torch  # lazy: --dry-run on a machine without the train group still prints commands

    ck = torch.load(latest, map_location="cpu", weights_only=False)
    rows = cfg["num_envs"] * cfg["rollout_len"]
    if smoke:
        rows = SMOKE_PPO["num_envs"] * SMOKE_PPO["rollout_len"]
        return ck["global_step"] >= SMOKE_UPDATES * rows
    return ck["global_step"] >= int(cfg["total_steps"] // rows) * rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out-root", type=Path, default=ROOT / "runs", help="default: runs/")
    ap.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    ap.add_argument(
        "--smoke", action="store_true", help="tiny settings for a quick end-to-end test"
    )
    ap.add_argument("--from", dest="first", default=None, metavar="STAGE", help="start at STAGE")
    ap.add_argument("--only", default=None, metavar="STAGE", help="run just this stage")
    ap.add_argument("--force", action="store_true", help="do not skip finished stages")
    args = ap.parse_args()

    stages = load_stages()
    names = [s["name"] for s in stages]
    for sel in (args.first, args.only):
        if sel and sel not in names:
            ap.error(f"unknown stage {sel!r}; stages: {', '.join(names)}")
    if args.first:
        stages = stages[names.index(args.first) :]
    if args.only:
        stages = [s for s in stages if s["name"] == args.only]

    out_root = args.out_root.resolve()
    if out_root.is_relative_to(
        ROOT
    ):  # keep the recorded paths repo-relative (commands run in ROOT)
        out_root = out_root.relative_to(ROOT)
    t_all = time.perf_counter()
    for cfg in stages:
        name = cfg["name"]
        if cfg.get("script") == "bc_smoke":
            cmd = bc_command(cfg, out_root, args.smoke)
        else:
            cmd = ppo_command(cfg, out_root, args.smoke)
        shown = "python -u " + " ".join(shlex.quote(c) for c in cmd)
        if not args.force and is_done(cfg, out_root, args.smoke):
            print(f"== {name}: finished, skipping")
            continue
        print(f"== {name}\n{shown}\n", flush=True)
        if args.dry_run:
            continue
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, "-u", *cmd], cwd=ROOT)
        if r.returncode != 0:
            print(f"== {name}: FAILED (exit {r.returncode}); rerun to resume", file=sys.stderr)
            return r.returncode
        print(f"== {name}: done in {time.perf_counter() - t0:.0f}s", flush=True)
    if not args.dry_run:
        print(f"chain finished in {(time.perf_counter() - t_all) / 3600:.2f} h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
