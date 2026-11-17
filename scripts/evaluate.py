"""Evaluate checkpoints on a fixed game set (default dev: 1000 games, seed ``TEST_SEED``) against baselines.

    uv run --no-sync python scripts/evaluate.py --checkpoint runs/e2e/best.pt --name e2e
    uv run --no-sync python scripts/evaluate.py --checkpoint runs/s1/best.pt --route play

Baselines (teacher = fair-information heuristic, BC checkpoint, random) are cached in
``runs/eval/baselines_<games>[_route][_<deck>_<stake>].json`` (the deck/stake part only for a
non-default ``--deck`` / ``--stake``, so the Red/White cache names are unchanged) and compared
*paired* (same games). With ``--route``
(a curriculum phase set such as ``play``) the model plays only those phases, the teacher the rest
(baselines are then teacher / BC under the same routing).
``--set {dev,holdout,val}`` / ``--seed-base`` pick the games (see ``evaluate.py`` docs; default dev
is bit-identical to the historical protocol); a non-dev set adds ``_<set>`` to the baseline cache
and result names. ``--memory {zero,shuffle}`` is the recurrent-memory knockout (adds ``_mem<mode>``).
Results go to ``runs/eval/<name>.json`` (``<name>`` gets ``_<deck>_<stake>`` for a non-default
configuration unless ``--name`` is given).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from balatro_rl.model import load_checkpoint
from balatro_rl.train import evaluate as ev

ROOT = Path(__file__).resolve().parents[1]


def _dump(r: ev.EvalResult) -> dict:
    d = r.summary()
    d["antes"], d["wins"] = r.antes, r.wins
    return d


def _load(d: dict) -> ev.EvalResult:
    antes, wins = d.pop("antes"), d.pop("wins")
    d["ante_hist"] = {int(k): v for k, v in d["ante_hist"].items()}
    return ev.EvalResult(**d, antes=antes, wins=wins)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument(
        "--set",
        dest="eval_set",
        choices=sorted(ev.EVAL_SETS),
        default="dev",
        help="named game set (dev = historical test set, holdout = 10k, val = 512)",
    )
    ap.add_argument("--seed-base", type=int, default=None, help="raw base seed (overrides --set)")
    ap.add_argument(
        "--games", type=int, default=None, help="default: the set's size (dev 1000, holdout 10000)"
    )
    ap.add_argument(
        "--memory",
        choices=ev.MEMORY_MODES,
        default="normal",
        help="recurrent-memory knockout: zero = reset hidden each step, shuffle = permute across envs",
    )
    ap.add_argument("--modes", default="det,sample", help="comma list of det,sample")
    ap.add_argument("--route", default=None, help="phases the model plays (rest: teacher)")
    ap.add_argument("--bc", type=Path, default=ROOT / "runs" / "bc" / "model.pt")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument(
        "--move-limit",
        type=int,
        default=None,
        help="evaluation repetition guard: mask MoveJoker after this many consecutive moves",
    )
    ap.add_argument("--max-episode-steps", type=int, default=ev.MAX_EPISODE_STEPS)
    ap.add_argument("--deck", default="red", help="deck name of every game (default red)")
    ap.add_argument("--stake", default="white", help="stake name of every game (default white)")
    ap.add_argument(
        "--endless", action="store_true", help="continue after the ante-8 win (see run_games)"
    )
    ap.add_argument("--endless-max-ante", type=int, default=12)
    args = ap.parse_args()
    set_seed, set_games = ev.EVAL_SETS[args.eval_set]
    if args.games is None:
        args.games = set_games
    seed = set_seed if args.seed_base is None else args.seed_base
    set_suffix = ""
    if args.seed_base is not None:
        set_suffix = f"_seed{args.seed_base}"
    elif args.eval_set != "dev":
        set_suffix = f"_{args.eval_set}"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    out_dir = ROOT / "runs" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    default_cfg = (args.deck, args.stake) == ("red", "white")
    cfg_suffix = "" if default_cfg else f"_{args.deck}_{args.stake}"
    mem_suffix = "" if args.memory == "normal" else f"_mem{args.memory}"
    if args.endless:
        cfg_suffix += f"_endless{args.endless_max_ante}"
    name = (args.name or args.checkpoint.parent.name + cfg_suffix) + set_suffix + mem_suffix
    kw = {
        "max_episode_steps": args.max_episode_steps,
        "deck": args.deck,
        "stake": args.stake,
        "seed": seed,
    }
    if args.endless:
        kw.update(endless=True, endless_max_ante=args.endless_max_ante)
    suffix = (f"_{args.route}" if args.route else "") + cfg_suffix + set_suffix

    base: dict[str, ev.EvalResult] = {}
    cache = out_dir / f"baselines_{args.games}{suffix}.json"
    if not args.no_baselines:
        if cache.exists():
            base = {k: _load(v) for k, v in json.loads(cache.read_text()).items()}
        else:
            bc, _ = load_checkpoint(args.bc, dev)
            if not args.route:
                base["random"] = ev.run_games("random", ev.random_actor, args.games, **kw)
            base["teacher"] = ev.run_games("teacher", ev.teacher_actor, args.games, **kw)
            base["bc_det"] = ev.evaluate_model(
                bc, args.games, True, args.route, name="bc_det", **kw
            )
            base["bc_sample"] = ev.evaluate_model(
                bc, args.games, False, args.route, name="bc_sample", **kw
            )
            cache.write_text(json.dumps({k: _dump(v) for k, v in base.items()}))
        for r in base.values():
            print(r.line())

    model, ck = load_checkpoint(args.checkpoint, dev)
    results = {}
    for mode in args.modes.split(","):
        det = mode == "det"
        r = ev.evaluate_model(
            model,
            args.games,
            det,
            args.route,
            name=f"{name}_{mode}",
            move_limit=args.move_limit,
            memory=args.memory,
            **kw,
        )
        results[mode] = r
        print(r.line())
        print("   ante histogram:", dict(sorted(r.ante_hist.items())))
        for bname, b in base.items():
            c = ev.compare(r, b)
            print(
                f"   vs {bname:<10} d_ante={c['ante_diff']:+.3f}+-{c['ante_diff_sem']:.3f} "
                f"d_win={100 * c['win_diff']:+.1f}+-{100 * c['win_diff_sem']:.1f}%"
            )
    payload = {m: _dump(r) for m, r in results.items()}
    payload["set"] = "custom" if args.seed_base is not None else args.eval_set
    payload["seed_base"], payload["memory"] = seed, args.memory
    payload["checkpoint"] = str(args.checkpoint)
    payload["deck"], payload["stake"] = args.deck, args.stake
    payload["update"] = ck.get("update")
    payload["global_step"] = ck.get("global_step")
    (out_dir / f"{name}.json").write_text(json.dumps(payload))


if __name__ == "__main__":
    main()
