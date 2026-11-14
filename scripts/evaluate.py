"""Evaluate checkpoints on the fixed test set (1000 games, seed ``TEST_SEED``) against baselines.

    uv run --no-sync python scripts/evaluate.py --checkpoint runs/e2e/best.pt --name e2e
    uv run --no-sync python scripts/evaluate.py --checkpoint runs/s1/best.pt --route play

Baselines (teacher = fair-information heuristic, BC checkpoint, random) are cached in
``runs/eval/baselines_<games>[_route].json`` and compared *paired* (same games). With ``--route``
(a curriculum phase set such as ``play``) the model plays only those phases, the teacher the rest
(baselines are then teacher / BC under the same routing).
Results go to ``runs/eval/<name>.json``.
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
    ap.add_argument("--games", type=int, default=ev.TEST_GAMES)
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
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    out_dir = ROOT / "runs" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or args.checkpoint.parent.name
    kw = {"max_episode_steps": args.max_episode_steps}
    suffix = f"_{args.route}" if args.route else ""

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
    payload["checkpoint"] = str(args.checkpoint)
    payload["update"] = ck.get("update")
    payload["global_step"] = ck.get("global_step")
    (out_dir / f"{name}.json").write_text(json.dumps(payload))


if __name__ == "__main__":
    main()
