"""Ensemble vote (D4) and cross-play head-swap matrix (D10) evaluation.

    # D4: vote of three checkpoints on N games from a seed base
    uv run --no-sync python scripts/ensemble_eval.py --checkpoints a.pt,b.pt,c.pt \
        --mode vote --games 10000 --seed-base 123 --name vote3

    # D10 single routing (model index per phase)
    uv run --no-sync python scripts/ensemble_eval.py --checkpoints a.pt,b.pt \
        --mode route --route-models shop=1,blind=0,play=0,pack=0 --games 1000 --name x

    # D10 full table: pure models + for each phase p and ordered pair (A, B): A plays all but p,
    # B plays p
    uv run --no-sync python scripts/ensemble_eval.py --checkpoints a.pt,b.pt,c.pt --matrix \
        --games 1000 --seed-base 123 --name xplay

Writes raw JSON (same layout as scripts/evaluate.py: ``{"det": {... antes, wins}, ...}``) to
``--out-dir`` (default: ``runs/eval`` of the main checkout).
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch
from balatro_rl.model import load_checkpoint
from balatro_rl.train import evaluate as ev
from balatro_rl.train.ensemble import PHASES, EnsembleActor, parse_route_models

MAIN_EVAL = Path("/home/yuse/Desktop/balatroRL/runs/eval")


def _dump(r: ev.EvalResult) -> dict:
    d = r.summary()
    d["antes"], d["wins"] = r.antes, r.wins
    return d


def _save(out_dir: Path, name: str, r: ev.EvalResult, extra: dict) -> None:
    payload = {"det": _dump(r), **extra}
    (out_dir / f"{name}.json").write_text(json.dumps(payload))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", required=True, help="comma list of checkpoint paths")
    ap.add_argument("--mode", choices=("vote", "route"), default="vote")
    ap.add_argument("--vote", choices=("mean_logp", "mean_prob"), default="mean_logp")
    ap.add_argument("--route-models", default=None, help="e.g. shop=1,blind=0,play=0,pack=0")
    ap.add_argument("--matrix", action="store_true", help="run the D10 cross-play table")
    ap.add_argument("--games", type=int, default=ev.TEST_GAMES)
    ap.add_argument("--seed-base", type=int, default=ev.TEST_SEED)
    ap.add_argument("--name", default="ensemble")
    ap.add_argument("--out-dir", type=Path, default=MAIN_EVAL)
    ap.add_argument("--deck", default="red")
    ap.add_argument("--stake", default="white")
    ap.add_argument("--max-episode-steps", type=int, default=ev.MAX_EPISODE_STEPS)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = [Path(p) for p in args.checkpoints.split(",")]
    models = [load_checkpoint(p, dev)[0] for p in paths]
    labels = [p.parent.name for p in paths]
    kw = {
        "seed": args.seed_base,
        "max_episode_steps": args.max_episode_steps,
        "deck": args.deck,
        "stake": args.stake,
    }
    base_extra = {
        "checkpoints": [str(p) for p in paths],
        "seed_base": args.seed_base,
        "deck": args.deck,
        "stake": args.stake,
    }

    def run(name, sel, mode, route_models=None):
        actor = EnsembleActor([models[i] for i in sel], mode, args.vote, route_models)
        r = ev.run_games(name, actor, args.games, **kw)
        extra = {
            **base_extra,
            "mode": mode,
            "vote": args.vote,
            "models": sel,
            "route_models": route_models,
        }
        _save(args.out_dir, name, r, extra)
        print(r.line(), flush=True)
        return r

    if not args.matrix:
        rm = parse_route_models(args.route_models, len(models)) if args.mode == "route" else None
        run(args.name, list(range(len(models))), args.mode, rm)
        return

    results: dict[str, ev.EvalResult] = {}
    n = len(models)
    for i in range(n):
        results[f"pure:{labels[i]}"] = run(f"{args.name}_pure_{labels[i]}", [i], "route")
    for p, ph in enumerate(PHASES):
        for a, b in itertools.permutations(range(n), 2):
            rm = [0] * len(PHASES)  # indices into the [a, b] sub-list: 0 = A, 1 = B
            rm[p] = 1
            key = f"{ph}:{labels[a]}<-{labels[b]}"
            results[key] = run(f"{args.name}_{ph}_{labels[a]}_by_{labels[b]}", [a, b], "route", rm)
    print("\nD10 cross-play (A plays all but phase, B plays phase): win%  ante  d_win vs A pure")
    for key, r in results.items():
        line = f"{key:<40} win={100 * r.win_rate:5.1f}%  ante={r.mean_ante:.2f}"
        if not key.startswith("pure:"):
            _, pair = key.split(":")
            a_lab = pair.split("<-")[0]
            c = ev.compare(r, results[f"pure:{a_lab}"])
            line += f"  d_win={100 * c['win_diff']:+.1f}+-{100 * c['win_diff_sem']:.1f}"
        print(line)


if __name__ == "__main__":
    main()
