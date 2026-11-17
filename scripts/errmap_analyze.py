"""Error maps: chess-engine-style analysis of Balatro games (docs/design/error_maps.md).

Analyse the lost games of a recorded dev run (default: first 3 losses of runs/analysis/ref_det.pkl):

    flock runs/.gpu.lock uv run --no-sync python scripts/errmap_analyze.py \
        --pkl runs/analysis/ref_det.pkl --losses 3 --out runs/errmaps/smoke

All dev losses:  ... --pkl runs/analysis/ref_det.pkl --losses all --out runs/errmaps/dev_losses
Fresh holdout games (plays the deterministic policy first, analyses its losses):
    ... --set holdout --n-games 200 --only-losses --out runs/errmaps/holdout200

Output: ``game_XXXXX.json`` per game, ``report.md``, ``timing.json`` in ``--out``.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import torch
from balatro_rl.analysis import errmap
from balatro_rl.analysis.features import PHASE_BLIND, PHASE_PACK, PHASE_SHOP, PolicyRunner
from balatro_rl.analysis.pwin import load_head
from balatro_rl.analysis.report import build_report, load_games
from balatro_rl.model import load_checkpoint
from balatro_rl.train import evaluate as ev

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("/home/yuse/Desktop/balatroRL/runs/e2e_long_fix/latest.pt"))
    ap.add_argument("--head", type=Path, default=Path("/home/yuse/Desktop/balatroRL/runs/errmaps/pwin_head.pt"))
    ap.add_argument("--set", dest="eval_set", choices=sorted(ev.EVAL_SETS), default="dev")
    ap.add_argument("--pkl", type=Path, default=None, help="recorded actions (dev only)")
    ap.add_argument("--games", default=None, help="explicit game indices, e.g. 3,17,42")
    ap.add_argument("--losses", default=None, help="analyse the first N lost games (or 'all')")
    ap.add_argument("--n-games", type=int, default=None, help="live det play of this many games")
    ap.add_argument("--only-losses", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chunk", type=int, default=48, help="games per batch")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--m", type=int, default=8, help="screening futures")
    ap.add_argument("--m-confirm", type=int, default=64)
    ap.add_argument("--n-confirm", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=1, help="boss fights the futures run through")
    ap.add_argument("--min-alt-prob", type=float, default=0.01)
    ap.add_argument("--threshold", type=float, default=0.03)
    ap.add_argument("--z", type=float, default=2.0)
    ap.add_argument("--pack", action="store_true", help="also analyse Pack decisions")
    ap.add_argument("--slots", type=int, default=2048)
    ap.add_argument("--sample-rollouts", action="store_true")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(a.checkpoint, dev)
    head = load_head(a.head, dev)
    runner = PolicyRunner(model, seed=1)
    base, _ = ev.EVAL_SETS[a.eval_set]
    phases = (PHASE_BLIND, PHASE_SHOP) + ((PHASE_PACK,) if a.pack else ())
    cfg = errmap.Config(
        k=a.k, m_screen=a.m, m_confirm=a.m_confirm, n_confirm=a.n_confirm, horizon=a.horizon,
        min_alt_prob=a.min_alt_prob, threshold=a.threshold, z_min=a.z, phases=phases,
        deterministic=not a.sample_rollouts, slots=a.slots,
    )  # fmt: skip

    t_start = time.perf_counter()
    if a.pkl is not None:
        d = pickle.loads(a.pkl.read_bytes())
        acts_all = {i: [tuple(x) for x in g] for i, g in enumerate(d["acts"])}
        codes = {r[0]: r[2] for r in d["res"]}
    else:
        n = a.n_games or ev.EVAL_SETS[a.eval_set][1]
        acts_l, code_l = errmap.record_games(runner, base, n)
        acts_all = dict(enumerate(acts_l))
        codes = dict(enumerate(code_l))
        print(f"recorded {n} games: win {sum(c == 2 for c in code_l) / n:.1%} "
              f"({time.perf_counter() - t_start:.0f}s)")  # fmt: skip
    if a.games:
        ids = [int(x) for x in a.games.split(",")]
    else:
        ids = sorted(codes)
        if a.losses or a.only_losses:
            ids = [i for i in ids if codes[i] != 2]
            if a.losses and a.losses != "all":
                ids = ids[: int(a.losses)]
    print(f"analysing {len(ids)} games: {ids[:10]}{'...' if len(ids) > 10 else ''}")

    a.out.mkdir(parents=True, exist_ok=True)
    pool = errmap.RolloutPool(runner, head, cfg)
    totals = {"env_steps": 0, "decisions": 0}
    t0 = time.perf_counter()
    for c0 in range(0, len(ids), a.chunk):
        chunk = ids[c0 : c0 + a.chunk]
        tc = time.perf_counter()
        decisions, info = errmap.run_trunk(runner, head, chunk, base, acts_all, cfg)
        t_trunk = time.perf_counter() - tc
        stats = errmap.screen_and_confirm(pool, decisions, cfg)
        for g in chunk:
            if info[g]["outcome"] is not None and info[g]["outcome"] != codes[g]:
                print(f"WARNING game {g}: replay outcome {info[g]['outcome']} != {codes[g]}")
            res = errmap.finalize_game(g, decisions[g], info[g], cfg)
            (a.out / f"game_{g:05d}.json").write_text(json.dumps(res))
        steps = sum(stats.get(k, {}).get("env_steps", 0) for k in ("screen", "confirm"))
        totals["env_steps"] += steps
        totals["decisions"] += stats["decisions"]
        print(
            f"chunk {c0 // a.chunk}: {len(chunk)} games, {stats['decisions']} decisions, "
            f"trunk {t_trunk:.1f}s, screen {stats.get('screen', {}).get('seconds', 0):.1f}s "
            f"({stats.get('screen', {}).get('env_steps', 0):,} steps), confirm "
            f"{stats.get('confirm', {}).get('seconds', 0):.1f}s "
            f"({stats.get('confirm', {}).get('env_steps', 0):,} steps); "
            f"capped {stats.get('screen', {}).get('capped', 0)}",
            flush=True,
        )
    totals["seconds"] = time.perf_counter() - t0
    totals["games"] = len(ids)
    totals["seconds_per_game"] = totals["seconds"] / max(len(ids), 1)
    (a.out / "timing.json").write_text(json.dumps(totals))
    games = load_games(a.out)
    (a.out / "report.md").write_text(build_report(games, totals))
    print(f"done: {totals}")


if __name__ == "__main__":
    main()
