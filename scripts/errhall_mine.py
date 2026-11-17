"""Error-hall mining: play games with a policy on TRAINING seeds, run the error-map pipeline
(screen + confirm with reseeded futures and common random numbers, ``analysis/errmap.py``) on the
decisions of the lost games (optionally also the won ones) and save, for every CONFIRMED mistake,
the exported state BEFORE the mistaken decision plus metadata, as a pool file for
``train_ppo.py --error-hall P --error-hall-pool FILE`` (``docs/design/error_hall.md``).

    flock runs/.gpu.lock uv run --no-sync python scripts/errhall_mine.py \
        --checkpoint runs/st3/latest.pt --head runs/errmaps/pwin_head_st3.pt \
        --games 1400 --seed-base 900000007 \
        --decks red,blue,yellow,green,black --stakes white,red,green,black,blue,purple,orange,gold \
        --out runs/errhall/st3.jsonl

``--estimate-only`` prints the cost estimate and exits. Cost knobs: ``--games``, ``--m`` (screen
futures), ``--m-confirm``, ``--n-confirm`` (decisions re-run with many futures per game), ``--k``
(played action + k-1 alternatives), ``--horizon``, ``--include-wins``.

The P(win) head must belong to the policy that plays the futures (its features are the critic
input of a model): train one per mining checkpoint with ``errmap_pwin.py`` (collect with the same
``--decks/--stakes``), or pass ``--head-model`` to read the head's features from a different
frozen model (a fixed ruler across checkpoints). The script warns when the head's recorded
checkpoint differs from the policy and prints a calibration check before the expensive part.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from balatro_rl.analysis import errmap, hall
from balatro_rl.analysis.features import PHASE_BLIND, PHASE_PACK, PHASE_SHOP, PolicyRunner
from balatro_rl.analysis.pwin import auc, load_head
from balatro_rl.analysis.report import build_report
from balatro_rl.model import load_checkpoint
from balatro_rl.train import evaluate as ev

DEFAULT_SEED_BASE = 900_000_007
# Cost model. Dev-loss run (docs/design/error_maps.md 2b, Red / White): 69.6 k rollout steps per
# lost game, 3.7 s per game. st2_v2_s1 on the st3 mix (5 decks x 8 stakes, losses without the
# looping games, 2048 slots, head model shadowed, measured): 25 decisions and 1133 futures of 21
# steps per game, 120 s per 48 games = 2.5 s per game at ~10 k steps/s (trunk included). The
# estimate is the pessimistic one (a game that hits the step limit is excluded: it holds ~1000
# near-identical decisions and would dominate the cost).
STEPS_PER_FUTURE = 22.0
DECISIONS_PER_GAME = 25
STEPS_PER_SEC = 10_000.0


def estimate(a, n_analysed: float) -> dict:
    futures = DECISIONS_PER_GAME * a.k * a.m + a.n_confirm * a.k * a.m_confirm
    steps = futures * STEPS_PER_FUTURE
    per_game = steps / STEPS_PER_SEC
    return {
        "futures_per_game": futures,
        "rollout_steps_per_game": steps,
        "seconds_per_analysed_game": per_game,
        "analysed_games": n_analysed,
        "minutes": n_analysed * per_game / 60,
    }


def env_kwargs(a) -> dict | None:
    kw = {}
    if a.decks:
        kw["decks"] = [x.strip() for x in a.decks.split(",") if x.strip()]
    if a.stakes:
        kw["stakes"] = [x.strip() for x in a.stakes.split(",") if x.strip()]
    return kw or None


def calibration_check(runner, head, head_runner, base, acts, codes, kw, n: int) -> None:
    """The head's P(win) at the first decision of ``n`` recorded games (all outcomes) against
    their win rate: a head that belongs to a different policy / game mix is visibly off."""
    n = min(n, len(acts))
    cfg = errmap.Config(phases=())
    _, info = errmap.run_trunk(
        runner, head, list(range(n)), base, dict(enumerate(acts)), cfg,
        env_kwargs=kw, head_runner=head_runner,
    )  # fmt: skip
    p = np.array([info[g]["curve"][0][3] for g in range(n)])
    y = np.array([float(codes[g] == 2) for g in range(n)])
    line = (
        f"head calibration check ({n} games): mean P(win) at the start {p.mean():.3f} "
        f"vs win rate {y.mean():.3f}"
    )
    if 0 < y.sum() < n:
        line += f", AUC of the start P(win) {auc(p, y):.3f}"
    print(line, flush=True)
    if abs(p.mean() - y.mean()) > 0.12:
        print(
            "WARNING: the P(win) head looks miscalibrated for this policy / game mix "
            "(train one with errmap_pwin.py collect --decks/--stakes, see error_hall.md)",
            file=sys.stderr,
            flush=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument(
        "--head", type=Path, default=Path("/home/yuse/Desktop/balatroRL/runs/errmaps/pwin_head.pt")
    )
    ap.add_argument("--head-model", type=Path, default=None,
                    help="checkpoint feeding the head (default: --checkpoint)")  # fmt: skip
    ap.add_argument(
        "--games", type=int, required=True, help="games to play (first episode of envs 0..N-1)"
    )
    ap.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    ap.add_argument("--decks", default=None, help="comma list (default: red)")
    ap.add_argument("--stakes", default=None, help="comma list (default: white)")
    ap.add_argument("--out", type=Path, required=True, help="pool file (.jsonl)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing pool file")
    ap.add_argument("--include-wins", action="store_true", help="analyse the won games too")
    ap.add_argument(
        "--include-truncated", action="store_true",
        help="also analyse games cut by the step limit (loops; excluded by default)",
    )  # fmt: skip
    ap.add_argument("--chunk", type=int, default=48, help="games per batch")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--m", type=int, default=8, help="screening futures")
    ap.add_argument("--m-confirm", type=int, default=64)
    ap.add_argument("--n-confirm", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--min-alt-prob", type=float, default=0.01)
    ap.add_argument("--threshold", type=float, default=0.03)
    ap.add_argument("--z", type=float, default=2.0)
    ap.add_argument("--pack", action="store_true", help="also analyse Pack decisions")
    ap.add_argument("--max-rollout-steps", type=int, default=300,
                    help="cap of one future (capped: valued by the head)")  # fmt: skip
    ap.add_argument("--slots", type=int, default=2048, help="parallel futures (VRAM ~ slots)")
    ap.add_argument(
        "--calib-games", type=int, default=96, help="head calibration check size (0 = skip)"
    )
    ap.add_argument(
        "--est-loss-frac", type=float, default=0.75, help="loss rate assumed by the estimate"
    )
    ap.add_argument("--estimate-only", action="store_true")
    ap.add_argument("--keep-games", type=Path, default=None,
                    help="also write the per-game error-map JSONs + report.md here")  # fmt: skip
    a = ap.parse_args()

    assert a.seed_base not in (ev.TEST_SEED, ev.HOLDOUT_SEED, ev.VAL_SEED), (
        "mine on training seeds, never on the dev / holdout / val sets"
    )
    n_an = a.games * (1.0 if a.include_wins else a.est_loss_frac)
    what = "all" if a.include_wins else f"losses, assumed {a.est_loss_frac:.0%}"
    est = estimate(a, n_an)
    print(
        f"estimate: {a.games} games, ~{n_an:.0f} analysed ({what}); "
        f"{est['futures_per_game']:.0f} futures = "
        f"{est['rollout_steps_per_game'] / 1000:.0f}k rollout steps "
        f"= {est['seconds_per_analysed_game']:.1f} s per analysed game "
        f"-> ~{est['minutes']:.0f} min "
        f"(measured on a shared GPU: pessimistic; slots {a.slots})",
        flush=True,
    )
    if a.estimate_only:
        return
    if a.out.exists() and not a.force:
        sys.exit(f"{a.out} exists (use --force to overwrite)")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(a.checkpoint, dev)
    runner = PolicyRunner(model, seed=1)
    head_runner = None
    if a.head_model is not None:
        hm, _ = load_checkpoint(a.head_model, dev)
        head_runner = PolicyRunner(hm, seed=2)
    head = load_head(a.head, dev)
    ck = torch.load(a.head, map_location="cpu", weights_only=False)["meta"].get("checkpoint")
    own = a.head_model or a.checkpoint
    if ck is not None and Path(ck).resolve() != Path(own).resolve():
        print(
            f"WARNING: the head was trained on features of {ck}, not of {own}; "
            "its P(win) may be off",
            file=sys.stderr,
            flush=True,
        )
    kw = env_kwargs(a)
    phases = (PHASE_BLIND, PHASE_SHOP) + ((PHASE_PACK,) if a.pack else ())
    cfg = errmap.Config(
        k=a.k, m_screen=a.m, m_confirm=a.m_confirm, n_confirm=a.n_confirm, horizon=a.horizon,
        min_alt_prob=a.min_alt_prob, threshold=a.threshold, z_min=a.z, phases=phases,
        slots=a.slots, max_rollout_steps=a.max_rollout_steps,
    )  # fmt: skip

    t_start = time.perf_counter()
    acts, codes_l = errmap.record_games(runner, a.seed_base, a.games, env_kwargs=kw)
    codes = dict(enumerate(codes_l))
    acts_all = dict(enumerate(acts))
    n_win = sum(c == 2 for c in codes_l)
    print(
        f"played {a.games} games on base {a.seed_base}: win {n_win / a.games:.1%} "
        f"({time.perf_counter() - t_start:.0f}s)",
        flush=True,
    )
    if a.calib_games > 0:
        calibration_check(runner, head, head_runner, a.seed_base, acts, codes, kw, a.calib_games)
    n_trunc = sum(c == 3 for c in codes_l)
    ids = [g for g in range(a.games) if a.include_wins or codes[g] != 2]
    if not a.include_truncated:
        # a looping game (3000 steps) holds ~1000 near-identical decisions and futures that loop
        # again: not strategy mistakes, and a large share of the cost
        ids = [g for g in ids if codes[g] != 3]
    print(
        f"{n_trunc} of {a.games} games hit the step limit"
        f"{' (analysed)' if a.include_truncated else ' (excluded)'}",
        flush=True,
    )
    real = estimate(a, len(ids))
    print(f"analysing {len(ids)} games, estimate ~{real['minutes']:.0f} min", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("")  # a killed run leaves the chunks written so far
    if a.keep_games:
        a.keep_games.mkdir(parents=True, exist_ok=True)
    pool = errmap.RolloutPool(runner, head, cfg, head_runner=head_runner)
    games_json: list[dict] = []
    records_all: list[dict] = []
    totals = {"env_steps": 0, "decisions": 0, "mistakes": 0}
    t0 = time.perf_counter()
    for c0 in range(0, len(ids), a.chunk):
        chunk = ids[c0 : c0 + a.chunk]
        tc = time.perf_counter()
        decisions, info = errmap.run_trunk(
            runner, head, chunk, a.seed_base, acts_all, cfg, env_kwargs=kw, head_runner=head_runner
        )
        stats = errmap.screen_and_confirm(pool, decisions, cfg)
        recs = []
        for g in chunk:
            if info[g]["outcome"] is not None and info[g]["outcome"] != codes[g]:
                print(f"WARNING game {g}: replay outcome {info[g]['outcome']} != {codes[g]}")
            res = errmap.finalize_game(g, decisions[g], info[g], cfg)
            by_step = {d.step: d.state for d in decisions[g]}
            for m in res["mistakes"]:
                recs.append(
                    hall.pool_record(
                        by_step[m["step"]], m, game=g, outcome=res["outcome"],
                        final_ante=res["final_ante"], seed_base=a.seed_base,
                        checkpoint=str(a.checkpoint),
                    )
                )  # fmt: skip
            if a.keep_games:
                (a.keep_games / f"game_{g:05d}.json").write_text(json.dumps(res))
                games_json.append(res)
        hall.write_pool(a.out, recs, append=True)
        records_all += recs
        steps = sum(stats.get(k, {}).get("env_steps", 0) for k in ("screen", "confirm"))
        capped = sum(stats.get(k, {}).get("capped", 0) for k in ("screen", "confirm"))
        futures = sum(stats.get(k, {}).get("jobs", 0) for k in ("screen", "confirm"))
        totals["capped"] = totals.get("capped", 0) + capped
        totals["futures"] = totals.get("futures", 0) + futures
        totals["env_steps"] += steps
        totals["decisions"] += stats["decisions"]
        totals["mistakes"] += len(recs)
        done = c0 + len(chunk)
        el = time.perf_counter() - t0
        print(
            f"chunk {c0 // a.chunk}: {len(chunk)} games, {stats['decisions']} decisions, "
            f"{len(recs)} confirmed mistakes ({len(recs) / len(chunk):.2f}/game), "
            f"{time.perf_counter() - tc:.0f}s, {steps:,} steps ({futures:,} futures, "
            f"{capped:,} capped) | total {totals['mistakes']} "
            f"mistakes in {done}/{len(ids)} games, {el / 60:.1f} min, "
            f"ETA {el / done * (len(ids) - done) / 60:.1f} min",
            flush=True,
        )
    totals.update(
        seconds=time.perf_counter() - t0,
        games_played=a.games,
        games_analysed=len(ids),
        win_rate=n_win / a.games,
        mistakes_per_analysed_game=totals["mistakes"] / max(len(ids), 1),
        seconds_per_analysed_game=(time.perf_counter() - t0) / max(len(ids), 1),
        rollout_steps_per_sec=totals["env_steps"] / max(time.perf_counter() - t0, 1e-9),
        estimate_minutes=real["minutes"],
        args={k: str(v) for k, v in vars(a).items()},
        pool=hall.summarize(records_all),
    )
    a.out.with_suffix(".summary.json").write_text(json.dumps(totals, indent=1))
    if a.keep_games and games_json:
        (a.keep_games / "report.md").write_text(build_report(games_json, totals))
    print(f"done: {json.dumps({k: v for k, v in totals.items() if k not in ('args', 'pool')})}")
    print(f"pool: {json.dumps(totals['pool']['category'])}")


if __name__ == "__main__":
    main()
