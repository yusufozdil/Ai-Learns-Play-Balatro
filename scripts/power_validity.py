"""D8: does the build-power meter (G1's Phi) predict the outcome of a game better than the xMult count?

Replays the recorded games of ``scripts/audit_agent.py`` (``Game(TEST_SEED, env_index=i)`` + the stored
action list; the simulator is deterministic), stops at the first Small-blind BlindSelect of every
ante, computes ``_engine.power`` there (several variants) and tabulates, per ante, the AUC of each
predictor for the final win (among the games that reach the ante).

    uv run --no-sync python scripts/power_validity.py collect --pkl runs/analysis/ref_det.pkl \
        --out runs/analysis/power_rows.pkl --procs 6
    uv run --no-sync python scripts/power_validity.py report --rows runs/analysis/power_rows.pkl

Predictors: ``xmult`` (number of xMult jokers, the set of ``analyze_losses.py``), ``money``,
``phi_v1`` (plain boss target), ``phi_v2`` (actual boss and rules), ``phi_v2s`` (v2, stop at the
target), ``phi_v2k32`` (v2 with K=32: the noise ceiling of the meter).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from balatro_rl import _engine

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_losses import XMULT  # noqa: E402

TEST_SEED = 10_000_019
VARIANTS = {  # name -> kwargs of _engine.power
    "phi_v1": dict(k=8, boss_aware=False, play_on=True),
    "phi_v2": dict(k=8, boss_aware=True, play_on=True),
    "phi_v2s": dict(k=8, boss_aware=True, play_on=False),
    "phi_v2k32": dict(k=32, boss_aware=True, play_on=True),
}
_DATA: dict = {}


def _work(i: int) -> list[dict]:
    d = _DATA["d"]
    acts = d["acts"][i]
    ev = d["games"][i]["ev"]
    fin = d["games"][i]["fin"]
    res = d["res"][i]
    outcome = {2: "win", 1: "loss"}.get(res[2], "trunc")
    stops = {}
    for j, e in enumerate(ev):
        if e["ph"] == "BlindSelect" and e["blind"] == "Small" and e["ante"] not in stops:
            stops[e["ante"]] = j
    at = {j: a for a, j in stops.items()}
    g = _engine.Game(TEST_SEED, env_index=i)
    rows = []
    for j, a in enumerate(acts):
        if j in at:
            s = json.loads(g.state_json())
            row = dict(
                game=i, ante=s["ante"], outcome=outcome, final_ante=fin["ante"], money=s["money"],
                n_jokers=len(s["jokers"]), boss=s["boss"],
                xmult=sum(x["id"] in XMULT for x in s["jokers"]),
            )  # fmt: skip
            sj = json.dumps(s)
            for name, kw in VARIANTS.items():
                r = _engine.power(sj, **kw)
                row[name] = r["phi"]
                if name == "phi_v2":
                    row["frac_beat"] = r["frac_beat"]
                    row["std"] = r["std"]
            rows.append(row)
        g.step_parts(*a)
    return rows


def collect(args: argparse.Namespace) -> None:
    _DATA["d"] = pickle.loads(Path(args.pkl).read_bytes())
    n = min(args.games, len(_DATA["d"]["acts"]))
    with Pool(args.procs) as p:
        out = p.map(_work, range(n), chunksize=4)
    rows = [r for g in out for r in g]
    Path(args.out).write_bytes(pickle.dumps(rows))
    print(f"{len(rows)} rows from {n} games -> {args.out}")


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """P(score of a winner > score of a loser), ties count 1/2 (Mann-Whitney)."""
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def auc_ci(score, y, rng, n=300):
    xs = []
    for _ in range(n):
        ix = rng.integers(0, len(y), len(y))
        xs.append(auc(score[ix], y[ix]))
    return np.nanpercentile(xs, [2.5, 97.5])


def logit_auc(X: np.ndarray, y: np.ndarray, folds: int = 5, seed: int = 0) -> float:
    """Cross-validated AUC of a (lightly ridged) logistic regression on standardised features."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    pred = np.zeros(len(y))
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        A = np.c_[np.ones(len(tr)), (X[tr] - mu) / sd]
        w = np.zeros(A.shape[1])
        for _ in range(30):
            p = 1 / (1 + np.exp(-A @ w))
            H = A.T @ (A * (p * (1 - p))[:, None]) + 1e-2 * np.eye(A.shape[1])
            w += np.linalg.solve(H, A.T @ (y[tr] - p) - 1e-2 * w)
        pred[te] = np.c_[np.ones(len(te)), (X[te] - mu) / sd] @ w
    return auc(pred, y)


def report(args: argparse.Namespace) -> None:
    rows = [r for r in pickle.loads(Path(args.rows).read_bytes()) if r["outcome"] != "trunc"]
    rng = np.random.default_rng(0)
    preds = ["xmult", "money", *VARIANTS]
    print("## AUC for the final win, per ante (games that reach the ante; 95% bootstrap CI)\n")
    print("| ante | n | win % | " + " | ".join(preds) + " | xmult+money | +phi_v2 |")
    print("|" + "---|" * (len(preds) + 5))
    for a in range(1, 9):
        rs = [r for r in rows if r["ante"] == a]
        if len(rs) < 30:
            continue
        y = np.array([r["outcome"] == "win" for r in rs], float)
        cells = []
        for p in preds:
            x = np.array([r[p] for r in rs], float)
            cells.append(f"{auc(x, y):.3f}" + (f" [{'-'.join(f'{v:.2f}' for v in auc_ci(x, y, rng))}]" if p in ("xmult", "phi_v2") else ""))  # fmt: skip
        base = np.array([[r["xmult"], r["money"]] for r in rs], float)
        both = np.c_[base, [r["phi_v2"] for r in rs]]
        cells.append(f"{logit_auc(base, y):.3f}")
        cells.append(f"{logit_auc(both, y):.3f}")
        print(f"| {a} | {len(rs)} | {100 * y.mean():.1f} | " + " | ".join(cells) + " |")
    print("\n## Mean predictor value, winners vs losers\n")
    print("| ante | xmult W / L | money W / L | phi_v1 W / L | phi_v2 W / L | frac_beat(v2) W / L | std(v2) |")
    print("|---|---|---|---|---|---|---|")
    for a in range(1, 9):
        rs = [r for r in rows if r["ante"] == a]
        if len(rs) < 30:
            continue
        w = [r for r in rs if r["outcome"] == "win"]
        l = [r for r in rs if r["outcome"] == "loss"]  # noqa: E741
        m = lambda g, k: np.mean([r[k] for r in g]) if g else float("nan")  # noqa: E731
        print(f"| {a} | " + " | ".join(f"{m(w, k):.2f} / {m(l, k):.2f}" for k in ("xmult", "money", "phi_v1", "phi_v2", "frac_beat")) + f" | {m(rs, 'std'):.2f} |")  # fmt: skip
    print("\n## Win rate by predictor tercile at ante 5 and 6\n")
    for a in (5, 6):
        rs = [r for r in rows if r["ante"] == a]
        if len(rs) < 30:
            continue
        y = np.array([r["outcome"] == "win" for r in rs], float)
        for p in ("xmult", "phi_v2"):
            x = np.array([r[p] for r in rs], float)
            q = np.quantile(x, [1 / 3, 2 / 3])
            b = np.digitize(x, q, right=True)
            print(f"ante {a} {p}: " + ", ".join(f"T{t + 1} {100 * y[b == t].mean():.1f}% (n={int((b == t).sum())})" for t in range(3) if (b == t).any()))  # fmt: skip
    print("\n## Phi split by boss at ante 5-7 (v2, mean Phi / win rate)\n")
    by: dict = {}
    for r in rows:
        if 5 <= r["ante"] <= 7:
            by.setdefault(r["boss"], []).append(r)
    for b, rs in sorted(by.items(), key=lambda kv: -len(kv[1]))[:12]:
        print(f"{b}: n={len(rs)} phi {np.mean([r['phi_v2'] for r in rs]):+.2f} win {100 * np.mean([r['outcome'] == 'win' for r in rs]):.0f}%")  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--pkl", default="runs/analysis/ref_det.pkl")
    c.add_argument("--out", default="runs/analysis/power_rows.pkl")
    c.add_argument("--games", type=int, default=1000)
    c.add_argument("--procs", type=int, default=6)
    c.set_defaults(fn=collect)
    r = sub.add_parser("report")
    r.add_argument("--rows", default="runs/analysis/power_rows.pkl")
    r.set_defaults(fn=report)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
