"""Error-hall probe: what does a checkpoint do at the mined mistake states?

Cheap (one forward pass per state, no rollouts, seconds): every pool state is imported, the policy
acts on the observation with a zero memory, and we report per category how often it repeats the
mistaken action, how often it picks the action the error map found better, and the probability
it gives to each. Run it on the held-out games of the pool (``--part holdout``, the games the
trainer left out with ``--error-hall-holdout``) to see generalisation, and on ``--part train``
for the states the hall trained on. Compare checkpoints (control / hall) on the same pool.

    uv run --no-sync python scripts/errhall_probe.py --pool runs/errhall/st3.jsonl --holdout 0.1 \
        --part holdout --checkpoint runs/st3_ctrl/latest.pt --checkpoint runs/st3_hall/latest.pt

Caveats: the memory is zero (the policy that made the mistake had a GRU state); ``better`` is
better *given that the reference policy plays on* and was measured with noise (about one in six
flagged mistakes may be a false flag), so a rise of P(better) is evidence, not proof.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from balatro_rl import BalatroVecEnv
from balatro_rl.analysis.features import PolicyRunner
from balatro_rl.analysis.hall import read_pool
from balatro_rl.model import load_checkpoint
from balatro_rl.model.inputs import prepare


def probe(runner: PolicyRunner, recs: list[dict], batch: int = 256) -> list[dict]:
    """Per record: ``repeat`` (greedy action == played), ``fix`` (== better), ``p_played``,
    ``p_better`` (policy probability of the structured action)."""
    dev = runner.dev
    out: list[dict] = []
    env = BalatroVecEnv(batch, seed=1, max_episode_steps=3000)
    env.reset()
    for b0 in range(0, len(recs), batch):
        chunk = recs[b0 : b0 + batch]
        for i, r in enumerate(chunk):
            env.import_state(i, r["state"])
        n = len(chunk)
        o = {k: v[:n] for k, v in env.obs.items()}
        m = {k: v[:n] for k, v in env.masks.items()}
        prepared = prepare(o, m, dev)
        h0 = runner.init_hidden(n)
        act, _, _, _ = runner.run(o, m, h0, True, prepared=prepared)
        act = act.cpu().numpy()
        played = torch.tensor([r["played_action"] for r in chunk], device=dev, dtype=torch.long)
        better = torch.tensor([r["better_action"] for r in chunk], device=dev, dtype=torch.long)
        lp_p, _ = runner.logp_of(prepared, h0, played)
        lp_b, _ = runner.logp_of(prepared, h0, better)
        for i, r in enumerate(chunk):
            out.append(
                {
                    "category": r["category"],
                    "repeat": bool((act[i] == np.array(r["played_action"])).all()),
                    "fix": bool((act[i] == np.array(r["better_action"])).all()),
                    "p_played": float(lp_p[i].exp()),
                    "p_better": float(lp_b[i].exp()),
                }
            )
    return out


def table(rows: list[dict], top: int) -> str:
    cats = defaultdict(list)
    for r in rows:
        cats["ALL"].append(r)
        cats[r["category"]].append(r)
    order = ["ALL"] + [c for c, v in sorted(cats.items(), key=lambda kv: -len(kv[1])) if c != "ALL"]
    lines = [
        "| category | n | repeats the mistake | picks the better action | P(played) | P(better) |"
    ]
    lines.append("|---|---|---|---|---|---|")
    for c in order[: top + 1]:
        v = cats[c]
        lines.append(
            f"| {c} | {len(v)} | {np.mean([x['repeat'] for x in v]):.1%} | "
            f"{np.mean([x['fix'] for x in v]):.1%} | {np.mean([x['p_played'] for x in v]):.3f} | "
            f"{np.mean([x['p_better'] for x in v]):.3f} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--pool", required=True, help="pool file(s), comma separated")
    ap.add_argument("--checkpoint", type=Path, action="append", required=True)
    ap.add_argument("--holdout", type=float, default=0.0, help="the trainer's --error-hall-holdout")
    ap.add_argument("--part", choices=["train", "holdout", "all"], default="holdout")
    ap.add_argument("--min-dq", type=float, default=0.0)
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()
    recs = read_pool(a.pool, min_dq=a.min_dq, holdout=a.holdout, part=a.part)
    recs = [r for r in recs if r.get("played_action") and r.get("better_action")]
    assert recs, "no pool records with actions (mine with the current errhall_mine.py)"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{len(recs)} pool states ({a.part}, holdout fraction {a.holdout})")
    for ck in a.checkpoint:
        model, _ = load_checkpoint(ck, dev)
        rows = probe(PolicyRunner(model, seed=1), recs)
        print(f"\n### {ck}\n")
        print(table(rows, a.top))


if __name__ == "__main__":
    main()
