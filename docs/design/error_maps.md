# Error maps: chess-engine-style analysis of Balatro games (package B4, `exp/error-maps`)

Question: for a recorded game of the reference policy (`runs/e2e_long_fix/latest.pt`), at which
decisions did a *different* action have a clearly higher chance of winning the run, and by how much?
(`fabledan_notlar.md` S2, findings 9 and 11: a single counterfactual future carries almost no
information because the seed explains only ~14% of the outcome variance, so every comparison here
averages many reseeded futures and is cut short by a calibrated P(win) head.)

Code: `python/balatro_rl/analysis/{features,pwin,errmap,describe,report}.py`,
`scripts/errmap_pwin.py` (Part 1), `scripts/errmap_analyze.py` (Part 2), `tests_py/test_errmap.py`.
Nothing existing was modified. Outputs live in `runs/errmaps/` (git-ignored).

## 1. Calibrated P(win) head (Part 1)

* **Features.** The input of the critic (`[encoder summary, GRU memory]`, 256 values) captured with a
  forward hook (`PolicyRunner`); the policy checkpoint is never modified or re-saved.
* **Data** (`errmap_pwin.py collect`): the reference policy plays the first episode of 3000 games per
  shard on base seed `333_000_001 + 1009 s` (never the dev / holdout / val bases); even shards are
  deterministic, odd shards sampled. Every BlindSelect and Shop decision is kept, other phases with
  probability 10% (about 100 rows per game). Label = the game was won (done code 2). Stored as
  float16 npz shards.
* **Head** (`pwin.PWinHead`): input standardisation, MLP 256-256-128-1 (GELU, dropout 0.1), BCE,
  AdamW + one-cycle, early stopping on validation BCE, then one scalar **temperature** fitted on the
  validation games. Split 80/10/10 *by game* (hash of the game id): train / validation (stopping,
  temperature) / test (all numbers below). Saved as `runs/errmaps/pwin_head.pt` (+
  `pwin_head_report.json`).
* Compared with the raw critic `V(s)` as a predictor of the win (AUC): V is trained on a shaped,
  discounted return, the head on the win bit.

**Training run actually done** (6 shards: 3000 + 5 x 1500 games = 10 500 games, 1 106 429 rows; test
split = 1076 games, 111 729 rows; head 256-128-64-1, dropout 0.3, AdamW wd 1e-2, lr 3e-4, 6 epochs,
fitted temperature T = 0.957, `runs/errmaps/pwin_head.pt`, 171 KB). Held-out games, all decisions:

| | rows | win rate | log-loss | const log-loss | AUC | ECE (10 bins) | Brier |
|---|---|---|---|---|---|---|---|
| P(win) head | 111 729 | 0.659 | **0.550** | 0.642 | **0.744** | **0.025** | 0.187 |
| raw critic V(s) | | | (not a probability) | | 0.580 | | |
| V + per-ante intercept, logistic (calibrated baseline) | | | 0.607 | | 0.708 | 0.084 | 0.210 |

By ante (test games; AUC within the ante, the critic's AUC next to it):

| ante | rows | win rate | log-loss | AUC head | AUC V | ECE |
|---|---|---|---|---|---|---|
| 1 | 10 220 | 0.629 | 0.655 | 0.564 | 0.571 | 0.031 |
| 2 | 12 584 | 0.643 | 0.630 | 0.634 | 0.632 | 0.033 |
| 3 | 14 398 | 0.643 | 0.607 | 0.684 | 0.679 | 0.032 |
| 4 | 15 567 | 0.636 | 0.575 | 0.733 | 0.731 | 0.030 |
| 5 | 16 045 | 0.645 | 0.551 | 0.756 | 0.762 | 0.026 |
| 6 | 16 011 | 0.643 | 0.537 | 0.775 | 0.782 | 0.035 |
| 7 | 14 744 | 0.691 | 0.503 | 0.781 | 0.794 | 0.024 |
| 8 | 12 031 | 0.754 | 0.351 | 0.887 | 0.884 | 0.022 |

Reliability (test, 10 equal-width bins of predicted P(win); predicted -> observed): 0.06 -> 0.07,
0.15 -> 0.20, 0.25 -> 0.26, 0.35 -> 0.33, 0.45 -> 0.42, 0.56 -> 0.52, 0.65 -> 0.62, 0.75 -> 0.74,
0.85 -> 0.82, 0.94 -> 0.93 (a slight over-prediction of about 3 points in the middle, within what 1076
correlated test games can resolve). By phase: BlindSelect AUC 0.735, Shop 0.746, Play 0.750, Pack
0.744. By collection mode: det 0.752 (log-loss 0.543), sampled 0.732 (0.561).
Full numbers: `runs/errmaps/pwin_head_report.json`.

Honest reading. (1) The head is a calibrated probability, the critic is not: overall AUC 0.744 vs
0.580 because V is a discounted shaped return that is not comparable across antes; *within an ante*
the head and V rank states equally well (AUC differences of +-0.01). (2) The head does not extract
information beyond the critic: a 2-feature MLP on (V, ante) reaches the same validation BCE
(0.5247 vs 0.5245; scratch experiment, a plain logistic on V is worse because V is not linear in
the logit). The 256 features overfit quickly (best validation epoch is 2 of 12 with the first
setup (256-256-128, dropout 0.1); the final regularised setup is flat after epoch 3), because only 10 500 games carry
independent labels. (3) The AUC ceiling of ~0.75 is mostly the outcome randomness of finding 11,
not a head defect; what the head buys is the calibration that lets the analyzer replace a 0 / 1
leaf (variance 0.23) by a probability.


## 2. Analyzer (Part 2)

`scripts/errmap_analyze.py` -> `analysis/errmap.py`. For each game:

1. **Trunk.** The game is replayed in a `VecEnv` (recorded actions from `ref_det.pkl`, or the
   deterministic policy live: game `i` = `Game(base_seed, env_index=i)`). The frozen policy runs on
   every step (recurrent state included) and the P(win) head scores every state: this is the **eval
   bar**. At each BlindSelect and Shop decision (`--pack` adds Pack decisions) the state is exported,
   every legal mask-conforming action is enumerated (`Game.legal_parts()` filtered by
   `select.conforms`) and scored with the policy's structured log-probability (kind, arg0, arg1 and
   card steps, via the model's own teacher-forced path); the **played action plus the top `k-1 = 2`
   alternatives** by probability are kept (alternatives below `--min-alt-prob 0.01` are dropped; a
   decision without alternatives is skipped).
2. **Screen.** For each kept action and each of `M = 8` futures: `import_state` of the decision state
   with `reseed = seed(game, decision, j, stage)` (**all streams**, draw pile order, face-down
   cards), apply the action, let the deterministic reference policy play on (its GRU memory
   continues from the trunk's memory at the decision, which holds only what it had observed), until
   the end of the next boss fight and the shop after it (`--horizon 1`; 2 = one ante more), and score
   the leaf with the P(win) head. A run that ends first counts 0 / 1 exactly. `Q(a)` = mean over
   futures; the advantage of alternative `a` is the **paired** difference to the played action and
   its standard error `std(d) / sqrt(M)`.
3. **Confirm.** Per game the ~5 most suspicious decisions (largest lower confidence bound
   `d - SE` of the best alternative, with `d > 0`) are re-run with `M = 64` futures and **fresh
   seeds** (stage 2 is a different seed stream, so selecting on screening noise does not bias the
   confirmation). A **mistake** is reported when the confirmed advantage of the best alternative is
   above `--threshold` (3 points of P(win)) and above `--z` (2) standard errors. Decisions that were
   only screened are never flagged.

**Fairness.** (a) Every future is reseeded (`import_state(reseed=...)`, streams `None` = all):
nothing of the real future is used; without the reseed a branch replays the real draws (a test
checks both properties). The policy only sees the observation, never the state. (b) **Common random
numbers**: all alternatives of a decision share the seed of future `j`, so they face the same
hidden draws as far as the streams they consume line up (the streams are independent, so an extra
purchase does not shift the deck shuffles), which makes the paired difference much tighter than the
difference of two means. (c) The leaf value comes from a head trained on the policy's own play,
so `Q` is `Q^pi` for this policy and horizon, not the value of optimal play: a "better action" means
better *given that the reference plays on*. (d) The position of Mr. Bones, Hieroglyph etc. is not
special-cased; the boss counter looks at the observation only.

## 2b. Cost (measured, **GPU shared and contended** by another training job: 100% utilisation, numbers
are pessimistic)

Settings of the table: k = 3, M = 8 screen, 5 x M = 64 confirm, horizon 1 boss fight, 2048 slots.

| run | games | decisions analysed | rollout steps | wall | per game |
|---|---|---|---|---|---|
| smoke (first 3 dev losses: games 9, 10, 16) | 3 | 151 | 115 k | 64 s (1024 slots, batches underfilled) | 21 s |
| 24 dev losses in one chunk | 24 | 1013 | 819 k (34 k per game) | 89 s | 3.7 s |

A future of one ante is about 28 env steps on average (a 186-step game is 24 rounds), so the
paper estimate of "25 steps" holds; the wall time is set by the number of *batched iterations*, not
by steps, so a chunk must be large (default 48 games) and the per-future step cap is 300
(`max_rollout_steps`; a MoveJoker loop would otherwise stretch the last wave to thousands of
iterations, a first run with a cap of 1500 took 3.4 min for 3 games). Under 0.1% of the futures
(20 of ~24 000 screened) hit the cap and are valued by the head at that state. **Extrapolation to all 315 dev losses:
315 x 3.7 s ~ 19 min** on the contended GPU (steady state, one chunk per 48 games); an idle GPU
should be faster. Per game: ~42 decisions analysed; the trunk pass costs ~14 s per chunk (included above). Memory: states are
kept as JSON strings (12 KB each), ~50 MB per 48-game chunk; VRAM < 3 GB at 2048 slots.

Commands (all under `flock runs/.gpu.lock` when the GPU is shared):

```
# (1) head at full size: collect (about 100 rows per game; 20 shards x 5000 games = ~10 M rows is
#     more than needed, 10 500 games gave the numbers above) and train
for s in 0 1 2 3 4 5 6 7; do flock runs/.gpu.lock uv run --no-sync python scripts/errmap_pwin.py \
  collect --checkpoint runs/e2e_long_fix/latest.pt --out runs/errmaps/pwin_data --shards $s --games 3000; done
flock runs/.gpu.lock uv run --no-sync python scripts/errmap_pwin.py train \
  --data runs/errmaps/pwin_data --out runs/errmaps/pwin_head.pt
# (2) all dev losses
flock runs/.gpu.lock uv run --no-sync python scripts/errmap_analyze.py --pkl runs/analysis/ref_det.pkl \
  --losses all --out runs/errmaps/dev_losses
# (3) N holdout games (plays the det policy live, analyses its losses)
flock runs/.gpu.lock uv run --no-sync python scripts/errmap_analyze.py --set holdout --n-games 500 \
  --only-losses --out runs/errmaps/holdout500
```

Note on statistical honesty: with the thresholds above (3 points and 2 SE one-sided, 5 rechecked
decisions x 2 alternatives per game) about 0.2 false flags per game are expected under the null, so
roughly one in six flagged mistakes in the 24-game run (31 flagged, 1.3 per game) may be noise;
the biggest ones (+10 points and more with SE ~ 2-4) are far from it. Raise `--z` / `--threshold`
or `--m-confirm` for fewer, surer flags. Consistency check on the smoke games: the mean `Q(played)`
over decisions is 0.357 against a mean trunk P(win) of 0.365 (correlation 0.88).


## 3. How to read the report

`report.md` (per directory of games): header with the settings, counts of analysed decisions and
confirmed mistakes, mistakes by phase and ante (rate per analysed decision), the most common
mistake types (`played -> better`, e.g. `EndShop -> Buy Joker` = left the shop without buying a
joker, `Reroll -> EndShop` = rerolled instead of leaving, `SellJoker -> ...` = sold a joker), the
top-20 blunders as one narrated line each (context, played vs better action with the policy's
probabilities, `Q` before / after, confirmed `dP(win) +- SE`, screening estimate, P(win) before the
decision), and a per-game table with an ASCII eval bar (P(win) at each BlindSelect / Shop point,
`▁` low to `█` high). `game_XXXXX.json` has everything: the full per-step curve (phase, ante, P(win),
critic value) and every analysed decision with all kept actions, `Q`, differences and SEs of both
stages.

Caveats: P(win) of a leaf is only as good as the head (ECE above); alternatives outside the policy's
top 3 are not examined; MoveJoker permutations are legal actions and may appear as alternatives
(they matter for xMult order but are cheap to take); a few futures can hit the rollout step cap
(a MoveJoker loop; counted in the log as `capped`, valued by the head); decisions inside Play
(which cards to play or discard) are not analysed.
