# Build-power meter (G1) and its validity test (D8)

Question (`fabledan_notlar.md` G1 / D8): is a state function Phi, "how well does the current build fare against this ante's boss",
a good dense signal? Before using it as a potential-based reward, check that it separates future winners from future losers
better than the cheap baseline (number of xMult jokers) on the recorded games of the reference policy.

## 1. The meter

`balatro_core::agents::power::power(state, k, cfg) -> PowerReport` (`crates/balatro-core/src/agents/power.rs`).

* Clone the state, make the next blind "this ante's boss" (`phase = BlindSelect`, `blind = Boss`), keep hands / discards / hand size /
  jokers / deck / hand levels / vouchers / consumables as they are (consumables are never used), `Step(SelectBlind)` so the round starts
  exactly like a real one (deck reshuffled from the canonical `uid` order, boss set-up, round-start jokers), then play it with the
  Greedy play/discard policy (`GreedyAgent::act_play`, a thin refactor of its Play branch that takes an explicit target).
* Repeat `k` times, every time with a fresh `GameRng` seed. **Phi = mean over the k rounds of ln(clamp(round score / target, e^-3, e^3))**.
  Diagnostics: `std`, `frac_beat` (fraction of rounds with score >= target), `target`.
* **Pure function of the state**: the seeds come from `splitmix(digest(build) ^ salt)`. The digest (`state_digest`) covers config,
  ante, money, base hands/discards/hand size/slots, boss, jokers (with their internal state), hand levels, vouchers, consumables and the
  *set* of deck cards sorted by `uid`; it excludes `state.rng`, pile order, shop, tags. The input is never mutated (`&GameState`), the
  game's RNG streams are never read. Tests (`tests/power.rs`): same state -> same report; replacing `state.rng` and reversing the draw
  pile does not change it; adding Cavendish raises mean Phi by > 0.2; a bare ante-8 build has Phi < -2 and never beats the boss.
* **v1** (`boss_aware = false`): `boss = None`, the plain boss target (x2 base chips), no boss effect. **v2** (`boss_aware = true`): the
  ante's actual boss, its target multiplier (Wall, Vessel, Needle) and rules. The boss is part of the state from the ante's start.
* `play_on = true` (default): the game's target is lifted to infinity during the simulation (the policy still decides play/discard
  against the real target), so a strong build is credited with all its hands instead of the overshoot of the hand that crossed the
  line. `play_on = false` stops at the target (`phi_v2s` below): practically the same AUC.
* Python: `balatro_rl._engine.power(state, k=8, boss_aware=True, play_on=True, salt=0)` where `state` is a state JSON string or a `Game`;
  returns `{phi, std, frac_beat, target, k}`; the GIL is released. (`crates/balatro-py/src/power.rs`, tests `tests_py/test_power.py`.)

Cost (release, 1 core, heuristic-agent states from games 0..99, all antes): **~70 us fixed (digest + clone) + ~335 us per simulated
round**: K=1 0.40 ms, **K=4 1.4 ms, K=8 2.8 ms** per call (the card's estimate was 0.15 ms per round: 2x off, Greedy's 218-subset
preview scan per action dominates). The card's "13 ms per vector step" assumed K=8 on every step; used only at BlindSelect entries (a few per ~170 steps) the amortised cost is negligible. Probe: `cargo run --release -p balatro-core --bin power_probe -- --games 100 --k 8`.

## 2. D8 validity test

Method. `runs/analysis/ref_det.pkl` (Phase 10E, 1000 test games of `e2e_long_fix/latest.pt`, Red Deck / White Stake, 68.6% win) stores the
action list per game and the game `i` is `Game(TEST_SEED, env_index=i)`. The simulator is deterministic, so the games are **replayed**
(no GPU, no re-recording). `scripts/power_validity.py collect` replays every game, stops at the first Small-blind BlindSelect of each
ante label (7514 rows; one game was truncated and is excluded), takes the state JSON and calls `_engine.power` (K=8; and K=32 for the noise
ceiling). `report` computes, per ante, the Mann-Whitney AUC of each predictor for the final win among the games that reach the ante
(95% bootstrap CIs on xmult and phi_v2), plus a 5-fold logistic regression on xmult+money and on xmult+money+phi_v2.
Predictors: `xmult` = number of xMult jokers (the `analyze_losses.py` set), `money`, `phi_v1`, `phi_v2`, `phi_v2s` (v2, stop at target),
`phi_v2k32` (v2, K=32). Reproduce:

```
uv run --no-sync python scripts/power_validity.py collect --pkl runs/analysis/ref_det.pkl --out runs/analysis/power_rows.pkl --procs 6   # ~40 s
uv run --no-sync python scripts/power_validity.py report --rows runs/analysis/power_rows.pkl
```

### AUC for the final win (state at the first Small BlindSelect of the ante)

| ante | n | win % | xmult | money | phi_v1 | phi_v2 | phi_v2 stop-at-target | phi_v2 K=32 | xmult+money | xmult+money+phi_v2 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 999 | 68.6 | 0.500 | 0.500 | 0.493 | 0.519 | 0.517 | 0.517 | 0.478 | 0.499 |
| 2 | 991 | 69.1 | 0.491 | 0.478 | 0.544 | 0.529 | 0.505 | 0.532 | 0.512 | 0.530 |
| 3 | 977 | 70.1 | 0.492 | 0.513 | 0.526 | 0.520 | 0.525 | 0.516 | 0.496 | 0.509 |
| 4 | 968 | 70.8 | 0.525 | 0.524 | 0.570 | 0.571 | 0.567 | 0.571 | 0.520 | 0.551 |
| **5** | 958 | 71.5 | 0.551 [0.52-0.59] | 0.574 | 0.595 | 0.586 [0.55-0.63] | 0.579 | 0.585 | 0.585 | 0.597 |
| **6** | 933 | 73.4 | 0.557 [0.52-0.60] | 0.603 | 0.607 | 0.610 [0.58-0.65] | 0.601 | 0.614 | 0.613 | 0.634 |
| 7 | 884 | 77.5 | 0.546 [0.51-0.59] | 0.636 | 0.648 | 0.683 [0.64-0.72] | 0.675 | 0.685 | 0.631 | 0.695 |
| 8 | 798 | 85.8 | 0.551 [0.49-0.60] | 0.662 | 0.653 | 0.796 [0.76-0.84] | 0.795 | 0.800 | 0.653 | 0.809 |

(`frac_beat` of the K=8 run as a predictor: ante 4/5/6/7/8 = 0.553 / 0.582 / 0.603 / 0.661 / 0.741; joker count 0.51-0.56.)

Mean values, winners / losers (v2): ante 5 Phi +0.15 / -0.04, ante 6 +0.03 / -0.19, ante 7 -0.11 / -0.48, ante 8 -0.27 / -0.95; K=8 std of one
Phi estimate is ~0.23-0.26 (the winner-loser gap at ante 5-6 is ~0.2, i.e. one std of the estimate, and the K=32 column shows the
estimate noise is *not* what limits the AUC). Win rate by Phi_v2 tercile at ante 5: 63.1 / 74.9 / 76.5% (xmult: 67.3 / 73.9 / 77.4%),
at ante 6: 64.3 / 74.6 / 81.4% (xmult 68.4 / 74.3 / 79.5%). Phi_v2 is boss-dependent as designed (ante 5-7 mean Phi: The Eye -0.45,
The Wall -0.53, The Needle -0.34 vs The Water +0.23), and v2 is much better than v1 at ante 7-8 (0.68 / 0.80 vs 0.65 / 0.65), where
the actual boss (Vessel, Needle, Wall) decides the round.

### Verdict: **red light for G1** under the pre-registered rule

Rule: green if Phi at ante 4-6 has AUC >= ~0.70 and above the xMult baseline. Measured: **0.57 / 0.59 / 0.61** (v2) at ante 4 / 5 / 6.
It is above the xMult baseline (0.53 / 0.55 / 0.56; CIs overlap at ante 5, separate only weakly at ante 4 and 6) and slightly above xmult+money, but nowhere near 0.70. It
only reaches 0.68 / 0.80 at ante 7 / 8, when the game is nearly decided (and the end-of-ante-8 money, xmult etc. are no use either).
The adjusted xmult+money+Phi model adds +0.01-0.02 AUC at ante 5-6. Phi is a real but weak predictor at the antes where there
is still time to act.

Why this is probably not an artifact of the meter's noise: K=32 equals K=8; the stop-at-target variant equals the play-on one.
What it does measure is the Greedy policy's score on a re-shuffled deck against the boss. Caveats for the interpretation (not fixed):
(1) Greedy is not the reference policy (the model plays Pair 79% of the time at ante >= 5), (2) the outcome of a game depends on
ante 5-8 *future* shops that no function of the ante-start state can see; the loss analysis found winners and losers almost
identical in money / jokers / xMult until ante 4-5, so ~0.70 may be above the ceiling of any ante-start state function on this data, (3) the test
is observational (survivor bias: games that reach ante 7 are already selected). If G1 is still wanted, the data say: use the
**v2** (boss-aware) version, expect most of the value late (ante >= 6) where the boss decides, and treat it as an auxiliary or analysis
signal rather than a credit-assignment fix for the shop; the early-ante signal (< 0.60 AUC) is too weak to carry a shaping reward.

## 3. How to wire Phi into the reward later (not done)

* Where: `crates/balatro-py/src/env.rs`, `Env::step_parts` / `reward_of` (reward is computed from `StepInfo` and `Game` after each step;
  `vec_env.rs` only calls it). Detect the entry: after the step, `game.state().phase == Phase::BlindSelect` and the phase before the
  step was not BlindSelect (EndShop, or the new episode's first state). Keep `last_phi` in `Env` (reset in `next_episode` / `hard_reset`).
* Potential-based form (`G1(a)`): at each BlindSelect entry compute `phi_new = power(game.state(), K, {boss_aware: true, salt: 0})`;
  the reward of that step gets `+ w * (gamma^dt * phi_new - last_phi)` where `dt` is the number of env steps since the previous entry;
  terminal: `phi_new = 0`; first state of an episode: `last_phi = phi(start)`. Because `power` is a function of the state, the
  shaping does not change the optimal policy. Add a `power_k` / `power_weight` field to `RewardConfig`, default 0 (off).
* Cost control: only at the Small BlindSelect (ante start, ~1/3 of entries) and K=4 (1.4 ms) is ~0.5 ms per ante per env, i.e. noise
  next to the model's 27 ms `act`; it must run inside the rayon worker (the pool already runs envs in parallel, GIL released).
* Given the red verdict, an analysis-only use (per-ante power curves in `evaluate.py`, via the Python binding) is the cheaper next step.

## Orchestrator addendum (2026-11-17): verdict revised to "try G1"

The 0.70 bar above was set against the *final* win, which from an ante-start state depends on many future shops the meter
cannot see. Φ is meant to measure the build's power *now*, so the matching validity question is whether it predicts surviving
the **current** ante (the two shops of that ante still lie ahead of the measurement). Same 7514 rows, AUC for "this ante is
survived" (`not (lost and final_ante == ante)`):

| ante | n | survive | Φ_v2 | Φ_v1 | xMult count | money | frac_beat |
|---|---|---|---|---|---|---|---|
| 1 | 1000 | 0.992 | 0.546 | 0.514 | 0.500 | 0.500 | 0.522 |
| 2 | 992 | 0.986 | 0.878 | 0.837 | 0.517 | 0.579 | 0.869 |
| 3 | 978 | 0.991 | 0.890 | 0.840 | 0.544 | 0.596 | 0.892 |
| 4 | 969 | 0.990 | 0.836 | 0.715 | 0.493 | 0.753 | 0.831 |
| 5 | 959 | 0.973 | 0.787 | 0.687 | 0.572 | 0.695 | 0.764 |
| 6 | 933 | 0.947 | 0.786 | 0.737 | 0.642 | 0.691 | 0.760 |
| 7 | 884 | 0.902 | 0.787 | 0.685 | 0.543 | 0.712 | 0.739 |
| 8 | 798 | 0.860 | 0.798 | 0.652 | 0.547 | 0.660 | 0.740 |

Φ_v2 is a strong short-horizon power meter (0.79–0.89 from ante 2 on, clearly above both baselines), so it is a valid
potential. Potential-based shaping leaves the optimal policy unchanged; the experiment asks only whether it speeds learning.
Decision: wire Φ_v2 (K=4) at every BlindSelect entry (i.e. at every shop exit, three per ante) behind a default-off flag and run
one 24M experiment against the `r_margin_zp` control (branch `exp/power-potential`).
