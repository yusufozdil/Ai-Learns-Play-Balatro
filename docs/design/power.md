# Build-power meter

A state function Phi, "how well does the current build fare against this ante's boss", exposed as a
free function in the simulator and in Python. It was evaluated as a possible dense training signal
(potential-based reward shaping); the shipped PPO training does **not** use it, the meter is kept as an
analysis tool.

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
  line. `play_on = false` stops at the target : practically the same AUC.
* Python: `balatro_rl._engine.power(state, k=8, boss_aware=True, play_on=True, salt=0)` where `state` is a state JSON string or a `Game`;
  returns `{phi, std, frac_beat, target, k}`; the GIL is released. (`crates/balatro-py/src/power.rs`, tests `tests_py/test_power.py`.)

Cost (release, 1 core, heuristic-agent states from games 0..99, all antes): **~70 us fixed (digest + clone) + ~335 us per simulated
round**: K=1 0.40 ms, **K=4 1.4 ms, K=8 2.8 ms** per call (Greedy's 218-subset preview scan per action dominates). Used only at BlindSelect entries (a few per ~170 steps) the amortised cost is negligible. Probe: `cargo run --release -p balatro-core --bin power_probe -- --games 100 --k 8`.

## 2. Validity (summary)

The meter was checked on 7514 recorded states (the first Small-blind BlindSelect of each ante, games of a
reference policy on Red Deck / White Stake, 68.6% win rate) by the Mann-Whitney AUC of Phi as a predictor,
against the cheap baseline "number of xMult jokers".

* **Final win** from an ante-start state: Phi_v2 (boss-aware) AUC 0.57 / 0.59 / 0.61 at ante 4 / 5 / 6, only
  0.68 / 0.80 at ante 7 / 8 (xMult count: 0.53 / 0.55 / 0.56 at ante 4-6). Weak early: the final outcome depends
  on the later shops, which no function of the ante-start state can see. More rounds per estimate (K=32 vs K=8)
  and the stop-at-target variant do not change the AUC, so it is not meter noise.
* **Surviving the current ante**: Phi_v2 AUC 0.79-0.89 from ante 2 on (Phi_v1 0.65-0.84, xMult count
  0.49-0.64, money 0.58-0.75), i.e. a strong short-horizon power meter. Phi_v2 is boss-dependent as designed
  (ante 5-7 mean Phi: The Eye -0.45, The Wall -0.53, The Needle -0.34 vs The Water +0.23) and much better
  than v1 at ante 7-8, where the actual boss decides the round.

## 3. How Phi could be wired into the reward (not done)

* Where: `Env::step_parts` / `reward_of` in `crates/balatro-py/src/env.rs`; detect a BlindSelect entry
  (phase after the step is BlindSelect, before it was not) and keep `last_phi` in `Env`.
* Potential-based form: at each entry compute `phi_new = power(state, K, {boss_aware: true, salt: 0})`; the
  step reward gets `+ w * (gamma^dt * phi_new - last_phi)` (`dt` = env steps since the previous entry; terminal:
  `phi_new = 0`). Because `power` is a pure function of the state, the shaping does not change the optimal
  policy.
* Cost: K=4 is ~1.4 ms per call; at the three BlindSelect entries per ante this is negligible next to the model's
  forward pass, and it runs inside the rayon workers (GIL released).
