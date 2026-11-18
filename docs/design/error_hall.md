# Error hall (H1): carry the error-map findings into training

Plan item 1 + 2 of `notes.md` §0.0 "Hata haritası bulgularını eğitime taşıma planı". Source of the
findings: `docs/results/error_maps_dev_losses.md` (281 confirmed mistakes in 315 dev losses of the
reference; the most frequent and most expensive: selling the wrong joker, +12.3 pts on average).

* Part A (section 1): observation-gap audit. Report only, nothing in the Rust encoder was changed.
* Part B (sections 2 to 6): the error hall itself (mining, trainer, measurement), all opt-in, the
  defaults are bit for bit the previous training.
* Section 7: ready-to-queue commands for the st3 checkpoint.

## 1. Observation-gap audit (Part A)

Question: for the jokers whose *current* value depends on the game state (the ones the policy sells
wrongly), does the observation expose what that value depends on? Sources: `crates/balatro-py/src/
encode.rs`, `docs/design/rl_interface.md` §3, the joker implementations in
`crates/balatro-core/src/content/jokers/*.rs`.

### 1.1 What a joker token carries

`joker_i`: item id, edition, `aux_suit` / `aux_rank` / `aux_hand`. `joker_f` (13): `pos`,
`sell_value` (signed log, `ln(1+x)/5`), eternal / perishable / rental flags, `perish_rounds_left`,
`debuffed`, `hidden`, `value_lin` (`clamp(value/10, -10, 10)`), `value_log`, `counter_lin`
(`min(counter,100)/10`), `counter_log`, `aux_byte`. Global scalars that matter here: `discards_left`,
`hands_left`, `hands_played_round`, `discards_used_round`, `hand_size`, `hand_count`, `joker_count`,
`joker_capacity`, `draw_pile`, `deck_size`, `money` (signed log), `base_hands`, `base_discards`
(both *before* joker modifiers), 12 + 12 + 12 hand-type counters (`ln(1+n)/3`), the 52-cell
draw-pile rank/suit histogram, enhancement / edition / seal counts of the draw pile, owned vouchers.

Two facts shape everything below.

1. `JokerInstance::value` / `counter` hold a *scaling* state (Green Joker's +mult, Constellation's
   xmult, Popcorn's decaying mult ...). A joker whose effect is **derived from the state when it
   scores** (Banner, Abstract, Blue Joker, Raised Fist, Swashbuckler, Burglar ...) never updates
   them: its `value` and `counter` columns are 0 for the whole run, the token carries only its id,
   edition, sell value and flags.
2. At Shop / Blind select the round-local quantities are **stale**: `hands_left` / `discards_left`
   are what was left at the end of the previous round (they are refilled at `SelectBlind`), the
   hand is empty and the draw pile is the whole deck. The next round's effective hands and
   discards (`round_hands()` / `round_discards()`: base + joker and voucher modifiers, Burglar's
   "+3 hands, no discards") are not in the observation. Sell decisions are taken exactly there.

### 1.2 Per joker

| joker | current value depends on | visible inputs | missing |
|---|---|---|---|
| **Banner** (should be sold x5) | +30 chips x discards left when it scores | Play: `discards_left`. Shop: leftover `discards_left` (stale), `base_discards`, owned vouchers, ids of Drunkard / Merry Andy / Burglar | effective discards of the next round (modifiers, Burglar -> 0); the resulting chips; `value` / `counter` are 0 |
| **Raised Fist** (x4) | 2 x chip value of the lowest held card (Stone excluded, ties: rightmost); "held" = not played | Play: every hand card's rank / chips (but which cards stay held depends on the play the model is choosing, and face-down cards are hidden). Shop / Blind select: the 52-cell deck histogram, `hand_size` | the lowest held rank (a minimum over a random subset of the deck at shop time), the resulting mult |
| **Abstract Joker** (x3) | +3 mult x number of jokers | `joker_count` (/10) and the joker tokens | nothing missing (linear in a visible scalar); the marginal effect of a sale is -3 mult but not stated |
| **Blue Joker** (x3) | +2 chips x cards left in the draw pile at scoring | `draw_pile` (/52), `deck_size`, `hand_size`, `hand_count` | nothing missing in Play; at shop time the score-time pile size (deck - hand size - draws) is implicit |
| **Swashbuckler** (wrongly sold x1) | +mult = sum of the sell values of all *other* jokers (Egg / Gift Card raise them) | each token's `sell_value`, as a signed **log** | the sum: it needs decoding every token and adding across tokens |
| **Burglar** (wrongly sold x3) | at blind select: +3 hands, all discards lost; interacts with Banner (-> 0), Delayed Gratification (-> 0), Mystic Summit (-> always on), Green Joker (never decays), Ramen, Hanging Chad ... | the id; Play: `hands_left`, `discards_left` after the trigger | the net effect: effective next-round hands / discards and what every other joker is worth under them |
| **Delayed Gratification** (wrongly sold x1) | $2 per unused discard if no discard was used this round | Play: `discards_left`, `discards_used_round`; the id | the expected payout (next-round discards, and it depends on the policy not discarding) |
| **Astronomer** (wrongly sold x1) | planet cards in the shop are free | offer price 0 and the `free` flag on planet offers; hand levels | nothing structural; its value (money saved) is only visible for the planets currently offered |
| **Scaling jokers** (Green, Ride the Bus, Constellation, Hologram, Vampire, Obelisk, Madness, Lucky Cat, Glass, Campfire, Square, Runner, Castle, Wee, Spare Trousers, Red Card, Flash Card, Ceremonial Dagger, Canio, Yorick, Hit the Road, decaying Popcorn / Ice Cream / Ramen / Turtle Bean, Seltzer countdown) | their own `value` / `counter` | `value_lin`, `value_log`, `counter_lin`, `counter_log` (the current amount; for xmult jokers the factor, start 1.0) | nothing (the id tells whether the value is +mult, +chips or x) |
| **Run-counter jokers** | Fortune Teller: tarots used this run; Throwback: blinds skipped; Satellite: unique planets used; Loyalty Card: `hands_played_run - offset` | Loyalty's offset (`counter`) and the 12 hand-type play counters (compressed) | `stats.tarots_used`, `stats.blinds_skipped`, `stats.unique_planets_used` are **not in the observation at all**; Loyalty's countdown only approximately |
| **Money / slot / deck-count jokers** (Bull, Bootstraps, Joker Stencil, Mystic Summit, Acrobat, Half Joker, Steel / Stone / Driver's License / Cloud 9 / Erosion, Supernova, Card Sharp) | money, empty slots, `discards_left == 0`, `hands_left == 1`, cards of a kind in the deck, plays of the hand type | all of them are in the global block (money as a signed log; deck counts from the histogram, enhancement counts; `hand_plays_run` / `hand_plays_round`) | nothing missing; computing the contribution is left to the network |

### 1.3 Reading

* Almost every *raw input* is visible. The exceptions are real information holes: (a) the effective
  hands / discards of the **next round** at Shop / Blind select, (b) the run counters `tarots_used`,
  `blinds_skipped`, `unique_planets_used` (and an exact `hands_played_run`), (c) the held-card
  distribution of Raised Fist at shop time (only the deck histogram).
* What is missing for the *dominant error cluster* (Banner, Raised Fist, Swashbuckler, Burglar,
  Delayed Gratification) is a **derived per-joker contribution**: the token's `value` is 0 by
  construction, and obtaining the value means cross-token arithmetic (Swashbuckler: decode the log
  sell values and sum them), next-round state (Banner, Delayed Gratification, Burglar) or a
  minimum over a random subset (Raised Fist). A network can learn that from 1 to 2 billion
  steps of data, but these decisions are rare (a sell at the shop with a specific pair of jokers),
  which is the situation where such a derived feature pays off.
* Whether the missing feature *causes* the mistakes is not known. The mistakes can also be
  value-estimation noise (the error map has about one in six false flags) or an exploration
  problem (the pair Burglar + Banner is rare). The error hall (below) attacks the second cause
  and is cheap; the feature attacks the first and is a shared-contract change.

### 1.4 Proposal (P1 and P2 implemented as interface v3, `rl_interface.md` §3.2 / §3.4 / §12)

**P0, a probe before any change (CPU or a few GPU minutes, no contract change).** On the mined
pool (or recorded Shop states) regress the *true* contribution of Banner / Raised Fist /
Swashbuckler / Abstract, computed with the Rust scorer on the same state, on the frozen encoder
summary (linear probe). A high R^2 means the network already represents the value (then the
mistakes are credit assignment and the hall is the right tool); a low R^2 means the feature is
missing and P1 / P2 are worth the contract change.

**P1, two global scalars (cheap, small).** `next_hands` and `next_discards` =
`GameState::round_hands()` / `round_discards()` with the blind-select effects of owned jokers
applied (Burglar: +3 / 0), `/ 8` like `hands_left`. They make Banner, Delayed Gratification,
Mystic Summit and Burglar readable at Shop / Blind select. One function, a few lines.

**P2, a per-joker "current contribution" block (the real fix).** A Rust function
`joker_contribution(state, slot) -> (chips, mult, xmult, money)` per joker at the *next round's
reference state* (effective discards, current money, joker count, the deck histogram for Raised
Fist's expected lowest held rank, the sum of sell values for Swashbuckler, the run counters for
Fortune Teller / Throwback / Satellite / Loyalty), with one arm for each of the roughly 25
derived-value jokers and a generic arm that reads `value` / `counter` for the scaling jokers, so
the block has the same meaning for every joker. Encoding: 4 columns appended to `joker_f`
(`sl()` scaled, the xmult column as `ln(x)`), written next to the existing ones in `write_jokers`.
This is an interface version bump (2 -> 3: `joker_f` 13 -> 17, `global_f` +2 for P1) handled by the
name-based checkpoint migration (`tests_py/test_migration.py`: new columns zero-initialised, so
the old policy is reproduced exactly at load time). Cost: O(jokers) per observation; the code
should call the same formulas as the scoring hooks (extract the closed forms into shared
functions) so the two cannot drift.

**P3, rejected for now: sell-ablation with the power meter** (`power(state) - power(state without
joker_i)`). It needs no per-joker code but costs 0.4 ms (K=1) to 1.4 ms (K=4) per call, 5 calls
per shop state: at least a doubling of the rollout cost.

## 2. Design of the error hall

**Idea.** The error-map pipeline (`analysis/errmap.py`: screen M futures, confirm with M2 fresh
futures, reseeded streams, common random numbers, P(win) head leaf) finds decisions where another
action was clearly better. Their *states* are saved and a fraction of the training episodes starts
there. **No label enters training**: the pool is only a set of start states; the policy gets its
usual reward from the new, freshly reseeded future. The roughly one in six false flags only
cost a few episodes from a normal state; and rare situations (RerollBoss with Director's Cut,
the sell choices) are seen far more often than in natural play.

**Pool file** (`analysis/hall.py`, JSON lines, one confirmed mistake per line): `state` (the
`export_state` JSON of the state BEFORE the mistaken decision) and `category`
(`played->better` coarse type, e.g. `SellJoker->SellJoker` = sold the wrong joker,
`SelectBlind->RerollBoss`, `BuyJoker->SellJoker`), `phase`, `ante`, `deck`, `stake`, `dq` (confirmed
dP(win) of the better action), `se`, `q_played`, `q_better`, `played` / `better` (text and the
structured 4-tuples), `game`, `step`, `pwin_before`, `outcome` / `final_ante` of the mined game,
`seed_base`, `checkpoint`. Next to it `<pool>.summary.json` (settings, timing, counts per
category / stake / deck / ante).

**Mining** (`scripts/errhall_mine.py`):

1. Play `--games N` games of the policy (deterministic, first episode of envs `0..N-1`) on a
   **training** seed base (default 900_000_007; the script refuses the dev / holdout / val bases)
   with the training mix `--decks` / `--stakes` (`BalatroVecEnv(decks=, stakes=)`).
2. Replay the lost games (all with `--include-wins`) as trunk, enumerate alternatives, screen,
   confirm, `finalize_game` (all existing code, unchanged apart from the additive arguments
   `env_kwargs` / `head_runner` and two extra keys in a mistake record).
3. For every confirmed mistake write a pool line. Chunks are appended as they finish (a killed run
   keeps what was done).
4. **P(win) head.** The head reads the critic input of a model, so it must belong to the policy
   that plays the futures. The reference head (`runs/errmaps/pwin_head.pt`) was trained on the
   reference's features on Red / White; for a stake-trained checkpoint train one on its mix
   (`errmap_pwin.py collect/train --decks --stakes`, about 10 minutes, section 7 step 0). The script
   warns when the head's recorded checkpoint differs from the policy and prints a calibration
   check (mean P(win) at the start vs the win rate of the played games, AUC) before the expensive
   part. `--head-model CKPT` reads the head's features from a different frozen model (a fixed
   ruler across checkpoints; the recurrent state of both models is carried together).
5. Cost knobs: `--games`, `--m` (screen futures, 8), `--m-confirm` (64), `--n-confirm` (5
   decisions re-run per game), `--k` (3 actions per decision), `--horizon`, `--slots`,
   `--include-wins`; `--estimate-only` prints the estimate (also printed at every start and
   re-computed with the real number of analysed games).

**Trainer** (`ppo.py`):

* `--error-hall P` and `--error-hall-pool FILE[,FILE]` (`--error-hall-min-dq`,
  `--error-hall-holdout F`: fraction of the pool's *games* kept out of training for the probe).
* Right after `env.step`, of the envs whose episode just ended **and whose new episode is natural**
  (not a scenario, not a boss-hall start), a fraction `q = P / (1 - scenario_prob - boss_hall)` is
  replaced by a random pool state of the same (deck, stake) as the fresh episode (else the same
  stake, else the reset stays natural and is counted as `errhall_fallback`) through
  `import_state(i, state, reseed=fresh)`: same visible state, new hidden future (all RNG streams).
  **Probabilities:** the shares of *all* episode starts are scenario `s`, boss hall `h`, error hall
  `P`, natural `1 - s - h - P`; the config asserts `s + h + P <= 1` (the existing `s <= 0.2` stays).
  The stake / deck mix keeps being governed by `--stake-curriculum auto` and the config pool,
  exactly as for the boss hall (the hall state is drawn for the stake the normal sampler picked).
* **Treated like a boss-hall start**: the step that ended the previous episode already carried
  `done`, so the GAE trace is cut and the recurrent state is reset (`first`); the imported run is
  an episode of its own (`imported=True`; steps, return and `max_episode_steps` count from the
  import). The terminal reward and `done` of the step are stored before the import rewrites the
  row. Pool states are static (loaded once, no FIFO).
* Natural statistics (`ep_*`, the stake curriculum's win-rate EMA) exclude hall episodes; hall
  episodes are logged separately.
* **Logging** (`log.jsonl`, per update): `errhall_pool`, `errhall_started`, `errhall_fallback`,
  `errhall_started/<category>`; of the finished hall episodes `errhall_episodes`, `errhall_win`,
  `errhall_ante`, `errhall_steps`, `errhall_return`, `errhall_trunc` ... and per category
  `errhall_n/<category>`, `errhall_win/<category>`. The win rate of hall episodes rising over the
  run is a *training-set* signal (do the mistaken states get solved), not generalisation.
* Default off: nothing is built, `import_state` is never called, the RNG is never touched; the
  weights after a short run are identical to the previous code (checked against `git show HEAD:`
  of `ppo.py`, v2 flags on) and the test below.

## 3. Measurement

Three levels, cheapest first.

1. **Probe at the mined states** (`scripts/errhall_probe.py`, seconds): for each pool state,
   the policy acts with zero memory; per category it reports how often the greedy action repeats the
   mistaken action, how often it is the action the error map found better, and the two
   probabilities. Run it on the held-out games (`--part holdout`, trainer with
   `--error-hall-holdout 0.1`) for generalisation and on `--part train` for the trained states, for
   the start checkpoint, the control and the hall run.
2. **Error rate on a fixed game set** (`errmap_analyze.py` + `errhall_compare.py`, ~20 min of GPU
   per model): play the dev seeds live with each checkpoint and run the error-map analysis on its
   losses, then compare mistakes per 1000 analysed decisions and per category.
   `errmap_analyze.py` got `--decks`, `--stakes` (live play only) and `--head-model` (a fixed
   ruler: pass the same head and `--head-model runs/st3/latest.pt` to every checkpoint). Each model
   is judged against *its own* continuation (Q^pi), the head is the same for all.
3. **Win rate**: `scripts/stake_eval.sh` on the 10k holdout for hall and control (paired).

## 4. Smoke results

Shared GPU (the queue was idle, nothing else of mine running), small settings; `st2_v2_s1` as the
policy, the reference `e2e_long_fix` features + `pwin_head.pt` as the ruler (`--head-model`), the
st3 mix (5 decks x 8 stakes). The head is therefore *not* the st3 head of step 0: the mistakes of
these smokes are a plumbing test, not a pool to train on.

| run | games played / analysed | settings | confirmed mistakes | wall | note |
|---|---|---|---|---|---|
| mining, Red, White + Red stake | 96 / 27 | M 4, confirm 16 x 3, 256 slots | 13 (0.48 per analysed game) | 52 s | 4.9 k rollout steps/s (underfilled batches) |
| mining, full st3 mix, defaults | 96 / 68 | M 8, confirm 64 x 5, 2048 slots | **53 (0.78 per analysed game)** | **190 s = 2.8 s per analysed game** | 79 016 futures of about 21 steps, 0.4% capped, 8.9 k steps/s with the second (head) model running |
| same, a looping game kept in | 40 / 32 | M 4, confirm 32 x 4 | 14 | 340 s | **one** game of the 40 hit the 3000-step limit and cost 5 of the 5.7 minutes: 41% of its futures hit the 300-step cap |
| probe, 53 pool states | | `st2_v2_s1` vs reference | | seconds | the policy repeats the mined mistake in 66% (reference 36%), picks the "better" action in 9% (26%) |
| training, 6 updates x 8192 rows, 256 envs | | `st2_v2_s1`, all v2 flags + `--error-hall 0.1`, 41 training states (12 held out), 60-step episodes | | 8 s | `errhall_started` 5 to 35 per update (about 10% of the starts), fallback 0, hall episodes logged per category |

Findings that changed the tool:

* **Looping games dominate the cost.** `--games` with the st3 mix contains an occasional game that
  hits the step limit (a deterministic MoveJoker / Reroll loop; here 1 in 96): it has about 1000
  nearly identical decisions and every future from it loops again (41% of that chunk's futures
  were capped). Games with done code 3 are therefore **excluded** from the mining
  (`--include-truncated` to keep them); the number is printed. Without the loop game the cost fell
  from 14.8 s to 2.5 s per analysed game.
* Measured yield on the mix: 0.7 to 0.8 confirmed mistakes per analysed game, 2.5 s per game: about
  1000 mistakes per 1000 analysed games, about 55 to 70 minutes of GPU for 1800 games played
  (about 1300 analysed, 70% losses). Mistakes occur at every stake including the hard ones, at
  least with the (mismatched) smoke head; with the real st3 head the hard-stake yield has to be read
  from `<pool>.summary.json` (`pool.stake`).
* The st3 head is needed (the reference head's start P(win) was 0.67 against a 0.28 win rate on the
  mix): the calibration check printed by `errhall_mine.py` shows exactly that.

## 5. Files

* `python/balatro_rl/analysis/hall.py`: pool records, files, summaries, game-level hold-out.
* `scripts/errhall_mine.py`, `scripts/errhall_probe.py`, `scripts/errhall_compare.py`.
* `python/balatro_rl/train/ppo.py`: `error_hall*` fields, `_init_error_hall`, `_errhall_resets`,
  `_errhall_annotate`, `errhall_log`.
* `python/balatro_rl/analysis/errmap.py`, `pwin.py`, `scripts/errmap_analyze.py`,
  `scripts/errmap_pwin.py`: additive options (`env_kwargs`, `head_runner`, `--decks`, `--stakes`,
  `--head-model`); defaults unchanged.
* Tests: `tests_py/test_error_hall.py` (pool round trip, reset import with episode boundary and
  Shop start, stake matching, default-off identity, probability composition and measured shares,
  hold-out split, probe, short training with every v2 flag).

## 6. Risks and open points

* **Pool size vs memorisation.** About 0.9 mistakes per lost game: 1500 games with 70% losses are
  roughly 900 states. With `--error-hall 0.1` over 30M steps each state is started about 20 times
  (different futures). The policy may learn "at exactly this state do X". Mitigations: the
  hold-out probe (generalisation to unseen games), mining more games or from several checkpoints
  (comma list of pool files), a lower P. The pool is static; a refreshed pool after the run
  (mine with the new checkpoint) is the natural second round.
* **The hard stakes give few mistakes.** At Purple / Orange / Gold the win probability is near 0 for
  every action, so the P(win) leaf differences are within noise and nothing is confirmed; the pool
  will be mostly White to Blue. This is a limit of the leaf value (horizon 1 boss fight, P(win)),
  not of the hall.
* **"Better" is relative** to the analysed policy playing on (`Q^pi`) and to the head; about one in
  six flags may be noise. The hall uses no labels, which is why the noise is harmless.
* **The head** must match the policy: train it for the mining checkpoint (step 0); the calibration
  check is the guard.
* **Endless**: pool states come from normal runs (`endless=False` in the state); do not combine
  with `--endless` training.
* The measurement of level 2 is expensive and noisy (a few hundred flagged mistakes per model); the
  probe (level 1) and the paired holdout win rate (level 3) are the sharper tools.

## 7. Ready-to-queue commands (checkpoint `runs/st3/latest.pt`, available after the st3 job)

All run from the repo root (queue jobs: `gpu_queue.py add --name N --cmd '...' --cwd <checkout>
--kind train|eval --notes '...'`). `MIX` is the st3 mix.

```
MIX="--decks red,blue,yellow,green,black --stakes white,red,green,black,blue,purple,orange,gold"
```

The st3 job file (`runs/queue/*/0*_st3*.json`) did not exist when this was written; the training
recipe below mirrors `runs/queue/done/010_st2_v2_s1.json` with the stake list extended. If the real
st3 job differs, copy ITS flags (everything except `--name`, `--init`, `--seed`, the hall flag).

**Step 0, P(win) head for st3 (about 10 min GPU).**

```
export PYTORCH_ALLOC_CONF=expandable_segments:True; \
uv run --no-sync python -u scripts/errmap_pwin.py collect --checkpoint runs/st3/latest.pt \
  --out runs/errmaps/pwin_data_st3 --shards 0-5 --games 3000 $MIX \
&& uv run --no-sync python -u scripts/errmap_pwin.py train --checkpoint runs/st3/latest.pt \
  --data runs/errmaps/pwin_data_st3 --out runs/errmaps/pwin_head_st3.pt
```

**(a) Mining, sized to about one hour of GPU** (1800 games, about 1300 analysed, about 900 to 1000
confirmed mistakes expected; measured 2.5 s per analysed game on the mix, the estimate printed at the
start is the pessimistic one; add `--include-wins` or raise `--games` for a bigger pool, lower
`--m-confirm 32` for about 35% less time; games that hit the step limit are skipped):

```
uv run --no-sync python -u scripts/errhall_mine.py --checkpoint runs/st3/latest.pt \
  --head runs/errmaps/pwin_head_st3.pt --games 1800 --seed-base 900000007 $MIX \
  --out runs/errhall/st3.jsonl --keep-games runs/errhall/st3_games
```

**(b) Hall run, 30M from st3 with the st3 recipe + the hall** (`--seed 3` for both arms: a seed
different from the st3 run's, so the env episodes do not replay):

```
export PYTORCH_ALLOC_CONF=expandable_segments:True; uv run --no-sync python -u scripts/train_ppo.py \
  --name st3_hall --init runs/st3/latest.pt --total-steps 30e6 \
  --decks red,blue,yellow,green,black --stakes white,red,green,black,blue,purple,orange,gold \
  --eval-deck red --eval-stake black --eval-set val --lr 5e-5 --kl-coef 0 --critic-warmup 2 \
  --reorder-cost -0.25 --margin 1.0 --margin-cap 4 --zero-play -0.3 --eval-every 8 \
  --eval-games 512 --seed 3 --adv-norm per_config --stake-curriculum auto --scenario-prob 0.1 \
  --boss-hall 0.2 --boss-hall-size 4096 \
  --error-hall 0.1 --error-hall-pool runs/errhall/st3.jsonl --error-hall-holdout 0.1 \
  > runs/st3_hall.log 2>&1 && scripts/stake_eval.sh runs/st3_hall/latest.pt stk_st3_hall
```

Shares of the episode starts: scenario 0.1, boss hall 0.2, error hall 0.1, natural 0.6.

**(c) Matched control** (the same line without the three `--error-hall*` flags, `--name st3_ctrl`,
the same `--seed 3`):

```
export PYTORCH_ALLOC_CONF=expandable_segments:True; uv run --no-sync python -u scripts/train_ppo.py \
  --name st3_ctrl --init runs/st3/latest.pt --total-steps 30e6 \
  --decks red,blue,yellow,green,black --stakes white,red,green,black,blue,purple,orange,gold \
  --eval-deck red --eval-stake black --eval-set val --lr 5e-5 --kl-coef 0 --critic-warmup 2 \
  --reorder-cost -0.25 --margin 1.0 --margin-cap 4 --zero-play -0.3 --eval-every 8 \
  --eval-games 512 --seed 3 --adv-norm per_config --stake-curriculum auto --scenario-prob 0.1 \
  --boss-hall 0.2 --boss-hall-size 4096 \
  > runs/st3_ctrl.log 2>&1 && scripts/stake_eval.sh runs/st3_ctrl/latest.pt stk_st3_ctrl
```

The control keeps the pool's hold-out irrelevant (it never reads it); both arms draw the same seeds.

**(d) Evaluation.**

```
# 1. cheap probe at the mined states (seconds): held-out games and trained games
uv run --no-sync python scripts/errhall_probe.py --pool runs/errhall/st3.jsonl --holdout 0.1 \
  --part holdout --checkpoint runs/st3/latest.pt --checkpoint runs/st3_ctrl/latest.pt \
  --checkpoint runs/st3_hall/latest.pt
uv run --no-sync python scripts/errhall_probe.py --pool runs/errhall/st3.jsonl --holdout 0.1 \
  --part train --checkpoint runs/st3/latest.pt --checkpoint runs/st3_ctrl/latest.pt \
  --checkpoint runs/st3_hall/latest.pt
# 2. paired holdout win rates per stake (the stake_eval.sh runs of (b) and (c) wrote
#    runs/eval/stk_st3_{hall,ctrl}_<stake>_holdout.json)
for s in white red green black blue purple orange gold; do
  uv run --no-sync python scripts/compare_evals.py runs/eval/stk_st3_ctrl_${s}_holdout.json \
    runs/eval/stk_st3_hall_${s}_holdout.json   # paired difference hall - control
done
# 3. error rate per category on a fixed set (dev seeds, the st3 head as the common ruler;
#    ~20 min GPU per model with 600 games)
for n in st3_ctrl st3_hall; do
  uv run --no-sync python -u scripts/errmap_analyze.py --checkpoint runs/$n/latest.pt \
    --head runs/errmaps/pwin_head_st3.pt --head-model runs/st3/latest.pt --set dev \
    --n-games 600 --only-losses $MIX --out runs/errmaps/${n}_dev
done
uv run --no-sync python scripts/errhall_compare.py runs/errmaps/st3_ctrl_dev \
  runs/errmaps/st3_hall_dev --names control,hall
```

Reading: the hall helps if (i) the held-out probe shows a lower repeat rate / higher P(better)
than the control, (ii) the paired holdout win rate rises (one run against one control is a
**provisional** result, as for stake v2: repeat with a second seed before merging anything), (iii) the
`SellJoker->SellJoker` and `SelectBlind->RerollBoss` rates per 1000 decisions fall. A win-rate
change of at most 1 to 2 points is within the run-to-run spread.
