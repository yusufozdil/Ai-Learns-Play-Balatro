# Training (Phase 10A): recurrent PPO, curriculum routing, evaluation protocol

Code: `python/balatro_rl/train/{ppo,curriculum,evaluate}.py`, scripts `scripts/{train_ppo,evaluate,summarize_run}.py`,
Makefile targets `ppo-e2e`, `ppo-s1`, `eval-ckpt`; tests `tests_py/test_ppo.py`. Model: `docs/design/model.md`;
interface: `docs/design/rl_interface.md` (`INTERFACE_VERSION = 1`; checkpoints record it and `load_checkpoint` asserts it).
All heads see the **same reward** (shared critic, shared advantage); only the log-probability comes from the head of
the phase the action was taken in.

## 1. PPO (`train/ppo.py`, CleanRL style)

One update = rollout, GAE, `epochs` passes over recurrent minibatches.

1. **Rollout** (`Trainer.collect`): `rollout_len` (T) steps in `num_envs` (N) envs (default 128 x 1024 = 131k rows). Per step:
   `prepare` (H2D copy of the env buffers), store the prepared observation + masks in compact GPU buffers `(T, N, ...)`
   (floats f32, categoricals u8, masks bool/i64: lossless, ~5 KB/row, 0.65 GB per default rollout), `model.act`
   (sampling), the **route** of the curriculum stage (section 2) decides which action is executed, `env.step`.
   The GRU state is stored at every `chunk`-th step (`h0`). `first[t]` = observation `t` starts an episode.
2. **GAE** (`compute_gae`): `delta_t = r_t + gamma * V_{t+1} * (1 - first_{t+1}) - V_t`, `A_t = delta_t + gamma * lam *
   (1 - first_{t+1}) * A_{t+1}`; the bootstrap value of the observation after the last step comes from `model.value`.
   Defaults `gamma = 0.997`, `lam = 0.95`. Rewards are multiplied by the constant `reward_scale` (0.1: an ante-5 run
   returns ~1.6, a win ~3; keeps the value loss on the scale of the policy loss, advantages are normalised anyway).
   **Truncation** (`done == 3`, only after `max_episode_steps` = 3000, ~0.3% of BC games): the env does not keep the last
   observation of the old episode (rl_interface section 8), so the exact `V(s_T)` is unavailable. Since Phase 10B the trace is
   cut as at a terminal but the target bootstraps with `V(s_{T-1})`, the value of the observation the last action was taken
   in (`compute_gae(..., trunc)`, `PPOConfig.bootstrap_truncation`, default on): a truncated policy is almost always in a loop,
   so the two states are nearly identical. Before 10B (runs `a_e2e`, `b_s1`) truncation was a plain terminal.
3. **Recurrent minibatches**: a minibatch is (a time chunk of `chunk` = 32 steps) x (a random group of `group` = 128
   envs) = 4096 rows. Its GRU state is the stored `h0` of the rollout (detached; the old parameters computed it),
   the chunk is scanned with `first` resets. Before any step `evaluate` therefore reproduces the rollout log-probability
   exactly (ratio = 1, test `test_recurrent_minibatch_reproduces_rollout_logprobs`, max diff < 1e-4 incl. all chunks).
   Env groups are re-drawn each epoch, the (chunk, group) pairs are shuffled. Since the GRU input is the observation only
   (never the action), the hidden state does not depend on who chose the action: teacher steps in a curriculum keep
   the memory consistent.
4. **Loss**: `pg` (clipped PPO surrogate on the *chain* log-probability = kind + arg0 + arg1 + card steps of the head
   of the row's phase) over model-controlled rows only; value MSE (0.5 x, optional PPO value clipping `clip_vloss`)
   over **all** rows; entropy bonus `ent_coef * chain entropy` (a sample estimate of the joint entropy) over model rows;
   Adam (`eps` 1e-5), grad-norm clip 0.5, LR linearly annealed to `lr_final_frac * lr`, advantage normalisation over the
   model rows of the whole rollout. Rows whose action is not expressible by the model's chain (`ok = False`) never enter
   the policy loss (`where`, not multiply: garbage-proof against NaN/inf, test with corrupted teacher rows).
   `target_kl` (optional) stops the epochs early.
5. **BC initialisation + KL to the BC policy**: `--init runs/bc/model.pt` loads the BC model (config included) instead of
   a fresh one. `--kl-coef c --kl-anneal-updates K` adds `beta_u * KL(pi || pi_ref)` with `beta_u = c * max(0, 1 - u / K)`
   (estimator `k3 = exp(d) - 1 - d`, `d = log pi_ref(a) - log pi(a)` on the sampled actions; the reference log-probs
   come from a frozen copy of the init checkpoint, computed once per rollout without gradient and not at all once
   `beta = 0`). `--critic-warmup W`: for the first `W` updates only the critic MLP is trained (the BC critic was fitted
   to unscaled gamma = 0.99 teacher returns; letting its garbage advantages move the shared encoder in the first updates
   would damage the BC policy). Test: encoder weights bit-identical after a warm-up update.
6. **Logging** (`runs/<name>/log.jsonl`, one JSON per update + one per evaluation; console line per update):
   steps/s, rollout/model/teacher/update seconds, teacher action count, mean reward, finished-episode statistics
   (`drain_finished`: episodes, ante, win rate, steps, return, blinds beaten, invalid), PPO statistics (pg, vloss,
   approx KL, clip fraction, explained variance, grad norm, LR, KL coefficient, KL to the reference), the entropy per
   **head x part** (`ent/<blind|play|shop|pack>/<kind|arg0|arg1|cards>`) and the number of policy rows per head, and the
   kind distribution of the executed model actions per head (`actions`). `scripts/summarize_run.py runs/<name>`
   prints the learning curves as a table. No TensorBoard: it would add a dependency (tensorboard + protobuf) for what
   JSONL + the table already give; `log.jsonl` loads into anything (pandas, matplotlib) if wanted.
7. **Checkpoints** (`latest.pt` every `save_every` updates, `best.pt` when the periodic validation improves): model
   `state_dict` + `config` (model), `interface_version`, `ppo_config`, optimiser, update counter, global step, best score.
   `--resume` (same `--name`/`--out-dir`) continues from `latest.pt` (the env restarts with a new base seed derived from the
   resume count: env state is not serialised, the first rollout after a resume starts new episodes).

Memory (GPU): default rollout 0.65 GB + ~7 GB for the activations of a 4096-row minibatch (peak ~9 GB). The two runs of
section 5 shared the GPU with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## 2. Curriculum (`train/curriculum.py`)

`--curriculum "play@5e6,play+pack+blind@10e6,all"`: stages separated by commas, each `phases[@global_env_steps]`
(the last without a limit). Phases `blind play shop pack`; aliases `s1` = play, `s2` = play+pack+blind, `s3` = `s4` =
all (the PRD's stage 3 adds SHOP; stage 4 "end to end" is the same routing, typically with KL/warm-up off). Per env step
the phase (`global_i[:, 0]`) selects who acts: the model (`act`) or the teacher (`env.teacher_actions(select)`).
Only model-chosen steps enter the policy loss, entropy and KL; the critic learns from every step. Consumable use, selling
and joker moves are legal in every phase, so a phase route also routes them (a teacher-controlled shop also does the
teacher's selling/moving; the model's Play phase can still use consumables).

**Teacher cost.** `teacher_actions` computed every env in the original API (4-8k actions/s aggregate). Added a minimal
Rust change (`EnvPool::teacher_actions_where`, optional `select` bool array of `VecEnv.teacher_actions`, interface
version unchanged, Rust test): only the selected envs are computed, the others are not even consulted. Measured
(1024 envs, 12 threads, aggregate over all threads): BlindSelect 150k actions/s, Play 30k/s, **Shop 2k/s**, PackOpen 3.8k/s.
So stage 1 (the model plays PLAY only) is dominated by the teacher's shop: ~0.5 ms of wall time per shop action; in the
runs below the teacher's share is reported as `teacher_seconds` per rollout.

## 3. Evaluation protocol (`train/evaluate.py`, `scripts/evaluate.py`)

* **Test set**: `TEST_SEED = 10_000_019`, 1000 games = the first episode of env `i = 0..999` (seed is a pure function of
  `(TEST_SEED, i, 0)`, independent of the number of envs / threads). Training uses `train_seed(seed, resume) = 20_000_003 +
  ...` (asserted `!= TEST_SEED`; episode seeds are hashes of `(base, env, episode)`), so test games are never trained on.
  Red Deck, White Stake, `max_episode_steps = 3000` (same as the BC evaluation: the BC numbers are directly comparable).
* **Validation during training**: the first `eval_games` (512) games of the *same* set (a literal prefix; test
  `test_eval_protocol_prefix_and_determinism`), deterministic (argmax) and, optionally, sampled, every `eval_every`
  updates. `best.pt` = best deterministic mean ante (ties: win rate) on the validation prefix. The model runs only
  on the games that have not yet finished, so a validation costs 8-20 s.
* **Metrics** (`EvalResult`): mean ante (std, standard error), win rate, ante histogram, invalid-action rate per step,
  episode length, blinds beaten, truncation rate, per-game arrays. `compare(a, b)` gives the **paired** difference over
  the shared games with its standard error (the standard error of a mean ante is ~0.06 at 1000 games, of the paired
  difference to a baseline usually less).
* `scripts/evaluate.py --checkpoint X [--route play]` evaluates deterministic and sampled on the 1000 games against the
  cached baselines (random, teacher, BC det/sampled; `runs/eval/baselines_1000.json`) and writes `runs/eval/<name>.json`.
  With `--route` the model plays only those phases and the teacher the rest (baselines: teacher / BC under the same routing).

## 4. Hyper-parameters

(see `PPOConfig`; every field is a CLI flag.)

| | default | notes |
|---|---|---|
| envs x rollout | 1024 x 128 | 131k rows per update |
| epochs / minibatch | 2 / 4096 rows (chunk 32 x 128 envs) | 64 minibatches per update; the update is compute-bound (~20k rows/s fwd+bwd, bigger batches do not help) |
| gamma / lambda | 0.997 / 0.95 | |
| clip / vf / ent | 0.2 / 0.5 / 0.003 | |
| lr | 1e-4 (BC init), 3e-4 (scratch) | linear to 10%; 1.5e-4 from BC gave KL 0.02 and KL-to-reference spikes |
| grad clip | 0.5 | |
| reward scale | 0.1 | env reward: blind +1, boss ante +1, win +10, loss 0, invalid -0.05, shaping 0 (flag `--shaping`) |
| BC KL | 0.3 -> 0 over 120-150 updates | only with `--init` |
| reorder cost | 0 (recommended -0.05; section 8) | env reward of every MoveJoker (`--reorder-cost`) |
| truncation bootstrap | on | `--bootstrap-truncation` / `--no-bootstrap-truncation` |
| validation guard | off (recommended `--eval-move-limit 2`) | repetition guard of the periodic validation |

## 5. First results (Phase 10A; RTX 5060 Ti, Ryzen 5 7600; Red Deck, White Stake)

Test = the fixed 1000-game set (section 3); baselines on the same games: random 1.00 / 0%, teacher (fair-info heuristic)
**5.91 / 16.1%**, BC deterministic 5.29 / 6.5%, BC sampled 4.72 / 3.5% (standard error of a mean ante ~0.06).

| run | env steps | wall time | 1000-game test (deterministic) | 1000-game test (sampled) |
|---|---|---|---|---|
| **(a) `a_e2e`**: BC init, end-to-end, KL 0.3 -> 0 (120 updates), lr 1e-4 | 24.0M | 72 min (alone on the GPU after the first 25 min; 2 other runs shared it for ~35 min) | `best.pt` (u160, picked on the validation prefix): **7.08 / 36.1%**; final `latest.pt` (u183, not selected): **7.04 / 37.9%** | best 6.95 / 34.4%; final **6.99 / 38.0%** |
| (b) `b_s1`: scratch, stage 1 (model plays PLAY, teacher the rest), `--shaping 1`, lr 3e-4, ent 0.01 | 2.2M (stopped by hand, still rising) | 36 min (shared GPU) | 3.38 / 1.4% (`--route play`; 6.8% truncated) | 3.20 / 1.8% |

(a) paired against the baselines (best.pt, deterministic): +1.77 ante / +29.6 pp win vs BC (+-0.07 / 1.7), **+1.17 ante /
+20.0 pp win vs the teacher** (+-0.07 / 1.8). Ante histogram: 1: 9, 2: 18, 3: 9, 4: 17, 5: 68, 6: 120, 7: 196, **8: 563**.
For stage 1 the fair baselines are the same routing: teacher 5.91 / 16.1%, BC plays PLAY + teacher the rest 5.38 / 8.0%:
(b) is far below both after 2.2M steps, but the curve is still climbing (see below).

**(a) learning curve** (validation prefix, 512 games, deterministic / sampled; training episodes are sampled play):

| updates | Msteps | det ante / win | sampled ante / win | train ep. ante / win | blinds/ep | explained var |
|---|---|---|---|---|---|---|
| BC (start) | 0 | 5.3 / 7% | 4.8 / 4% | 4.8 / 4% | 10.2 | |
| 24 | 3.2 | 5.65 / 10% | 5.14 / 6% | 4.8 / 4% | 10.3 | 0.75 |
| 64 | 8.4 | 5.93 / 14% | 5.28 / 7% | 5.2 / 6% | 11.4 | 0.81 |
| 96 | 12.6 | 6.35 / 17% | 5.90 / 11% | 5.7 / 10% | 13.1 | 0.84 |
| 120 | 15.7 | 6.72 / 29% | 6.40 / 19% | 6.1 / 15% | 15.6 | 0.86 |
| 144 | 18.9 | 7.02 / 35% | 6.79 / 29% | 6.6 / 23% | 18.8 | 0.89 |
| 176 | 23.1 | 7.10 / 39% | 6.98 / 36% | 7.0 / 36% | 20.6 | 0.91 |

Roughly: ~3M steps to separate from BC (the first 3 updates are critic warm-up), parity with the teacher at ~10M steps,
then a steep climb (win rate 17% -> 38% between 12M and 22M steps) that flattens near 7.1 at 22-24M. Behaviour changes
(`actions` in the log, sampled model actions per head): SkipBlind 22% -> 2.6% of Blind-phase decisions (the reward pays
every blind, skipping forfeits it), Reroll 4.6% -> 14% of Shop decisions, Buy 22% -> 14%, Discard 23% -> 20% of Play
decisions, episode length 72 -> 184 steps; the chain entropy fell 0.65 -> 0.47 (kind entropy of the Play head 0.18 -> 0.13,
card-set entropy 1.0 -> 0.75). KL to the BC policy (reference) stayed 0.03-0.15 with occasional k3 spikes (0.7-0.9) in the
first 40 updates at lr 1.5e-4 (the reason for lr 1e-4).

**(b) learning curve** (stage 1, validation 256 games): deterministic ante 1.77 (u15) -> 2.23 -> 2.62 -> 3.19 -> 3.41 (u135),
blinds beaten per game 2.4 -> 6.7, training episodes 1.0 -> 3.2 ante; entropy 4.7 -> 2.0. **A from-scratch policy gets no
signal without shaping**: the random policy beats 0.01 blinds (ante 1.00, value loss 1e-4, explained variance < 0), the
first attempt without shaping sat at ante 1.00 for 26 updates. With `shaping = 1` (dense progress toward the blind target)
learning starts after ~10 updates. The shaped reward is for this probe only (the (a) run uses the plain reward).

### Throughput

| | value |
|---|---|
| rollout (1024 envs, `prepare` + `act` + D2H + `step`) | 3.7-3.9 s per 128 steps = 33-35k env steps/s (the model, not the simulator) |
| update (2 epochs x 32 minibatches x 4096 rows, + 2 s reference-policy pass while the KL is on) | 13-16 s (~20k rows/s fwd+bwd; compute-bound: batch 2048..8192 rows all give the same rows/s; bf16 autocast would be 1.4x but breaks the exact ratio = 1) |
| overall (rollout + update) | 6.7k steps/s (7.8k once the KL reference pass is off), 131k steps per 17-20 s |
| periodic validation (512 games det + sampled) | 10-45 s |
| GPU memory | peak ~9 GB (default shapes); a second concurrent run needs `group 64, chunk 16` and expandable segments |
| teacher (stage 1, from the log of (b)) | 2.8k actions/s aggregate (Shop dominates: 2k/s; Blind 150k/s, Play 30k/s, Pack 3.8k/s); the teacher was 41% of the steps and **41% of the rollout wall time**; (b) ran at 1.2-3k steps/s overall. Stage 1 costs ~5x more wall time per env step than an end-to-end run. |

## 6. Long runs

```bash
# (a) end-to-end from BC, as above but longer (~55-70 min per 24M steps on an idle machine; resume-able)
PYTORCH_ALLOC_CONF=expandable_segments:True uv run --no-sync python -u scripts/train_ppo.py \
    --name e2e_long --init runs/bc/model.pt --lr 1e-4 --kl-coef 0.3 --kl-anneal-updates 120 \
    --critic-warmup 2 --total-steps 100e6 --eval-every 8 --eval-games 512 > runs/e2e_long.log 2>&1 &
# continue after an interruption (same name):   ... --name e2e_long --resume (all other flags as before)
# curve:   uv run --no-sync python scripts/summarize_run.py runs/e2e_long
# test:    uv run --no-sync python scripts/evaluate.py --checkpoint runs/e2e_long/best.pt --name e2e_long

# (b) from scratch, stage 1 -> 2 -> end-to-end curriculum (the teacher plays what the model does not)
uv run --no-sync python -u scripts/train_ppo.py --name cur --curriculum "play@10e6,play+pack+blind@20e6,all" \
    --shaping 1 --lr 3e-4 --ent-coef 0.01 --num-envs 512 --rollout-len 64 --total-steps 60e6   # ~8-10 h, teacher-bound

# memory ablation under RL: same recipe from the memoryless BC checkpoint (runs/bc_nomem/model.pt, 1.83M params)
uv run --no-sync python -u scripts/train_ppo.py --name e2e_nomem --init runs/bc_nomem/model.pt \
    --lr 1e-4 --kl-coef 0.3 --kl-anneal-updates 120 --critic-warmup 2 --total-steps 24e6
```

`make ppo-e2e NAME=.. STEPS=..`, `make ppo-s1`, `make eval-ckpt CKPT=..` wrap the first two and the evaluation.
`best.pt` is chosen on the validation prefix of the test set, so its test score is slightly optimistic (section 5 also
reports the unselected final checkpoint).

## 7. Caveats and open points

* **Deterministic truncation**: solved in Phase 10B (section 8): the loop is a MoveJoker swap cycle; the evaluation
  repetition guard removes it exactly, the `reorder` reward term reduces it in training.
* **Simulator exploits**: a 2x win rate over the heuristic in 24M steps is large; nothing in the logs says it is
  illegitimate (0 invalid actions, plausible shifts: no skipping, more rerolls), but the policy should be inspected game by
  game (replay, joker/voucher choices, rerolling) before the number is believed, and the weak spots of the simulator
  (UNVERIFIED items in notes section 4.1) are exactly what RL finds.
* The value critic is shared and normalised only by the constant `reward_scale`; explained variance reached 0.91.
* Truncation bootstrap is approximate (`V(s_{T-1})`, section 1); exact would need the env to keep the final observation.
* k3 KL estimator spikes (0.7-0.9) appear when a few sampled actions get probability collapses; harmless here, but watch it
  with larger learning rates.

## 8. Phase 10B: the deterministic-policy stall, its fix and the new knobs

**Diagnosis** (`runs/a_e2e/{best,latest}.pt`, 1000 test games, deterministic): all 49 (best) / 89 (latest) truncated games
are the *same* loop: the last 500+ steps consist exclusively of `MoveJoker` (kind 8) actions, a period-2 swap of the same
two joker slots (`MoveJoker 3 2`, `MoveJoker 2 3`, ...; period in the action sequence: 2 in 83/89, 3 in 3, 1 in 2
(the same `MoveJoker 3 0` forever: it rotates the joker order, so the order cycles with period 4-5), 6 in 1), in the Shop (76/89 and 27/49) or in Play (13/89 and 22/49), at any ante (1-8), money and
round unchanged. The loop starts after 20-530 steps (mean ~150). It is a legal but pointless action cycle the simulator
allows: reordering jokers costs neither money, a hand nor a discard and changes nothing but the order (which matters for
scoring: Blueprint/Brainstorm, chips-before-mult jokers), so it is the one action that can be repeated forever. No
reroll/sell/buy/pack/consumable cycle appears: those spend resources and are self-limiting. The argmax policy sits in a
fixed point: after a swap the order-aware observation changes, the argmax is the swap back. Sampling escapes by chance
(MoveJoker is 4% of the sampled Shop decisions), the training distribution never contains the cycle, so no reward can fix
the deterministic mode directly.

**Fix = a hard evaluation guard + a soft training cost + proper truncation handling** (all three are small and
independent):

* *Evaluation guard* (`ModelActor(move_limit=k)`, `evaluate_model(.., move_limit=k)`, `scripts/evaluate.py --move-limit k`,
  `PPOConfig.eval_move_limit` for the validation): after `k` consecutive MoveJoker actions of a game the MoveJoker kind is
  masked out of the next decision (never when it is the only kind; the count restarts with any other action and with a new
  game). Deterministic given the seed, no model change, and the state space is not touched, so the policy is still judged
  by its own argmax everywhere else. Default off (the raw number stays available); the recommended report is *raw* and
  *guard 2*. Checked on `a_e2e` (without any retraining): k = 1, 2, 4 give the same result within noise (table below),
  truncation 0% for all.
* *Training cost*: reward term `reorder` (rl_interface section 7; `--reorder-cost -0.05`), added to every valid MoveJoker.
  The sampled policy learns to reorder less (MoveJoker share of the Shop decisions 4.3% -> 2.5% -> 1.6% over the
  fine-tunes), the raw deterministic truncation falls 4.9-8.7% -> 3.2-3.6% (cost -0.05, 8M steps) -> 0.8% (-0.25, +6M
  steps) without hurting the sampled score. It does not reach exactly 0: states where MoveJoker is the argmax with
  probability < 1 are rarely visited in sampled training.
* *Truncation bootstrap*: section 1 (`trunc` flag in GAE). With the loop removed truncation is rare in training
  (`ep_trunc` ~ 0 in the logs).
* Not done: lowering `max_episode_steps` (it would only truncate sooner); low-temperature sampling (also seed-deterministic,
  but it would report a different policy than the argmax; the guard keeps the argmax everywhere else).

**Validation runs** (fine-tune from `a_e2e/latest.pt`, fresh Adam, lr 5e-5 -> 5e-6, no KL / warm-up, curriculum all, rollout
1024 x 128, `--eval-every 10`, new run dirs, `a_e2e` untouched): `c_loopfix` = `--reorder-cost -0.05`, 8.0M steps, 21.5 min;
`c_loopfix2` = from `c_loopfix/latest.pt`, `--reorder-cost -0.25 --seed 1`, 5.9M steps, 14.6 min (GPU peak ~9 GB). Test =
the same 1000 games, Red/White; teacher 5.91 / 16.1%, BC 5.29 / 6.5%:

| checkpoint | steps total | det raw: ante / win / trunc | det guard 2: ante / win / trunc | sampled: ante / win / trunc |
|---|---|---|---|---|
| `a_e2e/best` (10A) | 24M | 7.08 / 36.1% / 4.9% | 7.11 / 36.9% / 0% | 6.95 / 34.4% / 0% (guard 2: 6.93 / 34.3%) |
| `a_e2e/latest` (10A) | 24M | 7.04 / 37.9% / 8.7% | 7.15 / 40.9% / 0% | 6.99 / 38.0% / 0% (guard 2: 6.89 / 32.9%) |
| `c_loopfix/latest` | 32M | 7.22 / 42.9% / 3.2% | 7.25 / 43.4% / 0% | 7.06 / 39.0% / 0% |
| `c_loopfix/best` | 32M | 7.21 / 46.8% / 3.6% | 7.24 / 47.0% / 0% | 7.05 / 40.5% / 0% |
| **`c_loopfix2/latest`** | 38M | **7.24 / 44.2% / 0.8%** | **7.25 / 44.6% / 0%** | **7.15 / 43.4% / 0%** (guard 2: 7.06 / 39.9%) |
| `c_loopfix2/best` | 38M | 7.13 / 41.5% / 0.7% | 7.14 / 41.8% / 0% | 7.05 / 41.3% / 0% |

(standard error of a mean ante ~0.04-0.05, of a win rate ~1.6 points; `best.pt` is chosen on the 512-game validation prefix
and is not better than `latest` on the full set; the guarded sampled numbers vary by +-3 points just from a different random
stream.) Guard on a_e2e: k = 1: best 7.09 / 37.6%, latest 7.17 / 41.9%; k = 4: 7.11 / 37.1%, 7.16 / 41.2%. Mean steps per
deterministic game: 322-430 (a_e2e raw) -> 269 (c_loopfix raw) -> 197 (c_loopfix2 raw) vs 175-185 guarded / sampled. The
gain over `a_e2e` (+0.2 ante, +6-8 points win) is mostly *more training* (the 10A curve was still rising at 24M; no
no-cost control run was made), not the loop fix; the loop fix removes the 3-9% truncation bias and the instability of
the deterministic score. Note the guarded sampled score of `a_e2e` is ~equal to the unguarded one, i.e. the guard does not
help sampled play (it only matters where the argmax loops).

**Recommended recipe for the long run (100M+ steps)**: the 10A command (section 6) plus `--reorder-cost -0.05
--eval-move-limit 2` (validation and `best.pt` then ignore loop noise), and evaluate with `scripts/evaluate.py --move-limit 2`
(report the raw numbers too). If the raw truncation of the final checkpoint is still > 0.5%, a final 5M steps with
`--reorder-cost -0.25` (as `c_loopfix2`) brought it to 0.8% at no cost in score. The cost -0.05 is small against the +1
blind reward but real against the 0.003 entropy bonus; -0.25 did not reduce the sampled win rate in 6M steps, but only
tested on a policy that already knows how to use the jokers, so keep it for the end.

## 9. Phase 10C: the 100M-step run (2026-11-14)

**Run** `e2e_long` = the recommended recipe above from `runs/bc/model.pt`: `--lr 1e-4 --kl-coef 0.3 --kl-anneal-updates 120
--critic-warmup 2 --reorder-cost -0.05 --eval-move-limit 2 --total-steps 100e6 --eval-every 8 --eval-games 512`; 99.9M steps
in 3.75 h (~8.1k steps/s; GPU memory ~15.5 of 16.3 GB with `expandable_segments`). Then `e2e_long_fix` = from
`e2e_long/latest.pt`, `--lr 5e-5 --kl-coef 0 --critic-warmup 0 --seed 1 --reorder-cost -0.25`, 5.9M steps, 12.8 min
(raw deterministic truncation of `e2e_long` was 1.1-1.6% > 0.5%).

**Validation curve** (512 games, deterministic with guard 2): flat ~5.5-5.8 / 10-13% while the KL to BC anneals (first
~15M steps), then a fast rise once it reaches 0 (update 120): 7.14 / 32% at 20M (same as `a_e2e`), 7.1-7.2 / 42-48% at
24-37M (mean ante plateaus, the win rate keeps rising), 7.2-7.35 / 52-58% at 50-59M, 7.3-7.4 / 55-65% at 61-70M,
7.50 / 66-70% at 99M. Approx KL per update fell 0.013 -> 0.0016 and entropy 0.59 -> 0.55; the curve had flattened but not
stopped at 100M.

**Test** (the same 1000 games, Red/White; teacher 5.91 / 16.1%, BC 5.29 / 6.5%, previous best `c_loopfix2/latest`
7.24 / 44.2%):

| checkpoint | steps total | det raw: ante / win / trunc | det guard 2: ante / win / trunc | sampled: ante / win / trunc |
|---|---|---|---|---|
| `e2e_long/best` (val. update 504) | 66M | 7.53 / 67.3% / 1.6% | 7.54 / 67.8% / 0% | 7.40 / 63.4% / 0% |
| `e2e_long/latest` | 100M | 7.50 / 66.1% / 1.1% | 7.50 / 66.7% / 0% | 7.44 / 65.5% / 0% |
| `e2e_long_fix/best` | 102M | 7.47 / 66.4% / 0.5% | 7.47 / 66.7% / 0% | 7.31 / 61.1% / 0% |
| **`e2e_long_fix/latest`** | 106M | **7.51 / 68.6% / 0.1%** | **7.51 / 68.6% / 0%** | **7.46 / 65.9% / 0%** |

Paired vs the teacher (`e2e_long/latest`, raw det): +1.38 +- 0.07 ante, +38.5 +- 1.9 points win. ~80% of the test games
reach ante 8. The second fine-tune at cost -0.25 again removed the loop (truncation 0.1%, mean deterministic steps 202 ->
173) with no loss in score; `e2e_long_fix/latest` is the current reference checkpoint.

**Pitfall**: `scripts/evaluate.py --move-limit 0` does **not** disable the guard, it bans MoveJoker entirely (the guard
blocks after `moves >= limit`); omit the flag for raw numbers. With MoveJoker banned the same checkpoint drops to
~7.3 / 55% (det and sampled), i.e. joker ordering is worth ~11 points of win rate to this policy.

**Memory ablation** `e2e_nomem`: the same recipe and seed from `runs/bc_nomem/model.pt` (no GRU, 1.83M params), 24.0M steps,
58 min. Validation (512, det guard 2) tracks `e2e_long` within noise at every point, including the jump after the KL reaches
0 (update 160: 7.06 / 40.6% vs 7.03 / 37.9%; update 176: 7.12 / 40.8% vs 7.12 / 42.4%). Test (1000 games):

| checkpoint | steps | det raw: ante / win / trunc | det guard 2 | sampled |
|---|---|---|---|---|
| `e2e_nomem/best` | 23M | 7.00 / 36.3% / 9.7% | 7.11 / 40.1% / 0% | 7.03 / 34.6% / 0% |
| `e2e_nomem/latest` | 24M | 7.07 / 37.5% / 8.8% | 7.17 / 40.9% / 0% | 6.96 / 36.0% / 0% |
| `a_e2e/latest` (GRU, 24M, no reorder cost) | 24M | 7.04 / 37.9% / 8.7% | 7.15 / 40.9% / 0% | 6.99 / 38.0% / 0% |

(`e2e_long` kept no 24M checkpoint; `a_e2e` is the closest GRU reference with test numbers.) **Conclusion: at 24M steps the
cross-decision GRU memory gives no measurable gain in strength** -- the observation already carries what the policy uses
(run state, deck counts, levels, shop). One structural difference: the memoryless argmax is a pure function of the
observation, so a MoveJoker cycle that returns to the same observation can never be left; its raw truncation is 8.8-9.7%
even with the -0.05 reorder cost (the GRU run `c_loopfix` with the same cost reached 3.2%, `e2e_long` 1.1% at 100M).
Not tested: whether memory matters beyond 24M, where most of the `e2e_long` gain (42% -> 68%) happened; a 100M memoryless
run would answer it (~4 h).

Open: seed repeats; Phase 5E (other decks and stakes) deferred until after training.

## 11. Phase 10D: head freeze and seed repeats (2026-11-15)

**Head freeze** (`evaluate.py --route`, reference `e2e_long_fix/latest`, 1000 games, raw det; full model 7.51 / 68.6%):
teacher plays SHOP → 6.15 / 16.7% (teacher level), BLIND → 6.77 / 39.7%, PLAY → 7.39 / 60.2%, PACK → 7.41 / 60.6%.
Sampled mode gives the same picture. The gain over the teacher is almost entirely shop decisions, then not skipping blinds.
Drops are not additive (the other heads never saw teacher-induced states). Results: `runs/eval/freeze_no_*.json`.

**Seed repeats** (`scripts/chain_seeds.sh`: the `e2e_long` recipe, same BC init, PPO `--seed` 1 and 2, then the `_fix`
fine-tune):

| seed | 100M raw det | `_fix` raw det | `_fix` sampled |
|---|---|---|---|
| 0 (`e2e_long`) | 7.50 / 66.1% (trunc 1.1%) | 7.51 / 68.6% (0.1%) | 7.46 / 65.9% |
| 1 (`e2e_s1`) | 7.47 / 64.9% (1.3%) | 7.46 / 63.9% (0.2%) | 7.38 / 64.5% |
| 2 (`e2e_s2`) | 7.42 / 61.9% (1.6%) | 7.48 / 62.1% (0.5%) | 7.39 / 61.6% |
| **mean ± sd** | | **7.48 ± 0.02 / 64.9 ± 3.4%** | **7.41 ± 0.05 / 64.0 ± 2.2%** |

The recipe reproduces; seed 0 is on the lucky side. A 150M run was skipped (flat ante curve after 24M, win rate in a
62-65% band over the last 8M steps of each run).

## 12. Phase 10F: reward options (margin, zero-play, endless)

New `scripts/train_ppo.py` flags (all default to off; semantics in `rl_interface.md` section 7):

| flag | meaning |
|---|---|
| `--margin M` | adds `M * ln(min(score/target, cap))` per beaten blind; a typical 1.2-1.5x margin pays `M * 0.18..0.41` |
| `--margin-cap C` | cap of score/target (default 4) |
| `--zero-play Z` | reward (negative = cost) per hand a boss rejected (Psychic/Eye/Mouth) |
| `--endless` | training envs continue after the ante-8 win (`win` paid once) |
| `--endless-max-ante A` | endless episodes end after the ante-A boss is beaten (default 12) |

The reward terms land in `config.json` under `reward`, the endless flags as `endless` /
`endless_max_ante`. Evaluation (`scripts/evaluate.py`, the periodic validation inside training) always
builds its own non-endless env with the default reward. Endless episodes are longer (up to 36 blinds
instead of 24, and the weaker the policy the earlier they end), so check `max_episode_steps` (3000 by
default) and the steps/s of the rollout; `drain_finished` reports `done == 2` for any run that passed
ante 8.

## 13. Phase 10G: decks and stakes in training and evaluation

The observation carries the run's deck and stake (`INTERFACE_VERSION = 2`, `rl_interface.md`
section 12); v1 checkpoints (`--init` / `--resume`, BC data) load through an exact migration.

| flag | meaning |
|---|---|
| `--decks red,plasma` | training envs: every episode of every env draws its deck uniformly from the list (default `red`) |
| `--stakes white,red,green` | ... and its stake from this list (default `white`); the pairs are the product of both lists |
| `--eval-deck D` / `--eval-stake S` | configuration of the periodic validation (default `red` / `white`, the configuration of all earlier numbers) |
| `evaluate.py --deck D --stake S` | evaluation configuration (default red/white); the baseline cache is `baselines_<games>[_route]_<deck>_<stake>.json` for a non-default pair, the result file gets the same suffix |

Sampling happens in the Rust `VecEnv` (`decks=` / `stakes=` constructor arguments, a pure function of
`(seed, env, episode)`), so it costs nothing and is independent of the thread count. Without the flags
the env is the fixed Red/White one, bit for bit. The lists land in `config.json`.
