# Team model (Phase 9A): architecture, API, behaviour-cloning smoke test

Code: `python/balatro_rl/model/{inputs,encoder,heads,cards,team}.py` (model),
`python/balatro_rl/train/{bc,play}.py` + `scripts/bc_smoke.py` (behaviour cloning, game evaluation),
tests `tests_py/test_model.py`, `tests_py/test_bc.py`. Interface it consumes:
`docs/design/rl_interface.md` (`INTERFACE_VERSION = 1`; checkpoints record it and `load_checkpoint`
asserts it).

PyTorch is an opt-in dependency group: `make setup-train` (torch 2.9.1 + CUDA 12.8 wheels from
`download.pytorch.org/whl/cu128`, needed for Blackwell sm_120; ~6.6 GB in `.venv` including the
NVIDIA libraries). Everything torch-related is skipped by `pytest.importorskip("torch")` when it is
not installed, so `make test` works without it.

## 1. Architecture

```
 obs (14 arrays) ──► Encoder ──────────────────────────────────────────────┐
   global_f/i  1 tok   per-type Linear(f) + sum of Embedding(i columns)      │
   hand        nh tok  + token-type emb + slot-position emb                  │
   joker       nj tok  -> LayerNorm -> 4 x pre-norm Transformer block        │
   consumable  nc tok     (d=128, 4 heads, FF 4d, SDPA, padding mask) -> LN  │
   offer       no tok                                                        │
                                                                             ▼
                   summary = global-token output (n, d)   hand (n, nh, d)   ptr (n, 49, d)
                         │                                       │          joker|cons|offer
                         ▼                                       │               │
   hidden (n, 128) ──► GRUCell ──► h (n, 128)                    │               │
   (reset where first)       │                                   │               │
                             ├──► ctx = MLP([summary, h]) + LN (n, d)            │
                             └──► critic MLP([summary, h]) ──► value (n,)        │
                                                                                 │
   phase = global_i[:,0]  ──► head = {Blind, Play, Shop, Pack}[phase]  (rows of a batch are routed)
        Head(ctx, ptr, hand, masks):
          kind  ~ softmax(MLP(ctx))            masked by kind_mask
          arg0  ~ softmax(q0(ctx+emb(kind)) . k0(ptr[ARG0_POINTER[kind]]))   masked by arg0_mask[kind]
          arg1  ~ softmax(q1(z) . k1(joker tokens))   masked by arg1_mask minus arg0  (MoveJoker only)
          cards ~ up to 5 steps over hand tokens + STOP, canonical ascending order (section 3)
        action = [kind, arg0, arg1, card_mask];  logprob = sum of all step logprobs
```

* **Token embeddings.** One embedding table (`encoder.table`) with a row range per vocabulary
  (`LAYOUT["vocab"]`); columns with the same meaning share their range: item (joker / consumable /
  offer / pack item, 250), edition, rank, suit, enhancement, seal (hand and offer cards), hand type
  (last hand, boss hand, tag hand, To Do List hand). Each categorical column of a token is looked up
  and summed with `Linear(continuous features)`, then the token-type embedding (5 types) and a
  learned slot-position embedding (hand 32, joker 20, consumable 8, offer 21) are added. The global
  token uses a 2-layer MLP for its 203 features.
* **Slicing.** Every group is cut to the batch's highest valid row (`hand_valid`, `joker_valid`,
  ...) before the Transformer; joker/consumable/offer outputs are zero-padded back to the fixed
  capacities so the pointer table (joker 0..20 | consumable 20..28 | offer 28..49) has constant
  indices. A Play batch typically runs 25..45 tokens, a Shop batch up to ~60.
* **Memory.** `memory="gru"` (default): `GRUCell(d, 128)` on the global-token output, one hidden
  vector per env, carried across phases (blind select, play, shop, pack) and zeroed when an episode
  starts. The memory enters the policy and critic *after* the Transformer (`ctx`, critic input), so
  the encoder for a whole `(T, n)` rollout is one batched pass and only the (cheap) GRU is scanned in
  time. `memory="none"`: no GRU, `ctx`/critic see the summary only; hidden tensors have width 0 (the
  API is identical).
* **Heads.** One `Head` per phase (Blind, Play, Shop, Pack), each over all 17 kinds (the masks decide
  what is possible: consumables, selling and joker moves are legal in every phase). The head is
  chosen from the observed phase (GameOver/Won are never observed in a running env; they fall back
  to Blind). A batch is split by phase, each head runs on its rows, the results are scattered back.
  Pointer scores are scaled dot products `q . k / sqrt(d)`; kinds without a pointer have the single
  legal option `arg0 = 0` (logprob 0).
* **Critic.** One shared head: `MLP([summary, h]) -> value`.

### Sizes (`TeamModel.num_parameters()`)

Default (`d=128`, 4 layers, 4 attention heads, GRU 128): **1,964,233** parameters: encoder
903,808 (embeddings ~100k, Transformer 4 x 198k, projections), memory 99,072, ctx 49,664,
4 heads x 219,666, critic 33,025. `memory="none"`: 1,832,393. `d=192, 6 heads`: 4,322,569.
Configurable through `ModelConfig(d_model, layers, heads, ff_mult, dropout, memory, mem_dim)`.

## 2. API

```python
from balatro_rl.model import TeamModel, ModelConfig
model = TeamModel(ModelConfig()).to("cuda")
hidden = model.init_hidden(n)                         # zeros (n, mem_dim)
actions, logprob, entropy, value, hidden = model.act(obs, masks, hidden,
                                                     deterministic=False, first=None, generator=None)
logprob, entropy, value = model.evaluate(obs, masks, hidden_in, actions, first=None)
out = model.evaluate_full(obs, masks, hidden_in, actions, first=None)   # + parts, hidden, phase
value, new_hidden = model.value(obs, masks, hidden, first=None)         # bootstrap
```

* `obs` / `masks` are the dicts of the env (numpy, as returned by `BalatroVecEnv`) or prepared
  tensors (`balatro_rl.model.prepare(obs, masks, device)`: categoricals -> int64, floats ->
  float32, `uint32` bit sets -> int64; always a copy, env buffers are overwritten by the next step).
  Stored datasets may keep float16 / uint8 / int32-viewed bit sets; `prepare` converts.
* `actions` is int64 `(n, 4)` = `[kind, arg0, arg1, card_mask]`, directly usable by `env.step`
  (`actions.cpu().numpy()`); it is mask-conforming by construction (zero invalid actions in the
  tests, see section 5). `logprob` = log p(kind) + log p(arg0) + log p(arg1) + sum of the card
  steps (STOP included); `entropy` = sum of the step entropies of the same chain.
  `evaluate(act(...))` reproduces `logprob` and `entropy` to < 1e-5.
* **`first` and sequences.** `first[i] = True` marks an observation that starts a new episode (the
  previous step returned `done != 0`; the env has already auto-reset): the memory of that env is
  zeroed *before* the GRU update. `act(first=(n,))`. `evaluate`: `first=None` = one step per row
  with `hidden_in (B, mem_dim)` the state *before* the step; `first` of shape `(T, n)` = the
  time-major flattening `(T * n, ...)` of a rollout, `hidden_in (n, mem_dim)` the state before
  step 0, GRU scanned over `T` with resets. Store `hidden` of each rollout step *before* acting (or
  only at chunk starts for truncated BPTT); `evaluate_full(...)["hidden"]` is the state after the
  last step (carry it, detached, to the next chunk).
* `evaluate_full` additionally returns the per-part entropies `ent_kind, ent_arg0, ent_arg1, ent_cards` (their sum is
  `entropy`); PPO logs them per head (Phase 10A, `docs/design/training.md`).
* `generator`: a `torch.Generator` (same device as the model) makes sampling reproducible
  (Gumbel-max over the masked logits). `evaluate_full` also returns `lp_kind, lp_arg0, lp_arg1,
  lp_cards`, `hit_*` (teacher-forced argmax equals the target), `has_*` and `ok` (the action is
  expressible through the masks and the canonical order; false rows have logprob contributions 0 and
  must be masked in a loss).
* Checkpoints: `save_checkpoint(path, model, **extra)` / `load_checkpoint(path, device)`.

## 3. Canonical card order

The core's rule (`rl_interface.md` section 6) accepts cards in any order, so a set has up to `k!`
chains. The model builds sets in **ascending hand-position order**, so `logprob(set)` is one product
and the BC target is unique. State: `partial` bit set and `last` (highest chosen position, -1 at the
start). Allowed next positions `q` (`model/cards.py::canonical_step`):

```
q in candidates, q > last, q not in partial
q <= min(missing required)                     # a required card is never skipped
k + 1 <= max                                   # k = popcount(partial)
|missing required after q| <= max - k - 1      # required cards still fit
min - k - 1 <= #candidates above q             # the minimum count stays reachable (*)
STOP allowed  iff  all required chosen and min <= k <= max
```

(*) The core rule needs no such term because it is order-free; with ascending order, picking a high
card first could make `min` unreachable (targeted consumables have `sel_min` 1..2, Death needs 2),
so the term is required for "never dead-ends". By induction: a state with a legal completion either
can STOP or can pick the smallest card of that completion; every legal set is reached by exactly its
own ascending chain. Tests: 500 random rules (incl. several required cards and minimums > 1) by DFS
against brute-force enumeration (same sets, **exactly one chain per set**, no dead end), and on real
game states (Play, Discard, targeted consumables, pack picks, Cerulean Bell's forced card) the
model's `exp(lp_cards)` of all legal sets **sums to 1** and illegal sets are not expressible.

## 4. Behaviour cloning (`scripts/bc_smoke.py`, `make bc-smoke`)

Purpose: prove the whole pipeline (encoder, memory, heads, canonical card chain, masks) can learn.

* **Data**: the fair-information heuristic teacher (`teacher_actions`) plays 1024 envs x 1500 steps
  (seed 1, no noise): 1,536,000 labelled states in **375 s** (4.1k actions/s), teacher run
  statistics during collection: 12,222 episodes, mean ante 5.83, win rate 15.8%. Stored as
  time-major `.npy` memmaps `(T, N, ...)` in `runs/bc_data/` (float16 continuous, uint8
  categorical, 2.7 KB/sample, 4.15 GB; gitignored), loaded completely onto the GPU (4.15 GB).
* **Training**: maximum likelihood of the teacher's structured action
  (`-(lp_kind + lp_arg0 + lp_arg1 + lp_cards)`) plus 0.1 x smooth-L1 value regression onto the
  discounted teacher return (`gamma = 0.99`, pre-trains the critic). Stateful truncated BPTT:
  groups of 128 envs are swept in time order in chunks of 32 steps (batch = 4096 rows) with the GRU
  state carried (detached) between chunks and zeroed at episode starts, exactly like at play time;
  chunks of the 7 groups are interleaved, chunk boundaries move every epoch. 128 envs held out
  (validation), 896 for training. AdamW lr 7e-4 (200 warm-up updates, cosine to 5%), weight decay
  1e-4, grad clip 1.0, TF32, 12 epochs = 4018 updates, **72.8 s/epoch, 14.5 min** on the RTX 5060 Ti
  (12.5 GB GPU memory in use including the dataset).
* **Teacher-forced accuracy** on held-out envs (argmax equals the teacher at that step given the
  teacher's previous choices; all teacher actions were expressible: `expressible = 1.0`):

| epoch | train NLL | val NLL | kind | arg0 (rows with >1 option) | arg1 | cards (exact set) | full action |
|---|---|---|---|---|---|---|---|
| 1 | 2.091 | 1.421 | 0.852 | 0.809 | 0.898 | 0.351 | 0.638 |
| 2 | 1.229 | 1.171 | 0.878 | 0.822 | 0.913 | 0.451 | 0.680 |
| 4 | 0.872 | 0.879 | 0.897 | 0.855 | 0.938 | 0.572 | 0.733 |
| 6 | 0.737 | 0.783 | 0.910 | 0.866 | 0.936 | 0.613 | 0.761 |
| 8 | 0.660 | 0.734 | 0.915 | 0.872 | 0.942 | 0.638 | 0.774 |
| 10 | 0.603 | 0.721 | 0.915 | 0.873 | 0.939 | 0.647 | 0.775 |
| 12 | 0.574 | 0.696 | 0.920 | 0.874 | 0.943 | 0.656 | 0.783 |

  Per phase at epoch 12 (kind / arg0 / cards / full): Blind 0.990 / 0.857 / - / 0.990 (15% of rows);
  Play 0.909 / 0.874 / 0.635 / 0.592 (33%); Shop 0.885 / 0.904 / - / 0.853 (40%);
  Pack 0.980 / 0.823 / 0.993 / 0.823 (12%). Train/val gap (0.57 vs 0.70) shows the start of
  over-fitting on 1.5M samples; more data or noise injection would help.
* **Games** (1024 games = the first episode of each of 1024 envs, base seed 10000019, Red/White,
  `max_episode_steps` 3000; `scripts/bc_smoke.py --stage eval`):

| policy | mean ante (std) | win rate | blinds beaten | truncated | invalid / game |
|---|---|---|---|---|---|
| random (uniform mask-conforming) | 1.00 (0.00) | 0.0% | 0.01 | 0% | 0 |
| **BC deterministic** | **5.29 (1.87)** | **6.6%** | 11.4 | 0.3% | 0 |
| BC sampled | 4.80 (2.00) | 4.3% | 10.2 | 0% | 0 |
| teacher (fair-info heuristic) | 5.90 (1.86) | 16.2% | 13.1 | 0% | 0 |

  The BC policy never plays an invalid action and gets 90% of the teacher's average ante; the win
  rate (needs a long sequence of near-perfect plays against the showdown bosses) is a third of the
  teacher's. The teacher decides with preview-style score simulation that the network only learns
  approximately (Play exact-set accuracy 64%).

* **Memory ablation** (`--memory none`, 1.83M parameters, same data/hyper-parameters): epoch-12 val
  NLL 0.706 (GRU 0.696), kind 0.916 / arg0 0.874 / cards 0.654 / full 0.779 (GRU 0.920 / 0.874 /
  0.656 / 0.783); games: deterministic ante 5.39, win 6.4%, truncated 1.4%; sampled ante 4.75, win
  4.0%. **No measurable difference**: the teacher is almost Markov in the observation (its per-run
  state is largely visible: jokers, hand levels, vouchers, `last_hand`), so imitation cannot show
  the value of the memory. It has to be judged under RL (Phase 10).
## 5. Tests (`tests_py/test_model.py`, `test_bc.py`; skipped without torch)

Shapes/dtypes; **thousands of model actions (sampled and greedy) in the env with 0 invalid** and all
expressible via `select.conforms`; `evaluate` reproduces `act` logprob/entropy/value over a rollout
(|diff| < 1e-5, random `first` resets, all four phases seen); `(T, n)` sequence `evaluate` equals
step-wise `act` and returns the same final hidden state; memory is zeroed exactly on `first` rows;
gradients reach encoder, GRU, critic, ctx and the heads; CPU and CUDA agree; generator
reproducibility; canonical-order brute force and set-probability normalisation (section 3); a tiny
end-to-end BC pipeline (collect -> train -> loss falls, all teacher actions expressible).

## 6. Notes for the PPO trainer

(Implemented in Phase 10A: `docs/design/training.md`; the notes below are the design input.)

* Per env step: `prepare` once (H2D copy of ~2.7 KB/env), `act`, `env.step(actions.cpu().numpy())`.
  Keep `first = done != 0` for the next `act`/`evaluate` call.
* Store per step: prepared obs/masks, actions, logprob, value, reward, done, `first`; hidden only at
  chunk starts (carry `evaluate_full(...)["hidden"]` detached, or store `act`'s `hidden` before each
  chunk). Recurrent minibatches are env-groups x time-chunks as in `train/bc.py::train` (batch =
  chunk x group rows); a flat `(B,)` minibatch with stored per-row hidden also works
  (`first=None`).
* `act` costs 4 head launches + the card loop (<= 6 small steps) + 1 device sync for the slicing
  lengths and one per head for the row routing; BC measured 0.22 s per 4096-row update
  (forward+backward, fp32/TF32); a full env step with `act` at 1024 envs (prepare + act + D2H +
  `env.step`) takes ~27 ms, i.e. ~38k env steps/s: the model, not the simulator (4.5M steps/s),
  bounds PPO throughput. Larger rollouts per `act` (more envs) amortise the launch overhead.
