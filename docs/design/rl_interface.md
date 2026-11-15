# RL interface (Phase 8B): token observation, structured action codec, masks, reward

`INTERFACE_VERSION = 1` (`balatro_rl._engine.INTERFACE_VERSION`, also `balatro_rl.INTERFACE_VERSION`).
Bump it on ANY change of a buffer name, shape, dtype, feature order, scaling, vocabulary, action
semantics or reward semantics. Checkpoints must record it.

Code map: `crates/balatro-py/src/{layout,encode,env,sample,vec_env,single}.rs`,
`python/balatro_rl/{env,select}.py`. The feature lists below are the single source of truth for
*meaning*; the order of every list is the order of the columns, and `_engine.layout()` returns the
same names, shapes, dtypes and vocabulary sizes (a test checks that this document mentions each
buffer and feature name).

The simulator side is `balatro-core`; the player-view rules (what may be observed) are in
`crates/balatro-core/src/observe.rs`, the action structure in `crates/balatro-core/src/action_mask.rs`.

## 1. Conventions

* One env step = one *structured action* (section 4). Every buffer is caller-owned, C-contiguous,
  `(num_envs, ...)`, filled in place by Rust (GIL released). Names below are the dict keys.
* **Categorical features** are `int32` columns for embeddings. Rule: `value = index + 1`, and
  **0 means hidden / none / unknown** (an embedding row "no information"). Exceptions are listed
  in the vocabulary column (`phase`, `blind` are always defined and use the plain index).
  Padding rows are all-zero and are told apart by the `*_valid` mask, never by a 0 id.
* **Continuous features** are `float32`. Scalings (all documented per feature):
  * `L(x) = log10(1 + min(max(x, 0), 1e30)) / 10` for large non-negative magnitudes (chip
    targets, scores). `+inf` targets (ante 39+) saturate at 3.0.
  * `S(x) = sign(x) * ln(1 + |x|) / 5` for money-like values (~1.0 at $150, signed: debt).
  * `C(x) = ln(1 + x) / 3` for unbounded counts (hand plays; ~1.0 at 20).
  * otherwise `x / divisor` with the divisor given (not clamped; values may exceed 1).
* Booleans are `0.0 / 1.0` in float arrays.
* Masks are `bool`.
* Hidden information is encoded as "hidden" (all categoricals 0, continuous 0, an explicit
  `face_down` / `hidden` flag): section 9.

## 2. Capacities (compile-time, `_engine.layout()["caps"]`)

| token type | capacity | reasoning |
|---|---|---|
| global | 1 | |
| hand | 32 | = `MAX_SELECTABLE_HAND` of the core: lossless. The rule cap is 16 dealt cards, but Certificate/DNA/Familiar/Incantation add cards on top, and positions >= 32 cannot be selected by the core at all. Hand token `i` is hand position `i` is bit `i` of a card mask. Slice to the largest valid count of the batch before the Transformer (`hand_valid.sum(1).max()`). |
| joker | 20 | 5 base slots + Antimatter + Negative jokers; 16+ in any realistic run. Jokers at slots >= 20 are neither shown nor addressable. |
| consumable | 8 | 2 base + Crystal Ball + Negative copies. |
| offer | 21 | rows `0..8` shop card slots (2 base + Overstock x2 = 4, plus Uncommon/Rare-tag jokers that are appended), rows `8..12` shop booster slots (2), rows `12..16` shop voucher slots (ante voucher + Voucher-tag extras), rows `16..21` items of the open pack (max 5). Slots beyond a range are neither shown nor addressable (they are always optional purchases, never a forced move). |

`MAX_ARG = 20` (largest addressable source: jokers) is the width of `arg0_mask`.

## 3. Observation

### 3.1 Buffers

| name | dtype | per-env shape | content |
|---|---|---|---|
| `global_f` | f32 | (203,) | section 3.2 |
| `global_i` | i32 | (8,) | section 3.2 |
| `hand_f` | f32 | (32, 6) | section 3.3 |
| `hand_i` | i32 | (32, 5) | |
| `hand_valid` | bool | (32,) | |
| `joker_f` | f32 | (20, 13) | section 3.4 |
| `joker_i` | i32 | (20, 5) | |
| `joker_valid` | bool | (20,) | |
| `cons_f` | f32 | (8, 3) | section 3.5 |
| `cons_i` | i32 | (8, 1) | |
| `cons_valid` | bool | (8,) | |
| `offer_f` | f32 | (21, 7) | section 3.6 |
| `offer_i` | i32 | (21, 7) | |
| `offer_valid` | bool | (21,) | |

Observation = these 14 arrays (a `dict[str, np.ndarray]`). Mask buffers (section 5) and the step
outputs `reward`, `done`, `invalid` come from the same call but are separate dicts in the Python
wrapper.

Token count in the Transformer: 1 global + up to 32 hand + 20 joker + 8 consumable + 21 offer;
add a learned token-type embedding per type, and for jokers/consumables/offers the slot position
(`pos`, or the row index) is a feature.

### 3.2 Global token (`global_f`, `global_i`)

`global_i` (vocabulary size in brackets):

| col | name | vocab | meaning |
|---|---|---|---|
| 0 | `phase` | 6 | `Phase` index: BlindSelect 0, Play 1, Shop 2, PackOpen 3, GameOver 4, Won 5 (plain index) |
| 1 | `blind` | 3 | Small 0, Big 1, Boss 2 (the blind being played / next to select; plain index) |
| 2 | `boss` | 29 | boss id of this ante + 1 (the boss is shown on the blind screen from the start of the ante) |
| 3 | `last_hand` | 13 | hand type last played this run + 1, 0 none |
| 4 | `boss_hand` | 13 | The Ox's hand type + 1 while the boss is in effect, else 0 |
| 5 | `open_pack` | 16 | open booster spec + 1 (`kind * 3 + size`, kinds Arcana, Celestial, Spectral, Standard, Buffoon; sizes Normal, Jumbo, Mega), 0 when no pack is open |
| 6 | `offered_tag` | 25 | tag id + 1 that skipping the current blind would give; only in BlindSelect on a Small/Big blind, else 0 |
| 7 | `offered_tag_hand` | 13 | the Orbital tag's hand type + 1, else 0 |

`global_f` scalars (cols 0..27):

| col | name | scaling |
|---|---|---|
| 0 | `ante` | `ante / 8` |
| 1 | `round` | `round / 24` |
| 2 | `target_log` | `L(target)` |
| 3 | `score_log` | `L(round_score)` |
| 4 | `progress` | `min(round_score / target, 1)`; 0 if target is not finite or not positive |
| 5 | `score_ratio` | `clamp(round_score / target, 0, 4) / 4` |
| 6 | `money` | `S(money)` |
| 7 | `hands_left` | `/ 8` |
| 8 | `discards_left` | `/ 8` |
| 9 | `hand_size` | effective hand size, `/ 16` |
| 10 | `hand_count` | cards in hand, `/ 16` |
| 11 | `joker_count` | `/ 10` |
| 12 | `joker_capacity` | `/ 10` (base + Negative + modifiers) |
| 13 | `consumable_count` | `/ 4` |
| 14 | `consumable_capacity` | `/ 4` |
| 15 | `reroll_price` | `S(price)` (0 while a free reroll is left) |
| 16 | `interest_cap` | `/ 25` |
| 17 | `draw_pile` | cards left to draw, `/ 52` |
| 18 | `deck_size` | all playing cards of the run, `/ 52` |
| 19 | `hands_played_round` | `/ 8` |
| 20 | `discards_used_round` | `/ 8` |
| 21 | `boss_active` | boss effect in force (Play on a boss blind, not disabled) |
| 22 | `boss_disabled` | Chicot / Luchador / Verdant Leaf |
| 23 | `jokers_hidden` | Amber Acorn: jokers are face down |
| 24 | `showdown` | the boss is a Showdown boss |
| 25 | `won` | the ante-8 boss was beaten (Endless) |
| 26 | `picks_left` | picks left in the open pack, `/ 2` |

`global_f` vectors (cols 27..203), each named `<prefix>.<i>` in `layout()`:

| cols | name | length | content / scaling |
|---|---|---|---|
| 27..39 | `hand_level` | 12 | level of each hand type (HandType order: High Card ... Flush Five), `ln(level) / 3` |
| 39..51 | `hand_plays_round` | 12 | plays this round, `C(x)` |
| 51..63 | `hand_plays_run` | 12 | plays this run, `C(x)` |
| 63..115 | `draw_rank_suit` | 52 | non-Stone cards left in the draw pile, `count / 4`, index `rank_index * 4 + suit_index` (rank Two..Ace; suit Spades, Hearts, Clubs, Diamonds) |
| 115 | `draw_stone` | 1 | Stone cards in the draw pile, `count / 4` |
| 116..124 | `draw_enh` | 8 | draw pile by enhancement (Bonus, Mult, Wild, Glass, Steel, Stone, Gold, Lucky), `C(count)` |
| 124..128 | `draw_edition` | 4 | by edition (Foil, Holographic, Polychrome, Negative), `C(count)` |
| 128..132 | `draw_seal` | 4 | by seal (Gold, Red, Blue, Purple), `C(count)` |
| 132..164 | `voucher` | 32 | voucher owned (VoucherId order) |
| 164..188 | `tag` | 24 | number of held tags of that id (TagId order), `count / 2` |
| 188..203 | `pending_pack` | 15 | queued packs (from pack tags) by spec index, `count / 2` |

The draw-pile histogram is the in-game deck viewer (what is *left*): face-down hand cards are not in
it and their identity is not given anywhere. Everything about the order of the pile is hidden.

### 3.3 Hand tokens (`hand_f`, `hand_i`, `hand_valid`)

Row `i` = hand position `i` (hands are rank-sorted by the engine, left to right).

`hand_i`: `rank` (vocab 15: 0 hidden, 1..13 Two..Ace, 14 Stone), `suit` (vocab 6: 0 hidden,
1..4 Spades/Hearts/Clubs/Diamonds, 5 Stone), `enhancement` (10: 0 hidden, 1 None, 2.. Bonus, Mult,
Wild, Glass, Steel, Stone, Gold, Lucky), `edition` (6: 0 hidden, 1 Base, Foil, Holographic,
Polychrome, Negative), `seal` (6: 0 hidden, 1 None, Gold, Red, Blue, Purple).

`hand_f`: `pos` (`i / 16`), `chips` (rank chip value `/ 11`, 0 for Stone/hidden), `perma_bonus`
(`/ 50`), `debuffed` (visible cards only), `face_down` (1 = identity hidden: all categoricals 0 and
`chips = perma_bonus = debuffed = 0`), `forced` (Cerulean Bell's force-selected card: it must be part
of every Play/Discard; also encoded in the card-selection table).

### 3.4 Joker tokens (`joker_f`, `joker_i`, `joker_valid`)

Row = joker slot (left to right). While Amber Acorn is in effect (`jokers_hidden`) a joker row is
valid but hidden: `hidden = 1`, categoricals 0, everything else 0 except `pos`.

`joker_i`: `item` (shared item vocabulary, section 3.7; jokers are ids 1..150), `edition` (6, as
above), `aux_suit` (5: 0 none, 1..4; Ancient Joker/Castle target), `aux_rank` (14: 0 none, 1..13;
Idol), `aux_hand` (13: 0 none, 1..12; To Do List).

`joker_f`: `pos` (`/ 16`), `sell_value` (`S`), `eternal`, `perishable`, `perish_rounds_left`
(`/ 5`), `rental`, `debuffed`, `hidden`, `value_lin` (`clamp(value / 10, -10, 10)`), `value_log`
(`S(value)`), `counter_lin` (`min(counter, 100) / 10`), `counter_log` (`ln(1 + counter) / 5`),
`aux_byte` (`/ 255`, the joker's free state byte). `value`/`counter` are the joker's public running
state (Ride the Bus mult, Obelisk, Loyalty countdown ...).

### 3.5 Consumable tokens (`cons_f`, `cons_i`, `cons_valid`)

Row = consumable slot. `cons_i`: `item` (ids 151..202). `cons_f`: `pos` (`/ 4`), `sell_value`
(`S`), `negative`.

### 3.6 Offer tokens (`offer_f`, `offer_i`, `offer_valid`)

A fixed row layout, valid only where the slot is occupied and the phase shows it (shop slots only
in `Shop`, pack items only in `PackOpen`):

| rows | `source` | slot kinds |
|---|---|---|
| 0..8 | 1 shop card | joker / tarot / planet (spectral) / playing card |
| 8..12 | 2 booster | one of the 15 booster specs |
| 12..16 | 3 voucher | a voucher |
| 16..21 | 4 pack item | a consumable, a playing card or a joker of the open pack |

`offer_i`: `source` (5: 0 none, 1..4), `item` (item vocabulary: joker, consumable, voucher or
booster id; 0 for a playing card), `edition` (6; jokers and playing cards), `rank`, `suit`,
`enhancement`, `seal` (playing cards only, same vocabularies as hand tokens; 0 otherwise).
`offer_f`: `price` (`S`; 0 for pack items), `free`, `affordable` (`money >= price`; 1 for pack
items), `eternal`, `perishable`, `rental` (jokers), `sell_value` (jokers; `S`).

### 3.7 Shared item vocabulary (size 250)

`0` none/hidden, `1 + joker_index` (1..150), `151 + consumable_index` (151..202: 12 planets,
22 tarots, 18 spectrals), `203 + voucher_index` (203..234), `235 + booster_spec_index` (235..249).
Jokers, consumables, offers and pack items share one embedding table.

## 4. Action codec

An action is `int64 (n, 4) = [kind, arg0, arg1, card_mask]` per env (`ActionKind`, 17 kinds; the
numbering is `ActionKind::index`, stable):

| kind | name | arg0 | arg1 | cards |
|---|---|---|---|---|
| 0 | SelectBlind | | | |
| 1 | SkipBlind | | | |
| 2 | Play | | | 1..5 hand cards |
| 3 | Discard | | | 1..5 hand cards |
| 4 | SellJoker | joker slot | | |
| 5 | EndShop | | | |
| 6 | UseConsumable | consumable slot | | 0, or `min..=max` hand cards (targeted consumables) |
| 7 | SellConsumable | consumable slot | | |
| 8 | MoveJoker | from joker slot | to joker slot | |
| 9 | Buy | shop card slot | | |
| 10 | BuyAndUse | shop card slot | | |
| 11 | OpenPack | shop booster slot | | |
| 12 | BuyVoucher | shop voucher slot | | |
| 13 | Reroll | | | |
| 14 | PickPackItem | pack item | | 0, or `min..=max` hand cards (targeted consumable) |
| 15 | SkipPack | | | |
| 16 | RerollBoss | | | |

`card_mask` is a `u32` bit set of hand positions (bit `i` = hand token `i`). Unused operands are 0
(operands a kind does not use are ignored, so a stray value is harmless).

**Pointer targets** (what a head scores `arg0` against; `_engine.layout()["arg0_pointer"]`): the
`arg0` of a kind is a slot of its own source, and `arg0` is a *row* of the pointer token array:

| kind | token array | row |
|---|---|---|
| SellJoker, MoveJoker (arg0 = from, arg1 = to) | `joker_*` | `arg0` (`arg1`) |
| UseConsumable, SellConsumable | `cons_*` | `arg0` |
| Buy, BuyAndUse | `offer_*` | `arg0` (0..8) |
| OpenPack | `offer_*` | `8 + arg0` |
| BuyVoucher | `offer_*` | `12 + arg0` |
| PickPackItem | `offer_*` | `16 + arg0` |
| cards of Play/Discard/UseConsumable/PickPackItem | `hand_*` | bit position |

Kind `kind` itself is chosen from `kind_mask` with the global token (a categorical head).

**Validation.** The env builds `Action::from_parts` and runs the core `validate`. An action that
is malformed (kind >= 17, an operand outside its range, `card_mask` outside `u32`) or rejected by
`validate` is a **no-op**: the game state does not change, `invalid[i] = 1`, the reward is
`reward_invalid` (default -0.05), the step counts towards the episode step limit, and the
observation/masks are rewritten (unchanged). Invalid actions never raise; shape/dtype errors do.
An action built through the masks below is always valid (property-tested; the rate of invalid
actions of a mask-conforming policy is exactly 0 in the tests).

## 5. Masks (written after every step and reset)

| name | dtype | per-env shape | meaning |
|---|---|---|---|
| `kind_mask` | bool | (17,) | kinds with at least one legal action (never all zero in a running episode) |
| `arg0_mask` | bool | (17, 20) | legal `arg0` per kind. Kinds without an `arg0` have `{0}`; an illegal kind has an empty row. Slots >= 20 are not representable (section 2) |
| `arg1_mask` | bool | (20,) | `MoveJoker` destinations: every joker slot (no slot is excluded here; the model must exclude `arg0` itself, the destination is any slot `!= arg0` below the joker count). All zero if MoveJoker is not legal |
| `sel_candidates` | u32 | (15,) | per card-selection row, the hand positions (bit set) that may be part of the selection |
| `sel_required` | u32 | (15,) | positions that must be part of it (Cerulean Bell's forced card) |
| `sel_min` | u8 | (15,) | fewest cards |
| `sel_max` | u8 | (15,) | most cards; `0` = this row takes no cards (not a card action) |

Selection rows: row 0 = Play, row 1 = Discard, rows `2..10` = `UseConsumable` slot `arg0`, rows
`10..15` = `PickPackItem` item `arg0` (`sel_row(kind, arg0)` in `balatro_rl.select`). A row is
non-empty (`sel_max > 0`) exactly when the chosen `(kind, arg0)` takes hand cards. For a
targeted consumable `sel_min >= 1`; an untargeted consumable has `sel_max = 0` and is played
with `card_mask = 0`.

## 6. Sequential card selection (inside the model)

The model picks `kind`, then `arg0` (then `arg1` for MoveJoker). If `sel_max[row] == 0` the
action is complete (`card_mask = 0`). Otherwise it picks cards one at a time, at most 5 (Play/
Discard) or 3 (consumables) steps. State: `partial` (bit set, starts at 0). At every step:

```
k      = popcount(partial)
if k >= max:                           next = 0
else:
    pool    = candidates & ~partial
    missing = required & ~partial
    next    = missing if (missing != 0 and max - k == popcount(missing)) else pool
can_stop = (partial & ~candidates == 0) and (partial & required == required)
           and min <= k <= max
```

The model samples among `next` positions plus a STOP choice that is allowed iff `can_stop`;
the final `card_mask = partial`. By construction the selection never dead-ends (`next` is empty only
when `can_stop` holds; `can_stop` is false only when `next` is non-empty) and the result is
legal. This is **exactly** `CardSelect::next_candidates` / `CardSelect::can_finish` of the core;
`balatro_rl.select.next_candidates` / `can_finish` are the Python reference and
`tests_py/test_select.py` checks them against the core on thousands of random states
(`Game.select_next_candidates`, `Game.select_can_finish`). Order does not matter (the mask is a set).

## 7. Reward (`VecEnv(..., reward=...)`, one scalar per env, shared by all heads)

| term | default | when |
|---|---|---|
| `blind` | +1.0 | a blind is beaten (score, or a Mr. Bones save) |
| `ante` | +1.0 | the Boss blind is beaten (so a beaten boss pays `blind + ante`) |
| `win` | +10.0 | the run is won (ante-8 boss beaten, non-endless) |
| `loss` | 0.0 | the run is lost (set negative to penalise dying) |
| `shaping` | 0.0 | `shaping * (progress_after - progress_before)` of `min(round_score / target, 1)` within a round (telescopes to <= `shaping` per round; 0 = off) |
| `invalid` | -0.05 | per invalid action (the reward replaces the others: nothing else happens) |
| `reorder` | 0.0 | added to every *valid* `MoveJoker` action (set negative, e.g. -0.05, to break swap-back-and-forth loops: reordering changes nothing but the joker order and costs neither money nor a hand, so it is the one action a policy can repeat forever; added in Phase 10B, backward compatible, `INTERFACE_VERSION` unchanged) |
| `margin` | 0.0 | when a blind is beaten, `margin * ln(min(score / target, margin_cap))` (>= 0; a Mr. Bones save below target adds 0; 0 = off; Phase 10F) |
| `margin_cap` | 4.0 | cap of `score / target` inside the margin term |
| `zero_play` | 0.0 | added to every played hand that a boss rejected (score 0): The Psychic (not 5 cards), The Eye (repeated hand type), The Mouth (not the first hand type), i.e. every use of `PlayCtx::invalidate` (`HandReport.invalid`); set negative for a cost (Phase 10F) |
| `endless_max_ante` | 12 | endless runs only (see below) |

**Endless runs** (`VecEnv(..., endless=True, endless_max_ante=12)`): the ante-8 boss still pays
`blind + ante + win` (`win` exactly once per run), but the run continues; `blind`, `ante`, `margin`
and `zero_play` keep paying. The episode ends with `done` (never truncation, terminal value 0) when
the run is lost or the boss of `endless_max_ante` is beaten (must be >= 8). The done code is 2 (won)
whenever the run passed ante 8, also if it then dies; 1 only for runs lost before ante 8. In
`drain_finished`, `ante` is the ante of the final state (`endless_max_ante + 1` after a cap exit).
`endless_max_ante` is also accepted as a key of the `reward` dict (the Python constructor merges it
there). All Phase 10F options default to the previous behaviour bit for bit; observation and
`INTERFACE_VERSION` are unchanged.

Rewards are added when they happen in the same transition. Truncation (`max_episode_steps`) gives no
terminal reward and is *not* a loss.

## 8. Done and auto-reset

`done[i]`: 0 running, 1 lost, 2 won, 3 truncated (`max_episode_steps`, default 10000, counts
invalid actions too). On `done != 0` the env has already auto-reset: the observation and masks
belong to the **new** episode's first state, `reward[i]` to the finishing transition. The new
episode's seed is a pure function of `(base_seed, env, episode)`. `drain_finished()` yields
`(env, ante, done_code, steps, return, blinds_beaten, invalid_actions)` per finished episode.
For value bootstrapping across truncation the last observation of the old episode is not kept;
the PPO trainer therefore bootstraps a truncated step with the value of the observation the last
action was taken in (a truncated policy is almost always in a loop, so the two are nearly the same
state; `docs/design/training.md` section 7) and keeps `max_episode_steps` high.

## 9. Information-leak rules

Never in the observation: card/joker `uid`, draw order, any RNG state, the real
`Game::preview_play` (it reads the RNG and face-down cards), the contents of a pack before it is
opened, the future of the market. Face-down cards (The House, Wheel, Fish, Mark) are read through
`Card::visible_card()`; jokers through `GameState::visible_joker()` (Amber Acorn: count visible,
identity hidden). Residual leaks that exist in the real game as well and are not fixed: the
hand position of a face-down card tells its rank bracket (the hand is rank-sorted), and the
joker order under Amber Acorn is real. Tests: observation and masks are bit-identical for two
states that differ only in hidden information (face-down identities, draw order, RNG).

The `teacher_actions` heuristic uses `fair_info = true` (it decides on `Game::fair_view`).

## 10. Helpers

* `random_actions(out)`: structured uniformly-random mask-conforming actions: kind uniformly among
  `kind_mask`, `arg0`/`arg1` uniformly among the legal ones, then cards by the sequential rule
  choosing uniformly among `{STOP if can_stop} ∪ next`. Deterministic per env, independent of
  the thread count, reseeded by `reset`.
* `teacher_actions(out)`: the action the fair-information heuristic team agent would play in each
  env's current state (structured form), parallel and GIL-free. The agent is created lazily per
  env and dropped on auto-reset (its strategy state is per episode).
* `Game` (debug, single run, no auto-reset, same interface): `observation()` / `masks()` dicts
  (per-env shapes), `step_parts(kind, arg0, arg1, mask) -> (reward, done, invalid)`,
  `legal_parts()` (every legal action as parts), `teacher_parts()`, `random_parts()`,
  `select_next_candidates` / `select_can_finish` (the core's rule for a `(kind, arg0)`), and three
  test hooks that mutate the state: `debug_face_down(mask)`, `debug_scramble_hidden(seed)`
  (changes only what the player cannot see) and `debug_force_card(pos)` (Cerulean Bell).
  `Game(seed)` is episode 0 of env 0 of `VecEnv(.., seed)`.
* Python: `balatro_rl.BalatroVecEnv` (buffers owned; `obs`, `masks`, `reward`, `done`, `invalid`;
  `step` returns `(obs, masks, reward, done, invalid)`), `balatro_rl.LAYOUT`,
  `balatro_rl.select` (sequential rule, `sample_actions` reference sampler, `conforms` mask check,
  `unpack_bits`): the template for the model's autoregressive action head.

## 11. Notes for the model (Phase 9)

* Encode each token type with its own `Linear(f) + sum of Embedding(i)` (vocabulary sizes from
  `LAYOUT["vocab"]`; the shared item vocabulary lets jokers, consumables and offers share one
  table), add a token-type embedding, mask padding with `*_valid`, and slice the hand to the batch's
  largest valid count.
* Heads: `kind` (categorical over 17, logits masked by `kind_mask`, conditioned on the global token);
  `arg0` (pointer: score the tokens `ARG0_POINTER[kind]` names, masked by `arg0_mask[kind]`);
  `arg1` (pointer over joker tokens, masked by `arg1_mask` minus `arg0`); cards (pointer over hand
  tokens, masked by `next_candidates`, plus a STOP logit allowed iff `can_finish`, at most 5 steps).
  Log-probabilities of the sampled chain add up; the action written back is
  `[kind, arg0, arg1, partial]`.
* The action is built from the masks of the *same* step's observation; the env never needs more
  than one round trip per decision. The masks are exact: a conforming chain is never invalid.
* Phase heads can share the encoder: the phase is `global_i[:, 0]`, and `kind_mask` already
  restricts each phase to its own kinds.
