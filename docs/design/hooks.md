# Joker hook contract (Phase 1C) — guide for joker authors

This is the contract every joker (and later every tarot/boss/voucher effect that touches scoring)
is written against. **Phase 3 agents: follow this document exactly.** If a joker cannot be
expressed cleanly, do not hack around it — describe the gap in your report and the orchestrator
extends the contract.

Code: `crates/balatro-core/src/{joker.rs, hooks.rs, scoring.rs}`, registry
`src/content/jokers/mod.rs`, working examples `src/content/jokers/prototypes.rs`, tests
`tests/joker_prototypes.rs`. Mechanics: `docs/mechanics/scoring.md`, decisions `notes.md` §4.1.

---

## 1. Adding a joker in 6 steps

1. Find the id in `src/content/jokers/ids.rs` (`JokerId::YourJoker`). Never edit `ids.rs`
   (orchestrator-owned). Missing id → ask in your report.
2. Open **your** file `src/content/jokers/waveN.rs` (only that file). It has one
   `pub static DEFS: &[JokerDef] = &[ ... ];` table. It must be listed once in
   `content/jokers/mod.rs::SOURCES` (orchestrator adds the line).
3. Add a `JokerDef` entry (template in §2). Set `rarity`, `base_cost`, compatibility flags and
   initial state from the wiki page / `docs/mechanics/jokers.csv`.
4. Write the hook functions below the table, one block per joker, starting with a comment line
   `// --- Name: https://balatrowiki.org/w/Name ---`. Put every number in a named `const` at the
   top of the block (no magic numbers inside logic).
5. Pick hooks with the decision table (§3). Read context with the API in §5; write only through
   `ctx` and your own `j` (§6).
6. Test in `tests/jokers_waveN.rs` (template in §11): trigger, non-trigger, scaling/state, and a
   Blueprint case if the joker is copyable. Run `cargo test -p balatro-core`,
   `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --all`.

## 2. Definition template

```rust
use crate::card::Card;
use crate::content::jokers::JokerId;
use crate::enums::{HandType, Rank, Rarity, Suit};
use crate::hooks::{DiscardInfo, EventCtx, Hooks, Modifiers, RoundEndInfo};
use crate::joker::{JokerDef, JokerInstance};
use crate::scoring::ScoreCtx;

pub static DEFS: &[JokerDef] = &[
    JokerDef {
        // only the fields that differ from JokerDef::new defaults:
        init_value: 1.0,                 // initial j.value (xmult jokers start at 1.0)
        blueprint_compatible: true,      // default true; false for the 29 listed in §8
        eternal_compatible: true,        // false for self-destroying jokers
        hooks: Hooks {
            on_card_scored: Some(my_scored),
            joker_main: Some(my_main),
            ..Hooks::NONE
        },
        ..JokerDef::new(JokerId::MyJoker, "My Joker", Rarity::Uncommon, 6)
    },
];

// --- My Joker: https://balatrowiki.org/w/My_Joker ---------------------------------------
const MY_GAIN: f64 = 0.25;

fn my_scored(j: &mut JokerInstance, ctx: &mut ScoreCtx<'_>, card: &Card) { /* ... */ }
fn my_main(j: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) { /* ... */ }
```

`JokerDef` fields: `id`, `name`, `rarity`, `base_cost`, `blueprint_compatible`,
`eternal_compatible`, `perishable_compatible`, `init_value: f64`, `init_counter: u32`,
`copies: Option<CopyFn>` (only Blueprint/Brainstorm), `hooks: Hooks`.

## 3. Which hook? (decision table)

| The joker's text says… | Hook | Examples |
|---|---|---|
| flat +chips / +mult / xmult, or conditional on hand type, `contains`, money, deck, counts, held set | `joker_main` | Joker, Jolly, The Duo, Half Joker, Banner, Bull, Blackboard, Steel Joker, Abstract, Stencil, Card Sharp, Acrobat, Supernova |
| "played cards with X give …" (per scored card, repeats on retrigger) | `on_card_scored` | Greedy, Fibonacci, Scholar, Walkie Talkie, Photograph, Ancient, Idol, Triboulet, Bloodstone, Business Card, Rough Gem, Golden Ticket, 8 Ball |
| "retrigger played cards …" | `card_retriggers` (return extra count) | Hack, Sock and Buskin, Dusk, Seltzer, Hanging Chad (2) |
| "each card held in hand …" | `on_held` | Baron, Shoot the Moon, Raised Fist, Reserved Parking |
| "retrigger held-in-hand abilities" | `held_retriggers` | Mime |
| "each [rarity] joker gives …" | `on_other_joker` | Baseball Card |
| "when hand is played" before scoring: level up, change played cards, scaling that the current hand already benefits from | `before_scoring` | Space Joker, Midas Mask, Vampire, Ride the Bus, Runner, Green Joker (+1), Square, Spare Trousers, Obelisk, DNA, To Do List |
| per-hand decay / bookkeeping after the hand | `after_scoring` | Ice Cream, Seltzer countdown, Sixth Sense |
| "if played hand triggers the boss" also on rejected hands | `on_invalid_hand` (+ `joker_main`) | Matador |
| changes rules or round stats passively (no trigger) | `modifiers` | Four Fingers, Shortcut, Smeared, Pareidolia, Splash, Oops! All 6s, Juggler, Drunkard, Troubadour, Merry Andy, Turtle Bean, Stuntman (hand size part), Credit Card, Chaos the Clown, Astronomer, Showman, To the Moon |
| when bought/created: pick a random target, record an offset | `on_acquire` | Ancient, Idol, Castle, Mail-In Rebate, To Do List, Loyalty Card, Chicot |
| "when Blind is selected" | `on_blind_select` (`ctx.run().blind`) | Riff-raff, Cartomancer, Marble, Burglar, Madness, Ceremonial Dagger, Chicot |
| "when round begins" (first hand drawn) | `on_first_hand_drawn` | Certificate |
| "when a blind is skipped" | `on_blind_skip` | (per-joker skip counters) |
| discards | `on_discard` | Burnt, Faceless, Trading Card, Castle, Mail-In Rebate, Yorick, Hit the Road, Green Joker (−1), Ramen |
| playing cards destroyed | `on_cards_destroyed` | Glass Joker, Canio |
| playing cards added to deck | `on_cards_added` | Hologram |
| "at end of round" (money, decay, re-roll targets, resets) | `on_round_end` | Golden, Cloud 9, Rocket, Satellite, Delayed Gratification, Egg, Gift Card, Popcorn, Turtle Bean, Gros Michel, Cavendish, Invisible, Hit the Road reset, Campfire reset, Ancient/Idol/Castle/Mail-In/To Do List targets, Mr. Bones |
| shop | `on_shop_enter`, `on_shop_exit`, `on_reroll` | Perkeo (exit), Flash Card (reroll) |
| selling | `on_sell_self`, `on_sell_other` | Luchador, Diet Cola, Invisible (self); Campfire (other) |
| packs | `on_pack_open`, `on_pack_skip` | Hallucination (open), Red Card (skip) |
| consumable used | `on_consumable_used` | Constellation |

A joker may use several hooks (Ride the Bus: `before_scoring` scales, `joker_main` pays).
Scaling jokers whose gain should count for the current hand scale in `before_scoring` or
`on_card_scored` (Lucky Cat, Wee Joker) and pay in `joker_main`.

## 4. The pipeline (when each scoring hook runs)

Exact order of one `Scorer::score_hand` call (`scoring.md`, `notes.md` §4.1):

0. **Boss validation.** `run.boss.hand_invalid` → score 0: only `on_invalid_hand`, then
   `after_scoring` run. Nothing else (no card effects, no Glass rolls).
2. **`before_scoring`**, jokers left to right. `ctx.level_up_hand(ht, n)` here raises the current
   hand's level for this hand (Space Joker).
1+3. **Base** = hand type at its current level; The Flint (`run.boss.halve_base`) halves both,
   rounding up.
4. **Scoring cards**, left to right. Debuffed cards are skipped entirely (no triggers, no joker
   hooks). For each card: triggers = 1 + Red Seal + Σ `card_retriggers` (jokers left to right).
   Each trigger: a) rank chips + `perma_bonus`, b) enhancement (Bonus +30, Stone +50, Mult +4,
   Glass ×2, Lucky: two fresh rolls), c) Gold Seal +$3, d) edition (Foil +50, Holo +10,
   Poly ×1.5), e) **`on_card_scored`** jokers left to right.
5. **Held cards**, left to right, debuffed skipped. Triggers = 1 + Red Seal + Σ `held_retriggers`.
   Each trigger: Steel ×1.5, then **`on_held`** jokers left to right.
6. **Jokers**, left to right. For joker *J*: a) J's Foil/Holo, b) **`joker_main`** (a copier runs
   its target's at its own slot), c) **`on_other_joker`** of every other joker, given *J*'s slot
   (runs even if *J* is debuffed: Baseball counts debuffed jokers), d) J's Polychrome. A debuffed
   joker gets no a/b/d.
7. Observatory (×1.5 per held Planet of this hand type), then Plasma Deck balancing.
8. `score = round_score(chips, mult)` = `floor(chips × mult)`. **Final**: later chips/mult
   changes are ignored.
9. Glass break rolls (once per non-debuffed Glass card in the scoring set, 1 in 4), then
   **`after_scoring`** jokers.

End of round, `Scorer::round_end_held` handles held Gold cards ($3) and Blue Seals (Planet),
each repeated by Red Seal and `held_retriggers` (Mime, and its copies).

## 5. Reading context

### 5.1 `ScoreCtx` (scoring hooks)

| Method | Meaning |
|---|---|
| `run()` | `&RunView` (§5.3) |
| `eval()`, `hand_type()`, `contains(ht)` | hand evaluation; `contains` = "hand contains a Pair" semantics |
| `played()` | all played cards (scoring or not), left to right |
| `scoring_indices()`, `scoring_count()` | indices of scoring played cards (debuffed included) |
| `held()` | cards held in hand |
| `jokers()`, `joker_def(slot)` | all jokers (read-only) and their defs (rarity…) |
| `slot()`, `exec_slot()`, `copied()` | whose ability runs / where it executes / is this a Blueprint copy |
| `card_index()` | index of the current card in `played()` (scored hooks) or `held()` (held hooks) |
| `scoring_pos()` | position among scoring cards (0 = first scoring card; Hanging Chad) |
| `trigger()` | 0 = first trigger, 1.. = retriggers |
| `lucky_triggered()` | current trigger of a Lucky card hit a roll (Lucky Cat) |
| `chips()`, `mult()` | running values |
| `money()` | run money + money gained earlier in this hand (Bull, Bootstraps, Vagabond) |
| `deck()` | full-deck `DeckCounts`, updated for enhancement changes made this hand |
| `hand_level()` | level used for the base |
| `step()` | which pipeline step is running |
| `hand_invalid()` | the boss rejected this hand |
| `is_face(&card)` | face under joker rules: J/Q/K, everything with Pareidolia (**Stone included**), **never a debuffed card**. Use this, not `Card::is_face`. |
| `has_suit(&card, suit)` | Wild / Smeared / Stone aware |

### 5.2 `EventCtx` (event hooks)

`run()`, `deck()` (full `Deck`: `deck().iter_all()` to pick Idol/Mail-In/Castle targets),
`jokers()`, `joker_def(slot)`, `slot()`, `exec_slot()`, `copied()`, `money()`, `add_money()`,
`push_effect()`, `effects()`, `level_up_hand()`, `destroy_self()`, `destroy_joker(slot)`,
`is_doomed(slot)`, `chance()`, `rng_range()`, `rng()`, `is_face()`, `has_suit()`.
Payloads arrive as hook arguments: `DiscardInfo { cards, hand_type, forced }` (`forced` = discarded
by The Hook, not a discard action: "first discard" effects such as Burnt Joker and Trading Card
must ignore it),
`RoundEndInfo { won, saved, round_score, target }` (+ `boss_defeated(run)` = boss blind && (won || saved); `saved` = Mr. Bones' `PreventDeath`, probed by the game layer before the real dispatch), `SoldItem`, `PackKind`,
`ConsumableView` (= `consumable::ConsumableInstance`: `id: ConsumableId`, `negative`,
`sell_value`; `kind()`, `planet_hand()`), `&[Card]` for destroyed/added cards (snapshots before
removal).

### 5.3 `RunView` fields (both contexts)

All counts are taken **after the triggering action is paid for**.

| Field | Meaning |
|---|---|
| `money` | money before this hand/event (use `ctx.money()`) |
| `ante`, `round`, `blind: BlindKind`, `boss: BossView` | progress; `boss.{id: Option<BossId>, active, halve_base, triggered, hand_invalid}` (`id` is the ante's boss in every blind; check `blind == Boss` / `active`) |
| `in_round` | `phase == Play`. In the shop / at blind select `blind` already names the next blind; Luchador `on_sell_self` and Chicot `on_acquire` disable the boss only when `in_round` |
| `hands_left` | hands left **excluding** the hand being scored (`is_final_hand()` = 0: Dusk, Acrobat) |
| `discards_left` | discards left (Banner, Mystic Summit, Delayed Gratification) |
| `hands_played_round`, `hands_played_run` | **including** the current hand (DNA/Sixth Sense: `hands_played_round == 1`) |
| `discards_used_round` | **including** the current discard (Burnt/Trading Card: `== 1`; Delayed Grat.: `== 0` at round end) |
| `hand_levels[ht]`, `hand_plays_run[ht]`, `hand_plays_round[ht]` | play counts **include** the current hand (Supernova, Obelisk, Card Sharp `> 1`) |
| `joker_slots` | capacity incl. Negative bonuses (Joker Stencil) |
| `consumable_slots`, `consumables: &[ConsumableView]` | held consumables (Observatory, Perkeo) |
| `deck: DeckCounts` | `total`, `starting_size` (Erosion), `draw_pile` (Blue Joker), `by_enhancement` (Steel/Stone Joker), `by_rank` (Cloud 9), `by_suit`, `by_seal`, `enhanced()` (Driver's License) |
| `rules: RuleFlags`, `prob_mult` | active passives |
| `observatory`, `plasma_deck` | late-step switches |
| `stats: RunStats` | `blinds_skipped` (Throwback), `tarots_used` (Fortune Teller), `planets_used`, `unique_planets_used` (Satellite), `bosses_defeated` |

## 6. Writing: immediate vs deferred

| Change | How | When it takes effect |
|---|---|---|
| chips / mult | `ctx.add_chips`, `ctx.add_mult`, `ctx.x_mult` | immediately |
| money | `ctx.add_money(n)` | immediately visible via `ctx.money()`; game applies the delta after |
| own state (value/counter/aux/sell_value) | mutate `j` | written back after the hook (discarded for copies) |
| played card: enhancement | `ctx.set_enhancement(i, e)` (Midas, Vampire) | immediately; keeps `deck()` counts right |
| played card: perma chips | `ctx.add_perma_bonus(i, n)` (Hiker) | immediately; later triggers see it |
| other played-card fields | `ctx.played_mut(i)` | immediately |
| hand level | `ctx.level_up_hand(ht, n)` | queued `LevelUpHand`; in `before_scoring` also raises this hand's base |
| destroy self | `ctx.destroy_self()` | queued `DestroyJoker` (no-op in copies) |
| destroy another joker | `EventCtx::destroy_joker(slot)` | queued; refuses eternal; later hooks of the same event skip it |
| everything else | `ctx.push_effect(SideEffect::…)` | queued; applied by the game layer after the call, in order |

`SideEffect` variants: `LevelUpHand`, `CreateConsumable{kind, negative}`,
`AddConsumable{id, negative}`, `CreatePlanetFor{hand}`,
`CopyConsumable{index, negative}`, `CreateJoker{rarity}`, `CopyJoker{uid, strip_negative}`,
`DestroyJoker{uid}`, `DestroyCard{uid}`, `AddCard{card, to: Pile}` (uid reassigned),
`AddSellValue{target, amount}`, `AddHands(n)`, `AddDiscards(n)`, `ClearDiscards`,
`CreateTag{id}`, `DisableBoss`, `PreventDeath`, `SetRunFlag(RunFlag)`.
Creation effects that need a slot are dropped by the game layer when there is no room at
application time; you do not check room yourself. Consumable effects are implemented since
Phase 4.0 (pool rules: `docs/design/consumables_bosses.md` §2.5); only `CreateJoker` is still
dropped (Phase 5).

Never mutate other jokers, held cards, the deck or the run directly. Never mutate state of another
joker to "share" information.

## 7. State conventions

`JokerInstance` generic fields:

| Field | Use for | Examples (initial value in the def) |
|---|---|---|
| `value: f64` | the amount the joker gives now: +mult, +chips, or the xmult factor; or a decaying amount | Ride the Bus (+mult, 0), Runner/Square/Castle/Wee (+chips, 0), Hologram/Vampire/Constellation/Lucky Cat/Glass/Campfire/Obelisk/Madness/Hit the Road/Canio/Yorick (xmult, **`init_value: 1.0`**), Ice Cream (100), Popcorn (20), Ramen (2.0), Turtle Bean (5), Rocket (payout, 1), Ceremonial Dagger (+mult, 0), Red Card/Flash Card/Green/Spare Trousers (+mult, 0) |
| `counter: u32` | integer counts and countdowns | Seltzer hands left (`init_counter: 10`), Yorick discards to next step (23), Invisible rounds held, Loyalty Card offset (`hands_played_run` at acquire) |
| `aux` lanes | chosen targets | `aux_suit`/`set_aux_suit` (Ancient, Castle, Idol), `aux_rank` (Mail-In, Idol), `aux_hand` (To Do List), `aux_byte` (free) |
| `sell_value: i32` | own sell value | Egg (+3 per round) |

Rules:
- Document the meaning in the joker's comment block (`// value = current +mult`).
- Xmult jokers store the factor (start 1.0) and apply `ctx.x_mult(j.value)`; skip when it is 1.0.
- "Resets" set the field back explicitly (`j.value = 1.0`); `j.reset_state()` restores all `init_*`.
- Random targets: pick in `on_acquire` AND re-pick in `on_round_end` (end-of-round hooks are
  never copied, so no `copied()` check is needed there).
- Decay to zero: decrement, and at ≤ 0 call `ctx.destroy_self()` (Ice Cream, Popcorn, Turtle
  Bean, Seltzer, Ramen at ≤ 1).

## 8. Copiers (Blueprint, Brainstorm)

- A copier has `copies: Some(copy_right | copy_leftmost)` and no hooks. At its own slot it runs the
  **target's** hooks, with **its own** edition. Chains resolve (Blueprint → Blueprint → X); cycles,
  missing targets, debuffed links and targets with `blueprint_compatible: false` give nothing.
- Copied hooks: every scoring hook and every event except `modifiers`, `on_acquire` and
  `on_round_end` (balatrowiki.org/w/Blueprint: no passive or end-of-round effects).
- In a copy, `j` is a **scratch copy of the target's state**: reads see the target's current
  value; writes are discarded. So scaling never double-counts. You still must check
  `ctx.copied()` before effects that are not joker state and must not be duplicated:
  card mutations (Vampire stripping, Midas), destroying other jokers (Ceremonial Dagger, Madness).
  Effects that ARE duplicated by copies in the real game (money, chips/mult, level-ups, creating
  cards/consumables/jokers, Hiker's perma bonus, Burnt's level-up, Burglar, DNA) need no check.
- "Self" means `ctx.slot()` (the target's slot), e.g. Swashbuckler excludes `ctx.slot()`.
- Set `blueprint_compatible: false` for: Four Fingers, Credit Card, Chaos the Clown, Delayed
  Gratification, Pareidolia, Egg, Splash, Sixth Sense, Shortcut, Cloud 9, Rocket, Midas Mask,
  Gift Card, Turtle Bean, To the Moon, Juggler, Drunkard, Golden Joker, Trading Card, Mr. Bones,
  Troubadour, Smeared Joker, Showman, Merry Andy, Oops! All 6s, Invisible Joker, Satellite,
  Astronomer, Chicot.

## 9. RNG rules

- Only through `ctx.chance(num, den)` (applies `run.prob_mult` = Oops! All 6s) and
  `ctx.rng_range(lo, hi)` (uniform, Misprint). Both use `RngKey::Jokers`. Write odds as
  `const X_ODDS: (u32, u32) = (1, 4);`.
- Never `rand::random`, `thread_rng`, time, `HashMap` iteration order.
- Roll exactly when the game rolls (per scored card, per trigger, per event) — the number of rolls
  is part of determinism. Do not roll in retrigger hooks (they are read-only and get `&ScoreCtx`).
- Card-level randomness (Lucky, Glass) lives in the pipeline on `RngKey::Cards`; jokers never use it.

## 10. Debuffs, faces, suits

- A debuffed **joker** never runs any hook and has no edition effect (Baseball still counts it).
- A debuffed **card** is never passed to `on_card_scored` / `on_held` / retrigger hooks. It still
  appears in `played()`, `held()`, `scoring_indices()` and `DiscardInfo::cards`; check
  `card.debuffed` yourself when iterating (per the joker's wiki page).
- `ctx.is_face(card)` is false for debuffed cards (Ride the Bus, Photograph, Faceless).
- `card.rank` of a Stone card is meaningless: check `card.is_stone()` for rank effects
  (Hack, Fibonacci, Baron, Raised Fist). Stone has no suit (`has_suit` handles it).

## 11. Testing template

```rust
use balatro_core::card::Card;
use balatro_core::content::jokers::JokerId;
use balatro_core::enums::{HandType, Rank::*, Suit::*};
use balatro_core::hand_eval::HandEval;
use balatro_core::hooks::RunView;
use balatro_core::joker::JokerInstance;
use balatro_core::rng::GameRng;
use balatro_core::scoring::{ScoreInput, Scorer};

fn pair7() -> (Vec<Card>, HandEval) {
    let p = vec![Card::new(1, Seven, Spades), Card::new(2, Seven, Hearts)];
    (p, HandEval { hand_type: HandType::Pair, scoring_mask: 0b11, contains: 1 << HandType::Pair.index() })
}

#[test]
fn my_joker_triggers() {
    let (mut played, eval) = pair7();
    let mut jokers = vec![JokerInstance::new(100, JokerId::MyJoker)];
    let run = RunView::default();            // set fields you depend on
    let mut rng = GameRng::new(1);
    let mut scorer = Scorer::with_trace(true);
    let r = scorer.score_hand(ScoreInput {
        played: &mut played, held: &[], eval, jokers: &mut jokers, run: &run, rng: &mut rng,
    });
    // Pair 10 x 2, 7 + 7 = 24 chips; +N mult -> 24 x (2 + N)
    assert_eq!(r.score, 24.0 * (2.0 + N));
}
```

- Build `HandEval` by hand; do not call `hand_eval::evaluate` in joker tests.
- Force probabilities with `RunView { prob_mult: 1000, .. }` (always) / `prob_mult: 0` (never).
- Events: `fire_round_end(EventInput { jokers, run, deck: &Deck::standard(), rng, effects: &mut fx }, &info)`.
- Check side effects with `scorer.effects()`; check order with `scorer.trace()`
  (`TraceSource::Joker { slot, src }`, `TraceOp::{AddChips, AddMult, XMult, Money, ...}`).
- Write the arithmetic in a comment for every expected number.

## 12. Common pitfalls

1. Applying xmult as `add_mult` (or the reverse). Xmult with factor 1.0 is a no-op; skip it.
2. Scaling in `joker_main` when the current hand should already benefit → scale in `before_scoring`.
3. Reading your own state from `ctx.jokers()[ctx.slot()]` — that is stale during your hook; use `j`.
4. Forgetting that play counts include the current hand (`>= 1` always true for the played hand).
5. Using `Card::is_face` instead of `ctx.is_face` (debuffed/Stone rules).
6. Using `card.rank` of Stone cards.
7. Checking room for created cards (the game layer does it).
8. Randomness outside `ctx.chance` / `ctx.rng_range`; rolling in retrigger hooks.
9. Mutating cards in a copy (`ctx.copied()` check) for effects the real game does not duplicate.
10. Expecting chips/mult changes in `after_scoring` to count (the score is already final).
11. Returning the *total* from a retrigger hook — return only the EXTRA triggers this joker grants.
12. Adding passive effects in `on_acquire` (they would never be removed) — use `modifiers`.

## 13. Annotated examples (from `prototypes.rs`)

**Greedy Joker — per scored card.**
```rust
fn greedy_scored(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>, card: &Card) {
    if ctx.has_suit(card, Suit::Diamonds) {   // Wild and Smeared aware
        ctx.add_mult(GREEDY_MULT);            // runs again on every retrigger
    }
}
```

**Ride the Bus — scale before, pay in main.**
```rust
fn ride_the_bus_before(j: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    // value = current +mult. Debuffed faces do not count (ctx.is_face).
    let face_scored = ctx.scoring_indices().any(|i| ctx.is_face(&ctx.played()[i]));
    if face_scored { j.value = 0.0 } else { j.value += RIDE_THE_BUS_GAIN }
}
fn ride_the_bus_main(j: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if j.value > 0.0 { ctx.add_mult(j.value) }   // a Blueprint copy reads the scaled value
}
```

**Hack — retrigger count (read-only, extra triggers only).**
```rust
fn hack_retriggers(_: &JokerInstance, _: &ScoreCtx<'_>, card: &Card) -> u32 {
    let low = matches!(card.rank, Rank::Two | Rank::Three | Rank::Four | Rank::Five);
    u32::from(low && !card.is_stone())
}
```

**Ice Cream — decay after the hand, self-destruct.**
```rust
fn ice_cream_after(j: &mut JokerInstance, ctx: &mut ScoreCtx<'_>) {
    if ctx.copied() { return; }            // optional: copy writes are discarded anyway
    j.value -= ICE_CREAM_DECAY;            // also runs after a boss-invalid hand
    if j.value <= 0.0 { j.value = 0.0; ctx.destroy_self(); }   // queued DestroyJoker{uid}
}
```

**Baseball Card — joker-on-joker.**
```rust
fn baseball_other(_: &mut JokerInstance, ctx: &mut ScoreCtx<'_>, other: usize) {
    // called at every other joker's turn, after its effect, before its Polychrome
    if ctx.joker_def(other).is_some_and(|d| d.rarity == Rarity::Uncommon) {
        ctx.x_mult(BASEBALL_XMULT);
    }
}
```

**Burnt Joker — event with payload.**
```rust
fn burnt_discard(_: &mut JokerInstance, ctx: &mut EventCtx<'_>, d: &DiscardInfo<'_>) {
    if ctx.run().discards_used_round == 1 && !d.cards.is_empty() {   // first discard
        ctx.level_up_hand(d.hand_type, 1);                            // queued
    }
}
```

**Four Fingers — passive rule.**
```rust
fn four_fingers_modifiers(_: &JokerInstance, m: &mut Modifiers) { m.rules.four_fingers = true; }
```

## 14. All 150 jokers → hooks (survey)

`main` = `joker_main`, `scored` = `on_card_scored`, `held` = `on_held`, `retrig` =
`card_retriggers`, `before`/`after` = `before_scoring`/`after_scoring`, `mods` = `modifiers`,
`acq` = `on_acquire`, `blind` = `on_blind_select`, `end` = `on_round_end`, `disc` = `on_discard`.
"(bp✗)" = `blueprint_compatible: false`.

| Joker | Hooks / state |
|---|---|
| Joker, Jolly, Zany, Mad, Crazy, Droll, Sly, Wily, Clever, Devious, Crafty, The Duo/Trio/Family/Order/Tribe | main (`contains`) |
| Half Joker | main (`played().len() <= 3`) |
| Joker Stencil | main (`run.joker_slots - jokers().len()` + Stencils) |
| Four Fingers, Shortcut, Smeared, Pareidolia, Splash (bp✗) | mods `rules` |
| Mime | `held_retriggers` 1 |
| Credit Card (bp✗) | mods `debt_limit += 20` |
| Ceremonial Dagger | blind (`!copied`: `destroy_joker(slot+1)`, value += 2×sell) + main |
| Banner, Mystic Summit | main (`discards_left`) |
| Marble Joker | blind → `AddCard{Stone, Pile::Draw}` |
| Loyalty Card | acq (counter = `hands_played_run`) + main |
| 8 Ball | scored (rank 8, `chance(1,4)`) → `CreateConsumable(Tarot)` |
| Misprint | main `rng_range(0, 24)` |
| Dusk | retrig (`is_final_hand`) |
| Raised Fist | held (lowest non-Stone, rightmost tie) |
| Chaos the Clown (bp✗) | mods `free_rerolls` |
| Fibonacci, Even Steven, Odd Todd, Scholar, Walkie Talkie, Scary Face, Smiley Face, Greedy/Lusty/Wrathful/Gluttonous, Arrowhead, Onyx Agate, Triboulet | scored |
| Steel Joker, Stone Joker, Driver's License, Erosion, Blue Joker | main (`ctx.deck()`) |
| Abstract Joker | main (`jokers().len()`) |
| Delayed Gratification (bp✗) | end (`discards_used_round == 0`) |
| Hack, Sock and Buskin | retrig |
| Gros Michel | main + end (`chance(1,6)` → `destroy_self`, `SetRunFlag(GrosMichelExtinct)`) |
| Business Card, Reserved Parking (held), Rough Gem, Golden Ticket, Bloodstone | scored/held + money/`chance` |
| Supernova | main (`hand_plays_run`) |
| Ride the Bus, Runner, Square, Spare Trousers, Green Joker | before (scale) + main (Green: disc −1) |
| Space Joker | before (`chance`, `level_up_hand`) |
| Egg (bp✗) | end (`j.sell_value += 3`) |
| Burglar | blind → `AddHands(3)`, `ClearDiscards` |
| Blackboard | main (`held()` all Spades/Clubs) |
| Ice Cream, Seltzer (retrig + after countdown) | after decay + `destroy_self` |
| DNA | before (`hands_played_round == 1`, 1 card) → `AddCard{to: Hand}` |
| Sixth Sense (bp✗) | after → `DestroyCard`, `CreateConsumable(Spectral)` |
| Constellation | `on_consumable_used` (Planet) + main |
| Hiker | scored → `add_perma_bonus` |
| Faceless Joker | disc |
| Superposition | main (`contains(Straight)` + Ace) → `CreateConsumable(Tarot)` |
| To Do List | acq/end pick `aux_hand`; before (money) |
| Cavendish | main + end (`chance(1,1000)`) |
| Card Sharp | main (`hand_plays_round[ht] > 1`) |
| Red Card | `on_pack_skip` + main |
| Madness | blind (non-boss, `!copied`, random non-eternal non-doomed other) + main |
| Séance | main (`hand_type == StraightFlush`) → Spectral |
| Riff-raff | blind → 2 × `CreateJoker{Common}` |
| Vampire | before (`!copied`: `set_enhancement(None)`, scale) + main |
| Hologram | `on_cards_added` + main |
| Vagabond | main (`money() <= 4`) → Tarot |
| Baron, Shoot the Moon | held |
| Cloud 9 (bp✗) | end (`deck.rank(Nine)` dollars) |
| Rocket (bp✗) | end (pay `value`; `boss_defeated` → +2) |
| Obelisk | before (most played via `hand_plays_run`) + main |
| Midas Mask (bp✗) | before (`set_enhancement(Gold)` on scoring faces) |
| Luchador | `on_sell_self` → `DisableBoss` |
| Photograph | scored (first scoring `is_face` card == `card_index()`) |
| Gift Card (bp✗) | end → `AddSellValue{AllJokers}`, `{AllConsumables}` |
| Turtle Bean (bp✗) | mods `hand_size += value` + end decay |
| Mail-In Rebate, Castle | acq/end pick target from `deck()`; disc |
| To the Moon (bp✗) | mods `interest_per_5` |
| Hallucination | `on_pack_open` (`chance(1,2)`) |
| Fortune Teller | main (`stats.tarots_used`) |
| Juggler, Drunkard, Troubadour, Merry Andy (bp✗) | mods |
| Golden Joker (bp✗) | end |
| Lucky Cat | scored (`lucky_triggered()`) + main |
| Baseball Card | `on_other_joker` |
| Bull, Bootstraps | main (`money()`) |
| Diet Cola | `on_sell_self` → `CreateTag` |
| Trading Card (bp✗) | disc (first discard, 1 card) → `DestroyCard` + $3 |
| Flash Card | `on_reroll` + main |
| Popcorn | main + end decay |
| Ancient Joker, The Idol | acq/end pick `aux_*`; scored |
| Ramen | disc decay (per card) + main |
| Campfire | `on_sell_other` + end reset on boss + main |
| Mr. Bones (bp✗) | end (`!won`, score ≥ 25%) → `PreventDeath` + `destroy_self` |
| Acrobat | main (`is_final_hand`) |
| Swashbuckler | main (sell values except `slot()`) |
| Certificate | `on_first_hand_drawn` → `AddCard{to: Hand}` |
| Throwback | main (`stats.blinds_skipped`) |
| Hanging Chad | retrig (`scoring_pos() == 0` → 2) |
| Glass Joker, Canio | `on_cards_destroyed` + main |
| Showman, Astronomer (bp✗) | mods |
| Flower Pot, Seeing Double | main (scoring cards' suits) |
| Blueprint / Brainstorm | `copies: copy_right / copy_leftmost` |
| Wee Joker | scored (rank 2, scale) + main |
| Oops! All 6s (bp✗) | mods `prob_mult *= 2` |
| Matador | main + `on_invalid_hand` (`boss.triggered`) |
| Hit the Road | disc (Jacks) + end reset + main |
| Stuntman | main (+250) + mods (`hand_size -= 2`) |
| Invisible Joker (bp✗) | end (counter) + `on_sell_self` → `CopyJoker` |
| Satellite (bp✗) | end (`stats.unique_planets_used`) |
| Cartomancer | blind → Tarot |
| Burnt Joker | disc (first) |
| Yorick | disc (counter 23 → value += 1) + main |
| Chicot (bp✗) | blind (boss) / acq → `DisableBoss` |
| Perkeo | `on_shop_exit` → `CopyConsumable{negative: true}` |

## 15. Game-layer obligations (Phase 2A/4/5)

- Before scoring: pay the hand (`hands_left -= 1`, increment `hands_played_*`, `hand_plays_*`),
  call `evaluate(played, &mods.rules)`, run the boss pre-play hook (`boss::pre_play`: validity,
  The Hook's forced discard + its `fire_discard(forced: true)`, The Tooth/Ox money, The Arm
  level, `boss.hand_invalid/triggered`), fill `RunView`, then `score_hand` with
  `&mut deck.played` (card mutations persist). `Game::preview_play` runs the same pre-play hook
  and forced-discard event on copies. Card debuffs come from the boss (`boss::refresh_debuffs`,
  after every Play-phase action); Crimson Heart's joker debuff is set between hands.
- Consumables (Phase 4.0): `UseConsumable` runs the card's effect, applies its side effects,
  updates `stats.run.{tarots,planets,unique_planets}_used` and then fires
  `fire_consumable_used`; `SellConsumable` fires `fire_sell_other(SoldItem::Consumable)` first.
- After scoring: add `money_delta`; apply `scorer.effects()` in order (eternal jokers are never
  destroyed; creation needs room); fire `fire_cards_destroyed` with pre-removal snapshots and
  `fire_cards_added` with the added cards; persist level-ups.
- Events: call the matching `fire_*` with the current `RunView` (after paying the action), apply
  `money_delta` and new effects. `fire_sell_self` + `fire_sell_other` before removing a sold joker.
  `fire_acquire` after adding a joker. End of round: `Scorer::round_end_held`, then
  `fire_round_end` (also on a lost round with `won: false`; `PreventDeath` saves the run).
- Keep `JokerInstance::uid` unique per run; side effects address jokers and cards by uid.
- Recompute `collect_modifiers(&jokers)` whenever jokers, their order, editions or debuffs change.
