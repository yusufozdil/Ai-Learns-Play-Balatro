# Consumable and boss contract: guide for content authors

This is the contract the 52 consumables (12 Planets, 22 Tarots, 18 Spectrals) and the 28 boss
blinds are written against. **Follow this document exactly.** If something cannot be expressed
cleanly, do not hack around it: extend the contract (and this document) instead.

Code: `src/consumable.rs` (instance, def, `UseCheck`, `UseCtx`, pools), `src/boss.rs` (def,
hooks, `BossState`, contexts, selection, game glue), ids `src/content/ids.rs`, registry
`src/content/mod.rs`. Working examples: `content/{planets,tarots,spectrals,bosses}.rs`.
Contract tests: `tests/consumables.rs`, `tests/boss_contract.rs`. Rules:
`docs/mechanics/{consumables,bosses,cards}.md`.
Joker hooks that react to consumables/bosses: `docs/design/hooks.md`.

---

## 1. Who writes what

| Package | Your file (only this) | Your tests (new file) | Ids |
|---|---|---|---|
| 4A | `src/content/tarots.rs` (+ `planets.rs`: tests only, all 12 already defined) | `tests/consumables_planets_tarots.rs` | 18 Tarots not yet defined (all except The Fool, The Magician, The Hermit, Death) |
| 4B | `src/content/spectrals.rs` | `tests/consumables_spectrals.rs` | 16 Spectrals not yet defined (all except Black Hole, Ankh) |
| 4C | `src/content/bosses.rs` | `tests/bosses.rs` | 20 bosses not yet defined (all except The Hook, The Wall, The Psychic, The Plant, The Needle, The Flint, The Mark, Verdant Leaf) |

Never edit `ids.rs`, `content/mod.rs`, `consumable.rs`, `boss.rs`, `hooks.rs`, `state.rs`,
`game.rs`, `actions.rs`, `data/*`. Shared numbers you need are already in
`data::consumables` (`EDITION_ROLL`, `ADDED_CARD_ENHANCEMENTS`), `data::prices`,
`blinds::{MULT_WALL, MULT_VIOLET_VESSEL, MULT_NEEDLE}`. A number that belongs to one card is a
named `const` in your file, above the card's functions.

Workflow: add the def to `DEFS`, write the functions below the table in a block starting with
`// --- Name: https://balatrowiki.org/w/Name ---`, test it, then run
`cargo test -p balatro-core`, `cargo clippy --workspace --all-targets -- -D warnings`,
`cargo fmt -p balatro-core -- <your files>` (do not `cargo fmt --all` in a shared worktree).

## 2. Consumables

### 2.1 Definition

```rust
use crate::consumable::{ConsumableDef, UseCheck, UseCtx};
use crate::content::ConsumableId as C;

pub static DEFS: &[ConsumableDef] = &[
    ConsumableDef {
        min_targets: 1,                 // selected hand cards (0 = untargeted)
        max_targets: 2,                 // <= MAX_CONSUMABLE_TARGETS (3); targeted => min >= 1
        needs_hand: false,              // true for untargeted cards that act on the hand
        can_use: Some(my_can_use),      // extra condition, pure
        ..ConsumableDef::new(C::TheEmpress, "The Empress", empress)
    },
];
```

Price and sell value are not in the def: they follow the kind (`ConsumableKind::base_cost`:
Tarot/Planet $3, Spectral $4; sell `max(1, floor(cost/2))`).

### 2.2 What the framework already checks (do not re-check)

`actions::validate_use` (pure, also drives `legal_actions`):
1. phase is `Play` or `Shop`; the slot exists; the id has a def;
2. `requires_hand()` (= `needs_hand || max_targets > 0`) → only in `Play` with a non-empty hand;
3. untargeted → `targets == 0`; targeted → `min..=max` selected cards, all inside the hand;
4. then your `can_use`.

Card selections (`Play`/`Discard`/`UseConsumable.targets`/`PickPackItem.targets`) are `u32`
bitmasks (Phase 8A): bit i = hand card i, positions 0..32 (`MAX_SELECTABLE_HAND`; the hand can
hold more than the hand size, e.g. 17 cards with Certificate). `legal_actions` keeps a precomputed
table for hands up to 16 cards and generates the masks lazily beyond; the env codec uses the
structured action space instead (`action_mask`: `structured_mask`, `card_select`), which also
expresses card-dependent `can_use` checks of single-target cards (Aura) per hand position.

Your `can_use` only states the card's own condition: free slot (The Fool, High Priestess,
Emperor, Judgement, Wraith, The Soul, Ankh), eligible joker (Wheel, Ectoplasm, Hex), hand of at
least 2 cards (Familiar, Grim, Incantation, Sigil, Ouija, Immolate). It must not roll or mutate.

`UseCheck` API: `state()` (whole `GameState`, read-only), `targets()` (`Targets`: `len()`,
`positions()`, `cards()`, `get(k)`), `hand()`, `jokers()`, `free_consumable_slots()` (**after
the used card left its slot**: a full area still fits the Fool's copy), `free_joker_slots()`,
`uses_of(id)`.

### 2.3 Effect API (`UseCtx`)

The used card is already out of its slot when the effect runs. Positions (`target_pos(k)`) refer
to the hand as it is during the effect (deferred effects run afterwards).

| Need | Call | Timing |
|---|---|---|
| selected cards | `target_count()`, `target_pos(k)` (0 = leftmost), `targets()` | — |
| hand / jokers / state | `hand()`, `jokers()`, `state()`, `used()`, `money()` | read |
| enhancement / suit / rank / seal / edition of a hand card | `set_enhancement(pos, e)`, `set_suit`, `set_rank`, `set_seal`, `set_edition` | immediate |
| arbitrary card change (uid kept) | `edit_card(pos, \|c\| ...)` | immediate |
| Death | `copy_card(dst, src)` (rank, suit, enhancement, edition, seal, perma_bonus) | immediate |
| destroy a hand card | `destroy_card(pos)` → `DestroyCard` (fires `on_cards_destroyed`) | deferred |
| add a playing card | `add_card(card, Pile::Hand)` → `AddCard` (uid reassigned, fires `on_cards_added`) | deferred |
| money | `add_money(n)`, `set_money(n)` | immediate |
| hand level | `level_up(hand, n)` → `LevelUpHand` (floor 1) | deferred |
| permanent hand size | `add_hand_size(delta)` (base size; effective clamp 1..=16) | immediate |
| joker edition | `set_joker_edition(slot, e)` (Negative = +1 joker slot) | immediate |
| destroy a joker | `destroy_joker(slot)` (refuses Eternal, returns false, queues nothing) | deferred |
| copy a joker | `copy_joker(slot, strip_negative)` → `CopyJoker` (needs room at application) | deferred |
| random joker | `create_joker(rarity)` → `CreateJoker` (needs a free joker slot at application, shop pools; Phase 5) | deferred |
| random consumable of a kind | `create_consumable(kind)` (pool rules §2.5) | deferred |
| specific consumable | `add_consumable(id)` (The Fool) | deferred |
| free slots now | `free_consumable_slots()`, `free_joker_slots()` | read |
| run counter | `uses_of(id)` (before this use; Ectoplasm) | read |
| randomness | `chance(num, den)` (Oops! All 6s applies), `rng_range(lo, hi)`, `weighted_index(&w)` — all `RngKey::Consumables` | — |
| anything else | `push_effect(SideEffect::…)` | deferred |

Creations that need a slot are dropped by the game layer when there is no room; for "create up
to N" cards create `min(N, free_consumable_slots())` yourself (High Priestess, Emperor).

### 2.4 What the game layer does around your effect (do not do it yourself)

1. `validate` → 2. remove the card from its slot → 3. your effect → 4. refresh modifiers, apply
the queued effects with follow-up joker events → 5. counters: `consumable_use_counts[id]`,
`stats.run.tarots_used` / `planets_used` / `unique_planets_used`, `last_tarot_planet` (Tarot or
Planet, never The Fool, never a Spectral) → 6. `fire_consumable_used` (Constellation, …) → 7. in
`Play`: re-sort the hand, recompute boss debuffs (a suit change under The Club is debuffed at
once), refill if the hand became empty.

### 2.5 Generation pool (`consumable::random_consumable`)

Uniform over the kind's ids that are `in_random_pool` (not The Soul / Black Hole; secret Planets
only once their hand was played this run) and not currently held (unless Showman,
`mods.allow_duplicates`); fallback Strength / Pluto / Incantation. Used by `CreateConsumable`
(8 Ball, Cartomancer, Séance, Sixth Sense, Purple Seal, High Priestess, Emperor, …).

### 2.6 Test template (`Game::step`, as in `tests/consumables.rs`)

```rust
use balatro_core::actions::{Action, ActionError};
use balatro_core::consumable::ConsumableInstance;
use balatro_core::content::ConsumableId as C;
use balatro_core::enums::{Enhancement, Phase};
use balatro_core::game::Game;
use balatro_core::state::GameConfig;

fn in_play(seed: u64) -> Game {
    let mut g = Game::new(seed, GameConfig::default());
    g.step(Action::SelectBlind).expect("select");
    g
}

#[test]
fn the_empress_turns_targets_into_mult_cards() {
    let mut g = in_play(1);
    g.add_consumable(ConsumableInstance::new(C::TheEmpress)).expect("room");
    let uids: Vec<u32> = g.state().deck.hand.iter().map(|c| c.uid).collect();
    g.step(Action::UseConsumable { slot: 0, targets: 0b11 }).expect("use");
    // hand is re-sorted after a use: find cards by uid
    for c in &g.state().deck.hand {
        let hit = c.uid == uids[0] || c.uid == uids[1];
        assert_eq!(c.enhancement == Enhancement::Mult, hit);
    }
    assert_eq!(g.state().stats.run.tarots_used, 1);
    // validation: 3 targets are too many
    g.add_consumable(ConsumableInstance::new(C::TheEmpress)).expect("room");
    assert!(matches!(
        g.step(Action::UseConsumable { slot: 0, targets: 0b111 }),
        Err(ActionError::TargetCount { selected: 3, .. })
    ));
}
```

For each card test: the effect, every `can_use` failure (`ActionError::CannotUse`), target-count
errors for targeted cards, randomness (force with `g.state_mut().mods.prob_mult = 1000` only in
pure tests; in `Game` tests loop over seeds and assert the distribution or both outcomes),
and `check_invariants()`. Set up positions with `g.state_mut()` (hand cards, money, jokers via
`g.add_joker`). Get to the shop by beating the blind:
`g.state_mut().round_score = target - 1.0; g.step(Action::Play(1))`.

### 2.7 Pitfalls

1. Do not check slot room for creations (except "up to N" counts); do not check Eternal.
2. Untargeted cards that act on the hand need `needs_hand: true` (Sigil, Ouija, Familiar, Grim,
   Incantation, Immolate). Targeted cards need `min_targets >= 1`.
3. Stone cards have no rank/suit: Strength, Sigil, Ouija skip them.
4. Death: **left** (`target_pos(0)`) becomes the **right** (`target_pos(1)`).
5. Random distinct hand cards (Immolate, Familiar's victim): pick without replacement from the
   positions; queue `destroy_card` for each. Never destroy by mutating the hand.
6. Added cards: build with `Card::new(0, rank, suit)` + fields; the uid is reassigned.
7. Wheel of Fortune: the card is used up even on "Nope!" (roll `chance(1, 4)` inside the effect,
   eligibility in `can_use`). Only edition-less jokers; never Negative.
8. Ectoplasm's hand-size penalty grows: `-(uses_of(Ectoplasm) + 1)`; Ouija is a flat -1.
9. Ankh/Hex: queue the destructions **before** the copy (the copy's room check comes after).
10. Do not touch `stats`, `last_tarot_planet` or fire joker events: the game layer does.
11. Judgement / Wraith / The Soul queue `create_joker`; since Phase 5.0 it creates a real joker
    (Wraith Rare, The Soul Legendary): test the new joker's rarity and `effects_no_room == 0`.

## 3. Bosses

### 3.1 Definition

```rust
use crate::boss::{BossCtx, BossDef, BossHooks, DrawCtx, DrawKind, PlayCtx};
use crate::card::{Card, RuleFlags};
use crate::content::BossId as B;

pub static DEFS: &[BossDef] = &[
    BossDef {
        mult: MULT_WALL,                 // requirement multiplier (default x2)
        keep_mult_when_disabled: false,  // The Needle: true
        halve_base: false,               // The Flint
        hand_size: 0,                    // The Manacle: -1
        hooks: BossHooks { before_play: Some(my_before_play), ..BossHooks::NONE },
        ..BossDef::new(B::TheWall, "The Wall", 2)   // id, name, min ante (Any = 1, Showdown = 8)
    },
];
```

### 3.2 Hooks

All hooks run only while the boss **is in effect** (`GameState::boss_in_effect()`: boss blind,
round in play, not disabled), except `on_ante_start`.

| hook | signature | when | use for |
|---|---|---|---|
| `on_ante_start` | `fn(&mut BossCtx)` | the boss was chosen for a new ante | The Ox (fix `boss_mut().hand`) |
| `on_blind_start` | `fn(&mut BossCtx)` | boss blind selected: deck reset, hands/discards set, **before** joker `on_blind_select` (Burglar, Chicot) and the first draw | Water, Needle, Amber Acorn, Crimson Heart |
| `debuffs_card` | `fn(&Card, &RuleFlags) -> bool` (pure; card passed un-debuffed) | recomputed at blind start and after every Play-phase action, all piles | Club/Goad/Window/Head, Plant, Pillar, Verdant Leaf |
| `face_down` | `fn(&mut DrawCtx, &Card) -> bool` | per card as it is drawn (top card first) | House, Wheel, Fish, Mark |
| `draw_count` | `fn(DrawKind, default) -> usize` (pure) | before each draw | Serpent |
| `after_draw` | `fn(&mut BossCtx)` | after each draw | Cerulean Bell |
| `before_play` | `fn(&mut PlayCtx)` | hand paid and evaluated, before jokers score it; also run by `Game::preview_play` on copies | Psychic, Eye, Mouth, Hook, Tooth, Ox, Arm |
| `after_play` | `fn(&mut BossCtx)` | after the hand scored, only if the round continues, before the refill | Crimson Heart |
| `on_joker_sold` | `fn(&mut BossCtx)` | a joker was sold during the round | Verdant Leaf |
| `on_disable` | `fn(&mut BossCtx)` | disabled mid-round (Chicot, Luchador, Verdant Leaf); still in effect during the call | Water, Needle (give back) |

`DrawKind`: `RoundStart` (opening hand), `AfterPlay`, `AfterDiscard`, `Refill`.

### 3.3 Contexts

**`BossCtx`**: `state()`, `boss()` / `boss_mut()` (`BossState`), `hand()`, `jokers()`,
`set_hands_left`, `set_discards_left`, `add_hands`, `add_discards`,
`set_debuffed_joker(Option<uid>)` (restores the previous one; the game restores at round end and
on disable), `shuffle_jokers()`, `rng_range`, `chance` (`RngKey::Bosses`), `disable()` (the
game disables the boss after the hook).

**`DrawCtx`**: `kind()`, `is_face(card)` (Pareidolia-aware, ignores debuff), `chance(num, den)`
(Oops! All 6s applies).

**`PlayCtx`** (works on references so the preview can share it): `played()`, `hand_type()`,
`held()`, `money()`, `plays_round(ht)` (includes this hand), `level(ht)`, `boss()`;
`invalidate()` (score 0, also sets triggered), `trigger()` (Matador), `add_money`, `set_money`,
`add_levels(ht, delta)` (floor 1, returns the change), `discard_random_held(n)` (The Hook; the
game moves them to the discard pile and fires a **forced** discard event: Green Joker/Ramen/
Castle/… react, Burnt/Trading Card do not; Purple Seals fire).

**`BossState`** (generic fields; reset rules in `boss.rs`): `hand` (Ox, per ante), `card_uid`
(Cerulean Bell forced card: Play/Discard masks must contain it, enforced by `validate`),
`joker_uid` (set only via `set_debuffed_joker`), `removed` (hands/discards taken, Needle/Water),
`jokers_hidden` (Amber Acorn), `draws`, per-hand `triggered` / `hand_invalid`.

### 3.4 What the game layer does (do not do it yourself)

- Selection at ante start (`boss::choose_boss`): defined bosses only, Showdown iff
  `ante % 8 == 0` (falls back to the other pool if one has no defined boss), `min_ante <= ante`,
  least-seen first, uniform on `RngKey::Bosses`; `boss_seen` counts.
- Requirement: `def.mult`; disabled → x2 unless `keep_mult_when_disabled`.
- `triggered` (Matador) is set automatically when the boss halves the base or a played card is
  debuffed by the boss; set it yourself in `before_play` for everything else (`invalidate()`
  already does).
- Disable: `on_disable`, then generic clean-up (joker debuff restored, forced card cleared,
  jokers revealed, hand cards turned face up, debuffs and requirement recomputed).
- Round end: debuffs and face-down flags cleared, joker debuff restored; `Card::played_ante`
  (The Pillar) is set on every played card and cleared at ante start.
- Hand size: `def.hand_size` is added while in effect (no hook needed for The Manacle).

### 3.5 Test template (as in `tests/boss_contract.rs`)

```rust
fn win_round(g: &mut Game) {
    let t = g.state().target;
    g.state_mut().round_score = t - 1.0;
    g.step(Action::Play(0b11111)).expect("winning play"); // 5 cards: fine under The Psychic
    if g.state().phase == Phase::Shop { g.step(Action::EndShop).expect("end shop"); }
}

/// Beat Small and Big, force `boss`, select it.
fn at_boss(seed: u64, boss: BossId) -> Game {
    let mut g = Game::new(seed, GameConfig::default());
    for _ in 0..2 { g.step(Action::SelectBlind).expect("select"); win_round(&mut g); }
    g.state_mut().boss = Some(boss);
    g.step(Action::SelectBlind).expect("select boss");
    g
}
```

For each boss test: the effect, the non-trigger case, Matador (`triggered`), disable via
`g.add_joker(JokerInstance::new(0, JokerId::Chicot))` mid-round (and Chicot owned before the
blind), `preview_play` == `step` for every mask (copy the loop of
`preview_matches_step_under_every_defined_boss`; it already iterates every defined boss, so your
bosses are covered automatically), and `check_invariants()`. The smoke test
`random_games_against_every_defined_boss_keep_the_invariants` also picks up every new boss.

### 3.6 Pitfalls

1. Never mutate cards' `debuffed` / `face_down` directly: return them from `debuffs_card` /
   `face_down` (the game recomputes debuffs constantly and would overwrite you).
2. `before_play` must be deterministic given its inputs (it runs in previews on copies): use only
   `PlayCtx` (rolls through `discard_random_held`). Adding new randomness there → ask.
3. Suit bosses: `card.has_suit(suit, rules)` on the passed (un-debuffed) card makes Wild cards
   debuffed and respects Smeared.
4. Eye/Mouth compare `hand_type()`, not `contains`. Counts include this hand:
   The Eye rejects when `plays_round(ht) > 1`. `hand_plays_round` also counts rejected hands, so
   Eye and Mouth treat a rejected play as played: after a rejected hand of type T, T is still
   forbidden for the Eye. The Mouth must not use `plays_round`: it writes the first **valid**
   hand type to `boss_state.hand` in `after_play` (reset at blind start) and rejects any other
   type while `boss_state.hand` is set.
5. The Ox: tie → highest tier (iterate `HandType::ALL` in reverse).
6. The Serpent: return 3 for `AfterPlay`/`AfterDiscard`, `default` otherwise; the game caps the
   hand at 16 cards.
7. Water/Needle: record what you take in `boss_mut().removed` and give it back in `on_disable`.
8. Crimson Heart: never the same joker twice in a row unless it is the only one; pick in
   `on_blind_start` and `after_play`.
9. Cerulean Bell: in `after_draw`, keep `card_uid` if that card is still in hand, else pick a new
   random hand card.

## 4. Survey: every consumable and boss → API

### 4.1 Planets (all defined; 4A tests them)

| Card | Def | Effect |
|---|---|---|
| Pluto … Eris (12) | untargeted | `level_up(used().planet_hand(), 1)` |

### 4.2 Tarots

| # | Card | targets / flags | `can_use` | effect |
|---|---|---|---|---|
| 0 | The Fool ✅ | — | `last_tarot_planet.is_some() && free_consumable_slots() > 0` | `add_consumable(last)` |
| 1 | The Magician ✅ | 1–2 | — | `set_enhancement(Lucky)` |
| 2 | The High Priestess | — | `free_consumable_slots() > 0` | `min(2, free)` × `create_consumable(Planet)` |
| 3 | The Empress | 1–2 | — | `set_enhancement(Mult)` |
| 4 | The Emperor | — | free slot | `min(2, free)` × `create_consumable(Tarot)` |
| 5 | The Hierophant | 1–2 | — | `set_enhancement(Bonus)` |
| 6 | The Lovers | 1–1 | — | `set_enhancement(Wild)` |
| 7 | The Chariot | 1–1 | — | `set_enhancement(Steel)` |
| 8 | Justice | 1–1 | — | `set_enhancement(Glass)` |
| 9 | The Hermit ✅ | — | — | `add_money(money.clamp(0, 20))` |
| 10 | The Wheel of Fortune | — | any joker with `Edition::Base` | `chance(1,4)` → random eligible joker → `weighted_index(EDITION_ROLL)` → `set_joker_edition` |
| 11 | Strength | 1–2 | — | `set_rank(next)` (K→A, A→2; skip Stone) |
| 12 | The Hanged Man | 1–2 | — | `destroy_card(pos)` |
| 13 | Death ✅ | 2–2 | — | `copy_card(left, right)` |
| 14 | Temperance | — | — | `add_money(min(Σ joker sell_value, 50))` |
| 15 | The Devil | 1–1 | — | `set_enhancement(Gold)` |
| 16 | The Tower | 1–1 | — | `set_enhancement(Stone)` |
| 17–19, 21 | The Star / Moon / Sun / World | 1–3 | — | `set_suit(Diamonds / Clubs / Hearts / Spades)` |
| 20 | Judgement | — | `free_joker_slots() > 0` | `create_joker(None)` (Phase 5) |

### 4.3 Spectrals

| Card | targets / flags | `can_use` | effect |
|---|---|---|---|
| Familiar | `needs_hand` | hand ≥ 2 | destroy 1 random hand card; 3 × `add_card(J/Q/K random suit, random ADDED_CARD_ENHANCEMENTS, Hand)` |
| Grim | `needs_hand` | hand ≥ 2 | destroy 1 random; 2 × Ace |
| Incantation | `needs_hand` | hand ≥ 2 | destroy 1 random; 4 × rank 2–10 |
| Talisman / Deja Vu / Trance / Medium | 1–1 | — | `set_seal(Gold / Red / Blue / Purple)` |
| Aura | 1–1 | — | `set_edition(weighted EDITION_ROLL)` |
| Wraith | — | `free_joker_slots() > 0` | `create_joker(Some(Rare))`; `set_money(0)` |
| Sigil | `needs_hand` | hand ≥ 2 | one random suit (4, equal allowed) → `set_suit` on every non-Stone hand card |
| Ouija | `needs_hand` | hand ≥ 2 | one random rank → `set_rank` on non-Stone cards; `add_hand_size(-1)` |
| Ectoplasm | — | any edition-less joker | random eligible → `set_joker_edition(Negative)`; `add_hand_size(-(uses_of(Ectoplasm) + 1))` |
| Immolate | `needs_hand` | hand ≥ 2 | destroy `min(5, hand)` random distinct cards; `add_money(20)` |
| Ankh ✅ | — | a joker and a free slot | destroy others (non-Eternal), then `copy_joker(keep, true)` |
| Hex | — | any edition-less joker | random eligible → `Polychrome`; destroy every other non-Eternal joker |
| Cryptid | 1–1 | — | 2 × `add_card(copy of target, Hand)` |
| The Soul | — | `free_joker_slots() > 0` | `create_joker(Some(Legendary))` (Phase 5) |
| Black Hole ✅ | — | — | `level_up(h, 1)` for all 12 |

### 4.4 Bosses

| Boss | min ante | static fields | hooks |
|---|---|---|---|
| The Hook ✅ | 1 | — | `before_play`: `discard_random_held(2)`, trigger if > 0 |
| The Ox | 6 | — | `on_ante_start`: most played hand (`state().hand_plays_run`, tie → highest tier) → `boss_mut().hand`; `before_play`: `hand_type() == hand` → `set_money(0)`, trigger |
| The House | 2 | — | `face_down`: `kind() == RoundStart` |
| The Wall ✅ | 2 | `mult: MULT_WALL` | — |
| The Wheel | 2 | — | `face_down`: `chance(1, 7)` |
| The Arm | 2 | — | `before_play`: `add_levels(ht, -1) != 0` → trigger |
| The Club / Goad / Window / Head | 1 | — | `debuffs_card`: `has_suit(Clubs / Spades / Diamonds / Hearts)` |
| The Fish | 2 | — | `face_down`: `kind() == AfterPlay` |
| The Psychic ✅ | 1 | — | `before_play`: `played().len() != 5` → `invalidate()` |
| The Water | 2 | — | `on_blind_start`: `removed = discards_left`, `set_discards_left(0)`; `on_disable`: `add_discards(removed)` |
| The Manacle | 1 | `hand_size: -1` | — |
| The Eye | 3 | — | `before_play`: `plays_round(ht) > 1` → `invalidate()` |
| The Mouth | 2 | — | `before_play`: `hand_type() != boss_state.hand` (first valid type, set in `after_play`) → `invalidate()` |
| The Plant ✅ | 4 | — | `debuffs_card`: `card.is_face(rules)` |
| The Serpent | 5 | — | `draw_count`: 3 after play/discard |
| The Pillar | 1 | — | `debuffs_card`: `card.played_ante` |
| The Needle ✅ | 2 | `mult: MULT_NEEDLE`, `keep_mult_when_disabled` | `on_blind_start` / `on_disable` (hands, `removed`) |
| The Tooth | 3 | — | `before_play`: `add_money(-(played().len()))`, trigger |
| The Flint ✅ | 2 | `halve_base` | — |
| The Mark ✅ | 2 | — | `face_down`: `is_face(card)` |
| Amber Acorn | 8 | — | `on_blind_start`: `shuffle_jokers()`, `jokers_hidden = true` |
| Verdant Leaf ✅ | 8 | — | `debuffs_card`: true; `on_joker_sold`: `disable()` |
| Violet Vessel | 8 | `mult: MULT_VIOLET_VESSEL` | — |
| Crimson Heart | 8 | — | `on_blind_start` + `after_play`: `set_debuffed_joker(random other)` |
| Cerulean Bell | 8 | — | `after_draw`: keep / re-pick `boss_mut().card_uid` |

✅ = defined in Phase 4.0.
