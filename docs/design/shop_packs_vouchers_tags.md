# Market, pack, voucher and tag contract (Phase 5.0): guide for content authors

This is the contract the market, the 15 booster packs, the 32 vouchers and the 24 tags are written
against. **Phase 5B/5C agents: follow this document exactly and write only your own files**
(table in §1). If something cannot be expressed cleanly, do not hack around it: describe the gap
in your report and the orchestrator extends the contract.

Code: `src/shop.rs` (market, prices, rerolls, joker pools), `src/packs.rs` (pack kinds, sizes,
open pack, generation), `src/voucher.rs` (`VoucherDef`, `ShopMods`, `VoucherCtx`, `redeem`),
`src/tag.rs` (`TagDef`, `TagCtx`, `gain`/hook runners), `src/game_shop.rs` (the `Game` steps),
ids `src/content/ids.rs` (`VoucherId`, `TagId`; `PackKind` is in `hooks.rs`, `PackSize` in
`packs.rs`), registry `src/content/mod.rs`, numbers `src/data/shop_weights.rs` and
`src/data/prices.rs`. Working examples: `content/{vouchers,tags}.rs`. Contract tests:
`tests/{shop,packs,vouchers_tags}.rs`. Rules: `docs/mechanics/{shop,packs,vouchers,tags}.md`,
decisions `notes.md` §4.1d and §6 of this file. Joker hooks that react to the market
(`on_reroll`, `on_pack_open`, `on_pack_skip`, `on_shop_enter/exit`, `Modifiers::{free_rerolls,
free_planets, allow_duplicates, debt_limit}`): `docs/design/hooks.md`.

---

## 1. Who writes what

| Package | Your files (only these) | Ids |
|---|---|---|
| **5B tags + pack audit** | `src/content/tags.rs` (add defs), `tests/tags.rs` (new), `tests/packs_audit.rs` (new) | 18 tags not yet defined (all except Economy, Investment, Charm, Coupon, Double, Boss) |
| **5C vouchers** | `src/content/vouchers.rs` (add defs), `tests/vouchers.rs` (new) | 24 vouchers not yet defined (all except Overstock, Clearance Sale, Grabber, Seed Money, Crystal Ball, Telescope, Director's Cut, Retcon) |

The 15 packs need no content file: `packs::generate_items` already builds all of them from
`PackKind x PackSize`. 5B's pack work is an audit: tests for every rule of `packs.md` not yet
covered by `tests/packs.rs` and a report of any deviation (do not edit `packs.rs`).

Never edit `ids.rs`, `content/mod.rs`, `shop.rs`, `packs.rs`, `voucher.rs`, `tag.rs`,
`game_shop.rs`, `state.rs`, `game.rs`, `actions.rs`, `hooks.rs`, `data/*`. Numbers you need are
in `data::shop_weights` / `data::prices` (`WEIGHT_MERCHANT`, `WEIGHT_TYCOON`,
`REROLL_SURPLUS_BASE`, `REROLL_GLUT_BASE`, `INTEREST_CAP_MONEY_TREE`, `TAG_*`, ...). A number that
belongs to one tag/voucher is a named `const` in your file above its functions.

Workflow: add the def to `DEFS`, write the functions below the table in a block starting with
`// --- Name: https://balatrowiki.org/w/Name ---`, test it, then run
`cargo test -p balatro-core`, `cargo clippy --workspace --all-targets -- -D warnings`,
`cargo fmt -p balatro-core -- <your files>` (do not `cargo fmt --all` in a shared worktree).
Both packages edit different files, so they can run in parallel without conflicts.

## 2. Vouchers

### 2.1 Definition

```rust
use crate::content::ids::VoucherId as V;
use crate::voucher::{ShopMods, VoucherCtx, VoucherDef};

pub static DEFS: &[VoucherDef] = &[
    VoucherDef {                                      // passive only
        passive: ShopMods { card_slots: 1, ..ShopMods::NONE },
        ..VoucherDef::new(V::OverstockPlus, "Overstock Plus")   // id, name; cost = $10
    },
    VoucherDef {                                      // one-shot only
        on_redeem: Some(nacho_tong),
        ..VoucherDef::new(V::NachoTong, "Nacho Tong")
    },
];
fn nacho_tong(ctx: &mut VoucherCtx<'_>) { ctx.add_hands(1); }
```

Two kinds of effect, both optional, a voucher may use both:

* **passive** (`ShopMods`): read by the market while the voucher is owned. `GameState::shop_mods`
  is the fold of all owned vouchers (`state::fold_shop_mods`; sums for `card_slots`, best-of for
  the rest, lowest `reroll_base`). Recomputed on redeem and checked by `check_invariants`.
* **one-shot** (`on_redeem`, `VoucherCtx`): permanent writes done once when bought.

| `ShopMods` field | Meaning | Consumer (already written) |
|---|---|---|
| `card_slots: i32` | extra random card slots | `shop::card_slot_count`; buying mid-shop fills the slot at once |
| `discount: Discount` | `Clearance` 25% / `Liquidation` 50% | every price (cards, packs, vouchers); owned sell values are re-scaled on redeem |
| `edition_mult: u32` | 2 Hone, 4 Glow Up (1 = none) | joker and playing-card edition odds (Negative is not scaled) |
| `reroll_base: Option<i32>` | absolute start price (`REROLL_SURPLUS_BASE` 3, `REROLL_GLUT_BASE` 1) | `shop::reroll_price` (lowest wins) |
| `tarot_weight` / `planet_weight: Option<f64>` | `WEIGHT_MERCHANT` 9.6, `WEIGHT_TYCOON` 32 | card-slot type roll |
| `playing_cards: bool` | Magic Trick | playing cards in the shop (weight 4) |
| `illusion: bool` | Illusion | shop playing cards: 40% enhancement, 20% seal, edition odds |
| `telescope: bool` | Telescope | Celestial packs: item 0 = Planet of the most played hand |
| `omen_globe: bool` | Omen Globe | Arcana pack slot is a Spectral 20% of the time |
| `observatory: bool` | Observatory | `RunView::observatory` (scoring: held Planet x1.5 Mult) |
| `boss_reroll: BossReroll` | `OncePerAnte` Director's Cut / `Unlimited` Retcon | `Action::RerollBoss` ($10, BlindSelect) |

`VoucherCtx` (all clamped: hands >= 1, discards >= 0, hand size 1..=16): `add_hands(d)`,
`add_discards(d)`, `add_hand_size(d)`, `add_joker_slots(d)`, `add_consumable_slots(d)`,
`raise_interest_cap(cap)` (never lowers), `add_ante(d)` (recomputes the blind target; the ante's
boss and tags stay), `state()` (read).

### 2.2 What the game layer does (do not do it yourself)

`Action::BuyVoucher(slot)`: price check (`shop::voucher_price`: $10 with the discount, debt
limit respected) -> pay -> the ante slot empties (`ante_voucher = None`) -> `voucher::redeem`:
push to `vouchers`, refold `shop_mods`, run `on_redeem`, re-scale the sell values of owned items
if the discount changed, fill a card slot the voucher opened. **Never** push to `vouchers` or write
`shop_mods` yourself in a `Game` test: `Game::step` checks the cache against the fold in debug
builds. Use `voucher::redeem(g.state_mut(), V::X)`.

Which voucher is offered: at every ante start (`GameState::start_ante`, `RngKey::Shop`) uniformly
among **defined** vouchers that are not owned and whose base is owned (bases need nothing). The
voucher stays in every shop of the ante until bought; after the boss it is rolled again. All
vouchers are unlocked (notes.md §4.1d). An undefined voucher is never offered, so adding your
defs changes the rolled vouchers of every seed: never assert a specific seed's voucher.

### 2.3 Survey: every voucher -> API

| # | Voucher | Needs | Def |
|---|---|---|---|
| 1 | Overstock ✅ | - | `passive.card_slots = 1` |
| 2 | Overstock Plus | Overstock | `passive.card_slots = 1` (sum = 2) |
| 3 | Clearance Sale ✅ | - | `passive.discount = Clearance` |
| 4 | Liquidation | Clearance Sale | `passive.discount = Liquidation` |
| 5 | Hone | - | `passive.edition_mult = 2` |
| 6 | Glow Up | Hone | `passive.edition_mult = 4` |
| 7 | Reroll Surplus | - | `passive.reroll_base = Some(REROLL_SURPLUS_BASE)` |
| 8 | Reroll Glut | Reroll Surplus | `passive.reroll_base = Some(REROLL_GLUT_BASE)` |
| 9 | Crystal Ball ✅ | - | `on_redeem`: `add_consumable_slots(1)` |
| 10 | Omen Globe | Crystal Ball | `passive.omen_globe = true` |
| 11 | Telescope ✅ | - | `passive.telescope = true` |
| 12 | Observatory | Telescope | `passive.observatory = true` |
| 13 | Grabber ✅ | - | `on_redeem`: `add_hands(1)` |
| 14 | Nacho Tong | Grabber | `add_hands(1)` |
| 15 | Wasteful | - | `add_discards(1)` |
| 16 | Recyclomancy | Wasteful | `add_discards(1)` |
| 17 | Tarot Merchant | - | `passive.tarot_weight = Some(WEIGHT_MERCHANT)` |
| 18 | Tarot Tycoon | Tarot Merchant | `Some(WEIGHT_TYCOON)` |
| 19 | Planet Merchant | - | `passive.planet_weight = Some(WEIGHT_MERCHANT)` |
| 20 | Planet Tycoon | Planet Merchant | `Some(WEIGHT_TYCOON)` |
| 21 | Seed Money ✅ | - | `raise_interest_cap(INTEREST_CAP_SEED_MONEY)` |
| 22 | Money Tree | Seed Money | `raise_interest_cap(INTEREST_CAP_MONEY_TREE)` |
| 23 | Blank | - | nothing |
| 24 | Antimatter | Blank | `add_joker_slots(1)` |
| 25 | Magic Trick | - | `passive.playing_cards = true` |
| 26 | Illusion | Magic Trick | `passive.illusion = true` |
| 27 | Hieroglyph | - | `add_ante(-1)` + `add_hands(-1)` |
| 28 | Petroglyph | Hieroglyph | `add_ante(-1)` + `add_discards(-1)` |
| 29 | Director's Cut ✅ | - | `passive.boss_reroll = OncePerAnte` |
| 30 | Retcon ✅ | Director's Cut | `passive.boss_reroll = Unlimited` |
| 31 | Paint Brush | - | `add_hand_size(1)` |
| 32 | Palette | Paint Brush | `add_hand_size(1)` |

✅ = defined in Phase 5.0. Money Tree + To the Moon etc. need nothing from you (economy handles
it). Sell-value re-scaling for Liquidation is in `shop::rescale_sell_values` (see §6 UNVERIFIED).

### 2.4 Voucher test template (`tests/vouchers.rs`)

```rust
use balatro_core::actions::Action;
use balatro_core::content::VoucherId as V;
use balatro_core::game::Game;
use balatro_core::state::GameConfig;
use balatro_core::voucher;

fn in_shop(seed: u64) -> Game {                      // as in tests/vouchers_tags.rs
    let mut g = Game::new(seed, GameConfig::default());
    g.step(Action::SelectBlind).expect("select");
    let t = g.state().target;
    g.state_mut().round_score = t - 1.0;
    g.step(Action::Play(1)).expect("win");
    g
}

#[test]
fn nacho_tong_gives_a_second_extra_hand() {
    let mut g = in_shop(1);
    voucher::redeem(g.state_mut(), V::Grabber);       // owns the base
    voucher::redeem(g.state_mut(), V::NachoTong);
    assert_eq!(g.state().base_hands, 6);
    g.step(Action::EndShop).expect("end");
    g.step(Action::SelectBlind).expect("select");
    assert_eq!(g.state().hands_left, 6);
    g.state().check_invariants().expect("invariants");
}
```

For every voucher test: the effect (state field or behaviour), that the cache equals the fold
(`check_invariants`), buying it through `Action::BuyVoucher` once (put it into the slot with
`g.state_mut().shop.vouchers = vec![Some(ShopVoucher { id, from_ante: false })]`), that an upgrade
is offered only after its base (`shop::pick_voucher` on a state with/without the base), and the
numbers of the wiki table. Passive effects on prices/rolls can be tested without stepping on a
`GameState` clone with `shop_mods` edited (never step such a state).

## 3. Tags

### 3.1 Definition

```rust
use crate::content::ids::TagId as T;
use crate::tag::{TagCtx, TagDef};

pub static DEFS: &[TagDef] = &[
    TagDef {
        on_gain: Some(speed_on_gain),                 // hooks are all optional
        ..TagDef::new(T::Speed, "Speed Tag", 1)       // id, name, min ante
    },
];
fn speed_on_gain(ctx: &mut TagCtx<'_>) -> bool {      // true = consumed (removed from `tags`)
    let skipped = ctx.state().stats.run.blinds_skipped as i32;
    ctx.add_money(5 * skipped);
    true
}
```

| hook | when | use for |
|---|---|---|
| `on_gain` | the tag is gained (skip) | immediate effects, pack tags, Boss, Orbital, Top-up |
| `on_shop_enter` | the shop was generated (cards, packs, voucher exist) | Coupon, D6, Uncommon/Rare/edition jokers, Voucher |
| `on_boss_defeated` | a Boss Blind was beaten, after the cash-out | Investment |
| `on_round_start` | a blind was selected, before the first draw | Juggle |

All return `bool`: `true` = used up, `false` = stays held and is offered the same hook again
later (an edition tag finds no base-edition joker in the shop: the tag waits for the next shop).
Held tags run oldest first. While a hook runs, `state().tags` does not contain the tags that
are still waiting (do not read it).

### 3.2 `TagCtx` API

| Need | Call |
|---|---|
| read the run | `state()` (`money`, `stats.hands_played`, `stats.unused_discards`, `stats.run.blinds_skipped`, `ante`, `hand_plays_run`, `shop`, ...) |
| money | `money()`, `add_money(n)` |
| Orbital hand (picked when the tag was offered) | `aux_hand() -> Option<HandType>` |
| hand level | `level_up(hand, levels)` (queued `LevelUpHand`) |
| random joker in the joker area | `create_joker(rarity)` (queued; dropped + `effects_no_room` if the area is full) |
| open a pack at once | `queue_pack(PackSpec { kind, size })` |
| reroll the boss for free | `reroll_boss()` |
| hand size for one round | `add_round_hand_size(n)` (cleared when the round ends) |
| randomness | `rng_range(lo, hi)` (`RngKey::Tags`) |
| extra free joker in the shop | `shop_add_free_joker(rarity) -> bool` |
| first base-edition shop joker gets an edition and is free | `shop_edition_first_base_joker(edition) -> bool` (return its result: the tag is consumed only if it applied) |
| extra voucher in the shop | `shop_add_voucher() -> bool` |
| everything in the shop free (cards and packs, not vouchers) | `shop_make_free()` |
| rerolls start at this price in this shop | `shop_set_reroll_base(0)` |

### 3.3 What the game layer does (do not do it yourself)

* **Offering**: at ante start each of Small/Big gets a tag, uniformly among **defined** tags with
  `min_ante <= max(ante, 1)` (`RngKey::Tags`, `state.blind_tags`); an Orbital tag also picks its
  hand then (`blind_tag_hands`, an open hand or a secret one already played).
* **Gaining** (`Action::SkipBlind`): `tag::gain`: a held Double Tag is used up and the tag is
  gained **twice** (never for Double itself); each copy runs `on_gain` and is held unless
  consumed; then `blinds_skipped += 1` happened before (a Speed Tag counts its own skip), the joker
  `on_blind_skip` event fires, the blind advances, and queued packs open (`PackOpen`, back to
  `BlindSelect` when the last closes). `SideEffect::CreateTag` (Diet Cola) only pushes the tag.
* **Timing**: `on_shop_enter` runs inside `shop::enter_shop` right after generation (before the
  joker `on_shop_enter` event); `on_boss_defeated` after the boss cash-out (before the next
  ante's shop); `on_round_start` in `select_blind` before the first draw.
* Undefined tags (until 5B is done) are held forever and do nothing.

### 3.4 Survey: every tag -> API

| # | Tag | min ante | Hook and body |
|---|---|---|---|
| 0 | Uncommon | 1 | `on_shop_enter`: `shop_add_free_joker(Uncommon)` -> true |
| 1 | Rare | 1 | `on_shop_enter`: `shop_add_free_joker(Rare)` |
| 2 | Negative | 2 | `on_shop_enter`: `shop_edition_first_base_joker(Negative)` (consume iff true) |
| 3 | Foil | 1 | same, `Foil` |
| 4 | Holographic | 1 | same, `Holographic` |
| 5 | Polychrome | 1 | same, `Polychrome` |
| 6 | Investment ✅ | 1 | `on_boss_defeated`: `add_money(TAG_INVESTMENT)` |
| 7 | Voucher | 1 | `on_shop_enter`: `shop_add_voucher()` (consume iff true) |
| 8 | Boss ✅ | 1 | `on_gain`: `reroll_boss()` |
| 9 | Standard | 2 | `on_gain`: `queue_pack(Standard Mega)` |
| 10 | Charm ✅ | 1 | `on_gain`: `queue_pack(Arcana Mega)` |
| 11 | Meteor | 2 | `on_gain`: `queue_pack(Celestial Mega)` |
| 12 | Buffoon | 2 | `on_gain`: `queue_pack(Buffoon Mega)` |
| 13 | Handy | 2 | `on_gain`: `add_money(stats.hands_played)` ($1 per hand this run) |
| 14 | Garbage | 2 | `on_gain`: `add_money(stats.unused_discards)` |
| 15 | Ethereal | 2 | `on_gain`: `queue_pack(Spectral Normal)` (UNVERIFIED size) |
| 16 | Coupon ✅ | 1 | `on_shop_enter`: `shop_make_free()` |
| 17 | Double ✅ | 1 | no hook: handled by `tag::gain` |
| 18 | Juggle | 1 | `on_round_start`: `add_round_hand_size(TAG_JUGGLE_HAND_SIZE)` |
| 19 | D6 | 1 | `on_shop_enter`: `shop_set_reroll_base(0)` |
| 20 | Top-up | 2 | `on_gain`: `create_joker(Common)` x `TAG_TOP_UP_JOKERS` |
| 21 | Speed | 1 | `on_gain`: `add_money(TAG_SPEED_PER_SKIP * blinds_skipped)` |
| 22 | Orbital | 2 | `on_gain`: `level_up(aux_hand(), TAG_ORBITAL_LEVELS)` |
| 23 | Economy ✅ | 1 | `on_gain`: `add_money(money.clamp(0, TAG_ECONOMY_CAP))` |

### 3.5 Tag test template (`tests/tags.rs`)

```rust
fn skip_with(g: &mut Game, tag: TagId) {              // fix the offered tag: never rely on a seed
    g.state_mut().blind_tags = [tag, tag];
    g.step(Action::SkipBlind).expect("skip");
}
#[test]
fn speed_pays_five_per_blind_skipped() {
    let mut g = Game::new(1, GameConfig::default());
    skip_with(&mut g, TagId::Speed);                  // first skip: $5
    assert_eq!(g.state().money, 4 + 5);
    skip_with(&mut g, TagId::Speed);                  // second skip: 2 x $5
    assert_eq!(g.state().money, 4 + 5 + 10);
}
```

Shop-time tags: push the tag (`g.state_mut().tags.push(TagId::X)`) then beat a blind to enter the
shop, or call `shop::enter_shop(g.state_mut())` on a `GameState` in `Phase::Shop` and apply the
queued effects with `game::apply_side_effects`. Test per tag: the effect, the timing (held until
its hook), that it is consumed (or kept when it could not apply), Double Tag interplay, and that
`blind_tags` never offers it below its min ante (check `try_tag_def(id).min_ante`).

## 4. Packs (all 15 are implemented; this is what 5B audits)

| Kind | Normal | Jumbo | Mega | Items | Picking |
|---|---|---|---|---|---|
| Arcana | 3 | 5 | 5 | Tarots; The Soul 0.3% per slot; Omen Globe: 20% Spectral per slot; hand dealt | used at once on `targets` |
| Celestial | 3 | 5 | 5 | Planets (secret ones only once played); Black Hole 0.3%; Telescope: item 0 | used at once |
| Spectral | 2 | 4 | 4 | Spectrals; The Soul 0.3%, Black Hole 0.3% each; hand dealt | used at once on `targets` |
| Standard | 3 | 5 | 5 | playing cards: 40% enhancement (8 kinds, Stone included), 20% seal, edition 4/2.8/1.2% x Hone/Glow Up | added to the draw pile at a random position |
| Buffoon | 2 | 4 | 4 | jokers 70/25/5 rarity, shop edition odds | needs a joker slot (Negative needs none) |

Prices $4 / $6 / $8 (discount, Coupon, Astronomer for Celestial apply); picks 1 / 1 / 2.
Weights (shop): Standard/Arcana/Celestial 4 / 2 / 0.5, Buffoon 1.2 / 0.6 / 0.15, Spectral
0.6 / 0.3 / 0.07 (total 22.42), two independent draws; the very first shop of a run opens with a
Normal Buffoon pack in slot 0. No duplicates inside one pack unless Showman.

Flow: `OpenPack(slot)` (pay, `on_pack_open`, `PackOpen`; Arcana/Spectral shuffle the deck and deal
a hand of `hand_size` cards) -> `PickPackItem { index, targets }` (validated by
`actions::validate_pick`; positions never shift, a picked item leaves `None`) -> after the last
pick or `SkipPack` the hand is gathered back and the phase returns (Shop for a bought pack,
BlindSelect for a tag pack; queued tag packs open one after the other). `SkipPack` fires
`on_pack_skip` once (also after a Mega's first pick); finishing the picks is not a skip. In
`PackOpen` you may also sell jokers/consumables, move jokers and use slot consumables on the dealt
hand (e.g. to make room for a Buffoon pick). A consumable pick uses `UseCheck::new(state,
targets, None)` (slot `None`: free-slot counts are the raw ones).

## 5. Other contract changes of Phase 5.0

* `Action`: `Buy`, `BuyAndUse`, `OpenPack`, `BuyVoucher`, `Reroll`, `MoveJoker`,
  `PickPackItem`, `SkipPack` are live; new `RerollBoss`. `SellJoker`, `SellConsumable`,
  `UseConsumable`, `MoveJoker` work in BlindSelect, Play, Shop and PackOpen (the real game allows
  it; hand-needing consumables only while a hand is dealt).
* `GameStats.effects_dropped` was split: `effects_unimplemented` (always 0: every `SideEffect` is
  implemented) and `effects_no_room` (normal: full areas, eternal targets, exhausted joker pool,
  cascade cap). `SideEffect::CreateJoker` is real: needs a free slot, picks by the shop pools
  (`shop::roll_joker`, `RngKey::Jokers`): rarity `None` = 70/25/5, Wraith Rare, The Soul Legendary
  (pool: not owned unless Showman; Gros Michel leaves and Cavendish enters with
  `RunFlag::GrosMichelExtinct`), edition rolled with the shop odds, sell value from the
  (discounted) price.
* Market slot semantics: `ShopState` slots are positional with `None` holes; `Buy(slot)` indexes
  `cards`, `OpenPack(slot)` `packs`, `BuyVoucher(slot)` `vouchers`. Tag jokers (Uncommon/Rare)
  are appended after the random slots and vanish on reroll.
* `REPLAY_VERSION` is 3.

## 6. Decisions and UNVERIFIED points

| Topic | Decision |
|---|---|
| Sell value | `max(1, floor(price_paid / 2))`; free items sell for $1; created jokers use the discounted base price |
| Discount change (Clearance/Liquidation) | owned jokers and consumables are re-priced: new sell value = `sell_value(new price) + extra`, `extra = sell_value - sell_value(old price)` (Egg/Gift Card bonus inferred) — UNVERIFIED |
| Reroll | cost `base + paid rerolls`, free (Chaos) rerolls do not advance it; `base`: D6 override ($0), else Reroll Surplus/Glut, else $5; floor $0 |
| Buy & Use | only for consumables usable with no hand and no targets (Planets, Hermit, ...): the shop has no hand; takes no slot |
| Pack picks | consumables are used at once and never take a slot (also Planets); Standard cards go to the draw pile at a random position |
| Edition tags | modify the first base-edition shop joker (it becomes free); consumed only if one exists; Uncommon/Rare tags add an extra free joker |
| Voucher fallback | if no voucher is eligible, no voucher is offered (the wiki says Blank appears; irrelevant while Blank is unowned) |
| Orbital | hand picked when the tag is offered (notes.md §4.1c) |
| Speed | counts the skip that earned it (UNVERIFIED) |
| Boss reroll | `Action::RerollBoss` in BlindSelect: $10, a different boss of the same pool, the old one counts as unseen again, Director's Cut once per ante |
| Stickers | White Stake has none; `shop::roll_stickers` is the hook for Eternal/Perishable/Rental |
| Spectral cards in the shop | weight 0 (Ghost Deck only; `GameState::deck_spectral_in_shop`) |

## 7. Pitfalls

1. Never edit `shop_mods` / `vouchers` directly in a state you `step` (debug invariant); use
   `voucher::redeem`.
2. Seeds shift when content is added (offered vouchers/tags change): fix them in tests
   (`blind_tags`, `ante_voucher`, `shop.vouchers`, `shop.cards`).
3. `Game::step` after `SkipBlind` may leave `PackOpen`; a skip never needs a pack to be open for
   your test unless the tag queues one.
4. Hooks must not read `state().tags` and must return the consumption flag honestly.
5. `create_joker` / `level_up` are queued: the game layer applies them right after the hook.
6. Do not touch `stats` counters other than through the ctx; the game layer owns them.
7. A tag hook runs at most once per event per held tag; a Double-copied tag runs `on_gain` twice
   but is held twice for later hooks.
8. Tests that rely on randomness: loop over seeds and assert distributions, or fix the state.
