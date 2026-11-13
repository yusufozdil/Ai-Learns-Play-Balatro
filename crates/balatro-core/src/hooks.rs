//! Joker hook contract: hook signatures, the [`Hooks`] table of fn pointers, the read-only
//! [`RunView`] every hook sees, deferred [`SideEffect`]s, passive [`Modifiers`], copier
//! resolution, and the non-scoring event dispatchers (`fire_*`) with their [`EventCtx`].
//!
//! Scoring hooks run inside [`crate::scoring::Scorer::score_hand`] with a
//! [`ScoreCtx`]. Joker authors: read `docs/design/hooks.md` first.
//!
//! Shared file: owned by the orchestrator; content agents request new hooks via their report.

use crate::card::{Card, RuleFlags};
use crate::content::jokers::try_def;
use crate::deck::{Deck, Pile};
use crate::enums::{Enhancement, HandType, Rank, Rarity, Seal, Suit};
use crate::joker::{JokerDef, JokerInstance};
use crate::rng::{GameRng, RngKey};
use crate::scoring::ScoreCtx;

// ---------------------------------------------------------------------------------------------
// Hook signatures
// ---------------------------------------------------------------------------------------------

/// Scoring hook without a card: `before_scoring`, `joker_main`, `after_scoring`,
/// `on_invalid_hand`. `j` is this joker's own state (a scratch copy when run by a copier).
pub type JokerFn = fn(j: &mut JokerInstance, ctx: &mut ScoreCtx<'_>);
/// Per-trigger card hook: `on_card_scored` (played card) and `on_held` (held card). `card` is a
/// snapshot taken at the start of the trigger; its index is `ctx.card_index()`.
pub type CardFn = fn(j: &mut JokerInstance, ctx: &mut ScoreCtx<'_>, card: &Card);
/// Retrigger count for one card (`card_retriggers` / `held_retriggers`). Read-only; called once
/// per card before its first trigger. Return the number of EXTRA triggers this joker grants.
pub type RetriggerFn = fn(j: &JokerInstance, ctx: &ScoreCtx<'_>, card: &Card) -> u32;
/// Joker-on-joker hook (Baseball Card): called at joker `other`'s turn, after `other`'s own
/// effect and before `other`'s Polychrome. `other` is a slot index into `ctx.jokers()`.
pub type OtherJokerFn = fn(j: &mut JokerInstance, ctx: &mut ScoreCtx<'_>, other: usize);
/// Passive rule/stat provider (Four Fingers, Oops! All 6s, Juggler, ...). Never copied.
pub type ModifierFn = fn(j: &JokerInstance, m: &mut Modifiers);
/// Non-scoring event without payload.
pub type EventFn = fn(j: &mut JokerInstance, ctx: &mut EventCtx<'_>);
/// Discard event (once per discard action, with all discarded cards).
pub type DiscardFn = fn(j: &mut JokerInstance, ctx: &mut EventCtx<'_>, d: &DiscardInfo<'_>);
/// Playing cards destroyed / added to the deck (once per batch).
pub type CardsFn = fn(j: &mut JokerInstance, ctx: &mut EventCtx<'_>, cards: &[Card]);
/// End of round.
pub type RoundEndFn = fn(j: &mut JokerInstance, ctx: &mut EventCtx<'_>, info: &RoundEndInfo);
/// Another item was sold.
pub type SellFn = fn(j: &mut JokerInstance, ctx: &mut EventCtx<'_>, sold: &SoldItem);
/// Booster pack opened / skipped.
pub type PackFn = fn(j: &mut JokerInstance, ctx: &mut EventCtx<'_>, pack: PackKind);
/// Consumable used.
pub type ConsumableFn = fn(j: &mut JokerInstance, ctx: &mut EventCtx<'_>, c: &ConsumableView);

/// Hook table of one joker kind. Every field is optional; use `..Hooks::NONE`.
///
/// Column "copied" = whether Blueprint/Brainstorm run this hook of their target.
///
/// | hook | when | ctx | copied |
/// |---|---|---|---|
/// | `before_scoring` | step 2, before base chips/mult are read | Score | yes |
/// | `on_card_scored` | step 4e, every trigger of every non-debuffed scoring card | Score | yes |
/// | `card_retriggers` | step 4f, once per non-debuffed scoring card | Score (read) | yes |
/// | `on_held` | step 5b, every trigger of every non-debuffed held card | Score | yes |
/// | `held_retriggers` | step 5c (and end-of-round held effects) | Score (read) | yes |
/// | `joker_main` | step 6b, the joker's own turn | Score | yes |
/// | `on_other_joker` | step 6c, at every OTHER joker's turn | Score | yes |
/// | `after_scoring` | step 9c, after the hand (also after an invalid hand) | Score | yes |
/// | `on_invalid_hand` | step 0, boss rejected the hand (Matador) | Score | yes |
/// | `modifiers` | whenever the game recomputes passives | none | NO |
/// | `on_acquire` | this joker entered the joker area | Event | NO |
/// | `on_blind_select` | a blind was selected (`ctx.run().blind`) | Event | yes |
/// | `on_first_hand_drawn` | first hand of a round drawn | Event | yes |
/// | `on_blind_skip` | a blind was skipped | Event | yes |
/// | `on_discard` | a discard action | Event | yes |
/// | `on_cards_destroyed` | playing cards were destroyed | Event | yes |
/// | `on_cards_added` | playing cards were added to the deck | Event | yes |
/// | `on_round_end` | round won (or lost: Mr. Bones) | Event | NO |
/// | `on_shop_enter` / `on_shop_exit` / `on_reroll` | shop events | Event | yes |
/// | `on_sell_self` | this joker is being sold | Event | yes |
/// | `on_sell_other` | another joker or a consumable is being sold | Event | yes |
/// | `on_pack_open` / `on_pack_skip` | booster pack opened / skipped | Event | yes |
/// | `on_consumable_used` | a Tarot/Planet/Spectral was used | Event | yes |
#[derive(Debug, Clone, Copy)]
pub struct Hooks {
    /// Step 2 ("on played"): Space Joker, Midas Mask, Vampire, scaling counters, DNA.
    pub before_scoring: Option<JokerFn>,
    /// Step 4e: per trigger of each scoring card (Greedy, Scholar, Hiker, Photograph, ...).
    pub on_card_scored: Option<CardFn>,
    /// Step 4f: extra triggers for a scoring card (Hack, Sock and Buskin, Seltzer, Dusk, ...).
    pub card_retriggers: Option<RetriggerFn>,
    /// Step 5b: per trigger of each held card (Baron, Shoot the Moon, Raised Fist, ...).
    pub on_held: Option<CardFn>,
    /// Step 5c: extra triggers for a held card (Mime).
    pub held_retriggers: Option<RetriggerFn>,
    /// Step 6b: the joker's own effect (+chips, +mult, xmult, conditionals, money).
    pub joker_main: Option<JokerFn>,
    /// Step 6c: effect at another joker's turn (Baseball Card).
    pub on_other_joker: Option<OtherJokerFn>,
    /// Step 9c: after scoring (decay: Ice Cream, Seltzer; Sixth Sense). Chips/mult changes here
    /// are ignored (the score is final).
    pub after_scoring: Option<JokerFn>,
    /// Step 0: the boss blind rejected the hand (Psychic/Eye/Mouth). Only for effects that fire
    /// on such hands (Matador). `after_scoring` still runs afterwards.
    pub on_invalid_hand: Option<JokerFn>,
    /// Passive rules and stat deltas (Four Fingers, Shortcut, Smeared, Pareidolia, Splash,
    /// Oops! All 6s, hand size, hands, discards, debt, free rerolls, ...). Never copied.
    pub modifiers: Option<ModifierFn>,
    /// The joker entered the joker area (bought, created, duplicated): pick random targets
    /// (Ancient, Idol, Castle, Mail-In Rebate, To Do List), record offsets (Loyalty Card).
    pub on_acquire: Option<EventFn>,
    /// A blind was selected; `ctx.run().blind` tells which (Riff-raff, Cartomancer, Madness,
    /// Ceremonial Dagger, Burglar, Marble Joker, Chicot).
    pub on_blind_select: Option<EventFn>,
    /// The first hand of the round was drawn (Certificate).
    pub on_first_hand_drawn: Option<EventFn>,
    /// A blind was skipped (Throwback-like per-joker counters).
    pub on_blind_skip: Option<EventFn>,
    /// A discard action (Burnt, Faceless, Castle, Mail-In Rebate, Yorick, Hit the Road, Green
    /// Joker, Ramen, Trading Card).
    pub on_discard: Option<DiscardFn>,
    /// Playing cards were destroyed (Glass Joker, Canio).
    pub on_cards_destroyed: Option<CardsFn>,
    /// Playing cards were added to the deck (Hologram).
    pub on_cards_added: Option<CardsFn>,
    /// End of round (Golden Joker, Cloud 9, Rocket, Egg, Gift Card, Popcorn, Gros Michel, Hit
    /// the Road reset, Campfire reset, target re-rolls, Mr. Bones). Never copied.
    pub on_round_end: Option<RoundEndFn>,
    /// Entering the shop.
    pub on_shop_enter: Option<EventFn>,
    /// Leaving the shop (Perkeo).
    pub on_shop_exit: Option<EventFn>,
    /// Shop reroll (Flash Card).
    pub on_reroll: Option<EventFn>,
    /// This joker is being sold (Luchador, Diet Cola, Invisible Joker).
    pub on_sell_self: Option<EventFn>,
    /// Another joker or a consumable is being sold (Campfire).
    pub on_sell_other: Option<SellFn>,
    /// Booster pack opened (Hallucination).
    pub on_pack_open: Option<PackFn>,
    /// Booster pack skipped (Red Card).
    pub on_pack_skip: Option<PackFn>,
    /// Consumable used (Constellation, Fortune Teller-like counters).
    pub on_consumable_used: Option<ConsumableFn>,
}

impl Hooks {
    /// No hooks at all. Use as `Hooks { joker_main: Some(f), ..Hooks::NONE }`.
    pub const NONE: Hooks = Hooks {
        before_scoring: None,
        on_card_scored: None,
        card_retriggers: None,
        on_held: None,
        held_retriggers: None,
        joker_main: None,
        on_other_joker: None,
        after_scoring: None,
        on_invalid_hand: None,
        modifiers: None,
        on_acquire: None,
        on_blind_select: None,
        on_first_hand_drawn: None,
        on_blind_skip: None,
        on_discard: None,
        on_cards_destroyed: None,
        on_cards_added: None,
        on_round_end: None,
        on_shop_enter: None,
        on_shop_exit: None,
        on_reroll: None,
        on_sell_self: None,
        on_sell_other: None,
        on_pack_open: None,
        on_pack_skip: None,
        on_consumable_used: None,
    };
}

// ---------------------------------------------------------------------------------------------
// Read-only run view
// ---------------------------------------------------------------------------------------------

/// Kind of the current blind (defined in `blinds`, re-exported for hook authors).
pub use crate::blinds::BlindKind;

/// Boss-blind facts relevant to jokers and scoring. Placeholder until Phase 4 (`BossId`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BossView {
    /// Placeholder boss id (0 = none). TODO(orchestrator): `Option<BossId>` in Phase 4.
    pub id: u16,
    /// The current blind is a boss whose effect is active (not disabled by Chicot/Luchador).
    pub active: bool,
    /// The Flint: base chips and mult are halved, rounded up (scoring step 1).
    pub halve_base: bool,
    /// The boss ability was triggered by the current hand (Matador). Per hand.
    pub triggered: bool,
    /// The boss rejects the current hand (Psychic/Eye/Mouth): it scores 0 and fires no scoring
    /// effects (only `on_invalid_hand` and `after_scoring`). Per hand.
    pub hand_invalid: bool,
}

/// Run-long counters some jokers read (Throwback, Fortune Teller, Satellite, ...).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RunStats {
    /// Blinds skipped this run (Throwback).
    pub blinds_skipped: u32,
    /// Tarot cards used this run (Fortune Teller).
    pub tarots_used: u32,
    /// Planet cards used this run.
    pub planets_used: u32,
    /// Distinct Planet cards used this run (Satellite).
    pub unique_planets_used: u32,
    /// Boss blinds defeated this run.
    pub bosses_defeated: u32,
}

/// Aggregate counts over the FULL deck (all piles) plus the draw-pile size. Phase 2A keeps it
/// up to date (or rebuilds it with [`DeckCounts::from_deck`] before scoring/events).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeckCounts {
    /// Cards in the full deck.
    pub total: u32,
    /// Deck size at the start of the run (Erosion).
    pub starting_size: u32,
    /// Cards remaining in the draw pile (Blue Joker).
    pub draw_pile: u32,
    /// Count per enhancement, indexed by `Enhancement::index()` (Steel/Stone Joker, ...).
    pub by_enhancement: [u32; Enhancement::COUNT],
    /// Count per printed rank (non-Stone cards), indexed by `Rank::index()` (Cloud 9).
    pub by_rank: [u32; Rank::COUNT],
    /// Count per printed suit (non-Stone cards), indexed by `Suit::index()`.
    pub by_suit: [u32; Suit::COUNT],
    /// Count per seal, indexed by `Seal::index()`.
    pub by_seal: [u32; Seal::COUNT],
}

impl Default for DeckCounts {
    fn default() -> Self {
        DeckCounts {
            total: 0,
            starting_size: 52,
            draw_pile: 0,
            by_enhancement: [0; Enhancement::COUNT],
            by_rank: [0; Rank::COUNT],
            by_suit: [0; Suit::COUNT],
            by_seal: [0; Seal::COUNT],
        }
    }
}

impl DeckCounts {
    /// Count `cards`; `draw_pile` and `starting_size` are given separately.
    pub fn from_cards<'c>(
        cards: impl IntoIterator<Item = &'c Card>,
        draw_pile: u32,
        starting_size: u32,
    ) -> Self {
        let mut d = DeckCounts {
            draw_pile,
            starting_size,
            ..DeckCounts::default()
        };
        for c in cards {
            d.add(c);
        }
        d
    }

    /// Count every card of a [`Deck`] (all piles).
    pub fn from_deck(deck: &Deck, starting_size: u32) -> Self {
        Self::from_cards(deck.iter_all(), deck.draw_pile.len() as u32, starting_size)
    }

    /// Add one card to the counts.
    pub fn add(&mut self, c: &Card) {
        self.total += 1;
        self.by_enhancement[c.enhancement.index()] += 1;
        self.by_seal[c.seal.index()] += 1;
        if !c.is_stone() {
            self.by_rank[c.rank.index()] += 1;
            self.by_suit[c.suit.index()] += 1;
        }
    }

    /// Remove one card from the counts (saturating).
    pub fn remove(&mut self, c: &Card) {
        self.total = self.total.saturating_sub(1);
        let e = &mut self.by_enhancement[c.enhancement.index()];
        *e = e.saturating_sub(1);
        let s = &mut self.by_seal[c.seal.index()];
        *s = s.saturating_sub(1);
        if !c.is_stone() {
            let r = &mut self.by_rank[c.rank.index()];
            *r = r.saturating_sub(1);
            let s = &mut self.by_suit[c.suit.index()];
            *s = s.saturating_sub(1);
        }
    }

    /// Cards with any enhancement (Driver's License).
    pub fn enhanced(&self) -> u32 {
        self.total
            .saturating_sub(self.by_enhancement[Enhancement::None.index()])
    }

    /// Count of one enhancement.
    pub fn enhancement(&self, e: Enhancement) -> u32 {
        self.by_enhancement[e.index()]
    }

    /// Count of one printed rank (non-Stone).
    pub fn rank(&self, r: Rank) -> u32 {
        self.by_rank[r.index()]
    }
}

/// Kind of consumable. Placeholder until Phase 4.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConsumableKind {
    /// Tarot card.
    Tarot,
    /// Planet card.
    Planet,
    /// Spectral card.
    Spectral,
}

/// What jokers/scoring may know about one consumable. Placeholder until Phase 4 defines the
/// consumable ids; Phase 4 maps its types onto this view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConsumableView {
    /// Tarot / Planet / Spectral.
    pub kind: ConsumableKind,
    /// Placeholder id within its kind. TODO(orchestrator): typed id in Phase 4.
    pub id: u16,
    /// For Planets: the hand type it levels (Observatory, Satellite, Constellation).
    pub planet_hand: Option<HandType>,
    /// Negative edition (does not take a slot).
    pub negative: bool,
}

/// Booster pack kind. Placeholder until Phase 5.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PackKind {
    /// Tarots.
    Arcana,
    /// Planets.
    Celestial,
    /// Spectrals.
    Spectral,
    /// Playing cards.
    Standard,
    /// Jokers.
    Buffoon,
}

/// Read-only facts about the run that every hook can see (`ctx.run()`). A plain struct of
/// values and slices; Phase 2A fills it from `GameState` before each scoring call / event.
///
/// Counts are taken AFTER the triggering action was paid for: while scoring a hand,
/// `hands_left` is already decremented and `hands_played_*` / `hand_plays_*` already include
/// the hand being scored; during a discard, `discards_left` is decremented and
/// `discards_used_round` includes it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RunView<'a> {
    /// Money before this hand/event (hooks see `ctx.money()`, which adds pending gains).
    pub money: i32,
    /// Ante (1-based).
    pub ante: u32,
    /// Round number in the run (1-based).
    pub round: u32,
    /// Current (or just selected) blind kind.
    pub blind: BlindKind,
    /// Boss facts (placeholder).
    pub boss: BossView,
    /// Hands left this round, excluding the hand being scored (0 = this is the final hand).
    pub hands_left: u32,
    /// Discards left this round.
    pub discards_left: u32,
    /// Hands played this round, including the current one.
    pub hands_played_round: u32,
    /// Discards used this round, including the current one (first discard = 1).
    pub discards_used_round: u32,
    /// Hands played this run, including the current one.
    pub hands_played_run: u32,
    /// Level of every hand type, indexed by `HandType::index()` (level 1 = base).
    pub hand_levels: [u32; HandType::COUNT],
    /// Times each hand type was played this run, including the current hand (Supernova).
    pub hand_plays_run: [u32; HandType::COUNT],
    /// Times each hand type was played this round, including the current hand (Card Sharp).
    pub hand_plays_round: [u32; HandType::COUNT],
    /// Joker slot capacity, including +1 per Negative joker (Joker Stencil).
    pub joker_slots: u32,
    /// Consumable slot capacity.
    pub consumable_slots: u32,
    /// Held consumables, left to right (Observatory, Perkeo).
    pub consumables: &'a [ConsumableView],
    /// Full-deck counts and draw-pile size.
    pub deck: DeckCounts,
    /// Active rule flags (from [`collect_modifiers`]); also passed to `hand_eval::evaluate`.
    pub rules: RuleFlags,
    /// Probability numerator multiplier (1, Oops! All 6s doubles per copy).
    pub prob_mult: u32,
    /// Observatory voucher owned (scoring step 7).
    pub observatory: bool,
    /// Plasma Deck (scoring step 7: balance chips and mult).
    pub plasma_deck: bool,
    /// Run-long counters.
    pub stats: RunStats,
}

impl Default for RunView<'_> {
    /// Start-of-run defaults: ante 1, round 1, every hand level 1, 5 joker slots, 2 consumable
    /// slots, `prob_mult` 1, empty deck counts.
    fn default() -> Self {
        RunView {
            money: 0,
            ante: 1,
            round: 1,
            blind: BlindKind::Small,
            boss: BossView::default(),
            hands_left: 0,
            discards_left: 0,
            hands_played_round: 0,
            discards_used_round: 0,
            hands_played_run: 0,
            hand_levels: [1; HandType::COUNT],
            hand_plays_run: [0; HandType::COUNT],
            hand_plays_round: [0; HandType::COUNT],
            joker_slots: 5,
            consumable_slots: 2,
            consumables: &[],
            deck: DeckCounts::default(),
            rules: RuleFlags::default(),
            prob_mult: 1,
            observatory: false,
            plasma_deck: false,
            stats: RunStats::default(),
        }
    }
}

impl RunView<'_> {
    /// Level of a hand type (at least 1).
    pub fn hand_level(&self, ht: HandType) -> u32 {
        self.hand_levels[ht.index()].max(1)
    }

    /// The hand being scored is the last one of the round (Dusk, Acrobat).
    pub fn is_final_hand(&self) -> bool {
        self.hands_left == 0
    }

    /// Face card under joker semantics: J/Q/K, every card with Pareidolia (Stone included, per
    /// docs/mechanics/cards.md); debuffed cards are never face cards (balatrowiki.org/w/Ride_the_Bus:
    /// debuffed face cards do not reset it).
    pub fn is_face(&self, card: &Card) -> bool {
        !card.debuffed && (self.rules.pareidolia || (!card.is_stone() && card.rank.is_face_rank()))
    }

    /// Suit test with Wild/Smeared/Stone rules (`Card::has_suit`).
    pub fn has_suit(&self, card: &Card, suit: Suit) -> bool {
        card.has_suit(suit, &self.rules)
    }
}

// ---------------------------------------------------------------------------------------------
// Side effects
// ---------------------------------------------------------------------------------------------

/// What a `SellValue` side effect targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SellTarget {
    /// Every joker (Gift Card).
    AllJokers,
    /// Every held consumable (Gift Card).
    AllConsumables,
}

/// Run-level flags set by jokers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RunFlag {
    /// Gros Michel went extinct: Cavendish may now appear in the shop.
    GrosMichelExtinct,
}

/// A deferred change requested by a hook. Queued in order; the game layer (Phase 2A) applies
/// the queue after the scoring call / event, in order, and fires follow-up events
/// (`fire_cards_destroyed`, `fire_cards_added`, ...). Anything that changes a collection other
/// than the hook's own joker state, the played cards, chips/mult or money is deferred.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SideEffect {
    /// Persist a hand level change (Space Joker, Burnt Joker). During `before_scoring` the
    /// current hand's level is ALSO raised immediately for this hand's base values.
    LevelUpHand {
        /// Hand type.
        hand: HandType,
        /// Levels to add (may be negative).
        levels: i32,
    },
    /// Create a random consumable of `kind` (8 Ball, Cartomancer, Séance, Vagabond, Sixth
    /// Sense, Hallucination, Superposition). Non-negative ones need a free slot at application
    /// time; dropped otherwise.
    CreateConsumable {
        /// Tarot / Planet / Spectral.
        kind: ConsumableKind,
        /// Negative edition (needs no slot).
        negative: bool,
    },
    /// Create the Planet for `hand` (Blue Seal). Needs a free consumable slot.
    CreatePlanetFor {
        /// Hand type whose Planet is created.
        hand: HandType,
    },
    /// Copy the held consumable at `index` (Perkeo). Negative copies need no slot.
    CopyConsumable {
        /// Index into `RunView::consumables`.
        index: usize,
        /// Copy is Negative.
        negative: bool,
    },
    /// Create a random joker (Riff-raff: Common). Needs a free joker slot; dropped otherwise.
    CreateJoker {
        /// Rarity, `None` = normal shop odds.
        rarity: Option<Rarity>,
    },
    /// Duplicate the joker with `uid` (Invisible Joker). Needs a free slot.
    CopyJoker {
        /// Joker to copy.
        uid: u32,
        /// Remove Negative from the copy.
        strip_negative: bool,
    },
    /// Destroy the joker with `uid` (self-destruct, Ceremonial Dagger, Madness). Eternal
    /// jokers are never destroyed (the game layer checks). Idempotent.
    DestroyJoker {
        /// Joker uid.
        uid: u32,
    },
    /// Destroy the playing card with `uid` (Glass break, Sixth Sense, Trading Card).
    /// Idempotent. The game layer then fires `fire_cards_destroyed`.
    DestroyCard {
        /// Card uid.
        uid: u32,
    },
    /// Add a playing card (DNA, Marble Joker, Certificate). `card.uid` is ignored; the game
    /// layer allocates a fresh one, then fires `fire_cards_added`.
    AddCard {
        /// Card template.
        card: Card,
        /// Destination pile (`Pile::Hand` for DNA/Certificate, `Pile::Draw` for Marble).
        to: Pile,
    },
    /// Add `amount` to the sell value of a group (Gift Card).
    AddSellValue {
        /// Group.
        target: SellTarget,
        /// Dollars.
        amount: i32,
    },
    /// Add hands to the current round (Burglar).
    AddHands(i32),
    /// Add discards to the current round.
    AddDiscards(i32),
    /// Set the current round's discards to 0 (Burglar).
    ClearDiscards,
    /// Create a tag (Diet Cola: Double Tag). Placeholder id until Phase 5.
    CreateTag {
        /// Placeholder tag id. TODO(orchestrator): typed `TagId` in Phase 5.
        id: u16,
    },
    /// Disable the current boss blind (Chicot, Luchador).
    DisableBoss,
    /// Prevent the game over of a lost round (Mr. Bones).
    PreventDeath,
    /// Set a run flag.
    SetRunFlag(RunFlag),
}

// ---------------------------------------------------------------------------------------------
// Passive modifiers
// ---------------------------------------------------------------------------------------------

/// Passive contributions of all non-debuffed jokers (`Hooks::modifiers`). Deltas start at 0,
/// `prob_mult` at 1. The game layer recomputes this whenever the joker list, an edition or a
/// debuff changes, and feeds `rules` to `hand_eval::evaluate` and `rules`/`prob_mult` to
/// [`RunView`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Modifiers {
    /// Four Fingers, Shortcut, Smeared, Pareidolia, Splash.
    pub rules: RuleFlags,
    /// Probability numerator multiplier (Oops! All 6s: `*= 2` per copy).
    pub prob_mult: u32,
    /// Hand size delta (Juggler +1, Turtle Bean, Stuntman -2, Merry Andy -1, Troubadour +2).
    pub hand_size: i32,
    /// Hands per round delta (Troubadour -1).
    pub hands_per_round: i32,
    /// Discards per round delta (Drunkard +1, Merry Andy +3).
    pub discards_per_round: i32,
    /// Joker slot delta (reserved; Negative edition is handled by the game layer).
    pub joker_slots: i32,
    /// Consumable slot delta (reserved).
    pub consumable_slots: i32,
    /// How far money may go below 0 (Credit Card +20).
    pub debt_limit: i32,
    /// Free rerolls per shop (Chaos the Clown +1).
    pub free_rerolls: u32,
    /// Planets and Celestial packs are free in the shop (Astronomer).
    pub free_planets: bool,
    /// Duplicate cards may appear in shops/packs (Showman).
    pub allow_duplicates: bool,
    /// Extra interest dollars per $5 at end of round (To the Moon +1).
    pub interest_per_5: i32,
}

impl Default for Modifiers {
    fn default() -> Self {
        Modifiers {
            rules: RuleFlags::default(),
            prob_mult: 1,
            hand_size: 0,
            hands_per_round: 0,
            discards_per_round: 0,
            joker_slots: 0,
            consumable_slots: 0,
            debt_limit: 0,
            free_rerolls: 0,
            free_planets: false,
            allow_duplicates: false,
            interest_per_5: 0,
        }
    }
}

/// Fold the `modifiers` hooks of all non-debuffed jokers, left to right. Copiers never copy
/// passive modifiers (balatrowiki.org/w/Blueprint).
pub fn collect_modifiers(jokers: &[JokerInstance]) -> Modifiers {
    let mut m = Modifiers::default();
    for j in jokers.iter().filter(|j| !j.debuffed) {
        if let Some(f) = try_def(j.id).and_then(|d| d.hooks.modifiers) {
            f(j, &mut m);
        }
    }
    m
}

// ---------------------------------------------------------------------------------------------
// Copier resolution
// ---------------------------------------------------------------------------------------------

/// The slot whose hooks run at `slot`: `slot` itself for a normal joker; for a copier, the end
/// of its copy chain. `None` if the joker at `slot` is debuffed or unknown, or the copy chain
/// ends at nothing (no target, a cycle, a debuffed joker, or a Blueprint-incompatible joker).
pub fn resolve_source(jokers: &[JokerInstance], slot: usize) -> Option<usize> {
    let first = jokers.get(slot)?;
    if first.debuffed {
        return None;
    }
    let mut cur = slot;
    // A chain longer than the number of jokers must revisit a slot: cycle.
    for _ in 0..=jokers.len() {
        let d = try_def(jokers[cur].id)?;
        let Some(copy) = d.copies else {
            return Some(cur);
        };
        let target = copy(cur, jokers.len())?;
        let t = jokers.get(target)?;
        if target == cur || t.debuffed || !try_def(t.id)?.blueprint_compatible {
            return None;
        }
        cur = target;
    }
    None
}

// ---------------------------------------------------------------------------------------------
// Non-scoring events
// ---------------------------------------------------------------------------------------------

/// Payload of a discard event.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiscardInfo<'a> {
    /// The discarded cards, left to right (snapshots taken before they moved).
    pub cards: &'a [Card],
    /// Hand type of the discarded cards (`hand_eval::evaluate`, computed by the game layer;
    /// Burnt Joker levels it).
    pub hand_type: HandType,
}

/// Payload of the end-of-round event.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoundEndInfo {
    /// The blind was beaten. `false` = the run is about to end (Mr. Bones may save it).
    pub won: bool,
    /// Chips scored this round.
    pub round_score: f64,
    /// Chips required by the blind.
    pub target: f64,
}

impl RoundEndInfo {
    /// A boss blind was defeated (Rocket, Campfire reset). Needs the run's blind kind.
    pub fn boss_defeated(&self, run: &RunView<'_>) -> bool {
        self.won && run.blind == BlindKind::Boss
    }
}

/// Something being sold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SoldItem {
    /// A joker (state as of the sale).
    Joker(JokerInstance),
    /// A consumable.
    Consumable(ConsumableView),
}

/// Who is executing the current hook.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CallInfo {
    /// Slot of the joker whose ability runs (the copy target for copiers).
    pub slot: usize,
    /// Slot where the hook executes (the copier's slot for copies; equal to `slot` otherwise).
    pub exec_slot: usize,
}

impl CallInfo {
    /// The hook runs on behalf of Blueprint/Brainstorm.
    pub fn copied(&self) -> bool {
        self.slot != self.exec_slot
    }
}

/// Everything the game layer passes to a `fire_*` event dispatcher.
pub struct EventInput<'a> {
    /// The joker area, left to right. Hooks mutate only their own state.
    pub jokers: &'a mut [JokerInstance],
    /// Run facts.
    pub run: &'a RunView<'a>,
    /// Full deck (Idol / Mail-In Rebate / Castle pick targets from it).
    pub deck: &'a Deck,
    /// RNG (hooks use `RngKey::Jokers` through `ctx.chance` / `ctx.rng_range`).
    pub rng: &'a mut GameRng,
    /// Side-effect queue (appended to, never cleared by the dispatcher).
    pub effects: &'a mut Vec<SideEffect>,
}

/// Result of an event dispatch.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EventResult {
    /// Money gained/lost by hooks (apply to the run's money).
    pub money_delta: i32,
}

/// Context of a non-scoring event hook. Phase 2A may add fields (always through accessors).
///
/// What it promises: read-only [`RunView`], the full [`Deck`], all jokers (read-only), the
/// running money (`money()` = run money + gains in this dispatch), RNG, the side-effect queue,
/// and the call info (`slot`, `copied`).
pub struct EventCtx<'a> {
    run: &'a RunView<'a>,
    deck: &'a Deck,
    jokers: &'a [JokerInstance],
    rng: &'a mut GameRng,
    effects: &'a mut Vec<SideEffect>,
    money_delta: i32,
    call: CallInfo,
}

impl<'a> EventCtx<'a> {
    /// Run facts.
    pub fn run(&self) -> &RunView<'a> {
        self.run
    }

    /// Full deck (all piles).
    pub fn deck(&self) -> &Deck {
        self.deck
    }

    /// All jokers as they were when the event started (a hook's own changes are only in `j`).
    pub fn jokers(&self) -> &[JokerInstance] {
        self.jokers
    }

    /// Definition of the joker at `slot`.
    pub fn joker_def(&self, slot: usize) -> Option<&'static JokerDef> {
        self.jokers.get(slot).and_then(|j| try_def(j.id))
    }

    /// Slot of the joker whose ability runs (the copy target for copies).
    pub fn slot(&self) -> usize {
        self.call.slot
    }

    /// Slot where the hook executes (the copier's own slot for copies).
    pub fn exec_slot(&self) -> usize {
        self.call.exec_slot
    }

    /// The hook runs on behalf of Blueprint/Brainstorm. State changes to `j` are discarded
    /// automatically; check this before effects that must not be duplicated.
    pub fn copied(&self) -> bool {
        self.call.copied()
    }

    /// Money including gains made earlier in this dispatch.
    pub fn money(&self) -> i32 {
        self.run.money.saturating_add(self.money_delta)
    }

    /// Gain (or lose) money.
    pub fn add_money(&mut self, dollars: i32) {
        self.money_delta = self.money_delta.saturating_add(dollars);
    }

    /// Queue a side effect.
    pub fn push_effect(&mut self, e: SideEffect) {
        self.effects.push(e);
    }

    /// Side effects queued so far in this dispatch chain (including earlier dispatches that
    /// share the queue).
    pub fn effects(&self) -> &[SideEffect] {
        self.effects
    }

    /// Queue a hand level-up (Burnt Joker).
    pub fn level_up_hand(&mut self, hand: HandType, levels: i32) {
        self.effects.push(SideEffect::LevelUpHand { hand, levels });
    }

    /// Destroy the joker whose ability runs. No-op for copies (a copy never destroys its
    /// target).
    pub fn destroy_self(&mut self) {
        if !self.copied()
            && let Some(j) = self.jokers.get(self.call.slot)
        {
            self.effects.push(SideEffect::DestroyJoker { uid: j.uid });
        }
    }

    /// Destroy the joker at `slot` (Ceremonial Dagger, Madness). Eternal jokers are refused
    /// (returns false). Later hooks of the same dispatch skip a doomed joker.
    pub fn destroy_joker(&mut self, slot: usize) -> bool {
        match self.jokers.get(slot) {
            Some(j) if !j.stickers.eternal => {
                self.effects.push(SideEffect::DestroyJoker { uid: j.uid });
                true
            }
            _ => false,
        }
    }

    /// A `DestroyJoker` for the joker at `slot` is already queued.
    pub fn is_doomed(&self, slot: usize) -> bool {
        self.jokers
            .get(slot)
            .is_some_and(|j| is_doomed_uid(self.effects, j.uid))
    }

    /// Probability roll with the run's probability multiplier, on `RngKey::Jokers`.
    pub fn chance(&mut self, numerator: u32, denominator: u32) -> bool {
        self.rng
            .chance(RngKey::Jokers, numerator, denominator, self.run.prob_mult)
    }

    /// Uniform integer in `[lo, hi_exclusive)` on `RngKey::Jokers`.
    pub fn rng_range(&mut self, lo: usize, hi_exclusive: usize) -> usize {
        self.rng.range(RngKey::Jokers, lo, hi_exclusive)
    }

    /// Raw RNG access. Joker hooks must only use `RngKey::Jokers`.
    pub fn rng(&mut self) -> &mut GameRng {
        self.rng
    }

    /// Face card under joker semantics (see [`RunView::is_face`]).
    pub fn is_face(&self, card: &Card) -> bool {
        self.run.is_face(card)
    }

    /// Suit test under the active rules.
    pub fn has_suit(&self, card: &Card, suit: Suit) -> bool {
        self.run.has_suit(card, suit)
    }
}

fn is_doomed_uid(effects: &[SideEffect], uid: u32) -> bool {
    effects
        .iter()
        .any(|e| matches!(e, SideEffect::DestroyJoker { uid: u } if *u == uid))
}

/// Which slots an event reaches.
#[derive(Clone, Copy)]
enum Reach {
    /// Every joker, left to right.
    All,
    /// Only this slot (on_acquire, on_sell_self).
    Only(usize),
    /// Every joker except this slot (on_sell_other).
    Except(usize),
}

/// Generic event dispatcher: left to right, skipping debuffed and doomed jokers; resolves
/// copiers when `copies` is true; runs each hook on a scratch copy of the source joker's state
/// and writes it back unless the hook ran as a copy.
fn dispatch<F: Copy>(
    inp: EventInput<'_>,
    reach: Reach,
    copies: bool,
    get: impl Fn(&Hooks) -> Option<F>,
    mut call: impl FnMut(F, &mut JokerInstance, &mut EventCtx<'_>),
) -> EventResult {
    let EventInput {
        jokers,
        run,
        deck,
        rng,
        effects,
    } = inp;
    let mut money_delta = 0i32;
    for slot in 0..jokers.len() {
        match reach {
            Reach::Only(s) if s != slot => continue,
            Reach::Except(s) if s == slot => continue,
            _ => {}
        }
        let me = jokers[slot];
        if me.debuffed || is_doomed_uid(effects, me.uid) {
            continue;
        }
        let src = if copies {
            match resolve_source(jokers, slot) {
                Some(s) => s,
                None => continue,
            }
        } else {
            slot
        };
        let Some(f) = try_def(jokers[src].id).and_then(|d| get(&d.hooks)) else {
            continue;
        };
        let mut inst = jokers[src];
        let mut ctx = EventCtx {
            run,
            deck,
            jokers,
            rng,
            effects,
            money_delta,
            call: CallInfo {
                slot: src,
                exec_slot: slot,
            },
        };
        call(f, &mut inst, &mut ctx);
        money_delta = ctx.money_delta;
        if src == slot {
            jokers[slot] = inst;
        }
    }
    EventResult { money_delta }
}

/// The joker at `slot` entered the joker area (bought, created, duplicated). Not copied.
pub fn fire_acquire(inp: EventInput<'_>, slot: usize) -> EventResult {
    dispatch(
        inp,
        Reach::Only(slot),
        false,
        |h| h.on_acquire,
        |f, j, c| f(j, c),
    )
}

/// A blind was selected (`run.blind` = its kind). Fire before drawing the first hand.
pub fn fire_blind_select(inp: EventInput<'_>) -> EventResult {
    dispatch(
        inp,
        Reach::All,
        true,
        |h| h.on_blind_select,
        |f, j, c| f(j, c),
    )
}

/// The first hand of the round was drawn.
pub fn fire_first_hand_drawn(inp: EventInput<'_>) -> EventResult {
    dispatch(
        inp,
        Reach::All,
        true,
        |h| h.on_first_hand_drawn,
        |f, j, c| f(j, c),
    )
}

/// A blind was skipped (after `stats.blinds_skipped` was incremented).
pub fn fire_blind_skip(inp: EventInput<'_>) -> EventResult {
    dispatch(
        inp,
        Reach::All,
        true,
        |h| h.on_blind_skip,
        |f, j, c| f(j, c),
    )
}

/// A discard action (after `discards_left`/`discards_used_round` were updated).
pub fn fire_discard(inp: EventInput<'_>, info: &DiscardInfo<'_>) -> EventResult {
    dispatch(
        inp,
        Reach::All,
        true,
        |h| h.on_discard,
        |f, j, c| f(j, c, info),
    )
}

/// Playing cards were destroyed (after they were removed from the deck).
pub fn fire_cards_destroyed(inp: EventInput<'_>, cards: &[Card]) -> EventResult {
    dispatch(
        inp,
        Reach::All,
        true,
        |h| h.on_cards_destroyed,
        |f, j, c| f(j, c, cards),
    )
}

/// Playing cards were added to the deck.
pub fn fire_cards_added(inp: EventInput<'_>, cards: &[Card]) -> EventResult {
    dispatch(
        inp,
        Reach::All,
        true,
        |h| h.on_cards_added,
        |f, j, c| f(j, c, cards),
    )
}

/// End of round. Not copied (Blueprint does not copy end-of-round effects). On a lost round,
/// fire it with `won = false` before declaring game over; a queued `PreventDeath` saves the run.
pub fn fire_round_end(inp: EventInput<'_>, info: &RoundEndInfo) -> EventResult {
    dispatch(
        inp,
        Reach::All,
        false,
        |h| h.on_round_end,
        |f, j, c| f(j, c, info),
    )
}

/// Entering the shop.
pub fn fire_shop_enter(inp: EventInput<'_>) -> EventResult {
    dispatch(
        inp,
        Reach::All,
        true,
        |h| h.on_shop_enter,
        |f, j, c| f(j, c),
    )
}

/// Leaving the shop.
pub fn fire_shop_exit(inp: EventInput<'_>) -> EventResult {
    dispatch(inp, Reach::All, true, |h| h.on_shop_exit, |f, j, c| f(j, c))
}

/// A shop reroll was bought.
pub fn fire_reroll(inp: EventInput<'_>) -> EventResult {
    dispatch(inp, Reach::All, true, |h| h.on_reroll, |f, j, c| f(j, c))
}

/// The joker at `slot` is being sold (call before removing it). A sold copier runs its
/// target's `on_sell_self` (Blueprint + Luchador works).
pub fn fire_sell_self(inp: EventInput<'_>, slot: usize) -> EventResult {
    dispatch(
        inp,
        Reach::Only(slot),
        true,
        |h| h.on_sell_self,
        |f, j, c| f(j, c),
    )
}

/// An item is being sold; `sold_slot` = the sold joker's slot (skipped), `None` for a
/// consumable.
pub fn fire_sell_other(
    inp: EventInput<'_>,
    sold: &SoldItem,
    sold_slot: Option<usize>,
) -> EventResult {
    let reach = sold_slot.map_or(Reach::All, Reach::Except);
    dispatch(
        inp,
        reach,
        true,
        |h| h.on_sell_other,
        |f, j, c| f(j, c, sold),
    )
}

/// A booster pack was opened.
pub fn fire_pack_open(inp: EventInput<'_>, pack: PackKind) -> EventResult {
    dispatch(
        inp,
        Reach::All,
        true,
        |h| h.on_pack_open,
        |f, j, c| f(j, c, pack),
    )
}

/// A booster pack was skipped.
pub fn fire_pack_skip(inp: EventInput<'_>, pack: PackKind) -> EventResult {
    dispatch(
        inp,
        Reach::All,
        true,
        |h| h.on_pack_skip,
        |f, j, c| f(j, c, pack),
    )
}

/// A consumable was used (after its own effect).
pub fn fire_consumable_used(inp: EventInput<'_>, used: &ConsumableView) -> EventResult {
    dispatch(
        inp,
        Reach::All,
        true,
        |h| h.on_consumable_used,
        |f, j, c| f(j, c, used),
    )
}
