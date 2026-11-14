//! `GameState`: the single source of truth for a run. Plain data, cheap to `Clone` (small `Vec`s,
//! no boxed trait objects), `PartialEq` and serde-serializable (determinism checks, replays).
//!
//! The state machine that mutates it lives in [`crate::game`]; legal-action generation and
//! validation in [`crate::actions`]. This module holds the data, the config, derived values
//! (effective hand size, slots, blind target), the [`RunView`] builder and the invariants.

use crate::blinds::{self, BlindKind, MULT_BOSS_DEFAULT, StakeScaling};
use crate::data::prices::{INTEREST_CAP_BASE, STARTING_MONEY};
use crate::data::run_rules::{
    BASE_CONSUMABLE_SLOTS, BASE_DISCARDS, BASE_HAND_SIZE, BASE_HANDS, BASE_JOKER_SLOTS,
    MAX_HAND_SIZE, MIN_HAND_SIZE, MIN_HANDS_PER_ROUND, RED_DECK_EXTRA_DISCARDS,
    SHOWDOWN_ANTE_INTERVAL, STANDARD_DECK_SIZE, STARTING_ANTE, TAG_POOL_PLACEHOLDER,
};
use crate::deck::Deck;
use crate::enums::{Edition, HandType, Phase};
use crate::hooks::{
    BossView, ConsumableView, DeckCounts, Modifiers, RunFlag, RunStats, RunView, collect_modifiers,
};
use crate::joker::JokerInstance;
use crate::rng::{GameRng, RngKey};
use serde::{Deserialize, Serialize};

/// Starting deck. Only Red is implemented (prd.md §1: Red Deck + White Stake).
/// TODO(Phase 5E): move to `content/decks.rs` and add the other decks.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeckKind {
    /// +1 discard every round.
    #[default]
    Red,
}

/// Stake. Only White is implemented.
/// TODO(Phase 5E): move to `content/stakes.rs` and add the other stakes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StakeKind {
    /// Base difficulty.
    #[default]
    White,
}

impl StakeKind {
    /// Blind chip table of this stake.
    pub fn scaling(self) -> StakeScaling {
        match self {
            StakeKind::White => StakeScaling::White,
        }
    }

    /// Small blind pays no reward (Red Stake and above).
    pub fn no_small_reward(self) -> bool {
        match self {
            StakeKind::White => false,
        }
    }
}

/// Run options besides the seed. Extend with new fields (all must have a default).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GameConfig {
    /// Starting deck.
    pub deck: DeckKind,
    /// Stake.
    pub stake: StakeKind,
    /// Keep playing after the ante-8 boss is beaten (Endless Mode). Off by default: beating it
    /// ends the run with [`Outcome::Win`].
    pub endless: bool,
}

/// How a finished run ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Outcome {
    /// The ante-8 boss was beaten (Mr. Bones saves count, blinds.md §1).
    Win,
    /// A round was lost (out of hands below the target, or out of cards).
    Loss,
}

/// Placeholder skip-tag id (no effect yet). TODO(Phase 5C): typed `TagId` from `content/tags.rs`
/// (also `SideEffect::CreateTag`).
pub type TagId = u16;
/// Placeholder voucher id. TODO(Phase 5D): typed `VoucherId`.
pub type VoucherId = u16;

/// Bookkeeping counters (not all visible to hooks; [`RunStats`] is the hook-visible part).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct GameStats {
    /// Hook-visible run counters (Throwback, Fortune Teller, Satellite, ...).
    pub run: RunStats,
    /// Hands played this run (`RunView::hands_played_run`).
    pub hands_played: u32,
    /// Discards used this run.
    pub discards_used: u32,
    /// Rounds won (blinds beaten, including Mr. Bones saves).
    pub rounds_won: u32,
    /// Playing cards added to the deck this run.
    pub cards_added: u32,
    /// Playing cards destroyed this run.
    pub cards_destroyed: u32,
    /// Jokers sold this run.
    pub jokers_sold: u32,
    /// Best single-hand score this run.
    pub best_hand: f64,
    /// Side effects that were dropped: no room, not implemented yet, or past the cascade cap.
    pub effects_dropped: u32,
}

/// Placeholder for the shop of the current Shop phase. TODO(Phase 5A): shop slots, vouchers,
/// packs, reroll counter.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShopState {
    /// Rerolls bought in the current shop (reroll price; Phase 5A).
    pub rerolls: u32,
}

/// The whole run. Fields are public for observation encoding, tests and tools; only
/// [`crate::game::Game`] should mutate them during play. [`GameState::check_invariants`] states
/// what must always hold.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GameState {
    /// Options this run was created with.
    pub config: GameConfig,
    /// Current phase.
    pub phase: Phase,
    /// Set once the run is over (`phase` is `GameOver` or `Won`).
    pub outcome: Option<Outcome>,
    /// The ante-8 boss was beaten at some point (stays true in Endless Mode).
    pub won: bool,

    // --- progress --------------------------------------------------------------------------
    /// Ante (1-based; may drop to 0 or below later via Hieroglyph/Petroglyph).
    pub ante: i32,
    /// Rounds started this run (incremented when a blind is selected; skipped blinds do not
    /// count). 0 before the first blind.
    pub round: u32,
    /// The blind being played (Play) or the next one to select (BlindSelect, Shop).
    pub blind: BlindKind,
    /// Chips required to beat `blind`.
    pub target: f64,
    /// Tags offered for skipping this ante's Small and Big blind (rolled at ante start on
    /// `RngKey::Tags`, notes.md §4.1c).
    pub blind_tags: [TagId; 2],
    /// PLACEHOLDER boss id of this ante (0 = none). TODO(Phase 4): `Option<BossId>` chosen at
    /// ante start on `RngKey::Bosses`, plus per-boss state.
    pub boss_id: u16,
    /// The boss effect was disabled (Chicot, Luchador).
    pub boss_disabled: bool,

    // --- economy ---------------------------------------------------------------------------
    /// Money in dollars (may be negative only with debt allowance, `Modifiers::debt_limit`).
    pub money: i32,
    /// Interest cap in dollars (Seed Money / Money Tree raise it). TODO(Phase 5D): vouchers.
    pub interest_cap: i32,

    // --- round -----------------------------------------------------------------------------
    /// Hands left this round.
    pub hands_left: u32,
    /// Discards left this round.
    pub discards_left: u32,
    /// Hands played this round.
    pub hands_played_round: u32,
    /// Discards used this round.
    pub discards_used_round: u32,
    /// Chips scored this round.
    pub round_score: f64,
    /// Hand type of the last hand played this run (Blue Seal at round end).
    pub last_hand: Option<HandType>,

    // --- per-run base values (before joker modifiers) -------------------------------------
    /// Hands per round (deck/vouchers change it).
    pub base_hands: u32,
    /// Discards per round.
    pub base_discards: u32,
    /// Hand size.
    pub base_hand_size: u32,
    /// Joker slots, without Negative bonuses.
    pub joker_slots: u32,
    /// Consumable slots, without Negative bonuses.
    pub consumable_slots: u32,

    // --- collections -----------------------------------------------------------------------
    /// Joker area, left to right.
    pub jokers: Vec<JokerInstance>,
    /// Passive modifiers of `jokers` (cache of `collect_modifiers(&jokers)`; recomputed whenever
    /// jokers change).
    pub mods: Modifiers,
    /// Held consumables. PLACEHOLDER type until Phase 4 (`ConsumableInstance`).
    pub consumables: Vec<ConsumableView>,
    /// All playing cards of the run, in their piles. Outside a round every card is in
    /// `deck.draw_pile`; during a round: draw / hand / discard (`played` only inside a step).
    pub deck: Deck,
    /// Deck size at the start of the run (Erosion).
    pub starting_deck_size: u32,
    /// Level of every hand type, indexed by `HandType::index()` (1 = base).
    pub hand_levels: [u32; HandType::COUNT],
    /// Times each hand type was played this run.
    pub hand_plays_run: [u32; HandType::COUNT],
    /// Times each hand type was played this round.
    pub hand_plays_round: [u32; HandType::COUNT],
    /// Vouchers owned. PLACEHOLDER until Phase 5D.
    pub vouchers: Vec<VoucherId>,
    /// Tags held, oldest first. PLACEHOLDER (no effects) until Phase 5C.
    pub tags: Vec<TagId>,
    /// Run flags set by jokers (unique, in the order they were set).
    pub run_flags: Vec<RunFlag>,
    /// Shop state (Phase 5).
    pub shop: ShopState,

    // --- misc ------------------------------------------------------------------------------
    /// Counters.
    pub stats: GameStats,
    /// Next joker uid (joker uids are unique per run; card uids come from `deck`).
    pub next_joker_uid: u32,
    /// All randomness.
    pub rng: GameRng,
}

impl GameState {
    /// Fresh run: ante 1, Small blind selectable, $4, standard 52-card deck (unshuffled; the
    /// deck is shuffled when a blind is selected), tags rolled for ante 1.
    pub fn new(seed: u64, config: GameConfig) -> Self {
        let deck_extra_discards = match config.deck {
            DeckKind::Red => RED_DECK_EXTRA_DISCARDS,
        };
        let base_hands = BASE_HANDS;
        let base_discards = BASE_DISCARDS + deck_extra_discards;
        let mut s = GameState {
            config,
            phase: Phase::BlindSelect,
            outcome: None,
            won: false,
            ante: STARTING_ANTE,
            round: 0,
            blind: BlindKind::Small,
            target: 0.0,
            blind_tags: [0; 2],
            boss_id: 0,
            boss_disabled: false,
            money: STARTING_MONEY,
            interest_cap: INTEREST_CAP_BASE,
            hands_left: base_hands,
            discards_left: base_discards,
            hands_played_round: 0,
            discards_used_round: 0,
            round_score: 0.0,
            last_hand: None,
            base_hands,
            base_discards,
            base_hand_size: BASE_HAND_SIZE,
            joker_slots: BASE_JOKER_SLOTS,
            consumable_slots: BASE_CONSUMABLE_SLOTS,
            jokers: Vec::new(),
            mods: Modifiers::default(),
            consumables: Vec::new(),
            deck: Deck::standard(),
            starting_deck_size: STANDARD_DECK_SIZE,
            hand_levels: [1; HandType::COUNT],
            hand_plays_run: [0; HandType::COUNT],
            hand_plays_round: [0; HandType::COUNT],
            vouchers: Vec::new(),
            tags: Vec::new(),
            run_flags: Vec::new(),
            shop: ShopState::default(),
            stats: GameStats::default(),
            next_joker_uid: 0,
            rng: GameRng::new(seed),
        };
        s.start_ante();
        s
    }

    /// Ante setup: roll the Small/Big skip tags (and, from Phase 4, the boss), reset the blind
    /// to Small and refresh the target.
    pub(crate) fn start_ante(&mut self) {
        for t in &mut self.blind_tags {
            *t = self
                .rng
                .range(RngKey::Tags, 0, usize::from(TAG_POOL_PLACEHOLDER))
                as TagId;
        }
        // TODO(Phase 4): choose the boss on `RngKey::Bosses` (least-seen rule, notes.md §4.1c).
        self.boss_id = 0;
        self.boss_disabled = false;
        self.blind = BlindKind::Small;
        self.refresh_target();
    }

    /// Recompute `target` for the current ante and blind.
    pub(crate) fn refresh_target(&mut self) {
        self.target = blinds::requirement(
            self.ante,
            self.blind,
            self.boss_mult(),
            self.config.stake.scaling(),
        );
    }

    /// Boss blind multiplier. TODO(Phase 4): Wall x4 / Violet Vessel x6 / Needle x1 (Chicot
    /// reverts Wall and Vessel).
    pub fn boss_mult(&self) -> f64 {
        MULT_BOSS_DEFAULT
    }

    /// The current boss is a Showdown boss (ante % 8 == 0).
    pub fn is_showdown(&self) -> bool {
        self.blind == BlindKind::Boss && self.ante > 0 && self.ante % SHOWDOWN_ANTE_INTERVAL == 0
    }

    /// Recompute the passive-modifier cache. Call whenever jokers, their order, editions or
    /// debuffs change (hooks.md §15).
    pub fn refresh_modifiers(&mut self) {
        self.mods = collect_modifiers(&self.jokers);
    }

    /// Effective hand size: base + joker modifiers, clamped to `[MIN_HAND_SIZE, MAX_HAND_SIZE]`.
    pub fn hand_size(&self) -> u32 {
        clamp_delta(
            self.base_hand_size,
            self.mods.hand_size,
            MIN_HAND_SIZE,
            MAX_HAND_SIZE,
        )
    }

    /// Hands granted at round start: base + joker modifiers, at least `MIN_HANDS_PER_ROUND`.
    pub fn round_hands(&self) -> u32 {
        clamp_delta(
            self.base_hands,
            self.mods.hands_per_round,
            MIN_HANDS_PER_ROUND,
            u32::MAX,
        )
    }

    /// Discards granted at round start: base + joker modifiers, at least 0.
    pub fn round_discards(&self) -> u32 {
        clamp_delta(
            self.base_discards,
            self.mods.discards_per_round,
            0,
            u32::MAX,
        )
    }

    /// Joker capacity: base slots + 1 per Negative joker + modifier delta.
    pub fn joker_capacity(&self) -> u32 {
        let negatives = self
            .jokers
            .iter()
            .filter(|j| j.edition == Edition::Negative)
            .count() as u32;
        clamp_delta(
            self.joker_slots + negatives,
            self.mods.joker_slots,
            0,
            u32::MAX,
        )
    }

    /// Consumable capacity: base slots + 1 per Negative consumable + modifier delta.
    pub fn consumable_capacity(&self) -> u32 {
        let negatives = self.consumables.iter().filter(|c| c.negative).count() as u32;
        clamp_delta(
            self.consumable_slots + negatives,
            self.mods.consumable_slots,
            0,
            u32::MAX,
        )
    }

    /// A non-Negative joker fits.
    pub fn has_joker_room(&self) -> bool {
        (self.jokers.len() as u32) < self.joker_capacity()
    }

    /// A non-Negative consumable fits.
    pub fn has_consumable_room(&self) -> bool {
        (self.consumables.len() as u32) < self.consumable_capacity()
    }

    /// Lowest money allowed after a purchase (`-debt_limit`; Credit Card).
    pub fn money_floor(&self) -> i32 {
        -self.mods.debt_limit.max(0)
    }

    /// Spending `cost` keeps money at or above [`GameState::money_floor`].
    pub fn can_afford(&self, cost: i32) -> bool {
        self.money.saturating_sub(cost) >= self.money_floor()
    }

    /// The run is over.
    pub fn is_terminal(&self) -> bool {
        matches!(self.phase, Phase::GameOver | Phase::Won)
    }

    /// A run flag is set.
    pub fn has_flag(&self, f: RunFlag) -> bool {
        self.run_flags.contains(&f)
    }

    /// Allocate a joker uid.
    pub fn alloc_joker_uid(&mut self) -> u32 {
        let u = self.next_joker_uid;
        self.next_joker_uid += 1;
        u
    }

    /// Boss facts for hooks. Placeholder until Phase 4 (no boss has an effect yet).
    pub fn boss_view(&self) -> BossView {
        BossView {
            id: self.boss_id,
            active: self.blind == BlindKind::Boss && self.boss_id != 0 && !self.boss_disabled,
            ..BossView::default()
        }
    }

    /// Full-deck counts for hooks.
    pub fn deck_counts(&self) -> DeckCounts {
        DeckCounts::from_deck(&self.deck, self.starting_deck_size)
    }

    /// The [`RunView`] hooks see, built from this state.
    pub fn run_view(&self) -> RunView<'_> {
        self.run_view_with(&self.consumables)
    }

    /// [`GameState::run_view`] whose only borrow is `consumables` (normally
    /// `&self.consumables`), so the game layer can hold the view while mutably borrowing other
    /// fields (`jokers`, `deck`, `rng`) for a scoring call or an event.
    ///
    /// Count conventions (hooks.md §5.3) hold if the caller pays the action first: during
    /// scoring `hands_left` is already decremented and the play counters include the hand.
    pub fn run_view_with<'a>(&self, consumables: &'a [ConsumableView]) -> RunView<'a> {
        RunView {
            money: self.money,
            ante: u32::try_from(self.ante).unwrap_or(0),
            round: self.round,
            blind: self.blind,
            boss: self.boss_view(),
            hands_left: self.hands_left,
            discards_left: self.discards_left,
            hands_played_round: self.hands_played_round,
            discards_used_round: self.discards_used_round,
            hands_played_run: self.stats.hands_played,
            hand_levels: self.hand_levels,
            hand_plays_run: self.hand_plays_run,
            hand_plays_round: self.hand_plays_round,
            joker_slots: self.joker_capacity(),
            consumable_slots: self.consumable_capacity(),
            consumables,
            deck: self.deck_counts(),
            rules: self.mods.rules,
            prob_mult: self.mods.prob_mult,
            // TODO(Phase 5D): Observatory voucher. TODO(Phase 5E): Plasma Deck.
            observatory: false,
            plasma_deck: false,
            stats: self.stats.run,
        }
    }

    /// Check every invariant; `Err` names the first violation. `Game::step` runs this through
    /// `debug_assert!` after every successful action.
    pub fn check_invariants(&self) -> Result<(), String> {
        // Cards are conserved: start + added - destroyed.
        let expected = i64::from(self.starting_deck_size) + i64::from(self.stats.cards_added)
            - i64::from(self.stats.cards_destroyed);
        if self.deck.len() as i64 != expected {
            return Err(format!(
                "card count {} != start {} + added {} - destroyed {}",
                self.deck.len(),
                self.starting_deck_size,
                self.stats.cards_added,
                self.stats.cards_destroyed
            ));
        }
        let mut uids: Vec<u32> = self.deck.iter_all().map(|c| c.uid).collect();
        uids.sort_unstable();
        if uids.windows(2).any(|w| w[0] == w[1]) {
            return Err("duplicate card uid".into());
        }
        if uids.last().is_some_and(|&u| u >= self.deck.next_uid()) {
            return Err("card uid not allocated by the deck".into());
        }
        let mut juids: Vec<u32> = self.jokers.iter().map(|j| j.uid).collect();
        juids.sort_unstable();
        if juids.windows(2).any(|w| w[0] == w[1]) {
            return Err("duplicate joker uid".into());
        }
        if juids.last().is_some_and(|&u| u >= self.next_joker_uid) {
            return Err("joker uid not allocated".into());
        }
        if !self.deck.played.is_empty() {
            return Err("played pile not empty between steps".into());
        }
        match self.phase {
            Phase::Play => {
                if self.hands_left == 0 {
                    return Err("in Play with no hands left".into());
                }
                if self.deck.hand.is_empty() {
                    return Err("in Play with an empty hand".into());
                }
                if self.round_score >= self.target {
                    return Err("in Play with the blind already beaten".into());
                }
            }
            Phase::BlindSelect | Phase::Shop | Phase::PackOpen => {
                if !self.deck.hand.is_empty() || !self.deck.discard_pile.is_empty() {
                    return Err("cards outside the draw pile between rounds".into());
                }
            }
            Phase::GameOver | Phase::Won => {}
        }
        let expected_outcome = match self.phase {
            Phase::GameOver => Some(Outcome::Loss),
            Phase::Won => Some(Outcome::Win),
            _ => None,
        };
        if self.outcome != expected_outcome {
            return Err(format!(
                "outcome {:?} does not match phase {:?}",
                self.outcome, self.phase
            ));
        }
        // Money: purchases never go below the debt floor; nothing in Phase 2 spends money.
        // TODO(Phase 4): The Tooth / The Ox can push money below the floor; relax this then.
        if self.money < self.money_floor() {
            return Err(format!(
                "money {} below floor {}",
                self.money,
                self.money_floor()
            ));
        }
        if self.round_score.is_nan() || self.round_score < 0.0 {
            return Err("round score negative or NaN".into());
        }
        if self.mods != collect_modifiers(&self.jokers) {
            return Err("modifier cache out of date".into());
        }
        if self.jokers.len() as u32 > self.joker_capacity() {
            return Err("more jokers than joker slots".into());
        }
        if self.consumables.len() as u32 > self.consumable_capacity() {
            return Err("more consumables than consumable slots".into());
        }
        if self.hand_levels.contains(&0) {
            return Err("hand level 0".into());
        }
        if self
            .run_flags
            .iter()
            .enumerate()
            .any(|(i, f)| self.run_flags[..i].contains(f))
        {
            return Err("duplicate run flag".into());
        }
        Ok(())
    }
}

/// `base + delta` clamped to `[lo, hi]`.
fn clamp_delta(base: u32, delta: i32, lo: u32, hi: u32) -> u32 {
    let v = i64::from(base) + i64::from(delta);
    v.clamp(i64::from(lo), i64::from(hi)) as u32
}

/// Phase transitions allowed by the state machine (prd.md §3.1). `from == to` is allowed for
/// actions that stay in their phase (Play/Discard/Sell/Skip). PackOpen edges are for Phase 5.
pub fn transition_allowed(from: Phase, to: Phase) -> bool {
    use Phase::*;
    if from == to {
        return !matches!(from, GameOver | Won);
    }
    matches!(
        (from, to),
        (BlindSelect, Play)
            | (BlindSelect, GameOver) // nothing left to draw at round start
            | (BlindSelect, PackOpen) // pack tag opened on skip (Phase 5)
            | (Play, Shop)
            | (Play, GameOver)
            | (Play, Won)
            | (Shop, BlindSelect)
            | (Shop, PackOpen)
            | (PackOpen, Shop)
            | (PackOpen, BlindSelect)
    )
}
