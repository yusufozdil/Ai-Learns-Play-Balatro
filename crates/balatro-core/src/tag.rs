//! Skip tags: the static [`TagDef`] (hooks), the context hooks are written against
//! ([`TagCtx`]) and the game-layer runners. Ids: [`crate::content::ids::TagId`], definitions:
//! `content/tags.rs`. Guide: `docs/design/shop_packs_vouchers_tags.md`.
//!
//! A tag is gained by skipping a Small/Big blind (`Game::step(SkipBlind)`). Each hook returns
//! `true` when the tag is **consumed** (removed from `GameState::tags`) and `false` to stay held
//! for a later chance (an edition tag finds no base-edition joker in the shop). Held tags run
//! oldest first. Randomness is `RngKey::Tags`.

use crate::content::ids::TagId;
use crate::content::try_tag_def;
use crate::data::shop_weights::BOSS_REROLL_COST;
use crate::enums::{Edition, HandType, Rarity};
use crate::hooks::SideEffect;
use crate::packs::PackSpec;
use crate::rng::RngKey;
use crate::shop::{ShopCard, ShopCardKind, ShopVoucher, pick_voucher, roll_joker};
use crate::state::GameState;

/// A tag hook. Returns true if the tag is used up.
pub type TagFn = fn(&mut TagCtx<'_>) -> bool;

/// Static definition of one tag. Build with [`TagDef::new`] and struct update syntax:
///
/// ```ignore
/// TagDef {
///     on_gain: Some(economy_on_gain),
///     ..TagDef::new(TagId::Economy, "Economy Tag", 1)
/// }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct TagDef {
    /// Id (exactly one def per id across all `DEFS` tables).
    pub id: TagId,
    /// Display name (English, for logs/debug).
    pub name: &'static str,
    /// First ante the tag can be offered at (tags.md: 1 or 2).
    pub min_ante: i32,
    /// When the tag is gained (immediate effects, pack tags, Boss, Orbital, Top-up, ...).
    pub on_gain: Option<TagFn>,
    /// The shop was just generated (Coupon, D6, Uncommon/Rare/edition jokers, Voucher).
    pub on_shop_enter: Option<TagFn>,
    /// A Boss Blind was just beaten, after the cash-out (Investment).
    pub on_boss_defeated: Option<TagFn>,
    /// A round starts (Juggle).
    pub on_round_start: Option<TagFn>,
}

impl TagDef {
    /// A tag without any hook.
    pub const fn new(id: TagId, name: &'static str, min_ante: i32) -> Self {
        TagDef {
            id,
            name,
            min_ante,
            on_gain: None,
            on_shop_enter: None,
            on_boss_defeated: None,
            on_round_start: None,
        }
    }
}

/// What a [`TagFn`] reads and writes. Most changes are immediate; creations that need the
/// joker area (`create_joker`) and hand levels are queued side effects, applied by the game
/// layer right after the hook.
pub struct TagCtx<'a> {
    state: &'a mut GameState,
    fx: &'a mut Vec<SideEffect>,
    hand: Option<HandType>,
}

impl<'a> TagCtx<'a> {
    /// Build a context (the game layer and tests). `hand` is the Orbital hand picked when the
    /// tag was offered.
    pub fn new(
        state: &'a mut GameState,
        fx: &'a mut Vec<SideEffect>,
        hand: Option<HandType>,
    ) -> Self {
        TagCtx { state, fx, hand }
    }

    /// The run (read-only).
    pub fn state(&self) -> &GameState {
        self.state
    }

    /// Current money.
    pub fn money(&self) -> i32 {
        self.state.money
    }

    /// Gain (or lose) money.
    pub fn add_money(&mut self, dollars: i32) {
        self.state.money = self.state.money.saturating_add(dollars);
    }

    /// The hand picked for this tag when it was offered (Orbital).
    pub fn aux_hand(&self) -> Option<HandType> {
        self.hand
    }

    /// Queue a hand level change (Orbital +3).
    pub fn level_up(&mut self, hand: HandType, levels: i32) {
        self.fx.push(SideEffect::LevelUpHand { hand, levels });
    }

    /// Queue a random joker of `rarity` in the joker area (Top-up: Common). Dropped (counted as
    /// no room) when the area is full.
    pub fn create_joker(&mut self, rarity: Rarity) {
        self.fx.push(SideEffect::CreateJoker {
            rarity: Some(rarity),
        });
    }

    /// Queue a pack to open right after the current action (pack tags). Several queue up and
    /// open one after the other.
    pub fn queue_pack(&mut self, spec: PackSpec) {
        self.state.pending_packs.push(spec);
    }

    /// Reroll the ante's boss for free (Boss tag).
    pub fn reroll_boss(&mut self) {
        crate::boss::reroll_boss(self.state);
    }

    /// Hand size bonus for the coming round only (Juggle +3); cleared at the round's end.
    pub fn add_round_hand_size(&mut self, delta: i32) {
        self.state.round_hand_size = self.state.round_hand_size.saturating_add(delta);
    }

    /// Uniform integer in `[lo, hi_exclusive)` on `RngKey::Tags`.
    pub fn rng_range(&mut self, lo: usize, hi_exclusive: usize) -> usize {
        self.state.rng.range(RngKey::Tags, lo, hi_exclusive)
    }

    // ----- shop tags (only meaningful inside `on_shop_enter`) -------------------------------

    /// Add an extra free joker of `rarity` to the shop (Uncommon/Rare tag; the base game does
    /// not need a free slot for it). False if no such joker can be generated.
    pub fn shop_add_free_joker(&mut self, rarity: Rarity) -> bool {
        let shown: Vec<_> = self
            .state
            .shop
            .cards
            .iter()
            .flatten()
            .filter_map(|c| match c.kind {
                ShopCardKind::Joker(j) => Some(j.id),
                _ => None,
            })
            .collect();
        match roll_joker(self.state, RngKey::Tags, Some(rarity), &shown) {
            Some(j) => {
                self.state.shop.cards.push(Some(ShopCard {
                    kind: ShopCardKind::Joker(j),
                    free: true,
                }));
                true
            }
            None => false,
        }
    }

    /// Give the first base-edition joker of the shop `edition` and make it free (Negative, Foil,
    /// Holographic, Polychrome tags). False if the shop has none (the tag then stays held).
    pub fn shop_edition_first_base_joker(&mut self, edition: Edition) -> bool {
        let shop = &mut self.state.shop;
        for c in shop.cards.iter_mut().flatten() {
            if let ShopCardKind::Joker(j) = &mut c.kind
                && j.edition == Edition::Base
            {
                j.edition = edition;
                c.free = true;
                return true;
            }
        }
        false
    }

    /// Add one extra voucher to the shop (Voucher tag). False if none is eligible.
    pub fn shop_add_voucher(&mut self) -> bool {
        let present: Vec<_> = self
            .state
            .shop
            .vouchers
            .iter()
            .flatten()
            .map(|v| v.id)
            .collect();
        match pick_voucher(self.state, &present) {
            Some(id) => {
                self.state.shop.vouchers.push(Some(ShopVoucher {
                    id,
                    from_ante: false,
                }));
                true
            }
            None => false,
        }
    }

    /// Make every card and pack currently in the shop free (Coupon; vouchers excluded).
    pub fn shop_make_free(&mut self) {
        for c in self.state.shop.cards.iter_mut().flatten() {
            c.free = true;
        }
        for p in self.state.shop.packs.iter_mut().flatten() {
            p.free = true;
        }
    }

    /// Set this shop's starting reroll price (D6: 0).
    pub fn shop_set_reroll_base(&mut self, base: i32) {
        self.state.shop.reroll_base = Some(base);
    }
}

type Select = fn(&crate::tag::TagDef) -> Option<TagFn>;

/// Run one hook kind over the held tags, oldest first; consumed tags are removed. While a hook
/// runs, `state.tags` holds only the tags that are still to be processed or were kept.
fn run(s: &mut GameState, fx: &mut Vec<SideEffect>, select: Select, hand: Option<HandType>) {
    let held = std::mem::take(&mut s.tags);
    let mut kept: Vec<TagId> = Vec::with_capacity(held.len());
    for id in held {
        let hook = try_tag_def(id).and_then(select);
        let consumed = match hook {
            Some(f) => f(&mut TagCtx::new(s, fx, hand)),
            None => false,
        };
        if !consumed {
            kept.push(id);
        }
    }
    // A hook never adds tags, but be safe: keep anything pushed meanwhile after the survivors.
    kept.append(&mut s.tags);
    s.tags = kept;
}

/// Gain a tag: a held Double Tag is used up to add the tag twice (never for Double itself);
/// each copy runs `on_gain` and stays held unless consumed. `hand` = the Orbital hand offered
/// with the tag.
pub fn gain(s: &mut GameState, id: TagId, hand: Option<HandType>, fx: &mut Vec<SideEffect>) {
    let mut copies = 1;
    if id != TagId::Double
        && let Some(pos) = s.tags.iter().position(|&t| t == TagId::Double)
    {
        s.tags.remove(pos);
        copies = 2;
    }
    for _ in 0..copies {
        let consumed = match try_tag_def(id).and_then(|d| d.on_gain) {
            Some(f) => f(&mut TagCtx::new(s, fx, hand)),
            None => false,
        };
        if !consumed {
            s.tags.push(id);
        }
    }
}

/// Run `on_shop_enter` of the held tags (the shop was just generated).
pub fn fire_shop_enter(s: &mut GameState) -> Vec<SideEffect> {
    let mut fx = Vec::new();
    run(s, &mut fx, |d| d.on_shop_enter, None);
    fx
}

/// Run `on_boss_defeated` of the held tags.
pub fn fire_boss_defeated(s: &mut GameState, fx: &mut Vec<SideEffect>) {
    run(s, fx, |d| d.on_boss_defeated, None);
}

/// Run `on_round_start` of the held tags.
pub fn fire_round_start(s: &mut GameState, fx: &mut Vec<SideEffect>) {
    run(s, fx, |d| d.on_round_start, None);
}

/// Price of a paid boss reroll (Director's Cut / Retcon).
pub const fn boss_reroll_cost() -> i32 {
    BOSS_REROLL_COST
}
