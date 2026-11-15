//! The market: shop state, generation (card slots, booster slots, voucher slots), prices,
//! rerolls and joker creation pools. Phase 5.0. Rules: docs/mechanics/shop.md, notes.md §4.1d;
//! numbers: `data/shop_weights.rs`, `data/prices.rs`. Guide:
//! `docs/design/shop_packs_vouchers_tags.md`.
//!
//! Randomness: card slots, pack slots and vouchers use `RngKey::Shop`. Jokers created outside
//! the shop (Judgement, Wraith, The Soul, Riff-raff, Top-up tag) use `RngKey::Jokers` through
//! [`roll_joker`].
//!
//! Everything here works on `&mut GameState` / `&GameState`; the step functions that spend
//! money and fire joker events are in `game_shop.rs`.

use crate::card::Card;
use crate::consumable::{ConsumableInstance, ConsumableKind, random_consumable_excluding};
use crate::content::ids::{ConsumableId, VoucherId};
use crate::content::jokers::JokerId;
use crate::content::jokers::try_def;
use crate::content::try_voucher_def;
use crate::data::prices::{PLAYING_CARD, REROLL_BASE};
use crate::data::shop_weights::{
    BASE_CARD_SLOTS, BOOSTER_SLOTS, CARD_ENHANCEMENTS, CARD_SEALS, ILLUSION_ENHANCEMENT_CHANCE,
    ILLUSION_SEAL_CHANCE, RARITY_WEIGHTS, WEIGHT_JOKER, WEIGHT_PLANET, WEIGHT_PLAYING_CARD,
    WEIGHT_SPECTRAL_GHOST_DECK, WEIGHT_TAROT, card_edition, card_surcharge, edition_surcharge,
    joker_edition, pack_price, pack_weight,
};
use crate::data::stakes::{
    ETERNAL_CHANCE, PERISHABLE_CHANCE, PERISHABLE_ROUNDS, RENTAL_BUY_PRICE, RENTAL_CHANCE,
};
use crate::economy::{self, Discount};
use crate::enums::{Edition, Rank, Rarity, Suit};
use crate::hooks::{PackKind, RunFlag};
use crate::joker::JokerInstance;
use crate::packs::{PackSize, PackSpec};
use crate::rng::{GameRng, RngKey};
use crate::state::GameState;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

/// What a shop card slot holds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ShopCardKind {
    /// A joker (edition and sell value already rolled; the uid is assigned on purchase).
    Joker(JokerInstance),
    /// A Tarot, Planet (or, with a deck that allows it, Spectral) card.
    Consumable(ConsumableId),
    /// A playing card (Magic Trick); added to the deck on purchase.
    PlayingCard(Card),
}

/// One shop card slot.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ShopCard {
    /// The item.
    pub kind: ShopCardKind,
    /// Costs $0 (Coupon / free-joker tags). Astronomer's free Planets are not flagged: the
    /// price function applies them.
    pub free: bool,
}

/// One booster slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShopPack {
    /// Kind and size.
    pub spec: PackSpec,
    /// Costs $0 (Coupon).
    pub free: bool,
}

/// One voucher slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShopVoucher {
    /// The voucher.
    pub id: VoucherId,
    /// This is the ante's voucher (stays for the rest of the ante if not bought); extra
    /// vouchers from the Voucher tag are gone when the shop closes.
    pub from_ante: bool,
}

/// The market of the current Shop phase. Slots are positions: a bought item leaves `None`
/// behind, other slots do not shift (`Buy(slot)`, `OpenPack(slot)`, `BuyVoucher(slot)`).
/// Reset to the default when the shop is left.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ShopState {
    /// Card slots. The first `base_slots` are the random slots (rerolled); tag jokers are
    /// appended after them.
    pub cards: Vec<Option<ShopCard>>,
    /// Random card slots generated so far (Overstock grows it mid-shop).
    pub base_slots: usize,
    /// Booster slots (`BOOSTER_SLOTS`).
    pub packs: Vec<Option<ShopPack>>,
    /// Voucher slots: the ante voucher first (if any), then Voucher-tag extras.
    pub vouchers: Vec<Option<ShopVoucher>>,
    /// Paid rerolls this shop (each raises the price by $1).
    pub rerolls: u32,
    /// Free rerolls used this shop (Chaos the Clown); they do not raise the price.
    pub free_rerolls_used: u32,
    /// Starting reroll price override for this shop (D6 tag: $0).
    pub reroll_base: Option<i32>,
}

// ----- prices -----------------------------------------------------------------------------------

/// Undiscounted base cost of a card (before the edition surcharge).
pub fn card_base_cost(k: &ShopCardKind) -> i32 {
    match k {
        ShopCardKind::Joker(j) => j
            .def()
            .map_or(1, |d| i32::try_from(d.base_cost).unwrap_or(i32::MAX)),
        ShopCardKind::Consumable(id) => id.kind().base_cost(),
        ShopCardKind::PlayingCard(_) => PLAYING_CARD,
    }
}

/// Edition surcharge of a card.
pub fn card_edition_surcharge(k: &ShopCardKind) -> i32 {
    match k {
        ShopCardKind::Joker(j) => edition_surcharge(j.edition),
        ShopCardKind::Consumable(_) => 0,
        ShopCardKind::PlayingCard(c) => card_surcharge(c),
    }
}

/// What the card costs now: `$0` if flagged free or a Planet under Astronomer, else the
/// discounted price (`economy::shop_price`).
pub fn card_price(s: &GameState, c: &ShopCard) -> i32 {
    if c.free {
        return 0;
    }
    if let ShopCardKind::Consumable(id) = c.kind
        && id.kind() == ConsumableKind::Planet
        && s.mods.free_planets
    {
        return 0;
    }
    if let ShopCardKind::Joker(j) = c.kind
        && j.stickers.rental
    {
        return RENTAL_BUY_PRICE;
    }
    economy::shop_price(
        card_base_cost(&c.kind),
        card_edition_surcharge(&c.kind),
        s.shop_mods.discount,
    )
}

/// What the pack costs now (Astronomer makes Celestial packs free).
pub fn pack_price_of(s: &GameState, p: &ShopPack) -> i32 {
    if p.free || (p.spec.kind == PackKind::Celestial && s.mods.free_planets) {
        return 0;
    }
    economy::shop_price(pack_price(p.spec.size), 0, s.shop_mods.discount)
}

/// What a voucher costs now (the discount applies, notes.md §4.1d).
pub fn voucher_price(s: &GameState, id: VoucherId) -> i32 {
    let base = try_voucher_def(id).map_or(crate::data::prices::VOUCHER, |d| d.cost);
    economy::shop_price(base, 0, s.shop_mods.discount)
}

/// Price of the next reroll: `$0` while a free reroll is left (Chaos the Clown), otherwise
/// `base + paid rerolls` where `base` = D6 override, else the voucher base, else $5.
pub fn reroll_price(s: &GameState) -> i32 {
    if has_free_reroll(s) {
        return 0;
    }
    let base = s
        .shop
        .reroll_base
        .or(s.shop_mods.reroll_base)
        .unwrap_or(REROLL_BASE);
    economy::reroll_cost(s.shop.rerolls, base)
}

/// A free reroll (Chaos the Clown) is left this shop.
pub fn has_free_reroll(s: &GameState) -> bool {
    s.shop.free_rerolls_used < s.mods.free_rerolls
}

/// Number of random card slots the shop has: 2 + vouchers.
pub fn card_slot_count(s: &GameState) -> usize {
    (i64::from(BASE_CARD_SLOTS) + i64::from(s.shop_mods.card_slots)).clamp(0, 16) as usize
}

/// After a discount change (Clearance Sale, Liquidation): owned jokers and consumables are
/// re-priced (sell value = `max(1, floor(price / 2))` of the new price plus what Egg/Gift Card
/// added), like the original recalculating every card's cost (shop.md §6; UNVERIFIED for
/// `extra` bookkeeping, which is inferred as `sell_value - base part`).
pub fn rescale_sell_values(s: &mut GameState, old: Discount, new: Discount) {
    if old == new {
        return;
    }
    for j in &mut s.jokers {
        let Some(d) = j.def() else { continue };
        if j.stickers.rental {
            continue; // Rental jokers cost $1 whatever the discount: their sell value stays.
        }
        let base = i32::try_from(d.base_cost).unwrap_or(i32::MAX);
        let sur = edition_surcharge(j.edition);
        let old_part = economy::sell_value(economy::shop_price(base, sur, old), 0);
        let extra = (j.sell_value - old_part).max(0);
        j.sell_value = economy::sell_value(economy::shop_price(base, sur, new), extra);
    }
    for c in &mut s.consumables {
        let base = c.kind().base_cost();
        let old_part = economy::sell_value(economy::shop_price(base, 0, old), 0);
        let extra = (c.sell_value - old_part).max(0);
        c.sell_value = economy::sell_value(economy::shop_price(base, 0, new), extra);
    }
}

// ----- joker creation ----------------------------------------------------------------------------

/// Sticker roll of a new shop or pack joker (stakes.md §2). Below Black Stake nothing is rolled
/// (no RNG consumed). One poll decides Eternal (top 30%) and, from Orange Stake, Perishable (next
/// 30%), so they never combine; an incompatible joker (`JokerDef::eternal_compatible` /
/// `perishable_compatible`) just gets nothing from its band. Rental (Gold) is an independent
/// roll and can combine with either. Legendary jokers never get stickers (the caller skips this).
pub fn roll_stickers(s: &mut GameState, key: RngKey, id: JokerId) -> crate::joker::Stickers {
    let stake = s.config.stake;
    let mut st = crate::joker::Stickers::default();
    if !stake.eternal_jokers() {
        return st;
    }
    let def = try_def(id);
    let poll = s.rng.float(key);
    if poll >= 1.0 - ETERNAL_CHANCE {
        st.eternal = def.is_none_or(|d| d.eternal_compatible);
    } else if stake.perishable_jokers()
        && poll >= 1.0 - ETERNAL_CHANCE - PERISHABLE_CHANCE
        && def.is_none_or(|d| d.perishable_compatible)
    {
        st.perishable = true;
        st.perish_rounds_left = PERISHABLE_ROUNDS;
    }
    if stake.rental_jokers() {
        st.rental = s.rng.float(key) >= 1.0 - RENTAL_CHANCE;
    }
    st
}

/// A fresh joker instance of `id` with `edition`, sell value from the discounted price. The uid
/// is 0 (assigned when it enters the joker area).
pub fn joker_instance(s: &GameState, id: JokerId, edition: Edition) -> JokerInstance {
    let mut j = JokerInstance::new(0, id).with_edition(edition);
    let base = j
        .def()
        .map_or(1, |d| i32::try_from(d.base_cost).unwrap_or(i32::MAX));
    j.sell_value = economy::sell_value(
        economy::shop_price(base, edition_surcharge(edition), s.shop_mods.discount),
        0,
    );
    j
}

/// May `id` be generated at all (defined, Gros Michel / Cavendish rules).
fn joker_available(id: JokerId, extinct: bool) -> bool {
    match id {
        JokerId::GrosMichel => !extinct,
        JokerId::Cavendish => extinct,
        _ => true,
    }
}

/// Ids of `rarity` that may be generated, not owned / not in `extra_taken` unless
/// `allow_duplicates`, in id order.
fn joker_candidates(
    s: &GameState,
    rarity: Rarity,
    extra_taken: &[JokerId],
    allow_duplicates: bool,
) -> Vec<JokerId> {
    let extinct = s.has_flag(RunFlag::GrosMichelExtinct);
    // Per-rarity id lists (id order), built once: no scan of the whole registry per roll.
    static BY_RARITY: OnceLock<[Vec<JokerId>; Rarity::COUNT]> = OnceLock::new();
    let table = BY_RARITY.get_or_init(|| {
        std::array::from_fn(|i| {
            JokerId::ALL
                .into_iter()
                .filter(|&id| try_def(id).is_some_and(|d| d.rarity.index() == i))
                .collect()
        })
    });
    table[rarity.index()]
        .iter()
        .copied()
        .filter(|&id| {
            joker_available(id, extinct)
                && (allow_duplicates
                    || (!extra_taken.contains(&id) && !s.jokers.iter().any(|j| j.id == id)))
        })
        .collect()
}

/// Roll a random joker: rarity (`None` = shop odds 70/25/5; `Some` forces it, Legendary only
/// ever comes from here), then uniformly among the candidates not owned and not in `extra_taken`
/// (Showman lifts both), then the edition. With `rarity == None` an exhausted rarity falls back
/// to the next lower one. Returns `None` if nothing can be generated. Consumes rolls on `key`.
pub fn roll_joker(
    s: &mut GameState,
    key: RngKey,
    rarity: Option<Rarity>,
    extra_taken: &[JokerId],
) -> Option<JokerInstance> {
    let mut j = roll_joker_plain(s, key, rarity, extra_taken)?;
    if j.def().is_none_or(|d| d.rarity != Rarity::Legendary) {
        j.stickers = roll_stickers(s, key, j.id);
        if j.stickers.rental {
            j.sell_value = economy::sell_value(RENTAL_BUY_PRICE, 0);
        }
    }
    Some(j)
}

/// [`roll_joker`] without the stake stickers: jokers created outside the shop and packs
/// (Judgement, Wraith, The Soul, Riff-raff, Top-up tag) never carry them.
pub fn roll_joker_plain(
    s: &mut GameState,
    key: RngKey,
    rarity: Option<Rarity>,
    extra_taken: &[JokerId],
) -> Option<JokerInstance> {
    let allow_dup = s.mods.allow_duplicates;
    let order: Vec<Rarity> = match rarity {
        Some(r) => vec![r],
        None => {
            let w: Vec<f64> = RARITY_WEIGHTS.iter().map(|&(_, w)| w).collect();
            let first = s.rng.weighted_index(key, &w).unwrap_or(0);
            let mut o = vec![RARITY_WEIGHTS[first].0];
            o.extend(
                RARITY_WEIGHTS
                    .iter()
                    .map(|&(r, _)| r)
                    .filter(|&r| r != RARITY_WEIGHTS[first].0),
            );
            o
        }
    };
    for r in order {
        let cands = joker_candidates(s, r, extra_taken, allow_dup);
        if cands.is_empty() {
            continue;
        }
        let id = cands[s.rng.range(key, 0, cands.len())];
        let ed_roll = s.rng.float(key);
        let edition = joker_edition(ed_roll, s.shop_mods.edition_mult);
        return Some(joker_instance(s, id, edition));
    }
    None
}

// ----- playing cards -----------------------------------------------------------------------------

/// A random playing card (uid 0): random rank and suit, then enhancement / seal with the given
/// chances (one roll each) and the card edition odds scaled by `edition_mult`.
pub fn random_playing_card(
    rng: &mut GameRng,
    key: RngKey,
    enhancement_chance: f64,
    seal_chance: f64,
    edition_mult: u32,
) -> Card {
    let rank = Rank::ALL[rng.range(key, 0, Rank::COUNT)];
    let suit = Suit::ALL[rng.range(key, 0, Suit::COUNT)];
    let mut c = Card::new(0, rank, suit);
    if rng.float(key) < enhancement_chance {
        c.enhancement = CARD_ENHANCEMENTS[rng.range(key, 0, CARD_ENHANCEMENTS.len())];
    }
    if rng.float(key) < seal_chance {
        c.seal = CARD_SEALS[rng.range(key, 0, CARD_SEALS.len())];
    }
    c.edition = card_edition(rng.float(key), edition_mult);
    debug_assert!(c.edition != Edition::Negative);
    c
}

/// Plain card (no enhancement/seal/edition) of random rank and suit, for Magic Trick without
/// Illusion.
fn plain_playing_card(rng: &mut GameRng, key: RngKey) -> Card {
    let rank = Rank::ALL[rng.range(key, 0, Rank::COUNT)];
    let suit = Suit::ALL[rng.range(key, 0, Suit::COUNT)];
    Card::new(0, rank, suit)
}

// ----- slots --------------------------------------------------------------------------------------

/// Type weights of a card slot: `[joker, tarot, planet, playing card, spectral]`.
pub fn card_type_weights(s: &GameState) -> [f64; 5] {
    let tarot = s.shop_mods.tarot_weight.unwrap_or(WEIGHT_TAROT);
    let planet = s.shop_mods.planet_weight.unwrap_or(WEIGHT_PLANET);
    [
        WEIGHT_JOKER,
        tarot,
        planet,
        if s.shop_mods.playing_cards {
            WEIGHT_PLAYING_CARD
        } else {
            0.0
        },
        if s.deck_spectral_in_shop() {
            WEIGHT_SPECTRAL_GHOST_DECK
        } else {
            0.0
        },
    ]
}

/// Jokers currently shown in the shop (excluded from the next draw without Showman).
fn shown_jokers(s: &GameState) -> Vec<JokerId> {
    s.shop
        .cards
        .iter()
        .flatten()
        .filter_map(|c| match c.kind {
            ShopCardKind::Joker(j) => Some(j.id),
            _ => None,
        })
        .collect()
}

/// Consumable ids held or shown in the shop (excluded without Showman).
fn taken_consumables(s: &GameState) -> Vec<ConsumableId> {
    let mut v: Vec<ConsumableId> = s
        .consumables
        .iter()
        .map(|c: &ConsumableInstance| c.id)
        .collect();
    v.extend(s.shop.cards.iter().flatten().filter_map(|c| match c.kind {
        ShopCardKind::Consumable(id) => Some(id),
        _ => None,
    }));
    v
}

/// Roll one random card slot (type by weight, then the card). `RngKey::Shop`.
pub fn roll_card(s: &mut GameState) -> ShopCard {
    let key = RngKey::Shop;
    let w = card_type_weights(s);
    let ty = s.rng.weighted_index(key, &w).unwrap_or(1);
    let kind = match ty {
        0 => {
            let shown = shown_jokers(s);
            match roll_joker(s, key, None, &shown) {
                Some(j) => ShopCardKind::Joker(j),
                None => ShopCardKind::Consumable(ConsumableId::Strength),
            }
        }
        3 => {
            let illusion = s.shop_mods.illusion;
            // Wiki (Illusion): shop playing-card editions ignore Hone / Glow Up.
            let em = 1;
            let c = if illusion {
                random_playing_card(
                    &mut s.rng,
                    key,
                    ILLUSION_ENHANCEMENT_CHANCE,
                    ILLUSION_SEAL_CHANCE,
                    em,
                )
            } else {
                plain_playing_card(&mut s.rng, key)
            };
            ShopCardKind::PlayingCard(c)
        }
        t => {
            let kind = match t {
                1 => ConsumableKind::Tarot,
                2 => ConsumableKind::Planet,
                _ => ConsumableKind::Spectral,
            };
            let taken = taken_consumables(s);
            let id = random_consumable_excluding(
                kind,
                &taken,
                &s.hand_plays_run,
                s.mods.allow_duplicates,
                key,
                &mut s.rng,
            );
            ShopCardKind::Consumable(id)
        }
    };
    ShopCard { kind, free: false }
}

/// Roll the kind and size of one booster slot by the packs.md §2 weights (independent draws).
pub fn roll_pack_spec(rng: &mut GameRng) -> PackSpec {
    let mut combos: Vec<PackSpec> = Vec::with_capacity(15);
    let mut w: Vec<f64> = Vec::with_capacity(15);
    for kind in PackKind::ALL {
        for size in PackSize::ALL {
            combos.push(PackSpec { kind, size });
            w.push(pack_weight(kind, size));
        }
    }
    let i = rng.weighted_index(RngKey::Shop, &w).unwrap_or(0);
    combos[i]
}

/// Pick the voucher for an ante: uniformly among the defined vouchers not owned whose base is
/// owned (or that need none), excluding `exclude`. `None` if nothing is eligible.
pub fn pick_voucher(s: &mut GameState, exclude: &[VoucherId]) -> Option<VoucherId> {
    let eligible: Vec<VoucherId> = VoucherId::ALL
        .into_iter()
        .filter(|&id| {
            try_voucher_def(id).is_some()
                && !s.vouchers.contains(&id)
                && !exclude.contains(&id)
                && id.requires().is_none_or(|b| s.vouchers.contains(&b))
        })
        .collect();
    if eligible.is_empty() {
        return None;
    }
    Some(eligible[s.rng.range(RngKey::Shop, 0, eligible.len())])
}

// ----- shop life cycle ---------------------------------------------------------------------------

/// Generate the market when a shop is entered: random card slots, 2 booster slots (the very
/// first shop of the run opens with a Normal Buffoon pack), the ante's voucher; then the held
/// tags' `on_shop_enter` hooks run (Coupon, D6, free jokers, Voucher). Resets any old shop.
pub fn enter_shop(s: &mut GameState) -> Vec<crate::hooks::SideEffect> {
    s.stats.shops_entered += 1;
    let first = s.stats.shops_entered == 1;
    s.shop = ShopState::default();
    fill_card_slots(s);
    for i in 0..BOOSTER_SLOTS {
        let spec = if first && i == 0 {
            PackSpec {
                kind: PackKind::Buffoon,
                size: PackSize::Normal,
            }
        } else {
            roll_pack_spec(&mut s.rng)
        };
        s.shop.packs.push(Some(ShopPack { spec, free: false }));
    }
    if let Some(id) = s.ante_voucher {
        s.shop.vouchers.push(Some(ShopVoucher {
            id,
            from_ante: true,
        }));
    }
    crate::tag::fire_shop_enter(s)
}

/// Roll random card slots until the shop has `card_slot_count` of them (Overstock bought
/// mid-shop fills the new slot at once).
pub fn fill_card_slots(s: &mut GameState) {
    let n = card_slot_count(s);
    while s.shop.base_slots < n {
        let c = roll_card(s);
        // Keep random slots in front of tag-created ones.
        s.shop.cards.insert(s.shop.base_slots, Some(c));
        s.shop.base_slots += 1;
    }
}

/// Replace every card slot (tag jokers included) with fresh random cards. Does not touch
/// packs or vouchers.
pub fn reroll_cards(s: &mut GameState) {
    s.shop.cards.clear();
    s.shop.base_slots = 0;
    fill_card_slots(s);
}
