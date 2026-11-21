//! Randomized-but-plausible mid-run states ("scenarios") for curriculum starts.
//!
//! [`random_scenario`] builds a run that sits at a Blind select, as if the player had reached a
//! random ante (1-8) with a build that fits it: money, joker count, hand levels, vouchers and
//! deck edits grow with the ante, jokers of every rarity (legendary ones included, rarely) with
//! editions and the stickers of the run's stake (the market's own sticker roll, so Eternal /
//! Perishable / Rental appear with their real frequencies and only from the stake that allows
//! them), random consumables. It is a pure function of `(seed, config)`.
//!
//! The scenario is a *tool for training starts* (a harder or rarer position than a natural run
//! reaches often), not a claim about what a good player's state looks like. Every generated state
//! passes `GameState::check_invariants` (checked here; the fallback is an ordinary fresh run).

use crate::blinds::BlindKind;
use crate::card::Card;
use crate::consumable::ConsumableInstance;
use crate::content::ConsumableId;
use crate::content::jokers::{JokerId, all_defs};
use crate::deck::Pile;
use crate::enums::{Edition, Enhancement, HandType, Rank, Rarity, Seal, Suit};
use crate::game::Game;
use crate::rng::{GameRng, RngKey};
use crate::state::GameConfig;

/// Seed salt of the scenario stream (kept apart from every game stream).
const SCENARIO_SALT: u64 = 0x5CE7_A210_5EED_0001;

/// Upper bound of the scenario ante.
pub const MAX_SCENARIO_ANTE: i32 = 8;

fn pick<T: Copy>(r: &mut GameRng, items: &[T]) -> T {
    items[r.range(RngKey::Misc, 0, items.len())]
}

fn chance(r: &mut GameRng, percent: usize) -> bool {
    r.range(RngKey::Misc, 0, 100) < percent
}

/// Jokers by rarity (display-order independent: sorted by index for determinism).
fn jokers_of(rarity: Rarity) -> Vec<JokerId> {
    let mut v: Vec<JokerId> = all_defs()
        .filter(|d| d.rarity == rarity)
        .map(|d| d.id)
        .collect();
    v.sort_by_key(|id| id.index());
    v.dedup();
    v
}

fn edition(r: &mut GameRng) -> Edition {
    match r.range(RngKey::Misc, 0, 100) {
        0..=77 => Edition::Base,
        78..=85 => Edition::Foil,
        86..=91 => Edition::Holographic,
        92..=95 => Edition::Polychrome,
        _ => Edition::Negative,
    }
}

/// A scenario run for `(seed, config)`; falls back to a fresh run if an invariant fails (it must
/// not: the tests assert that [`try_random_scenario`] never errors).
pub fn random_scenario(seed: u64, config: GameConfig) -> Game {
    try_random_scenario(seed, config).unwrap_or_else(|_| Game::new(seed, config))
}

/// Like [`random_scenario`], returning the invariant violation instead of falling back.
pub fn try_random_scenario(seed: u64, config: GameConfig) -> Result<Game, String> {
    let mut r = GameRng::new(seed ^ SCENARIO_SALT);
    let mut g = Game::new(seed, config);

    let ante = 1 + r.range(RngKey::Misc, 0, MAX_SCENARIO_ANTE as usize) as i32;
    let blind_idx = match r.range(RngKey::Misc, 0, 10) {
        0..=2 => 0usize,
        3..=5 => 1,
        _ => 2,
    };
    let a = ante as usize;

    // Vouchers first (a redeemed voucher must not be offered again by the ante set-up below).
    for _ in 0..r.range(RngKey::Misc, 0, 2 + a / 2) {
        if let Some(v) = crate::shop::pick_voucher(g.state_mut(), &[]) {
            crate::voucher::redeem(g.state_mut(), v);
        }
    }

    // Progress: ante, blind, rounds behind us; a fresh ante set-up (tags, boss, voucher, target).
    {
        let s = g.state_mut();
        s.ante = ante;
        s.round = (3 * (a - 1) + blind_idx) as u32;
        s.stats.rounds_won = s.round;
        s.stats.hands_played = s.round * 3;
        s.stats.shops_entered = s.round;
        s.start_ante();
        s.blind = match blind_idx {
            0 => BlindKind::Small,
            1 => BlindKind::Big,
            _ => BlindKind::Boss,
        };
        s.refresh_target();
        // Economy that fits the ante (interest and joker purchases push it up to ~$80 by ante 8).
        s.money = 4 + r.range(RngKey::Misc, 0, 6 + 9 * a) as i32;
    }

    // Hand levels: a couple of favoured hand types get most of the planets.
    {
        let hands: Vec<HandType> = HandType::ALL
            .into_iter()
            .filter(|h| !h.is_secret())
            .collect();
        let fav = [pick(&mut r, &hands), pick(&mut r, &hands)];
        for _ in 0..r.range(RngKey::Misc, 0, 2 * a + 1) {
            let h = if chance(&mut r, 70) {
                fav[r.range(RngKey::Misc, 0, 2)]
            } else {
                pick(&mut r, &hands)
            };
            g.state_mut().hand_levels[h.index()] += 1;
        }
    }

    // Jokers: count grows with the ante, rarity mix with a rare legendary.
    let pools = [
        jokers_of(Rarity::Common),
        jokers_of(Rarity::Uncommon),
        jokers_of(Rarity::Rare),
        jokers_of(Rarity::Legendary),
    ];
    let slots = g.state().joker_slots as i32;
    let want = (a.div_ceil(2) as i32 + r.range(RngKey::Misc, 0, 3) as i32 - 1).clamp(0, slots);
    let weights = [
        50.0,
        28.0,
        12.0 + 1.5 * a as f64,
        if a >= 2 { 4.0 } else { 0.0 },
    ];
    let mut owned: Vec<JokerId> = Vec::new();
    for _ in 0..want {
        let Some(tier) = r.weighted_index(RngKey::Misc, &weights) else {
            break;
        };
        let pool = &pools[tier];
        if pool.is_empty() {
            continue;
        }
        let id = pick(&mut r, pool);
        if owned.contains(&id) {
            continue;
        }
        let ed = edition(&mut r);
        let mut j = crate::shop::joker_instance(g.state(), id, ed);
        if tier != 3 {
            // The market's own sticker roll: nothing below Black, Eternal / Perishable from
            // Black / Orange, Rental from Gold (legendary jokers never get stickers).
            j.stickers = crate::shop::roll_stickers(g.state_mut(), RngKey::Shop, id);
        }
        if g.add_joker(j).is_some() {
            owned.push(id);
        }
    }

    // Consumables.
    for _ in 0..r.range(RngKey::Misc, 0, 3) {
        let mut c = ConsumableInstance::new(
            ConsumableId::ALL[r.range(RngKey::Misc, 0, ConsumableId::COUNT)],
        );
        c.negative = chance(&mut r, 8);
        let _ = g.add_consumable(c);
    }

    // Deck edits: enhancements / editions / seals, a few added and destroyed cards.
    if chance(&mut r, 40 + 5 * a) {
        let s = g.state_mut();
        let n = s.deck.draw_pile.len();
        for _ in 0..r.range(RngKey::Misc, 0, 3 + 2 * a) {
            let c = &mut s.deck.draw_pile[r.range(RngKey::Misc, 0, n)];
            c.enhancement = pick(&mut r, &Enhancement::ALL);
            if chance(&mut r, 50) {
                c.seal = pick(&mut r, &Seal::ALL);
            }
            if chance(&mut r, 40) {
                c.edition = pick(
                    &mut r,
                    &[Edition::Foil, Edition::Holographic, Edition::Polychrome],
                );
            }
        }
        for _ in 0..r.range(RngKey::Misc, 0, 1 + a / 2) {
            if s.deck.draw_pile.len() <= 40 {
                break;
            }
            let i = r.range(RngKey::Misc, 0, s.deck.draw_pile.len());
            let uid = s.deck.draw_pile[i].uid;
            if s.deck.remove_card(uid).is_some() {
                s.stats.cards_destroyed += 1;
            }
        }
        for _ in 0..r.range(RngKey::Misc, 0, 1 + a / 2) {
            let uid = s.deck.alloc_uid();
            let mut c = Card::new(uid, pick(&mut r, &Rank::ALL), pick(&mut r, &Suit::ALL));
            if chance(&mut r, 50) {
                c.enhancement = pick(&mut r, &Enhancement::ALL);
            }
            s.deck.add_card(c, Pile::Draw);
            s.stats.cards_added += 1;
        }
    }

    g.state().check_invariants()?;
    Ok(g)
}
