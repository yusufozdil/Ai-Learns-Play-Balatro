//! Phase 5E: the fifteen starting decks (docs/mechanics/decks.md). One test per effect; every
//! expected number is derived in a comment.

use balatro_core::actions::Action;
use balatro_core::blinds::BlindKind;
use balatro_core::content::{ConsumableId, TagId, VoucherId};
use balatro_core::enums::{Phase, Rank, Suit};
use balatro_core::game::{Game, StepInfo};
use balatro_core::rng::RngKey;
use balatro_core::shop::{self, ShopCardKind};
use balatro_core::state::{DeckKind, GameConfig, StakeKind};
use std::collections::HashMap;

fn game(deck: DeckKind, seed: u64) -> Game {
    Game::new(
        seed,
        GameConfig {
            deck,
            ..GameConfig::default()
        },
    )
}

fn beat(g: &mut Game) -> StepInfo {
    let t = g.state().target;
    g.state_mut().round_score = t - 1.0;
    g.step(Action::Play(0b1)).expect("winning play")
}

fn beat_and_leave_shop(g: &mut Game) {
    g.step(Action::SelectBlind).expect("select");
    beat(g);
    if g.state().phase == Phase::Shop {
        g.step(Action::EndShop).expect("end shop");
    }
}

#[test]
fn deck_names_round_trip() {
    assert_eq!(DeckKind::ALL.len(), 15);
    for d in DeckKind::ALL {
        assert_eq!(DeckKind::from_name(d.name()), Some(d));
        assert_eq!(DeckKind::ALL[d.index()], d);
    }
    assert_eq!(DeckKind::from_name("Plasma"), Some(DeckKind::Plasma));
    assert_eq!(DeckKind::from_name("blue deck"), None);
}

#[test]
fn every_deck_starts_valid() {
    for d in DeckKind::ALL {
        let g = game(d, 1);
        g.state()
            .check_invariants()
            .unwrap_or_else(|e| panic!("{d:?}: {e}"));
        assert_eq!(
            g.state().deck.len() as u32,
            g.state().starting_deck_size,
            "{d:?}"
        );
        assert!(
            g.state().deck.len() == 52 || d == DeckKind::Abandoned,
            "{d:?} has {} cards",
            g.state().deck.len()
        );
    }
}

#[test]
fn red_deck_extra_discard() {
    let s = game(DeckKind::Red, 1);
    // 3 + 1 discards, 4 hands, $4, hand 8, 5 joker / 2 consumable slots.
    let s = s.state();
    assert_eq!(
        (s.round_hands(), s.round_discards(), s.hand_size()),
        (4, 4, 8)
    );
    assert_eq!(s.money, 4);
}

#[test]
fn blue_deck_extra_hand() {
    let g = game(DeckKind::Blue, 1);
    let s = g.state();
    // 4 + 1 hands, 3 discards (no Red bonus).
    assert_eq!((s.round_hands(), s.round_discards()), (5, 3));
    let mut g = game(DeckKind::Blue, 1);
    g.step(Action::SelectBlind).unwrap();
    assert_eq!((g.state().hands_left, g.state().discards_left), (5, 3));
}

#[test]
fn yellow_deck_starts_with_ten_extra_dollars() {
    let g = game(DeckKind::Yellow, 1);
    assert_eq!(g.state().money, 14);
    assert_eq!(game(DeckKind::Red, 1).state().money, 4);
}

#[test]
fn green_deck_pays_per_hand_and_discard_and_no_interest() {
    // Money 25 would earn $5 interest on another deck. Green: 4 hands, 3 discards; one hand
    // played -> 3 hands ($2 each = 6) + 3 discards ($1 each = 3) + Small $3 = 12, interest 0.
    let mut g = game(DeckKind::Green, 1);
    g.state_mut().money = 25;
    g.step(Action::SelectBlind).unwrap();
    let co = beat(&mut g).round.unwrap().cash_out.unwrap();
    assert_eq!(co.interest, 0);
    assert_eq!(co.hands, 9);
    assert_eq!(co.blind_reward, 3);
    assert_eq!(g.state().money, 37);
    // Discards used are not paid.
    let mut g = game(DeckKind::Green, 1);
    g.state_mut().money = 25;
    g.step(Action::SelectBlind).unwrap();
    g.step(Action::Discard(0b1)).unwrap();
    let co = beat(&mut g).round.unwrap().cash_out.unwrap();
    assert_eq!(co.hands, 3 * 2 + 2);
    // Red Deck for comparison: interest 5, hands $3, Small $3, discards pay nothing.
    let mut g = game(DeckKind::Red, 1);
    g.state_mut().money = 25;
    g.step(Action::SelectBlind).unwrap();
    let co = beat(&mut g).round.unwrap().cash_out.unwrap();
    assert_eq!((co.interest, co.hands), (5, 3));
}

#[test]
fn black_deck_joker_slot_and_one_hand_less() {
    let g = game(DeckKind::Black, 1);
    let s = g.state();
    assert_eq!(s.round_hands(), 3);
    assert_eq!(s.joker_capacity(), 6);
    assert_eq!(s.round_discards(), 3);
}

#[test]
fn magic_deck_crystal_ball_and_two_fools() {
    let g = game(DeckKind::Magic, 1);
    let s = g.state();
    assert_eq!(s.vouchers, vec![VoucherId::CrystalBall]);
    assert_eq!(s.consumable_capacity(), 3);
    let ids: Vec<ConsumableId> = s.consumables.iter().map(|c| c.id).collect();
    assert_eq!(ids, vec![ConsumableId::TheFool, ConsumableId::TheFool]);
    // The ante voucher is never the starting one.
    assert_ne!(s.ante_voucher, Some(VoucherId::CrystalBall));
}

#[test]
fn nebula_deck_telescope_and_one_consumable_slot_less() {
    let g = game(DeckKind::Nebula, 1);
    let s = g.state();
    assert_eq!(s.vouchers, vec![VoucherId::Telescope]);
    assert_eq!(s.consumable_capacity(), 1);
    assert!(s.shop_mods.telescope);
}

#[test]
fn ghost_deck_has_spectrals_in_the_shop_and_starts_with_hex() {
    let mut g = game(DeckKind::Ghost, 3);
    assert_eq!(g.state().consumables.len(), 1);
    assert_eq!(g.state().consumables[0].id, ConsumableId::Hex);
    assert!(g.state().deck_spectral_in_shop());
    // Weights: joker 20, tarot 4, planet 4, spectral 2 (playing cards 0).
    assert_eq!(
        shop::card_type_weights(g.state()),
        [20.0, 4.0, 4.0, 0.0, 2.0]
    );
    let n = 3_000;
    let mut spectral = 0;
    for _ in 0..n {
        let c = shop::roll_card(g.state_mut());
        if let ShopCardKind::Consumable(id) = c.kind
            && id.kind() == balatro_core::consumable::ConsumableKind::Spectral
        {
            spectral += 1;
        }
    }
    // 2 / 30 = 6.7%.
    let p = 100.0 * f64::from(spectral) / f64::from(n);
    assert!((5.0..8.5).contains(&p), "spectral {p:.1}%");
    let mut r = game(DeckKind::Red, 3);
    for _ in 0..500 {
        let c = shop::roll_card(r.state_mut());
        if let ShopCardKind::Consumable(id) = c.kind {
            assert_ne!(
                id.kind(),
                balatro_core::consumable::ConsumableKind::Spectral
            );
        }
    }
}

#[test]
fn ghost_deck_hex_with_a_joker_makes_it_polychrome() {
    let mut g = game(DeckKind::Ghost, 3);
    g.add_joker(balatro_core::joker::JokerInstance::new(
        0,
        balatro_core::content::jokers::JokerId::Joker,
    ))
    .unwrap();
    g.step(Action::SelectBlind).unwrap();
    g.step(Action::UseConsumable {
        slot: 0,
        targets: 0,
    })
    .unwrap();
    assert_eq!(
        g.state().jokers[0].edition,
        balatro_core::enums::Edition::Polychrome
    );
    assert!(g.state().consumables.is_empty());
}

#[test]
fn abandoned_deck_has_no_face_cards() {
    let g = game(DeckKind::Abandoned, 1);
    let d = &g.state().deck;
    assert_eq!(d.len(), 40);
    assert_eq!(g.state().starting_deck_size, 40);
    assert!(d.iter_all().all(|c| !c.rank.is_face_rank()));
    // Every other rank in every suit.
    for s in Suit::ALL {
        for r in Rank::ALL.into_iter().filter(|r| !r.is_face_rank()) {
            assert_eq!(
                d.iter_all().filter(|c| c.rank == r && c.suit == s).count(),
                1
            );
        }
    }
    let mut g = g;
    g.step(Action::SelectBlind).unwrap();
    assert_eq!(g.state().deck.hand.len(), 8);
    assert_eq!(g.state().deck.draw_pile.len(), 32);
}

#[test]
fn checkered_deck_is_26_spades_and_26_hearts() {
    let g = game(DeckKind::Checkered, 1);
    let d = &g.state().deck;
    assert_eq!(d.len(), 52);
    assert_eq!(d.iter_all().filter(|c| c.suit == Suit::Spades).count(), 26);
    assert_eq!(d.iter_all().filter(|c| c.suit == Suit::Hearts).count(), 26);
    for r in Rank::ALL {
        assert_eq!(
            d.iter_all()
                .filter(|c| c.rank == r && c.suit == Suit::Spades)
                .count(),
            2
        );
    }
}

#[test]
fn zodiac_deck_three_vouchers() {
    let g = game(DeckKind::Zodiac, 1);
    let s = g.state();
    assert_eq!(
        s.vouchers,
        vec![
            VoucherId::TarotMerchant,
            VoucherId::PlanetMerchant,
            VoucherId::Overstock
        ]
    );
    // Overstock: 3 card slots; Tarot/Planet Merchant raise their shop weight to 9.6.
    assert_eq!(shop::card_slot_count(s), 3);
    let w = shop::card_type_weights(s);
    assert_eq!((w[1], w[2]), (9.6, 9.6));
    assert!(
        ![
            VoucherId::TarotMerchant,
            VoucherId::PlanetMerchant,
            VoucherId::Overstock
        ]
        .contains(&s.ante_voucher.unwrap())
    );
    // The first shop shows 3 card slots.
    let mut g = game(DeckKind::Zodiac, 1);
    beat_and_leave_shop_only_enter(&mut g);
    assert_eq!(g.state().shop.base_slots, 3);
}

/// Win the Small blind and stay in the shop.
fn beat_and_leave_shop_only_enter(g: &mut Game) {
    g.step(Action::SelectBlind).unwrap();
    beat(g);
    assert_eq!(g.state().phase, Phase::Shop);
}

#[test]
fn painted_deck_hand_size_up_joker_slot_down() {
    let mut g = game(DeckKind::Painted, 1);
    let s = g.state();
    assert_eq!(s.hand_size(), 10);
    assert_eq!(s.joker_capacity(), 4);
    g.step(Action::SelectBlind).unwrap();
    assert_eq!(g.state().deck.hand.len(), 10);
}

#[test]
fn anaglyph_deck_double_tag_after_each_boss() {
    let mut g = game(DeckKind::Anaglyph, 4);
    // No Double Tag after the Small and Big blinds.
    for _ in 0..2 {
        beat_and_leave_shop(&mut g);
        assert!(!g.state().tags.contains(&TagId::Double));
    }
    assert_eq!(g.state().blind, BlindKind::Boss);
    beat_and_leave_shop(&mut g);
    assert_eq!(
        g.state()
            .tags
            .iter()
            .filter(|&&t| t == TagId::Double)
            .count(),
        1
    );
    // A second boss adds a second one (the first is still held).
    for _ in 0..3 {
        beat_and_leave_shop(&mut g);
    }
    assert_eq!(
        g.state()
            .tags
            .iter()
            .filter(|&&t| t == TagId::Double)
            .count(),
        2
    );
    // Other decks get none.
    let mut r = game(DeckKind::Red, 4);
    for _ in 0..3 {
        beat_and_leave_shop(&mut r);
    }
    assert!(!r.state().tags.contains(&TagId::Double));
}

#[test]
fn plasma_deck_doubles_blinds_and_balances_score() {
    let g = game(DeckKind::Plasma, 1);
    assert!(g.state().run_view().plasma_deck);
    // Ante 1 Small: 300 x 2.
    assert_eq!(g.state().target, 600.0);
    assert_eq!(game(DeckKind::Red, 1).state().target, 300.0);
    assert!(!game(DeckKind::Red, 1).state().run_view().plasma_deck);
    // All three blinds (Small x1, Big x1.5, Boss x2 before the deck factor): 600 / 900 / 1200
    // unless the boss has its own multiplier (The Wall x4, Violet Vessel x6, The Needle x1).
    let mut g = game(DeckKind::Plasma, 1);
    beat_and_leave_shop(&mut g);
    assert_eq!(g.state().target, 900.0);
    beat_and_leave_shop(&mut g);
    let boss_mult = g.state().boss_mult();
    assert_eq!(g.state().target, 300.0 * boss_mult * 2.0);
    // Scoring: a lone Ace = High Card 5 chips + 11 = 16 chips x 1 mult. Plasma balances to
    // (16 + 1) / 2 = 8.5 each -> 8.5 x 8.5 = 72.25 -> 72.
    let mut p = game(DeckKind::Plasma, 1);
    p.step(Action::SelectBlind).unwrap();
    set_first_card(&mut p, Rank::Ace);
    let rep = p.preview_play(0b1).unwrap();
    assert_eq!(rep.score, 72.0, "{rep:?}");
    let mut r = game(DeckKind::Red, 1);
    r.step(Action::SelectBlind).unwrap();
    set_first_card(&mut r, Rank::Ace);
    assert_eq!(r.preview_play(0b1).unwrap().score, 16.0);
}

fn set_first_card(g: &mut Game, rank: Rank) {
    let c = &mut g.state_mut().deck.hand[0];
    c.rank = rank;
    c.suit = Suit::Spades;
}

#[test]
fn plasma_blind_size_survives_ante_changes() {
    // refresh_target keeps the deck factor (Hieroglyph/Petroglyph move the ante).
    let mut g = game(DeckKind::Plasma, 1);
    balatro_core::voucher::redeem(g.state_mut(), VoucherId::Hieroglyph);
    assert_eq!(g.state().ante, 0);
    assert_eq!(g.state().target, 200.0); // ante 0 base 100 x 2
}

#[test]
fn erratic_deck_is_random_but_deterministic() {
    let sig = |g: &Game| -> Vec<(u8, usize)> {
        g.state()
            .deck
            .draw_pile
            .iter()
            .map(|c| (c.rank as u8, c.suit.index()))
            .collect()
    };
    let a = sig(&game(DeckKind::Erratic, 11));
    assert_eq!(a.len(), 52);
    assert_eq!(a, sig(&game(DeckKind::Erratic, 11)), "same seed, same deck");
    assert_ne!(a, sig(&game(DeckKind::Erratic, 12)));
    // A random deck has repeats (collision probability 1 - 52!/52^52 ~ 1).
    let mut counts: HashMap<(u8, usize), u32> = HashMap::new();
    for k in &a {
        *counts.entry(*k).or_default() += 1;
    }
    assert!(counts.values().any(|&c| c > 1));
    // Over many seeds all ranks and suits occur uniformly-ish: the suit share is ~25% each.
    let mut suits = [0u32; 4];
    for seed in 0..100 {
        for (_, s) in sig(&game(DeckKind::Erratic, seed)) {
            suits[s] += 1;
        }
    }
    for s in suits {
        assert!((1_100..1_500).contains(&s), "{suits:?}"); // 5200 / 4 = 1300
    }
    // Unique card uids and a playable run.
    let g = game(DeckKind::Erratic, 5);
    g.state().check_invariants().unwrap();
    let mut g = g;
    g.step(Action::SelectBlind).unwrap();
    assert_eq!(g.state().deck.hand.len(), 8);
}

#[test]
fn starting_vouchers_are_not_offered_again() {
    for d in [DeckKind::Magic, DeckKind::Nebula, DeckKind::Zodiac] {
        for seed in 0..30u64 {
            let g = game(d, seed);
            let s = g.state();
            if let Some(v) = s.ante_voucher {
                assert!(!s.vouchers.contains(&v), "{d:?} seed {seed}: {v:?}");
            }
        }
    }
}

#[test]
fn deck_effects_stack_with_stake_effects() {
    // Blue Deck on Gold Stake: 5 hands, 3 - 1 = 2 discards, rental etc. enabled.
    let g = Game::new(
        1,
        GameConfig {
            deck: DeckKind::Blue,
            stake: StakeKind::Gold,
            endless: false,
        },
    );
    let s = g.state();
    assert_eq!((s.round_hands(), s.round_discards()), (5, 2));
    // Plasma on Purple: Ante 2 Small = 1,000 x 2.
    let mut g = Game::new(
        1,
        GameConfig {
            deck: DeckKind::Plasma,
            stake: StakeKind::Purple,
            endless: false,
        },
    );
    for _ in 0..3 {
        beat_and_leave_shop(&mut g);
    }
    assert_eq!(g.state().target, 2_000.0);
    // And the RNG key list is still intact.
    assert_eq!(RngKey::COUNT, 9);
}
