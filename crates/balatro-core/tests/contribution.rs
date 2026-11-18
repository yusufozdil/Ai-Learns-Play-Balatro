//! `contribution`: next-round hands / discards and the per-joker contribution block of the
//! observation (interface v3).

use balatro_core::actions::Action;
use balatro_core::blinds::BlindKind;
use balatro_core::card::Card;
use balatro_core::content::ids::BossId;
use balatro_core::content::jokers::JokerId;
use balatro_core::contribution::{ContribCtx, Contribution, joker_contribution, next_round_budget};
use balatro_core::enums::{Enhancement, Phase, Rank, Suit};
use balatro_core::game::Game;
use balatro_core::joker::JokerInstance;
use balatro_core::state::GameConfig;

fn game(jokers: &[JokerId]) -> Game {
    let mut g = Game::new(7, GameConfig::default());
    g.state_mut().joker_slots = 30;
    for (i, &id) in jokers.iter().enumerate() {
        g.add_joker(JokerInstance::new(100 + i as u32, id)).unwrap();
    }
    assert_eq!(g.state().phase, Phase::BlindSelect);
    g
}

fn c(g: &Game, slot: usize) -> Contribution {
    joker_contribution(g.state(), slot)
}

#[test]
fn budget_red_white_and_modifiers() {
    assert_eq!(next_round_budget(game(&[]).state()), (4, 3 + 1));
    assert_eq!(
        next_round_budget(game(&[JokerId::Drunkard]).state()),
        (4, 5)
    );
    assert_eq!(
        next_round_budget(game(&[JokerId::MerryAndy]).state()),
        (4, 7)
    );
    assert_eq!(
        next_round_budget(game(&[JokerId::Troubadour]).state()),
        (3, 4)
    );
    assert_eq!(next_round_budget(game(&[JokerId::Burglar]).state()), (7, 0));
    // Blueprint copies Burglar's blind-select hook; a debuffed Burglar does nothing.
    let g = game(&[JokerId::Blueprint, JokerId::Burglar, JokerId::Drunkard]);
    assert_eq!(next_round_budget(g.state()), (10, 0));
    let mut g = game(&[JokerId::Burglar]);
    g.state_mut().jokers[0].debuffed = true;
    assert_eq!(next_round_budget(g.state()), (4, 4));
    // Deck / stake modifiers.
    let g = Game::new(
        1,
        GameConfig {
            deck: balatro_core::state::DeckKind::Blue,
            stake: balatro_core::state::StakeKind::Blue,
            endless: false,
        },
    );
    assert_eq!(next_round_budget(g.state()), (5, 2)); // Blue Deck +1 hand, Blue Stake -1 discard
}

/// The formula agrees with what `SelectBlind` actually gives, bosses included.
#[test]
fn budget_matches_the_engine_at_blind_select() {
    let sets: &[&[JokerId]] = &[
        &[],
        &[JokerId::Burglar],
        &[JokerId::Burglar, JokerId::Drunkard, JokerId::Brainstorm],
        &[JokerId::MerryAndy, JokerId::Troubadour],
        &[JokerId::Chicot, JokerId::Drunkard],
        &[JokerId::Chicot, JokerId::Burglar],
    ];
    for jokers in sets {
        for boss in [
            None,
            Some(BossId::TheNeedle),
            Some(BossId::TheWater),
            Some(BossId::TheHook),
        ] {
            let mut g = game(jokers);
            if let Some(b) = boss {
                let s = g.state_mut();
                s.blind = BlindKind::Boss;
                s.boss = Some(b);
            }
            let want = next_round_budget(g.state());
            g.step(Action::SelectBlind).unwrap();
            let s = g.state();
            assert_eq!((s.hands_left, s.discards_left), want, "{jokers:?} {boss:?}");
            // In the round the budget is the round's own start.
            assert_eq!(next_round_budget(s), want, "{jokers:?} {boss:?} (Play)");
        }
    }
}

#[test]
fn banner_mystic_summit_delayed_gratification_and_burglar() {
    let g = game(&[
        JokerId::Banner,
        JokerId::MysticSummit,
        JokerId::DelayedGratification,
    ]);
    assert_eq!(c(&g, 0).chips, 30.0 * 4.0);
    assert_eq!(c(&g, 1), Contribution::NONE);
    assert_eq!(c(&g, 2).money, 8.0);
    // Burglar: no discards next round -> Banner 0, Summit always on, no payout.
    let g = game(&[
        JokerId::Banner,
        JokerId::MysticSummit,
        JokerId::DelayedGratification,
        JokerId::Burglar,
    ]);
    assert_eq!(c(&g, 0), Contribution::NONE);
    assert_eq!(c(&g, 1).mult, 15.0);
    assert_eq!(c(&g, 2), Contribution::NONE);
    assert_eq!(
        c(&g, 3),
        Contribution::NONE,
        "Burglar itself is in the budget"
    );
    // In the round: live discards, Delayed Gratification needs no discard used.
    let mut g = game(&[JokerId::Banner, JokerId::DelayedGratification]);
    g.step(Action::SelectBlind).unwrap();
    assert_eq!(c(&g, 0).chips, 120.0);
    let s = g.state_mut();
    s.discards_left = 1;
    s.discards_used_round = 3;
    assert_eq!(c(&g, 0).chips, 30.0);
    assert_eq!(c(&g, 1), Contribution::NONE);
}

#[test]
fn abstract_swashbuckler_and_debuffed() {
    let mut g = game(&[
        JokerId::AbstractJoker,
        JokerId::Swashbuckler,
        JokerId::Joker,
        JokerId::Egg,
    ]);
    {
        let s = g.state_mut();
        s.jokers[0].sell_value = 4;
        s.jokers[1].sell_value = 100; // its own sell value does not count
        s.jokers[2].sell_value = 1;
        s.jokers[3].sell_value = 9;
    }
    assert_eq!(c(&g, 0).mult, 12.0);
    assert_eq!(c(&g, 1).mult, 4.0 + 1.0 + 9.0);
    assert_eq!(c(&g, 2).mult, 4.0);
    assert_eq!(c(&g, 3), Contribution::NONE);
    g.state_mut().jokers[1].debuffed = true;
    assert_eq!(c(&g, 1), Contribution::NONE);
    // A debuffed joker still counts for Abstract (it is in the joker area).
    assert_eq!(c(&g, 0).mult, 12.0);
}

#[test]
fn raised_fist_play_uses_the_lowest_visible_card() {
    let mut g = game(&[JokerId::RaisedFist]);
    g.step(Action::SelectBlind).unwrap();
    let s = g.state_mut();
    s.deck.hand[0] = Card::new(900, Rank::Seven, Suit::Hearts);
    s.deck.hand[1] = Card::new(901, Rank::Three, Suit::Spades);
    for (i, card) in s.deck.hand.iter_mut().enumerate().skip(2) {
        *card = Card::new(910 + i as u32, Rank::King, Suit::Clubs);
    }
    assert_eq!(c(&g, 0).mult, 6.0);
    // A face-down card is not seen; a Stone card never counts.
    g.state_mut().deck.hand[1].face_down = true;
    assert_eq!(c(&g, 0).mult, 14.0);
    g.state_mut().deck.hand[0].enhancement = Enhancement::Stone;
    assert_eq!(c(&g, 0).mult, 20.0);
    // The lowest card debuffed: Raised Fist triggers on it and does nothing.
    g.state_mut().deck.hand[2].rank = Rank::Two;
    g.state_mut().deck.hand[2].debuffed = true;
    assert_eq!(c(&g, 0).mult, 0.0);
}

#[test]
fn raised_fist_shop_is_the_exact_expectation() {
    let g = game(&[JokerId::RaisedFist]);
    let s = g.state();
    let got = c(&g, 0).mult;
    // Monte Carlo over the same deck and hand size.
    let ranks: Vec<Option<Rank>> = s
        .deck
        .iter_all()
        .map(|c| (!c.is_stone()).then_some(c.rank))
        .collect();
    let h = s.hand_size() as usize;
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = |n: usize| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x % n as u64) as usize
    };
    let trials = 200_000;
    let mut sum = 0.0;
    let mut idx: Vec<usize> = (0..ranks.len()).collect();
    for _ in 0..trials {
        for i in 0..h {
            let j = i + next(idx.len() - i);
            idx.swap(i, j);
        }
        let low = idx[..h].iter().filter_map(|&i| ranks[i]).min();
        sum += low.map_or(0.0, |r| 2.0 * f64::from(r.chip_value()));
    }
    let mc = sum / f64::from(trials);
    assert!(
        (got - mc).abs() < 0.05,
        "analytic {got} vs Monte Carlo {mc}"
    );
    // 8 of 52 cards: the minimum is almost surely low (~2 x 3.0).
    assert!(got > 4.0 && got < 8.0, "{got}");

    // Degenerate decks: only Kings -> exactly 20; Kings plus Stones -> 20 x P(any King drawn).
    let mut g = game(&[JokerId::RaisedFist]);
    for c in g.state_mut().deck.draw_pile.iter_mut() {
        c.rank = Rank::King;
    }
    assert!((c(&g, 0).mult - 20.0).abs() < 1e-12);
    let n = g.state().deck.len();
    for (i, card) in g.state_mut().deck.draw_pile.iter_mut().enumerate() {
        if i >= 4 {
            card.enhancement = Enhancement::Stone;
        }
    }
    let p_none: f64 = (0..8)
        .map(|i| (n - 4 - i) as f64 / (n - i) as f64)
        .product();
    assert!((c(&g, 0).mult - 20.0 * (1.0 - p_none)).abs() < 1e-12);
}

#[test]
fn deck_money_and_counter_jokers() {
    let mut g = game(&[
        JokerId::Bull,
        JokerId::Bootstraps,
        JokerId::JokerStencil,
        JokerId::FortuneTeller,
        JokerId::Throwback,
        JokerId::Satellite,
        JokerId::SteelJoker,
        JokerId::StoneJoker,
        JokerId::Cloud9,
        JokerId::Erosion,
        JokerId::BlueJoker,
    ]);
    {
        let s = g.state_mut();
        s.money = 23;
        s.stats.run.tarots_used = 5;
        s.stats.run.blinds_skipped = 2;
        s.stats.run.unique_planets_used = 3;
        s.deck.draw_pile[0].enhancement = Enhancement::Steel;
        s.deck.draw_pile[1].enhancement = Enhancement::Steel;
        s.deck.draw_pile[2].enhancement = Enhancement::Stone;
        s.deck.draw_pile.pop();
        s.joker_slots = 30;
    }
    let s = g.state();
    let nines = s
        .deck
        .iter_all()
        .filter(|c| !c.is_stone() && c.rank == Rank::Nine)
        .count() as f64;
    assert_eq!(c(&g, 0).chips, 46.0);
    assert_eq!(c(&g, 1).mult, 8.0);
    assert_eq!(c(&g, 2).xmult, 30.0 - 11.0 + 1.0);
    assert_eq!(c(&g, 3).mult, 5.0);
    assert_eq!(c(&g, 4).xmult, 1.5);
    assert_eq!(c(&g, 5).money, 3.0);
    assert!((c(&g, 6).xmult - 1.4).abs() < 1e-12);
    assert_eq!(c(&g, 7).chips, 25.0);
    assert_eq!(c(&g, 8).money, nines);
    assert_eq!(c(&g, 9).mult, 4.0);
    assert_eq!(c(&g, 10).chips, 2.0 * (51.0 - 8.0));
}

#[test]
fn scaling_jokers_copiers_and_hidden_jokers() {
    let mut g = game(&[
        JokerId::Blueprint,
        JokerId::GreenJoker,
        JokerId::Constellation,
        JokerId::Runner,
        JokerId::Brainstorm,
    ]);
    {
        let s = g.state_mut();
        s.jokers[1].value = 7.0;
        s.jokers[2].value = 2.5;
        s.jokers[3].value = 45.0;
    }
    assert_eq!(c(&g, 1).mult, 8.0, "the next hand adds +1 first");
    assert_eq!(c(&g, 2).xmult, 2.5);
    assert_eq!(c(&g, 3).chips, 45.0);
    assert_eq!(
        c(&g, 0),
        c(&g, 1),
        "Blueprint copies the joker to its right"
    );
    // Brainstorm copies the leftmost (Blueprint -> Green Joker).
    assert_eq!(c(&g, 4).mult, 8.0);
    // Amber Acorn: every joker hidden.
    g.step(Action::SelectBlind).unwrap();
    let s = g.state_mut();
    s.blind = BlindKind::Boss;
    s.boss = Some(BossId::AmberAcorn);
    s.boss_state.jokers_hidden = true;
    let ctx = ContribCtx::new(g.state());
    for i in 0..5 {
        assert_eq!(ctx.joker(i), Contribution::NONE);
    }
}

#[test]
fn round_end_money_is_not_copied() {
    let g = game(&[JokerId::Blueprint, JokerId::GoldenJoker, JokerId::Rocket]);
    assert_eq!(c(&g, 1).money, 4.0);
    assert_eq!(c(&g, 0), Contribution::NONE);
    assert_eq!(c(&g, 2).money, 1.0);
}

#[test]
fn acrobat_card_sharp_supernova_loyalty() {
    let mut g = game(&[
        JokerId::Acrobat,
        JokerId::CardSharp,
        JokerId::Supernova,
        JokerId::LoyaltyCard,
    ]);
    assert_eq!(c(&g, 0), Contribution::NONE);
    assert_eq!(c(&g, 1), Contribution::NONE, "first hand of a round");
    assert_eq!(c(&g, 2).mult, 1.0);
    // Loyalty was acquired at 0 hands played: the 6th hand fires.
    g.state_mut().stats.hands_played = 5;
    assert_eq!(c(&g, 3).xmult, 4.0);
    g.state_mut().stats.hands_played = 6;
    assert_eq!(c(&g, 3), Contribution::NONE);
    g.step(Action::SelectBlind).unwrap();
    let s = g.state_mut();
    s.hands_left = 1;
    s.hand_plays_round[3] = 1;
    s.hand_plays_run[3] = 4;
    assert_eq!(c(&g, 0).xmult, 3.0);
    assert_eq!(c(&g, 1).xmult, 3.0);
    assert_eq!(c(&g, 2).mult, 5.0);
}

/// In a round, the block equals what the scorer's own trace attributes to each joker, for jokers
/// whose effect does not depend on the played cards (the hand is the single highest card, so
/// Raised Fist's lowest card stays held).
#[test]
fn play_phase_matches_the_scoring_trace() {
    use balatro_core::scoring::{Scorer, TraceOp, TraceSource};
    let ids = [
        JokerId::Banner,
        JokerId::AbstractJoker,
        JokerId::Swashbuckler,
        JokerId::Bull,
        JokerId::Bootstraps,
        JokerId::JokerStencil,
        JokerId::FortuneTeller,
        JokerId::Throwback,
        JokerId::SteelJoker,
        JokerId::StoneJoker,
        JokerId::Erosion,
        JokerId::BlueJoker,
        JokerId::Acrobat,
        JokerId::RaisedFist,
        JokerId::Supernova,
        JokerId::CardSharp,
        JokerId::LoyaltyCard,
        JokerId::Joker,
        JokerId::Stuntman,
        JokerId::DriversLicense,
        JokerId::MysticSummit,
        JokerId::Blueprint,
        JokerId::GreenJoker,
    ];
    for variant in 0..3u8 {
        let mut g = game(&ids);
        g.step(Action::SelectBlind).unwrap();
        {
            let s = g.state_mut();
            s.money = 17 + i32::from(variant);
            s.stats.run.tarots_used = 3;
            s.stats.run.blinds_skipped = 1;
            s.stats.hands_played = 5 + u32::from(variant == 1);
            for (i, j) in s.jokers.iter_mut().enumerate() {
                j.sell_value = 1 + i as i32 % 4;
            }
            s.jokers[ids.len() - 1].value = 6.0;
            s.deck.draw_pile[0].enhancement = Enhancement::Steel;
            s.deck.draw_pile[1].enhancement = Enhancement::Stone;
            s.deck.draw_pile.pop();
            if variant == 1 {
                s.hands_left = 1;
                s.discards_left = 0;
                s.hand_plays_round[0] = 1;
                for c in s.deck.draw_pile.iter_mut().take(16) {
                    c.enhancement = Enhancement::Bonus;
                }
            }
            if variant == 2 {
                s.discards_left = 2;
                s.hand_plays_run = [3; 12];
            }
        }
        let s = g.state();
        let hi = (0..s.deck.hand.len())
            .max_by_key(|&i| (s.deck.hand[i].rank, i))
            .unwrap();
        let ctx = ContribCtx::new(s);
        let mut scorer = Scorer::with_trace(true);
        let rep = g.preview_play_with(&mut scorer, 1 << hi).unwrap();
        assert_eq!(rep.hand_type, balatro_core::enums::HandType::HighCard);
        for (slot, id) in ids.iter().enumerate() {
            let mut got = Contribution::NONE;
            for ev in scorer.trace() {
                if let TraceSource::Joker { slot: k, .. } = ev.source
                    && usize::from(k) == slot
                {
                    match ev.op {
                        TraceOp::AddChips(x) => got.chips += x,
                        TraceOp::AddMult(x) => got.mult += x,
                        TraceOp::XMult(x) => got.xmult *= x,
                        _ => {}
                    }
                }
            }
            let want = ctx.joker(slot);
            for (a, b, what) in [
                (got.chips, want.chips, "chips"),
                (got.mult, want.mult, "mult"),
                (got.xmult, want.xmult, "xmult"),
            ] {
                assert!(
                    (a - b).abs() < 1e-9,
                    "variant {variant} {:?} {what}: scored {a}, block {b}",
                    id
                );
            }
        }
    }
}
