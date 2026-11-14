//! Phase 8A: fair-information previews and the "what the player sees" readers.
//!
//! `preview_play_fair` / `fair_view` must not read (a) the real future RNG, (b) the identity of
//! face-down cards, (c) the draw order; and the observation readers must hide face-down cards and
//! Amber Acorn's jokers.

use balatro_core::actions::Action;
use balatro_core::agents::{Agent, GreedyAgent, HeuristicAgent, HeuristicConfig};
use balatro_core::card::Card;
use balatro_core::content::jokers::JokerId;
use balatro_core::enums::{Enhancement, Phase, Rank, Suit};
use balatro_core::game::Game;
use balatro_core::joker::JokerInstance;
use balatro_core::observe::CardHistogram;
use balatro_core::rng::GameRng;
use balatro_core::state::GameConfig;
use std::collections::BTreeMap;

fn game_in_play(seed: u64) -> Game {
    let mut g = Game::new(seed, GameConfig::default());
    g.step(Action::SelectBlind).unwrap();
    assert_eq!(g.state().phase, Phase::Play);
    g
}

/// Hand of Lucky cards plus Misprint: lots of probabilistic scoring.
fn lucky_game(seed: u64) -> Game {
    let mut g = game_in_play(seed);
    for c in g.state_mut().deck.hand.iter_mut() {
        c.enhancement = Enhancement::Lucky;
    }
    g.add_joker(JokerInstance::new(0, JokerId::Misprint))
        .expect("room");
    g
}

fn masks(g: &Game) -> Vec<u32> {
    (1u32..(1 << g.state().deck.hand.len().min(8)))
        .filter(|m| m.count_ones() <= 5)
        .collect()
}

#[test]
fn fair_preview_does_not_depend_on_the_real_rng() {
    for seed in 0..6u64 {
        let a = lucky_game(seed);
        let mut b = a.clone();
        b.state_mut().rng = GameRng::new(0xDEAD_BEEF ^ seed);
        let mut full_differs = false;
        for m in masks(&a).into_iter().step_by(7) {
            for fair_seed in 0..3u64 {
                let (ra, rb) = (
                    a.preview_play_fair(m, fair_seed).unwrap(),
                    b.preview_play_fair(m, fair_seed).unwrap(),
                );
                assert_eq!(ra, rb, "mask {m:#b} seed {fair_seed}");
            }
            // Sanity: the full-information preview does read the real RNG.
            full_differs |= a.preview_play(m).unwrap() != b.preview_play(m).unwrap();
        }
        assert!(
            full_differs,
            "the full preview should see a different future"
        );
    }
}

#[test]
fn fair_preview_is_random_across_seeds_and_not_the_real_outcome() {
    let g = lucky_game(2);
    let m = 0b1_1111u32;
    let mut scores: BTreeMap<u64, u32> = BTreeMap::new();
    for s in 0..60u64 {
        *scores
            .entry(g.preview_play_fair(m, s).unwrap().score.to_bits())
            .or_default() += 1;
    }
    assert!(scores.len() >= 3, "fair previews should vary: {scores:?}");
    // Deterministic for a fixed seed.
    assert_eq!(
        g.preview_play_fair(m, 9).unwrap(),
        g.preview_play_fair(m, 9).unwrap()
    );
    // The real outcome (what `step` will produce) is the full preview, and the fair previews
    // are not all equal to it.
    let real = g.preview_play(m).unwrap();
    let mut played = g.clone();
    let stepped = played.step(Action::Play(m)).unwrap().hand.unwrap();
    assert_eq!(real.score, stepped.score);
    assert!(
        (0..60u64).any(|s| g.preview_play_fair(m, s).unwrap().score != real.score),
        "the fair preview leaks the real outcome"
    );
}

#[test]
fn fair_preview_equals_full_preview_without_chance() {
    let g = game_in_play(4);
    for m in masks(&g) {
        assert_eq!(
            g.preview_play_fair(m, 1).unwrap(),
            g.preview_play(m).unwrap()
        );
    }
    assert!(g.preview_play_fair(0, 1).is_err());
    assert!(g.preview_play_fair(0b11_1111, 1).is_err());
}

/// A copy of `g` with the identities of the hand cards flagged in `hidden` swapped with draw-pile
/// cards, the draw pile reordered and the RNG replaced: the same game for the player.
fn scramble_hidden(g: &Game, hidden: &[usize]) -> Game {
    let mut b = g.clone();
    let s = b.state_mut();
    for (k, &i) in hidden.iter().enumerate() {
        let j = 3 + 5 * k;
        let (mut h, mut p): (Card, Card) = (s.deck.hand[i], s.deck.draw_pile[j]);
        std::mem::swap(&mut h, &mut p);
        h.face_down = true;
        p.face_down = false;
        s.deck.hand[i] = h;
        s.deck.draw_pile[j] = p;
    }
    s.deck.draw_pile.reverse();
    s.rng = GameRng::new(4242);
    b
}

fn hide(g: &mut Game, hidden: &[usize]) {
    for &i in hidden {
        g.state_mut().deck.hand[i].face_down = true;
    }
}

#[test]
fn face_down_identity_and_draw_order_are_not_used() {
    for seed in 0..5u64 {
        let mut a = lucky_game(seed);
        // Give the face-down cards distinctive identities so a leak would show.
        let hidden = [1usize, 4];
        a.state_mut().deck.hand[1].enhancement = Enhancement::Glass;
        a.state_mut().deck.hand[4].enhancement = Enhancement::Steel;
        hide(&mut a, &hidden);
        let b = scramble_hidden(&a, &hidden);
        let mut full_differs = false;
        for m in masks(&a).into_iter().step_by(5) {
            for fair_seed in 0..3u64 {
                assert_eq!(
                    a.preview_play_fair(m, fair_seed).unwrap(),
                    b.preview_play_fair(m, fair_seed).unwrap(),
                    "mask {m:#b} seed {fair_seed}"
                );
            }
            full_differs |= a.preview_play(m).unwrap() != b.preview_play(m).unwrap();
        }
        assert!(
            full_differs,
            "the hidden identities should matter to the full preview"
        );
        // The fair views are the same state.
        assert!(
            a.fair_view(5).state() == b.fair_view(5).state(),
            "the fair views of the same player-visible game differ"
        );
    }
}

#[test]
fn fair_view_replaces_only_hidden_cards_order_and_rng() {
    let mut g = game_in_play(8);
    let hidden = [0usize, 2, 7];
    hide(&mut g, &hidden);
    let v = g.fair_view(77);
    let (s, t) = (g.state(), v.state());
    // Visible cards, hand length, piles' contents and every other field are untouched.
    assert_eq!(s.deck.hand.len(), t.deck.hand.len());
    for (i, (x, y)) in s.deck.hand.iter().zip(&t.deck.hand).enumerate() {
        if hidden.contains(&i) {
            assert!(y.face_down);
        } else {
            assert_eq!(x, y);
        }
    }
    // The unseen cards (draw pile + hidden hand cards) are the same set before and after; the
    // draw pile is in canonical order.
    let unseen = |st: &balatro_core::state::GameState| {
        let mut v: Vec<u32> = st
            .deck
            .draw_pile
            .iter()
            .map(|c| c.uid)
            .chain(hidden.iter().map(|&i| st.deck.hand[i].uid))
            .collect();
        v.sort_unstable();
        v
    };
    assert_eq!(unseen(s), unseen(t));
    let b: Vec<u32> = t.deck.draw_pile.iter().map(|c| c.uid).collect();
    assert!(b.windows(2).all(|w| w[0] < w[1]), "canonical draw order");
    // Everything else is untouched.
    let mut t2 = t.clone();
    let mut s2 = s.clone();
    t2.deck.hand = s2.deck.hand.clone();
    t2.deck.draw_pile = s2.deck.draw_pile.clone();
    t2.rng = s2.rng.clone();
    s2.rng = t2.rng.clone();
    assert!(
        s2 == t2,
        "the fair view changed more than hidden cards, order and rng"
    );
    // Sampled hidden cards come from the unseen pool: draw pile + the hidden cards.
    let pool: Vec<u32> = s
        .deck
        .draw_pile
        .iter()
        .map(|c| c.uid)
        .chain(hidden.iter().map(|&i| s.deck.hand[i].uid))
        .collect();
    let mut seen = Vec::new();
    for &i in &hidden {
        assert!(pool.contains(&t.deck.hand[i].uid));
        seen.push(t.deck.hand[i].uid);
    }
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), hidden.len(), "sampled without replacement");
}

#[test]
fn hidden_card_samples_are_uniform_over_the_unseen_cards() {
    let mut g = game_in_play(12);
    hide(&mut g, &[3]);
    let pool = g.state().deck.draw_pile.len() + 1;
    let mut counts: BTreeMap<u32, u32> = BTreeMap::new();
    let n = 4000u32;
    for s in 0..u64::from(n) {
        let v = g.fair_view(s);
        *counts.entry(v.state().deck.hand[3].uid).or_default() += 1;
    }
    assert_eq!(counts.len(), pool, "every unseen card is possible");
    let expect = f64::from(n) / pool as f64;
    for (&uid, &c) in &counts {
        assert!(
            f64::from(c) > expect * 0.4 && f64::from(c) < expect * 1.9,
            "card {uid}: {c} draws, expected ~{expect:.0}"
        );
    }
}

#[test]
fn fair_agents_ignore_hidden_information() {
    // Same game for the player: hidden identities swapped, draw order reversed, other RNG.
    let cfg = HeuristicConfig {
        shop_samples: 3,
        arrange_samples: 2,
        mc_samples: 3,
        ..HeuristicConfig::default()
    };
    assert!(cfg.fair_info, "fair information is the default");
    let mut checked = 0;
    for seed in 0..12u64 {
        let mut a = lucky_game(seed);
        a.state_mut().target = 1.0e9;
        let hidden = [2usize, 5];
        hide(&mut a, &hidden);
        let b = scramble_hidden(&a, &hidden);
        let (mut h1, mut h2) = (
            HeuristicAgent::with_config(seed, cfg.clone()),
            HeuristicAgent::with_config(seed, cfg.clone()),
        );
        assert_eq!(h1.act(&a), h2.act(&b), "heuristic, seed {seed}");
        let (mut g1, mut g2) = (
            GreedyAgent::new().with_fair_info(true),
            GreedyAgent::new().with_fair_info(true),
        );
        assert_eq!(g1.act(&a), g2.act(&b), "greedy, seed {seed}");
        checked += 1;
    }
    assert_eq!(checked, 12);
}

#[test]
fn fair_agents_are_deterministic_and_legal() {
    let cfg = HeuristicConfig {
        shop_samples: 2,
        arrange_samples: 2,
        mc_samples: 2,
        ..HeuristicConfig::default()
    };
    let run = |seed: u64, greedy: bool| -> Vec<Action> {
        let mut g = Game::new(seed, GameConfig::default());
        let mut agent: Box<dyn Agent> = if greedy {
            Box::new(GreedyAgent::new().with_fair_info(true))
        } else {
            Box::new(HeuristicAgent::with_config(seed, cfg.clone()))
        };
        let mut log = Vec::new();
        while !g.is_terminal() {
            assert!(log.len() < 5000, "seed {seed}: no termination");
            let a = agent.act(&g);
            assert!(g.legal_actions().contains(&a), "seed {seed}: {a:?}");
            g.step(a).unwrap();
            log.push(a);
        }
        log
    };
    for seed in 0..10u64 {
        for greedy in [false, true] {
            assert_eq!(
                run(seed, greedy),
                run(seed, greedy),
                "seed {seed} greedy={greedy}"
            );
        }
    }
}

#[test]
fn visible_card_hides_face_down_cards() {
    let mut g = game_in_play(1);
    hide(&mut g, &[2]);
    let hand = &g.state().deck.hand;
    let vis: Vec<Option<Card>> = g.state().visible_hand().collect();
    assert_eq!(vis.len(), hand.len());
    assert!(vis[2].is_none());
    for (i, v) in vis.iter().enumerate() {
        if i != 2 {
            let v = v.expect("face-up");
            assert_eq!(v.uid, 0, "uid must not reach the observation");
            assert_eq!((v.rank, v.suit), (hand[i].rank, hand[i].suit));
        }
    }
}

#[test]
fn amber_acorn_hides_jokers() {
    use balatro_core::blinds::BlindKind;
    use balatro_core::content::BossId;
    let mut g = game_in_play(3);
    g.add_joker(JokerInstance::new(0, JokerId::Joker)).unwrap();
    assert!(!g.state().jokers_hidden());
    assert!(g.state().visible_joker(0).is_some());
    {
        let s = g.state_mut();
        s.boss = Some(BossId::AmberAcorn);
        s.blind = BlindKind::Boss;
        s.boss_disabled = false;
        s.boss_state.jokers_hidden = true;
    }
    assert!(g.state().jokers_hidden());
    assert!(g.state().visible_joker(0).is_none());
    assert_eq!(g.state().jokers.len(), 1, "the count stays visible");
    // Disabled boss: jokers are visible again.
    g.state_mut().boss_disabled = true;
    assert!(g.state().visible_joker(0).is_some());
}

#[test]
fn histograms_count_the_draw_pile() {
    let g = game_in_play(6);
    let s = g.state();
    let h = s.draw_pile_histogram();
    assert_eq!(h.total as usize, s.deck.draw_pile.len());
    let by_cells: u32 = h
        .rank_suit
        .iter()
        .flatten()
        .map(|&c| u32::from(c))
        .sum::<u32>()
        + u32::from(h.stone);
    assert_eq!(by_cells, h.total);
    let whole = s.deck_histogram();
    assert_eq!(whole.total as usize, s.deck.len());
    assert_eq!(whole.count(Rank::Ace, Suit::Spades), 1);
    let manual = CardHistogram::from_cards(&s.deck.draw_pile);
    assert_eq!(manual, h);
    assert_eq!(s.discard_histogram().total, 0);
    // Readers that wrap prices.
    assert_eq!(
        s.hand_chips_mult(balatro_core::enums::HandType::Pair),
        (10.0, 2.0)
    );
}
