//! `Game::reseed_hidden` (branching API): the visible state is untouched, the hidden one is
//! re-sampled, per-stream mode only touches the chosen streams, the result is a pure function
//! of the seed.

use balatro_core::agents::{Agent, GreedyAgent};
use balatro_core::game::Game;
use balatro_core::rng::RngKey;
use balatro_core::serialize::state_to_json;
use balatro_core::state::{GameConfig, GameState};

/// The state with everything hidden normalised: draw pile in `uid` order, face-down cards and
/// RNG blanked.
fn visible(s: &GameState) -> String {
    let mut s = s.clone();
    s.deck.draw_pile.sort_unstable_by_key(|c| c.uid);
    for c in s.deck.hand.iter_mut().filter(|c| c.face_down) {
        c.rank = balatro_core::enums::Rank::Two;
        c.suit = balatro_core::enums::Suit::Spades;
        c.uid = 0;
    }
    s.rng = balatro_core::rng::GameRng::new(0);
    if s.stats.shops_entered == 0 {
        s.ante_voucher = None; // hidden until the first shop
    }
    state_to_json(&s, false)
}

fn next(g: &Game, key: RngKey) -> u64 {
    g.state().rng.clone().next_u64(key)
}

#[test]
fn reseed_keeps_what_is_visible_and_changes_what_is_hidden() {
    let mut changed_pile = 0;
    let mut checked = 0;
    for seed in 0..6u64 {
        let mut g = Game::new(seed, GameConfig::default());
        let mut agent = GreedyAgent::new();
        for step in 0..600 {
            if g.is_terminal() {
                break;
            }
            if step % 5 == 0 {
                let mut c = g.clone();
                c.reseed_hidden(seed * 1000 + step, None);
                assert_eq!(
                    visible(g.state()),
                    visible(c.state()),
                    "seed {seed} step {step}"
                );
                c.state().check_invariants().unwrap();
                // A reseeded game is a normal game: it can be played on.
                let mut p = c.clone();
                let mut pa = GreedyAgent::new();
                for _ in 0..40 {
                    if p.is_terminal() {
                        break;
                    }
                    let a = pa.act(&p);
                    p.step(a).unwrap();
                }
                p.state().check_invariants().unwrap();
                assert_ne!(
                    state_to_json(g.state(), false),
                    state_to_json(c.state(), false)
                );
                if c.state().deck.draw_pile != g.state().deck.draw_pile {
                    changed_pile += 1;
                }
                // Same seed, same result; another seed, another one.
                let mut d = g.clone();
                d.reseed_hidden(seed * 1000 + step, None);
                assert_eq!(
                    state_to_json(c.state(), false),
                    state_to_json(d.state(), false)
                );
                d.reseed_hidden(seed * 1000 + step + 1, None);
                assert_ne!(
                    state_to_json(c.state(), false),
                    state_to_json(d.state(), false)
                );
                checked += 1;
            }
            let a = agent.act(&g);
            g.step(a).unwrap();
        }
    }
    assert!(checked > 20, "{checked}");
    assert!(
        changed_pile > 10,
        "the draw order should change: {changed_pile}"
    );
}

#[test]
fn all_streams_equal_listing_every_stream_and_matches_new() {
    let mut g = Game::new(3, GameConfig::default());
    let mut agent = GreedyAgent::new();
    for _ in 0..40 {
        let a = agent.act(&g);
        g.step(a).unwrap();
    }
    let mut a = g.clone();
    let mut b = g.clone();
    a.reseed_hidden(9, None);
    b.reseed_hidden(9, Some(&RngKey::ALL));
    assert_eq!(
        state_to_json(a.state(), false),
        state_to_json(b.state(), false)
    );
    // Every stream is the one `GameRng::new(9)` has, advanced only by the pile shuffle (Deck).
    let fresh = balatro_core::rng::GameRng::new(9);
    for k in RngKey::ALL {
        if k != RngKey::Deck {
            assert_eq!(next(&a, k), fresh.clone().next_u64(k), "{k:?}");
        }
    }
}

#[test]
fn partial_reseed_only_touches_the_chosen_streams() {
    let mut g = Game::new(5, GameConfig::default());
    let mut agent = GreedyAgent::new();
    for _ in 0..30 {
        let a = agent.act(&g);
        g.step(a).unwrap();
    }
    // Shop only: cards and every other stream untouched.
    let mut c = g.clone();
    c.reseed_hidden(77, Some(&[RngKey::Shop]));
    assert_eq!(c.state().deck.draw_pile, g.state().deck.draw_pile);
    assert_eq!(c.state().deck.hand, g.state().deck.hand);
    for k in RngKey::ALL {
        if k == RngKey::Shop {
            assert_ne!(next(&c, k), next(&g, k));
        } else {
            assert_eq!(next(&c, k), next(&g, k), "{k:?}");
        }
    }
    // Deck: the pile is reshuffled (same cards), the other streams stay.
    let mut d = g.clone();
    d.reseed_hidden(77, Some(&[RngKey::Deck]));
    let mut p0 = g.state().deck.draw_pile.clone();
    let mut p1 = d.state().deck.draw_pile.clone();
    assert_ne!(p0, p1);
    p0.sort_unstable_by_key(|c| c.uid);
    p1.sort_unstable_by_key(|c| c.uid);
    assert_eq!(p0, p1);
    for k in RngKey::ALL {
        if k != RngKey::Deck {
            assert_eq!(next(&d, k), next(&g, k), "{k:?}");
        }
    }
}

#[test]
fn unreseeded_clone_is_the_same_future() {
    let mut g = Game::new(8, GameConfig::default());
    let mut agent = GreedyAgent::new();
    for _ in 0..25 {
        let a = agent.act(&g);
        g.step(a).unwrap();
    }
    let mut c = g.clone();
    for _ in 0..300 {
        if g.is_terminal() {
            break;
        }
        let a = agent.act(&g);
        let (x, y) = (g.step(a).unwrap(), c.step(a).unwrap());
        assert_eq!(format!("{x:?}"), format!("{y:?}"));
    }
    assert_eq!(
        state_to_json(g.state(), false),
        state_to_json(c.state(), false)
    );
}
