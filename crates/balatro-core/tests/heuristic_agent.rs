//! Heuristic team agent (Phase 5D): legality, determinism, robustness over many games, and a few
//! obvious decisions (win the round, buy a strong joker, put xmult to the right, sell the weakest
//! joker for a better one, play the best hand when out of discards).

use balatro_core::actions::Action;
use balatro_core::agents::{Agent, GreedyAgent, HeuristicAgent, HeuristicConfig};
use balatro_core::content::jokers::JokerId;
use balatro_core::enums::{Edition, Phase, Rank, Suit};
use balatro_core::game::Game;
use balatro_core::joker::JokerInstance;
use balatro_core::shop::{ShopCard, ShopCardKind};
use balatro_core::state::GameConfig;
use std::sync::atomic::{AtomicU64, Ordering};

const MAX_STEPS: usize = 5_000;

/// Cheaper sampling so thousands of games fit in a test run; same decision structure.
fn cheap_cfg() -> HeuristicConfig {
    HeuristicConfig {
        shop_samples: 3,
        arrange_samples: 2,
        mc_samples: 2,
        ..HeuristicConfig::default()
    }
}

/// Play one game; every chosen action must validate and step. Returns (actions, final ante, won).
fn play(seed: u64, cfg: HeuristicConfig) -> (Vec<Action>, i32, bool) {
    let mut g = Game::new(seed, GameConfig::default());
    let mut a = HeuristicAgent::with_config(seed, cfg);
    let mut actions = Vec::new();
    while !g.is_terminal() {
        assert!(actions.len() < MAX_STEPS, "seed {seed}: no termination");
        let act = a.act(&g);
        assert!(
            g.validate(act).is_ok(),
            "seed {seed}: illegal {act:?} in {:?}: {:?}",
            g.state().phase,
            g.validate(act)
        );
        assert!(
            g.legal_actions().contains(&act),
            "seed {seed}: {act:?} not listed as legal"
        );
        g.step(act).expect("step");
        actions.push(act);
    }
    (actions, g.state().ante, g.state().phase == Phase::Won)
}

fn game_in_play(seed: u64) -> Game {
    let mut g = Game::new(seed, GameConfig::default());
    g.step(Action::SelectBlind).unwrap();
    assert_eq!(g.state().phase, Phase::Play);
    g
}

/// Overwrite ranks/suits of the dealt hand in place (card conservation stays intact).
fn set_hand(g: &mut Game, cards: &[(Rank, Suit)]) {
    let st = g.state_mut();
    assert_eq!(st.deck.hand.len(), cards.len());
    for (dst, &(r, s)) in st.deck.hand.iter_mut().zip(cards) {
        dst.rank = r;
        dst.suit = s;
    }
}

fn in_shop(seed: u64) -> Game {
    let mut g = game_in_play(seed);
    let t = g.state().target;
    g.state_mut().round_score = t - 1.0;
    g.step(Action::Play(1)).unwrap();
    assert_eq!(g.state().phase, Phase::Shop);
    g
}

/// Empty the shop and put one joker in card slot 0.
fn offer_joker(g: &mut Game, id: JokerId) {
    let uid = g.state().next_joker_uid + 100;
    let s = g.state_mut();
    s.shop.cards = vec![Some(ShopCard {
        kind: ShopCardKind::Joker(JokerInstance::new(uid, id)),
        free: false,
    })];
    s.shop.packs = vec![None, None];
    s.shop.vouchers = vec![None];
}

#[test]
fn only_legal_actions_and_terminates() {
    for seed in 0..40 {
        play(seed, HeuristicConfig::default());
    }
}

#[test]
fn deterministic_same_seed_same_game() {
    for seed in [1u64, 7, 23, 99] {
        let (a1, ante1, w1) = play(seed, HeuristicConfig::default());
        let (a2, ante2, w2) = play(seed, HeuristicConfig::default());
        assert_eq!(a1, a2, "seed {seed}");
        assert_eq!((ante1, w1), (ante2, w2));
    }
}

#[test]
fn no_panic_over_thousands_of_games() {
    let next = AtomicU64::new(0);
    let total = 1500u64;
    let threads = std::thread::available_parallelism().map_or(2, |n| n.get().min(8));
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= total {
                        break;
                    }
                    play(10_000 + i, cheap_cfg());
                }
            });
        }
    });
}

#[test]
fn reusing_one_agent_across_games_stays_legal() {
    let mut a = HeuristicAgent::new(3);
    for seed in 0..4u64 {
        a.reset();
        let mut g = Game::new(seed, GameConfig::default());
        let mut n = 0;
        while !g.is_terminal() && n < MAX_STEPS {
            let act = a.act(&g);
            g.step(act).expect("legal");
            n += 1;
        }
        assert!(g.is_terminal());
    }
}

#[test]
fn plays_the_hand_that_beats_the_blind() {
    use Rank::*;
    use Suit::*;
    let mut g = game_in_play(5);
    set_hand(
        &mut g,
        &[
            (Two, Hearts),
            (Five, Hearts),
            (Nine, Hearts),
            (Jack, Hearts),
            (King, Hearts),
            (Three, Spades),
            (Seven, Clubs),
            (Four, Diamonds),
        ],
    );
    // A pair of twos is not needed: the flush clears this target, a lone high card does not.
    g.state_mut().target = 200.0;
    let mut a = HeuristicAgent::new(0);
    let Action::Play(mask) = a.act(&g) else {
        panic!("expected a Play");
    };
    let r = g.preview_play(mask).unwrap();
    assert!(r.score >= 200.0, "played a hand worth {}", r.score);
}

#[test]
fn out_of_discards_plays_the_best_previewed_hand() {
    use Rank::*;
    use Suit::*;
    let mut g = game_in_play(6);
    set_hand(
        &mut g,
        &[
            (Two, Hearts),
            (Two, Spades),
            (Nine, Clubs),
            (Jack, Diamonds),
            (King, Hearts),
            (Four, Spades),
            (Seven, Clubs),
            (Six, Diamonds),
        ],
    );
    g.state_mut().discards_left = 0;
    g.state_mut().target = 1.0e9;
    let mut a = HeuristicAgent::new(0);
    let Action::Play(mask) = a.act(&g) else {
        panic!("expected a Play");
    };
    let chosen = g.preview_play(mask).unwrap().score;
    let best = (1u16..(1 << 8))
        .filter(|m| m.count_ones() <= 5)
        .filter_map(|m| g.preview_play(m).ok())
        .map(|r| r.score)
        .fold(0.0, f64::max);
    assert!(
        (chosen - best).abs() < 1e-9,
        "chose {chosen}, best is {best}"
    );
}

#[test]
fn buys_a_strong_xmult_joker_it_can_afford() {
    let mut g = in_shop(11);
    g.state_mut().money = 20;
    offer_joker(&mut g, JokerId::TheTribe);
    let mut a = HeuristicAgent::new(0);
    assert_eq!(a.act(&g), Action::Buy(0));
}

#[test]
fn does_not_buy_what_it_cannot_afford() {
    let mut g = in_shop(11);
    g.state_mut().money = 3;
    offer_joker(&mut g, JokerId::TheTribe);
    let mut a = HeuristicAgent::new(0);
    let act = a.act(&g);
    assert!(g.validate(act).is_ok());
    assert_ne!(act, Action::Buy(0));
}

#[test]
fn moves_the_xmult_joker_to_the_right() {
    // xmult first, +mult after it: swapping them is strictly better.
    let mut g = in_shop(12);
    g.add_joker(JokerInstance::new(0, JokerId::TheDuo)).unwrap();
    g.add_joker(JokerInstance::new(0, JokerId::Joker)).unwrap();
    g.state_mut().shop.cards = vec![None, None];
    g.state_mut().shop.packs = vec![None, None];
    g.state_mut().shop.vouchers = vec![None];
    g.state_mut().money = 0;
    let mut a = HeuristicAgent::new(0);
    // The agent may first use or sell something else, but it must arrange before leaving.
    let mut moved = false;
    for _ in 0..6 {
        let act = a.act(&g);
        if matches!(act, Action::MoveJoker { .. }) {
            moved = true;
        }
        let end = act == Action::EndShop;
        g.step(act).unwrap();
        if end {
            break;
        }
    }
    assert!(moved, "no MoveJoker before leaving the shop");
    assert_eq!(g.state().jokers[1].id, JokerId::TheDuo);
    assert_eq!(g.state().jokers[0].id, JokerId::Joker);
}

#[test]
fn replaces_its_weakest_joker_when_full() {
    let mut g = in_shop(13);
    // Five weak jokers, then a strong rare one on offer.
    for id in [
        JokerId::GreedyJoker,
        JokerId::LustyJoker,
        JokerId::WrathfulJoker,
        JokerId::GluttonousJoker,
        JokerId::Joker,
    ] {
        g.add_joker(JokerInstance::new(0, id)).unwrap();
    }
    g.state_mut().money = 30;
    offer_joker(&mut g, JokerId::TheTrio);
    let mut a = HeuristicAgent::new(0);
    let first = a.act(&g);
    assert!(
        matches!(first, Action::SellJoker(_)),
        "expected to sell a joker first, got {first:?}"
    );
    g.step(first).unwrap();
    let second = a.act(&g);
    assert_eq!(second, Action::Buy(0), "then buy the strong joker");
}

#[test]
fn takes_the_best_joker_of_a_buffoon_pack() {
    use balatro_core::hooks::PackKind;
    use balatro_core::packs::{OpenPack, PackItem, PackOrigin, PackSize, PackSpec};
    let mut g = in_shop(14);
    let weak = JokerInstance::new(900, JokerId::Egg);
    let strong = JokerInstance::new(901, JokerId::Joker).with_edition(Edition::Base);
    let spec = PackSpec {
        kind: PackKind::Buffoon,
        size: PackSize::Normal,
    };
    g.state_mut().pack = Some(OpenPack {
        spec,
        items: vec![Some(PackItem::Joker(weak)), Some(PackItem::Joker(strong))],
        picks_left: 1,
        origin: PackOrigin::Shop,
    });
    g.state_mut().phase = Phase::PackOpen;
    let mut a = HeuristicAgent::new(0);
    assert_eq!(
        a.act(&g),
        Action::PickPackItem {
            index: 1,
            targets: 0
        }
    );
}

#[test]
fn beats_greedy_on_average() {
    // A coarse regression guard (cheap config, 24 seeds): the heuristic reaches a higher mean
    // ante than the Greedy baseline. The release evaluation (`eval_agents`) has the real numbers.
    let n = 24u64;
    let (mut h_sum, mut g_sum) = (0i64, 0i64);
    for seed in 0..n {
        h_sum += i64::from(play(20_000 + seed, cheap_cfg()).1);
        let mut g = Game::new(20_000 + seed, GameConfig::default());
        let mut a = GreedyAgent::new();
        while !g.is_terminal() {
            let act = a.act(&g);
            g.step(act).unwrap();
        }
        g_sum += i64::from(g.state().ante);
    }
    assert!(
        h_sum > g_sum + n as i64 / 2,
        "heuristic {h_sum} vs greedy {g_sum} over {n} games"
    );
}
