//! Agent tests (Phase 2D): Greedy picks obvious hands, `preview_play` equals `step`, agents
//! only produce legal actions.

use balatro_core::actions::Action;
use balatro_core::agents::{Agent, GreedyAgent, RandomAgent};
use balatro_core::card::Card;
use balatro_core::content::jokers::JokerId;
use balatro_core::enums::{HandType, Phase, Rank, Suit};
use balatro_core::game::Game;
use balatro_core::joker::JokerInstance;
use balatro_core::rng::{GameRng, RngKey};
use balatro_core::state::GameConfig;

fn game_in_play(seed: u64) -> Game {
    let mut g = Game::new(seed, GameConfig::default());
    g.step(Action::SelectBlind).unwrap();
    assert_eq!(g.state().phase, Phase::Play);
    g
}

fn set_hand(g: &mut Game, cards: &[(Rank, Suit)]) {
    let hand: Vec<Card> = cards
        .iter()
        .enumerate()
        .map(|(i, &(r, s))| Card::new(9000 + i as u32, r, s))
        .collect();
    // Keep card conservation: move the old hand's uids out would break invariants, so only
    // overwrite ranks/suits in place.
    let st = g.state_mut();
    assert_eq!(st.deck.hand.len(), hand.len());
    for (dst, src) in st.deck.hand.iter_mut().zip(&hand) {
        dst.rank = src.rank;
        dst.suit = src.suit;
    }
}

#[test]
fn greedy_plays_flush_when_available() {
    use Rank::*;
    use Suit::*;
    let mut g = game_in_play(1);
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
    let mut a = GreedyAgent::new();
    let act = a.act(&g);
    let Action::Play(mask) = act else {
        panic!("expected Play, got {act:?}");
    };
    assert_eq!(mask, 0b0001_1111);
    let r = g.preview_play(mask).unwrap();
    assert_eq!(r.hand_type, HandType::Flush);
}

#[test]
fn greedy_discards_toward_flush_when_hand_is_weak() {
    use Rank::*;
    use Suit::*;
    let mut g = game_in_play(2);
    // Four spades but no pair/straight; target far above anything playable now.
    set_hand(
        &mut g,
        &[
            (Two, Spades),
            (Five, Spades),
            (Nine, Spades),
            (Jack, Spades),
            (Three, Hearts),
            (Seven, Clubs),
            (Eight, Diamonds),
            (King, Hearts),
        ],
    );
    g.state_mut().target = 1.0e9;
    let mut a = GreedyAgent::new();
    let act = a.act(&g);
    // Keeps the four spades (positions 0-3), discards the other four.
    assert_eq!(act, Action::Discard(0b1111_0000));
}

#[test]
fn greedy_plays_best_pair_over_high_card() {
    use Rank::*;
    use Suit::*;
    let mut g = game_in_play(3);
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
    let mut a = GreedyAgent::new();
    let Action::Play(mask) = a.act(&g) else {
        panic!("expected Play");
    };
    let r = g.preview_play(mask).unwrap();
    // Pair of twos with the best kickers outscores any single card.
    assert_eq!(r.hand_type, HandType::Pair);
}

#[test]
fn preview_matches_step_for_every_mask() {
    for seed in 0..40u64 {
        let mut g = game_in_play(seed);
        if seed % 2 == 0 {
            let mut p = GameRng::new(seed);
            for _ in 0..p.range(RngKey::Misc, 1, 6) {
                let id = JokerId::ALL[p.range(RngKey::Misc, 0, JokerId::COUNT)];
                let _ = g.add_joker(JokerInstance::new(0, id));
            }
        }
        let mut agent = RandomAgent::new(seed);
        let mut guard = 0;
        while g.state().phase == Phase::Play && guard < 30 {
            guard += 1;
            let n = g.state().deck.hand.len();
            for mask in 1u32..(1 << n) {
                if mask.count_ones() > 5 {
                    continue;
                }
                let before = g.state().clone();
                let pv = g.preview_play(mask);
                assert_eq!(*g.state(), before, "preview mutated the state");
                if g.state().hands_left == 0 {
                    assert!(pv.is_err());
                    continue;
                }
                let pv = pv.unwrap();
                let mut copy = g.clone();
                let info = copy.step(Action::Play(mask)).unwrap();
                assert_eq!(info.hand.unwrap(), pv, "seed {seed} mask {mask:b}");
            }
            let a = agent.act(&g);
            g.step(a).unwrap();
        }
    }
}

#[test]
fn preview_errors_match_validate() {
    let g = game_in_play(5);
    assert!(g.preview_play(0).is_err());
    assert!(g.preview_play(0b11_1111).is_err()); // 6 cards
    assert!(g.preview_play(1 << 12).is_err()); // past the hand
    let fresh = Game::new(5, GameConfig::default());
    assert!(fresh.preview_play(1).is_err()); // wrong phase
}

#[test]
fn agents_only_produce_legal_actions() {
    for seed in 0..60u64 {
        let mut g = Game::new(seed, GameConfig::default());
        let mut agents: Vec<Box<dyn Agent>> = vec![
            Box::new(RandomAgent::new(seed)),
            Box::new(RandomAgent::with_play_weight(seed, 8.0)),
            Box::new(GreedyAgent::new()),
        ];
        let which = (seed % 3) as usize;
        let mut steps = 0;
        while !g.is_terminal() {
            assert!(steps < 5000);
            let a = agents[which].act(&g);
            assert!(
                g.legal_actions().contains(&a),
                "seed {seed}: {a:?} not legal"
            );
            assert!(g.validate(a).is_ok());
            g.step(a).unwrap();
            steps += 1;
        }
    }
}

#[test]
fn greedy_beats_random() {
    let mut greedy_antes = 0;
    let mut random_antes = 0;
    for seed in 0..100u64 {
        for (greedy, total) in [(true, &mut greedy_antes), (false, &mut random_antes)] {
            let mut g = Game::new(seed, GameConfig::default());
            let mut a: Box<dyn Agent> = if greedy {
                Box::new(GreedyAgent::new())
            } else {
                Box::new(RandomAgent::new(seed))
            };
            while !g.is_terminal() {
                let act = a.act(&g);
                g.step(act).unwrap();
            }
            *total += g.state().ante;
        }
    }
    assert!(
        greedy_antes > random_antes + 50,
        "{greedy_antes} vs {random_antes}"
    );
}
