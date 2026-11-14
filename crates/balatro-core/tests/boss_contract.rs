//! Phase 4.0 boss contract: selection (min ante, showdown, least-seen), requirement multipliers,
//! disabling (Chicot, Luchador, Verdant Leaf), per-hand flags (Matador), preview mirroring, and
//! the prototypes (The Hook, The Wall, The Psychic, The Plant, The Needle, The Flint, The Mark,
//! Verdant Leaf). A smoke test runs random games against every defined boss.
//! Guide: docs/design/consumables_bosses.md.

use balatro_core::actions::{Action, ActionError};
use balatro_core::agents::{Agent, RandomAgent};
use balatro_core::blinds::{self, BlindKind, MULT_BOSS_DEFAULT, StakeScaling};
use balatro_core::boss::choose_boss;
use balatro_core::content::jokers::JokerId;
use balatro_core::content::{BossId as B, boss_def_count, bosses, try_boss_def};
use balatro_core::enums::{HandType, Phase, Rank, Suit};
use balatro_core::game::Game;
use balatro_core::joker::JokerInstance;
use balatro_core::rng::{GameRng, RngKey};
use balatro_core::state::GameConfig;

// ----- helpers ------------------------------------------------------------------------------

fn new_game(seed: u64) -> Game {
    Game::new(seed, GameConfig::default())
}

fn win_round(g: &mut Game) {
    let t = g.state().target;
    g.state_mut().round_score = t - 1.0;
    let n = g.state().deck.hand.len().min(5);
    let mask = (1u32 << n) - 1; // 5 cards: valid under The Psychic
    let info = g.step(Action::Play(mask)).expect("winning play");
    assert!(info.round.expect("round ended").won);
    if g.state().phase == Phase::Shop {
        g.step(Action::EndShop).expect("end shop");
    }
}

/// From a fresh game: beat Small and Big, force `boss`, select it.
fn at_boss(seed: u64, boss: B) -> Game {
    let mut g = new_game(seed);
    for _ in 0..2 {
        g.step(Action::SelectBlind).expect("select");
        win_round(&mut g);
    }
    assert_eq!(g.state().blind, BlindKind::Boss);
    g.state_mut().boss = Some(boss);
    g.step(Action::SelectBlind).expect("select boss");
    assert!(g.state().boss_in_effect());
    g
}

fn add(g: &mut Game, id: JokerId) -> usize {
    g.add_joker(JokerInstance::new(0, id)).expect("room")
}

fn defined() -> Vec<B> {
    B::ALL
        .into_iter()
        .filter(|&b| try_boss_def(b).is_some())
        .collect()
}

// ----- registry and selection ----------------------------------------------------------------

#[test]
fn boss_registry_is_consistent() {
    assert_eq!(B::COUNT, 28);
    assert_eq!(B::ALL.iter().filter(|b| b.is_showdown()).count(), 5);
    for d in bosses::DEFS {
        assert_eq!(boss_def_count(d.id), 1, "{} defined twice", d.name);
        assert_eq!(try_boss_def(d.id).expect("registered").name, d.name);
        assert!(d.mult > 0.0);
        if d.id.is_showdown() {
            assert_eq!(d.min_ante, 8, "{}", d.name);
        } else {
            assert!((1..=8).contains(&d.min_ante), "{}", d.name);
        }
    }
}

#[test]
fn selection_respects_min_ante_and_showdown_antes() {
    let mut rng = GameRng::new(9);
    for ante in -1..=24 {
        for _ in 0..50 {
            let mut seen = [0u32; B::COUNT];
            let b = choose_boss(ante, 8, &mut seen, &mut rng).expect("bosses are defined");
            let d = try_boss_def(b).expect("only defined bosses are chosen");
            assert!(d.min_ante <= ante.max(1), "{b:?} at ante {ante}");
            let showdown_ante = ante > 0 && ante % 8 == 0;
            let any_showdown = defined().iter().any(|b| b.is_showdown());
            if any_showdown {
                assert_eq!(b.is_showdown(), showdown_ante, "{b:?} at ante {ante}");
            }
            assert_eq!(seen[b.index()], 1);
        }
    }
}

#[test]
fn selection_never_repeats_before_the_eligible_pool_is_exhausted() {
    let mut rng = GameRng::new(3);
    let pool: Vec<B> = defined()
        .into_iter()
        .filter(|&b| !b.is_showdown() && try_boss_def(b).is_some_and(|d| d.min_ante <= 5))
        .collect();
    let mut seen = [0u32; B::COUNT];
    for round in 0..3 {
        let mut got = Vec::new();
        for _ in 0..pool.len() {
            got.push(choose_boss(5, 8, &mut seen, &mut rng).expect("boss"));
        }
        got.sort();
        got.dedup();
        assert_eq!(
            got.len(),
            pool.len(),
            "round {round}: every boss exactly once"
        );
    }
}

#[test]
fn selection_is_uniform_among_the_least_seen() {
    let mut rng = GameRng::new(1);
    let ante1: Vec<B> = defined()
        .into_iter()
        .filter(|&b| !b.is_showdown() && try_boss_def(b).is_some_and(|d| d.min_ante <= 1))
        .collect();
    let mut counts = [0u32; B::COUNT];
    let n = 4000;
    for _ in 0..n {
        let mut seen = [0u32; B::COUNT];
        counts[choose_boss(1, 8, &mut seen, &mut rng)
            .expect("boss")
            .index()] += 1;
    }
    let expect = n as f64 / ante1.len() as f64;
    for b in ante1 {
        let c = f64::from(counts[b.index()]);
        assert!((c - expect).abs() < expect * 0.15, "{b:?}: {c} vs {expect}");
    }
}

#[test]
fn games_choose_a_boss_every_ante_and_count_it() {
    let g = new_game(5);
    let s = g.state();
    let b = s.boss.expect("ante 1 boss");
    assert_eq!(s.boss_seen.iter().sum::<u32>(), 1);
    assert_eq!(s.boss_seen[b.index()], 1);
    assert!(try_boss_def(b).expect("defined").min_ante <= 1);
    // Ante 2 rolls a new one.
    let mut g = new_game(5);
    for _ in 0..3 {
        g.step(Action::SelectBlind).expect("select");
        g.state_mut().boss = Some(B::TheWall);
        win_round(&mut g);
    }
    assert_eq!(g.state().ante, 2);
    assert_eq!(g.state().boss_seen.iter().sum::<u32>(), 2);
}

// ----- multipliers and disabling ----------------------------------------------------------

#[test]
fn the_wall_quadruples_the_requirement_until_disabled() {
    let base = blinds::base_chips(1, StakeScaling::White);
    let mut g = at_boss(1, B::TheWall);
    assert_eq!(g.state().target, base * 4.0);
    // Chicot added mid-round disables it: back to x2.
    add(&mut g, JokerId::Chicot);
    assert!(g.state().boss_disabled);
    assert_eq!(g.state().target, base * MULT_BOSS_DEFAULT);
}

#[test]
fn the_needle_allows_one_hand_and_keeps_x1_when_disabled() {
    let base = blinds::base_chips(1, StakeScaling::White);
    let mut g = at_boss(2, B::TheNeedle);
    assert_eq!(g.state().hands_left, 1);
    assert_eq!(g.state().target, base);
    add(&mut g, JokerId::Chicot);
    // The hands come back; the requirement stays x1 (notes.md §4.2).
    assert_eq!(g.state().hands_left, g.state().round_hands());
    assert_eq!(g.state().target, base);
    // Chicot owned before the blind: disabled during selection, hands never lost.
    let mut g = new_game(3);
    add(&mut g, JokerId::Chicot);
    for _ in 0..2 {
        g.step(Action::SelectBlind).expect("select");
        win_round(&mut g);
    }
    g.state_mut().boss = Some(B::TheNeedle);
    g.step(Action::SelectBlind).expect("boss");
    assert!(g.state().boss_disabled);
    assert_eq!(g.state().hands_left, g.state().round_hands());
}

#[test]
fn luchador_sold_in_the_boss_round_disables_it() {
    let mut g = new_game(4);
    add(&mut g, JokerId::Luchador);
    for _ in 0..2 {
        g.step(Action::SelectBlind).expect("select");
        win_round(&mut g);
    }
    g.state_mut().boss = Some(B::ThePlant);
    g.step(Action::SelectBlind).expect("boss");
    assert!(g.state().deck.iter_all().any(|c| c.debuffed));
    g.step(Action::SellJoker(0)).expect("sell Luchador");
    assert!(g.state().boss_disabled);
    assert!(g.state().deck.iter_all().all(|c| !c.debuffed));
}

// ----- prototypes ----------------------------------------------------------------------------

#[test]
fn the_psychic_rejects_hands_that_are_not_five_cards() {
    let mut g = at_boss(10, B::ThePsychic);
    add(&mut g, JokerId::Matador);
    let m0 = g.state().money;
    let h = g.step(Action::Play(0b1)).expect("play").hand.expect("hand");
    assert!(h.invalid);
    assert_eq!(h.score, 0.0);
    assert_eq!(
        g.state().money,
        m0 + 8,
        "Matador pays when the boss triggers"
    );
    let h = g
        .step(Action::Play(0b11111))
        .expect("play")
        .hand
        .expect("hand");
    assert!(!h.invalid);
    assert!(h.score > 0.0);
}

#[test]
fn the_hook_discards_two_held_cards_before_scoring() {
    let mut g = at_boss(11, B::TheHook);
    add(&mut g, JokerId::Ramen);
    add(&mut g, JokerId::Matador);
    let m0 = g.state().money;
    let before: Vec<u32> = g.state().deck.hand.iter().map(|c| c.uid).collect();
    let info = g.step(Action::Play(0b1)).expect("play");
    let s = g.state();
    // 1 played + 2 hooked in the discard pile; the hand was refilled to 8.
    assert_eq!(s.deck.discard_pile.len(), 3);
    assert_eq!(s.deck.hand.len(), 8);
    assert!(s.deck.discard_pile.iter().all(|c| before.contains(&c.uid)));
    // Forced discard: no discard used, Ramen loses 0.01 per card (2.0 -> 1.98).
    assert_eq!(s.discards_used_round, 0);
    assert!((s.jokers[0].value - 1.98).abs() < 1e-9);
    assert_eq!(s.money, m0 + 8, "Matador");
    assert!(info.hand.expect("hand").score > 0.0);
}

#[test]
fn the_hook_does_not_count_as_a_first_discard() {
    // Burnt Joker levels the first discard's hand; The Hook's discards are not discards.
    let mut g = at_boss(12, B::TheHook);
    add(&mut g, JokerId::BurntJoker);
    g.step(Action::Discard(0b1)).expect("discard");
    let levels = g.state().hand_levels;
    g.step(Action::Play(0b1)).expect("play");
    assert_eq!(g.state().hand_levels, levels);
}

#[test]
fn the_flint_halves_base_chips_and_mult() {
    let mut g = at_boss(13, B::TheFlint);
    // Pair of Kings: base 10 x 2 -> 5 x 1; chips 5 + 10 + 10 = 25 -> 25.
    {
        let h = &mut g.state_mut().deck.hand;
        h[0].rank = Rank::King;
        h[0].suit = Suit::Spades;
        h[1].rank = Rank::King;
        h[1].suit = Suit::Hearts;
        for c in h.iter_mut().take(2) {
            c.enhancement = Default::default();
            c.edition = Default::default();
            c.seal = Default::default();
            c.perma_bonus = 0;
        }
    }
    let h = g
        .step(Action::Play(0b11))
        .expect("play")
        .hand
        .expect("hand");
    assert_eq!(h.hand_type, HandType::Pair);
    assert_eq!((h.chips, h.mult, h.score), (25.0, 1.0, 25.0));
}

#[test]
fn the_plant_debuffs_face_cards_and_pareidolia_makes_all_faces() {
    let g = at_boss(14, B::ThePlant);
    for c in g.state().deck.iter_all() {
        assert_eq!(c.debuffed, c.rank.is_face_rank() && !c.is_stone(), "{c:?}");
    }
    let mut g = at_boss(14, B::ThePlant);
    add(&mut g, JokerId::Pareidolia);
    assert!(g.state().deck.iter_all().all(|c| c.debuffed));
    // Round end clears every debuff.
    win_round(&mut g);
    assert!(g.state().deck.iter_all().all(|c| !c.debuffed));
}

#[test]
fn verdant_leaf_debuffs_everything_until_a_joker_is_sold() {
    let mut g = new_game(15);
    add(&mut g, JokerId::Joker);
    for _ in 0..2 {
        g.step(Action::SelectBlind).expect("select");
        win_round(&mut g);
    }
    g.state_mut().boss = Some(B::VerdantLeaf);
    g.step(Action::SelectBlind).expect("boss");
    assert!(g.state().deck.iter_all().all(|c| c.debuffed));
    let h = g.step(Action::Play(0b1)).expect("play").hand.expect("hand");
    assert_eq!(
        h.chips, 5.0,
        "debuffed card adds nothing: High Card base only"
    );
    g.step(Action::SellJoker(0)).expect("sell");
    assert!(g.state().boss_disabled);
    assert!(g.state().deck.iter_all().all(|c| !c.debuffed));
}

#[test]
fn the_mark_draws_face_cards_face_down() {
    for seed in 0..10 {
        let mut g = at_boss(20 + seed, B::TheMark);
        for c in &g.state().deck.hand {
            assert_eq!(c.face_down, c.rank.is_face_rank(), "{c:?}");
        }
        // Refills follow the rule too; round end turns everything face up.
        g.step(Action::Discard(0b111)).expect("discard");
        for c in &g.state().deck.hand {
            assert_eq!(c.face_down, c.rank.is_face_rank(), "{c:?}");
        }
        win_round(&mut g);
        assert!(g.state().deck.iter_all().all(|c| !c.face_down));
    }
}

#[test]
fn disabling_turns_face_down_cards_up() {
    let mut g = at_boss(30, B::TheMark);
    add(&mut g, JokerId::Chicot);
    assert!(g.state().deck.hand.iter().all(|c| !c.face_down));
}

#[test]
fn cards_played_this_ante_are_flagged_and_reset_at_ante_start() {
    let mut g = new_game(31);
    g.step(Action::SelectBlind).expect("select");
    let uid = g.state().deck.hand[0].uid;
    win_round(&mut g);
    let flagged = |g: &Game| {
        g.state()
            .deck
            .iter_all()
            .find(|c| c.uid == uid)
            .expect("card")
            .played_ante
    };
    assert!(flagged(&g));
    for _ in 0..2 {
        g.step(Action::SelectBlind).expect("select");
        g.state_mut().boss = Some(B::TheWall);
        win_round(&mut g);
    }
    assert_eq!(g.state().ante, 2);
    assert!(!flagged(&g));
}

// ----- preview, smoke, determinism -----------------------------------------------------------

#[test]
fn preview_matches_step_under_every_defined_boss() {
    let jokers = [
        JokerId::Ramen,
        JokerId::GreenJoker,
        JokerId::Matador,
        JokerId::Bull,
        JokerId::Baron,
        JokerId::AbstractJoker,
    ];
    for boss in defined() {
        for seed in 0..6u64 {
            let mut g = at_boss(100 + seed, boss);
            for &j in &jokers[..(seed as usize % jokers.len()) + 1] {
                let _ = g.add_joker(JokerInstance::new(0, j));
            }
            let n = g.state().deck.hand.len();
            for mask in 1u32..(1 << n) {
                if mask.count_ones() > 5 {
                    continue;
                }
                let m = mask;
                let pv = match g.preview_play(m) {
                    Ok(p) => p,
                    Err(e) => {
                        assert!(g.validate(Action::Play(m)).is_err(), "{e}");
                        continue;
                    }
                };
                let mut copy = g.clone();
                let info = copy.step(Action::Play(m)).expect("valid play");
                assert_eq!(info.hand.expect("hand"), pv, "{boss:?} seed {seed} {m:b}");
            }
        }
    }
}

#[test]
fn random_games_against_every_defined_boss_keep_the_invariants() {
    for boss in defined() {
        let mut boss_steps = 0;
        for seed in 0..40u64 {
            let mut g = new_game(seed);
            let mut agent = RandomAgent::with_play_weight(seed, 4.0);
            let mut pick = GameRng::new(seed);
            if seed % 2 == 0 {
                for _ in 0..pick.range(RngKey::Misc, 1, 5) {
                    let j = JokerId::ALL[pick.range(RngKey::Misc, 0, JokerId::COUNT)];
                    let _ = g.add_joker(JokerInstance::new(0, j));
                }
            }
            let mut steps = 0;
            while !g.is_terminal() && steps < 600 {
                if g.state().phase == Phase::BlindSelect && g.state().blind == BlindKind::Boss {
                    g.state_mut().boss = Some(boss);
                }
                // Nudge the score so runs reach boss blinds.
                if g.state().phase == Phase::Play && pick.range(RngKey::Misc, 0, 3) == 0 {
                    let t = g.state().target;
                    g.state_mut().round_score = g.state().round_score.max(t * 0.9);
                }
                let a = agent.act(&g);
                g.step(a).expect("legal");
                if let Err(e) = g.state().check_invariants() {
                    panic!("{boss:?} seed {seed}: {e}");
                }
                boss_steps += usize::from(g.state().boss_in_effect());
                steps += 1;
            }
        }
        assert!(
            boss_steps > 20,
            "{boss:?}: only {boss_steps} steps in the boss round"
        );
    }
}

#[test]
fn boss_games_are_deterministic() {
    let run = |seed: u64| {
        let mut g = new_game(seed);
        let mut agent = RandomAgent::with_play_weight(seed, 6.0);
        let mut steps = 0;
        while !g.is_terminal() && steps < 300 {
            if g.state().phase == Phase::Play && steps % 4 == 0 {
                let t = g.state().target;
                g.state_mut().round_score = g.state().round_score.max(t * 0.95);
            }
            let a = agent.act(&g);
            g.step(a).expect("legal");
            steps += 1;
        }
        g.into_state()
    };
    for seed in [1u64, 2, 99] {
        assert_eq!(run(seed), run(seed));
    }
}

#[test]
fn forced_card_is_required_in_play_and_discard_masks() {
    // No Cerulean Bell yet (4C): emulate it through the boss state of an active boss.
    let mut g = at_boss(40, B::TheWall);
    let uid = g.state().deck.hand[2].uid;
    g.state_mut().boss_state.card_uid = Some(uid);
    assert_eq!(
        g.validate(Action::Play(0b1)),
        Err(ActionError::ForcedCardMissing { position: 2 })
    );
    assert!(g.validate(Action::Play(0b101)).is_ok());
    assert!(g.legal_actions().iter().all(|a| match *a {
        Action::Play(m) | Action::Discard(m) => m & 0b100 != 0,
        _ => true,
    }));
}

#[test]
fn smoke_covers_all_28_bosses_and_each_appears_naturally_with_the_ante_shortcut() {
    // The forced-boss smoke test above iterates `defined()`: it must be the whole roster.
    assert_eq!(defined().len(), B::COUNT);
    // Natural selection: start at ante 1..=8 (shortcut), nudge scores so rounds are won, and
    // record every boss that is in effect. Showdowns come from `choose_boss` at ante 8.
    let mut seen = [false; B::COUNT];
    for seed in 0..600u64 {
        let mut g = new_game(seed);
        g.state_mut().ante = 1 + (seed % 8) as i32;
        let mut agent = RandomAgent::with_play_weight(seed, 4.0);
        let mut steps = 0;
        while !g.is_terminal() && steps < 800 {
            if g.state().phase == Phase::Play {
                let t = g.state().target;
                g.state_mut().round_score = g.state().round_score.max(t - 1.0);
            }
            let a = agent.act(&g);
            g.step(a).expect("legal");
            if g.state().boss_in_effect()
                && let Some(b) = g.state().boss
            {
                seen[b.index()] = true;
            }
            steps += 1;
        }
    }
    for b in B::ALL {
        assert!(seen[b.index()], "{b:?} never active in a natural game");
    }
}
