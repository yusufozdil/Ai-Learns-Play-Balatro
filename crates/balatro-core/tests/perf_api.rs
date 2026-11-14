//! Phase 7A API additions: `legal_count`, `selection_space`, `reset_with_seed` agree with
//! `legal_actions` / `Game::new`.

use balatro_core::actions::Action;
use balatro_core::blinds::BlindKind;
use balatro_core::content::BossId;
use balatro_core::enums::Phase;
use balatro_core::game::Game;
use balatro_core::rng::{GameRng, RngKey};
use balatro_core::state::GameConfig;

/// Expand a selection space exactly as `legal_actions` lists it.
fn expand(g: &Game) -> Vec<Action> {
    let sp = g.selection_space();
    let mut v = Vec::new();
    if sp.can_play {
        v.extend(sp.iter().map(Action::Play));
    }
    if sp.can_discard {
        v.extend(sp.iter().map(Action::Discard));
    }
    v
}

fn check(g: &Game, scratch: &mut Vec<Action>) {
    let legal = g.legal_actions();
    assert_eq!(g.legal_count(scratch), legal.len());
    let sel: Vec<Action> = legal
        .iter()
        .copied()
        .filter(|a| matches!(a, Action::Play(_) | Action::Discard(_)))
        .collect();
    assert_eq!(expand(g), sel);
    assert_eq!(g.selection_space().len(), sel.len());
    if g.state().phase != Phase::Play {
        assert!(g.selection_space().is_empty());
    }
}

#[test]
fn counts_and_spaces_match_legal_actions() {
    let mut scratch = Vec::new();
    for seed in 0..60u64 {
        let mut g = Game::new(seed, GameConfig::default());
        let mut rng = GameRng::new(seed);
        for _ in 0..400 {
            if g.is_terminal() {
                check(&g, &mut scratch);
                break;
            }
            check(&g, &mut scratch);
            // Cerulean Bell: a forced card in hand.
            if g.state().phase == Phase::Play && rng.range(RngKey::Misc, 0, 4) == 0 {
                let uid = g.state().deck.hand[0].uid;
                let probe = {
                    let mut c = g.clone();
                    let s = c.state_mut();
                    s.boss = Some(BossId::CeruleanBell);
                    s.blind = BlindKind::Boss;
                    s.boss_disabled = false;
                    s.boss_state.card_uid = Some(uid);
                    c
                };
                check(&probe, &mut scratch);
                assert!(probe.selection_space().forced.is_some());
            }
            let legal = g.legal_actions();
            let a = legal[rng.range(RngKey::Misc, 0, legal.len())];
            g.step(a).expect("legal action");
        }
    }
}

#[test]
fn reset_with_seed_equals_new() {
    let mut g = Game::new(1, GameConfig::default());
    for _ in 0..30 {
        let legal = g.legal_actions();
        let Some(&a) = legal.last() else { break };
        g.step(a).unwrap();
    }
    g.reset_with_seed(77, GameConfig::default());
    assert_eq!(g.state(), Game::new(77, GameConfig::default()).state());
}
