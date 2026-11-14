//! Phase 2C: state JSON round-trip in every phase, replay determinism, error paths.

use balatro_core::actions::{Action, ActionError};
use balatro_core::enums::Phase;
use balatro_core::game::Game;
use balatro_core::rng::{GameRng, RngKey};
use balatro_core::serialize::{
    REPLAY_VERSION, RecordingGame, Replay, ReplayError, SerializeError, state_from_json,
    state_to_json,
};
use balatro_core::state::{GameConfig, GameState};

fn new_game(seed: u64) -> Game {
    Game::new(seed, GameConfig::default())
}

fn roundtrip(state: &GameState) {
    for pretty in [false, true] {
        let json = state_to_json(state, pretty);
        let back = state_from_json(&json).expect("decode");
        assert_eq!(&back, state, "round-trip (pretty={pretty})");
    }
}

/// Beat the current blind: put the score one below the target and play one card.
fn beat_blind(g: &mut Game) {
    let t = g.state().target;
    g.state_mut().round_score = t - 1.0;
    g.step(Action::Play(0b1)).expect("winning play");
}

#[test]
fn roundtrip_blind_select() {
    roundtrip(new_game(1).state());
}

#[test]
fn roundtrip_mid_play() {
    let mut g = new_game(2);
    g.step(Action::SelectBlind).unwrap();
    roundtrip(g.state());
    g.step(Action::Discard(0b111)).unwrap();
    g.step(Action::Play(0b11111)).unwrap();
    assert_eq!(g.state().phase, Phase::Play);
    roundtrip(g.state());
}

#[test]
fn roundtrip_shop() {
    let mut g = new_game(3);
    g.step(Action::SelectBlind).unwrap();
    beat_blind(&mut g);
    assert_eq!(g.state().phase, Phase::Shop);
    roundtrip(g.state());
}

#[test]
fn roundtrip_game_over() {
    let mut g = new_game(4);
    g.step(Action::SelectBlind).unwrap();
    // Four hands of one card each will not reach 300 chips.
    while !g.is_terminal() {
        g.step(Action::Play(0b1)).unwrap();
    }
    assert_eq!(g.state().phase, Phase::GameOver);
    roundtrip(g.state());
}

#[test]
fn resumed_game_continues_identically() {
    let mut a = new_game(5);
    a.step(Action::SelectBlind).unwrap();
    a.step(Action::Play(0b101)).unwrap();
    let mut b = Game::from_state(state_from_json(&state_to_json(a.state(), false)).unwrap());
    for act in [
        Action::Play(0b11),
        Action::Discard(0b1),
        Action::Play(0b111),
    ] {
        let ra = a.step(act);
        let rb = b.step(act);
        assert_eq!(ra, rb);
    }
    assert_eq!(a.state(), b.state());
}

#[test]
fn random_games_roundtrip_and_replay() {
    let mut buf = Vec::new();
    for seed in 0..40u64 {
        let mut rg = RecordingGame::new(seed, GameConfig::default());
        let mut rng = GameRng::new(seed ^ 0xA5A5);
        // Mid-run snapshots must round-trip too.
        for n in 0..400 {
            rg.game().legal_actions_into(&mut buf);
            let Some(i) = rng.choose_index(RngKey::Misc, buf.len()) else {
                break;
            };
            rg.step(buf[i]).expect("listed action must be accepted");
            if n % 7 == 0 {
                roundtrip(rg.game().state());
            }
        }
        let replayed = rg.replay().play().expect("replay");
        assert_eq!(replayed.state(), rg.game().state(), "seed {seed}");
        let json = rg.replay().to_json();
        assert_eq!(&Replay::from_json(&json).unwrap(), rg.replay());
        roundtrip(replayed.state());
    }
}

#[test]
fn scripted_full_round_replay() {
    // Reaches the Shop and back through the real rules (score shortcut is not used).
    let mut rg = RecordingGame::new(9, GameConfig::default());
    rg.step(Action::SelectBlind).unwrap();
    rg.step(Action::Play(0b11111)).unwrap();
    rg.step(Action::Discard(0b1)).unwrap();
    let r = rg.replay().clone();
    assert_eq!(r.version, REPLAY_VERSION);
    assert_eq!(r.actions.len(), 3);
    let g = r.play().unwrap();
    assert_eq!(g.state(), rg.game().state());
}

#[test]
fn rejected_action_is_not_recorded() {
    let mut rg = RecordingGame::new(1, GameConfig::default());
    assert!(rg.step(Action::EndShop).is_err());
    assert!(rg.replay().actions.is_empty());
}

#[test]
fn invalid_replay_reports_failing_step() {
    let mut r = Replay::record(1, GameConfig::default());
    r.push(Action::SelectBlind);
    r.push(Action::Play(0b1));
    r.push(Action::EndShop); // wrong phase
    r.push(Action::Play(0b1));
    match r.play() {
        Err(ReplayError::InvalidAction {
            step,
            action,
            source,
        }) => {
            assert_eq!(step, 2);
            assert_eq!(action, Action::EndShop);
            assert!(matches!(source, ActionError::WrongPhase { .. }));
        }
        other => panic!("expected InvalidAction, got {other:?}"),
    }
}

#[test]
fn wrong_version_is_rejected() {
    let mut r = Replay::record(1, GameConfig::default());
    r.version = REPLAY_VERSION + 1;
    assert!(matches!(r.play(), Err(ReplayError::Version { .. })));
}

#[test]
fn broken_json_returns_errors() {
    let good = state_to_json(new_game(1).state(), false);
    let truncated = &good[..good.len() / 2];
    let bad_inputs = [
        "",
        "null",
        "{}",
        "[]",
        "{\"phase\": 5}",
        "not json at all",
        truncated,
    ];
    for s in bad_inputs {
        assert!(
            matches!(state_from_json(s), Err(SerializeError::Json(_))),
            "{s:.30}"
        );
        assert!(Replay::from_json(s).is_err(), "{s:.30}");
    }
    // Wrong type for a field.
    let wrong = good.replacen("\"money\":4", "\"money\":\"lots\"", 1);
    assert_ne!(wrong, good);
    assert!(state_from_json(&wrong).is_err());
}

#[test]
fn state_violating_invariants_is_rejected() {
    let mut s = new_game(1).state().clone();
    s.starting_deck_size += 7; // cards no longer conserved
    let json = state_to_json(&s, false);
    assert!(matches!(
        state_from_json(&json),
        Err(SerializeError::Invariant(_))
    ));
}

#[test]
fn garbage_replay_json_does_not_panic() {
    let bad = [
        "{\"version\":1,\"seed\":1,\"config\":{},\"actions\":[{\"Nope\":1}]}",
        "{\"version\":1,\"seed\":-1,\"config\":{},\"actions\":[]}",
        "{\"version\":1}",
    ];
    for s in bad {
        assert!(Replay::from_json(s).is_err(), "{s}");
    }
}
