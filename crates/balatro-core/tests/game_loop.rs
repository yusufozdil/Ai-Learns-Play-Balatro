//! Phase 2A: GameState + phase state machine. Flow (select -> play -> cash out -> shop -> ...),
//! skip -> tag, boss skip rejected, game over, win, cash-out math, invalid actions leave the
//! state unchanged, determinism, joker event/side-effect integration, side-effect application.
//! Every expected number is derived in a comment.

use balatro_core::actions::{Action, ActionError};
use balatro_core::blinds::BlindKind;
use balatro_core::card::Card;
use balatro_core::content::BossId;
use balatro_core::content::jokers::JokerId;
use balatro_core::deck::Pile;
use balatro_core::enums::{Edition, Enhancement, HandType, Phase, Rank, Suit};
use balatro_core::game::{Game, StepInfo, apply_side_effects};
use balatro_core::hooks::{RunFlag, SellTarget, SideEffect};
use balatro_core::joker::JokerInstance;
use balatro_core::rng::{GameRng, RngKey};
use balatro_core::state::{GameConfig, GameState, Outcome, transition_allowed};

// ----- helpers -----

fn new_game(seed: u64) -> Game {
    Game::new(seed, GameConfig::default())
}

fn select(g: &mut Game) -> StepInfo {
    g.step(Action::SelectBlind).expect("select blind")
}

/// Beat the current blind with one hand: put the round score one chip below the target, then
/// play the leftmost card (any hand scores at least High Card 5 + 2 chips x 1 mult = 7 >= 1).
fn beat_blind(g: &mut Game) -> StepInfo {
    let t = g.state().target;
    g.state_mut().round_score = t - 1.0;
    g.step(Action::Play(0b1)).expect("winning play")
}

fn add(g: &mut Game, id: JokerId) -> usize {
    g.add_joker(JokerInstance::new(0, id))
        .expect("room for joker")
}

/// Step must fail with `err` and leave the state exactly as it was.
fn assert_rejected(g: &mut Game, a: Action, err: ActionError) {
    let before = g.state().clone();
    assert_eq!(g.step(a), Err(err), "{a:?}");
    assert_eq!(g.state(), &before, "state changed by rejected {a:?}");
}

// ----- start of run -----

#[test]
fn initial_state_is_red_deck_white_stake() {
    let g = new_game(1);
    let s = g.state();
    assert_eq!(s.phase, Phase::BlindSelect);
    assert_eq!((s.ante, s.round, s.blind), (1, 0, BlindKind::Small));
    assert_eq!(s.money, 4);
    // 4 hands, 3 + 1 (Red Deck) discards, hand size 8, 5 joker / 2 consumable slots.
    assert_eq!(
        (s.round_hands(), s.round_discards(), s.hand_size()),
        (4, 4, 8)
    );
    assert_eq!((s.joker_capacity(), s.consumable_capacity()), (5, 2));
    assert_eq!(s.deck.len(), 52);
    assert_eq!(s.deck.draw_pile.len(), 52);
    assert_eq!(s.target, 300.0); // ante 1 base 300 x Small 1
    assert_eq!(s.hand_levels, [1; HandType::COUNT]);
    assert!(s.check_invariants().is_ok());
    assert_eq!(
        g.legal_actions(),
        vec![Action::SelectBlind, Action::SkipBlind]
    );
    assert!(!g.is_terminal());
    assert_eq!(g.outcome(), None);
}

#[test]
fn select_blind_starts_round_and_draws_hand() {
    let mut g = new_game(2);
    let info = select(&mut g);
    let s = g.state();
    assert_eq!(
        (info.phase_before, info.phase_after),
        (Phase::BlindSelect, Phase::Play)
    );
    assert_eq!(s.round, 1);
    assert_eq!((s.hands_left, s.discards_left), (4, 4));
    assert_eq!(s.deck.hand.len(), 8);
    assert_eq!(s.deck.draw_pile.len(), 44);
    assert_eq!(s.round_score, 0.0);
    // 8 cards: 8 + 28 + 56 + 70 + 56 = 218 selections, as Play and as Discard.
    let legal = g.legal_actions();
    assert_eq!(legal.len(), 436);
    assert!(legal.contains(&Action::Play(0b1_1111)));
    assert!(!legal.contains(&Action::Play(0b11_1111)));
    // RunView reflects the state.
    let v = s.run_view();
    assert_eq!((v.ante, v.round, v.blind), (1, 1, BlindKind::Small));
    assert_eq!((v.hands_left, v.discards_left, v.money), (4, 4, 4));
    assert_eq!(
        (v.deck.total, v.deck.draw_pile, v.deck.starting_size),
        (52, 44, 52)
    );
    assert_eq!(v.joker_slots, 5);
}

// ----- the full ante loop -----

#[test]
fn full_ante_small_big_boss_then_next_ante() {
    let mut g = new_game(3);

    // Small: one hand played -> 3 hands left. Cash out: interest on $4 = 0, blind $3, hands $3.
    // 4 + 0 + 3 + 3 = 10.
    select(&mut g);
    let info = beat_blind(&mut g);
    let r = info.round.expect("round ended");
    assert!(r.won && !r.saved);
    assert_eq!((r.ante, r.blind), (1, BlindKind::Small));
    let co = r.cash_out.expect("cash out");
    assert_eq!(
        (co.interest, co.blind_reward, co.hands, co.total),
        (0, 3, 3, 6)
    );
    let s = g.state();
    assert_eq!(s.phase, Phase::Shop);
    assert_eq!(s.money, 10);
    assert_eq!(s.blind, BlindKind::Big); // next blind already set up
    assert_eq!(s.target, 450.0); // 300 x 1.5
    assert_eq!(s.deck.draw_pile.len(), 52); // all cards back in the deck
    assert_eq!(g.legal_actions(), vec![Action::EndShop]);
    let info = g.step(Action::EndShop).expect("end shop");
    assert_eq!(
        (info.phase_before, info.phase_after),
        (Phase::Shop, Phase::BlindSelect)
    );

    // Big: interest floor(10/5) = 2, blind $4, hands $3 -> 10 + 9 = 19.
    select(&mut g);
    beat_blind(&mut g);
    assert_eq!(g.state().money, 19);
    g.step(Action::EndShop).expect("end shop");
    assert_eq!(g.state().blind, BlindKind::Boss);
    assert_eq!(g.state().target, 600.0); // 300 x 2
    assert_eq!(g.legal_actions(), vec![Action::SelectBlind]); // no boss skip
    // A boss whose ability cannot stop a one-card win (the ante's real boss may be The Psychic).
    g.state_mut().boss = Some(BossId::TheFlint);

    // Boss: interest floor(19/5) = 3, blind $5, hands $3 -> 19 + 11 = 30. Ante 2 begins.
    select(&mut g);
    let info = beat_blind(&mut g);
    assert_eq!(info.round.expect("round").blind, BlindKind::Boss);
    let s = g.state();
    assert_eq!(s.money, 30);
    assert_eq!((s.ante, s.blind), (2, BlindKind::Small));
    assert_eq!(s.target, 800.0);
    assert_eq!(s.stats.rounds_won, 3);
    assert_eq!(s.stats.run.bosses_defeated, 1);
    assert_eq!(s.round, 3);
    assert_eq!(s.stats.hands_played, 3);
    g.step(Action::EndShop).expect("end shop");
    assert_eq!(g.state().phase, Phase::BlindSelect);
}

#[test]
fn round_continues_until_target_and_refills_hand() {
    let mut g = new_game(4);
    select(&mut g);
    // A single low card cannot reach 300 (High Card 5 + at most 11 chips, x1).
    let info = g.step(Action::Play(0b1000_0000)).expect("play");
    let h = info.hand.expect("hand report");
    assert_eq!(h.hand_type, HandType::HighCard);
    assert!(h.score < 300.0 && h.score >= 7.0);
    let s = g.state();
    assert_eq!(s.phase, Phase::Play);
    assert_eq!(s.round_score, h.score);
    assert_eq!(s.hands_left, 3);
    assert_eq!(s.hands_played_round, 1);
    assert_eq!(s.hand_plays_round[HandType::HighCard.index()], 1);
    assert_eq!(s.hand_plays_run[HandType::HighCard.index()], 1);
    assert_eq!(s.deck.hand.len(), 8); // refilled
    assert_eq!(s.deck.discard_pile.len(), 1); // the played card
    assert_eq!(s.deck.draw_pile.len(), 43);
    assert_eq!(s.last_hand, Some(HandType::HighCard));

    // Discard 3 cards: discard pile 4, hand refilled to 8, draw pile 40.
    let info = g.step(Action::Discard(0b111)).expect("discard");
    assert!(info.hand.is_none());
    let s = g.state();
    assert_eq!((s.discards_left, s.discards_used_round), (3, 1));
    assert_eq!(s.deck.discard_pile.len(), 4);
    assert_eq!(s.deck.hand.len(), 8);
    assert_eq!(s.deck.draw_pile.len(), 40);
}

#[test]
fn played_cards_are_taken_left_to_right_in_hand_order() {
    let mut g = new_game(5);
    select(&mut g);
    let hand = g.state().deck.hand.clone();
    g.step(Action::Play(0b1010_0001)).expect("play");
    let disc: Vec<u32> = g.state().deck.discard_pile.iter().map(|c| c.uid).collect();
    assert_eq!(disc, vec![hand[0].uid, hand[5].uid, hand[7].uid]);
}

// ----- skip / tags -----

#[test]
fn skip_gives_tag_and_moves_to_next_blind() {
    let mut g = new_game(6);
    let tags = g.state().blind_tags;
    let info = g.step(Action::SkipBlind).expect("skip small");
    assert_eq!(info.tag, Some(tags[0]));
    let s = g.state();
    assert_eq!(s.phase, Phase::BlindSelect);
    assert_eq!(s.blind, BlindKind::Big);
    assert_eq!(s.target, 450.0);
    assert_eq!(s.tags, vec![tags[0]]);
    assert_eq!(s.stats.run.blinds_skipped, 1);
    assert_eq!(s.money, 4); // skipping pays nothing
    assert_eq!(s.round, 0); // no round was played

    let info = g.step(Action::SkipBlind).expect("skip big");
    assert_eq!(info.tag, Some(tags[1]));
    assert_eq!(g.state().blind, BlindKind::Boss);
    assert_eq!(g.state().tags, vec![tags[0], tags[1]]);
    assert_rejected(&mut g, Action::SkipBlind, ActionError::CannotSkipBoss);
}

#[test]
fn tags_are_rolled_per_ante_from_the_tag_stream() {
    // Same seed -> same tags; the tag roll uses RngKey::Tags only (deck order unaffected).
    let a = new_game(7);
    let b = new_game(7);
    assert_eq!(a.state().blind_tags, b.state().blind_tags);
    let mut differ = false;
    for seed in 0..20 {
        differ |= new_game(seed).state().blind_tags != a.state().blind_tags;
    }
    assert!(differ);
}

// ----- losing and winning -----

#[test]
fn out_of_hands_below_target_is_game_over() {
    let mut g = new_game(8);
    select(&mut g);
    // Four single-card hands score at most 4 x 16 = 64 < 300.
    let mut last = None;
    for _ in 0..4 {
        last = Some(g.step(Action::Play(0b1000_0000)).expect("play"));
    }
    let info = last.expect("played");
    assert_eq!(info.phase_after, Phase::GameOver);
    let r = info.round.expect("round report");
    assert!(!r.won && !r.saved && !r.out_of_cards);
    assert!(r.cash_out.is_none());
    assert!(g.is_terminal());
    assert_eq!(g.outcome(), Some(Outcome::Loss));
    assert!(g.legal_actions().is_empty());
    assert_rejected(&mut g, Action::SelectBlind, ActionError::GameOver);
    assert_rejected(&mut g, Action::Play(1), ActionError::GameOver);
}

#[test]
fn running_out_of_cards_is_game_over() {
    let mut g = new_game(9);
    select(&mut g);
    {
        // Only the 8 hand cards remain (the rest destroyed), and the target is out of reach.
        let s = g.state_mut();
        let n = s.deck.draw_pile.len() as u32;
        s.deck.draw_pile.clear();
        s.stats.cards_destroyed += n;
        s.target = 1e12;
    }
    let info = g.step(Action::Play(0b1_1111)).expect("play 5");
    assert_eq!(info.phase_after, Phase::Play); // 3 cards left, nothing to draw
    assert_eq!(g.state().deck.hand.len(), 3);
    let info = g.step(Action::Play(0b111)).expect("play 3");
    let r = info.round.expect("round ended");
    assert!(!r.won && r.out_of_cards);
    assert_eq!(g.state().hands_left, 2); // hands were left, but no cards
    assert_eq!(g.outcome(), Some(Outcome::Loss));
}

#[test]
fn beating_the_ante_8_boss_wins() {
    let mut g = new_game(10);
    {
        let s = g.state_mut();
        s.ante = 8;
        s.blind = BlindKind::Boss;
    }
    select(&mut g);
    assert_eq!(g.state().target, 100_000.0); // 50,000 x 2
    let info = beat_blind(&mut g);
    let co = info.round.expect("round").cash_out.expect("cash out");
    assert_eq!(co.blind_reward, 8); // Showdown boss
    assert_eq!(info.phase_after, Phase::Won);
    assert_eq!(g.outcome(), Some(Outcome::Win));
    assert!(g.state().won);
    assert!(g.is_terminal());
    assert!(g.legal_actions().is_empty());
}

#[test]
fn endless_mode_continues_after_the_win() {
    let mut g = Game::new(
        11,
        GameConfig {
            endless: true,
            ..GameConfig::default()
        },
    );
    {
        let s = g.state_mut();
        s.ante = 8;
        s.blind = BlindKind::Boss;
        // Plain boss so a one-card play is never rejected (The Psychic etc.).
        s.boss = Some(BossId::TheWall);
    }
    select(&mut g);
    beat_blind(&mut g);
    let s = g.state();
    assert!(s.won);
    assert_eq!(s.phase, Phase::Shop);
    assert_eq!(s.ante, 9);
    assert_eq!(s.target, 110_000.0); // endless base, Small x1
    assert_eq!(g.outcome(), None);
}

// ----- cash-out math -----

#[test]
fn cash_out_interest_uses_money_before_rewards() {
    // $23: interest floor(23/5) = 4; blind $3; 3 hands left $3 -> 23 + 10 = 33.
    let mut g = new_game(12);
    select(&mut g);
    g.state_mut().money = 23;
    let co = beat_blind(&mut g).round.expect("r").cash_out.expect("co");
    assert_eq!((co.interest_base, co.interest), (23, 4));
    assert_eq!(g.state().money, 33);

    // $40: interest capped at $5; 2 hands used -> 2 left: 40 + 5 + 3 + 2 = 50.
    let mut g = new_game(13);
    select(&mut g);
    g.state_mut().money = 40;
    g.step(Action::Play(0b1000_0000)).expect("play");
    let co = beat_blind(&mut g).round.expect("r").cash_out.expect("co");
    assert_eq!((co.interest, co.hands), (5, 2));
    assert_eq!(g.state().money, 50);
}

#[test]
fn gold_cards_held_pay_before_interest() {
    // Two Gold cards held at round end: +$6 before the cash-out, so interest sees $4 + $6 = $10
    // -> 2. Total: 4 + 6 + 2 (interest) + 3 (blind) + 3 (hands) = 18.
    let mut g = new_game(14);
    select(&mut g);
    {
        let s = g.state_mut();
        // The leftmost card is played by `beat_blind`; positions 6 and 7 stay in hand.
        s.deck.hand[6].enhancement = Enhancement::Gold;
        s.deck.hand[7].enhancement = Enhancement::Gold;
    }
    let r = beat_blind(&mut g).round.expect("r");
    assert_eq!(r.held_money, 6);
    let co = r.cash_out.expect("co");
    assert_eq!((co.interest_base, co.interest), (10, 2));
    assert_eq!(g.state().money, 18);
}

// ----- invalid actions -----

#[test]
fn invalid_actions_are_rejected_without_changing_state() {
    let mut g = new_game(15);
    let wrong = |a, phase| ActionError::WrongPhase { action: a, phase };
    for a in [
        Action::Play(1),
        Action::Discard(1),
        Action::EndShop,
        Action::SellJoker(0),
    ] {
        assert_rejected(&mut g, a, wrong(a, Phase::BlindSelect));
    }
    assert_rejected(
        &mut g,
        Action::Reroll,
        ActionError::NotImplemented {
            action: Action::Reroll,
            phase_of_plan: 5,
        },
    );

    select(&mut g);
    assert_rejected(&mut g, Action::Play(0), ActionError::EmptySelection);
    assert_rejected(
        &mut g,
        Action::Play(0b11_1111),
        ActionError::TooManyCards {
            selected: 6,
            max: 5,
        },
    );
    assert_rejected(
        &mut g,
        Action::Discard(1 << 8),
        ActionError::SelectionOutOfRange {
            mask: 1 << 8,
            hand_len: 8,
        },
    );
    for a in [Action::SelectBlind, Action::SkipBlind, Action::EndShop] {
        assert_rejected(&mut g, a, wrong(a, Phase::Play));
    }
    assert_rejected(
        &mut g,
        Action::SellJoker(0),
        ActionError::InvalidJoker { index: 0, len: 0 },
    );
    let use_c = Action::UseConsumable {
        slot: 0,
        targets: 1,
    };
    assert_rejected(
        &mut g,
        use_c,
        ActionError::InvalidConsumable { index: 0, len: 0 },
    );
    g.state_mut().discards_left = 0;
    assert_rejected(&mut g, Action::Discard(1), ActionError::NoDiscardsLeft);
    assert!(
        !g.legal_actions()
            .iter()
            .any(|a| matches!(a, Action::Discard(_)))
    );
}

#[test]
fn eternal_jokers_cannot_be_sold() {
    let mut g = new_game(16);
    add(&mut g, JokerId::Joker);
    select(&mut g);
    g.state_mut().jokers[0].stickers.eternal = true;
    assert_rejected(&mut g, Action::SellJoker(0), ActionError::EternalJoker);
    assert!(!g.legal_actions().contains(&Action::SellJoker(0)));
}

#[test]
fn every_legal_action_steps_ok() {
    let mut g = new_game(17);
    add(&mut g, JokerId::Joker);
    for _ in 0..3 {
        if g.is_terminal() {
            break;
        }
        for a in g.legal_actions() {
            let mut c = g.clone();
            assert!(c.step(a).is_ok(), "legal {a:?} failed");
        }
        let a = g.legal_actions()[0];
        g.step(a).expect("first legal action");
    }
}

// ----- determinism -----

/// Play a seeded pseudo-random policy for at most `max_steps` steps.
fn random_run(seed: u64, policy_seed: u64, max_steps: usize) -> (Game, Vec<StepInfo>) {
    let mut g = new_game(seed);
    let mut policy = GameRng::new(policy_seed);
    let mut infos = Vec::new();
    let mut legal = Vec::new();
    for _ in 0..max_steps {
        g.legal_actions_into(&mut legal);
        if legal.is_empty() {
            break;
        }
        let a = legal[policy.range(RngKey::Misc, 0, legal.len())];
        infos.push(g.step(a).expect("legal action"));
    }
    (g, infos)
}

#[test]
fn same_seed_and_actions_give_same_state() {
    let (a, ia) = random_run(21, 5, 300);
    let (b, ib) = random_run(21, 5, 300);
    assert_eq!(a.state(), b.state());
    assert_eq!(ia, ib);
    let ja = serde_json::to_string(a.state()).expect("serialize");
    let jb = serde_json::to_string(b.state()).expect("serialize");
    assert_eq!(ja, jb);
    // Round trip through JSON gives the same state.
    let back: GameState = serde_json::from_str(&ja).expect("deserialize");
    assert_eq!(&back, a.state());
}

#[test]
fn different_seeds_shuffle_differently() {
    let mut a = new_game(1);
    let mut b = new_game(2);
    select(&mut a);
    select(&mut b);
    assert_ne!(a.state().deck.hand, b.state().deck.hand);
}

#[test]
fn cloned_game_replays_identically() {
    let mut g = new_game(22);
    select(&mut g);
    g.step(Action::Discard(0b11)).expect("discard");
    let mut c = g.clone();
    for a in [
        Action::Play(0b1),
        Action::Discard(0b101),
        Action::Play(0b11),
    ] {
        let x = g.step(a);
        let y = c.step(a);
        assert_eq!(x, y);
    }
    assert_eq!(g.state(), c.state());
}

// ----- jokers: events and side effects through the game loop -----

#[test]
fn golden_joker_pays_at_cash_out_outside_interest() {
    // Golden Joker +$4 is a cash-out item: interest on $4 = 0; 4 + 0 + 3 + 3 + 4 = 14.
    let mut g = new_game(30);
    add(&mut g, JokerId::GoldenJoker);
    select(&mut g);
    let co = beat_blind(&mut g).round.expect("r").cash_out.expect("co");
    assert_eq!((co.jokers, co.interest), (4, 0));
    assert_eq!(g.state().money, 14);
}

#[test]
fn burnt_joker_levels_first_discarded_hand() {
    let mut g = new_game(31);
    add(&mut g, JokerId::BurntJoker);
    select(&mut g);
    // One card = High Card; first discard of the round -> LevelUpHand applied to the state.
    g.step(Action::Discard(0b1)).expect("discard");
    assert_eq!(g.state().hand_levels[HandType::HighCard.index()], 2);
    // Second discard: no level-up (discards_used_round == 2).
    g.step(Action::Discard(0b1)).expect("discard");
    assert_eq!(g.state().hand_levels[HandType::HighCard.index()], 2);
    // The next hand uses the new level: High Card level 2 = 15 chips x 2 mult.
    let h = g.step(Action::Play(0b1)).expect("play").hand.expect("h");
    assert_eq!(h.level, 2);
}

#[test]
fn faceless_joker_pays_on_discard_event() {
    let mut g = new_game(32);
    add(&mut g, JokerId::FacelessJoker);
    select(&mut g);
    for c in &mut g.state_mut().deck.hand[..3] {
        c.rank = Rank::King;
        c.enhancement = Enhancement::None;
    }
    let info = g.step(Action::Discard(0b111)).expect("discard");
    assert_eq!(info.money_after - info.money_before, 5);
    assert_eq!(g.state().money, 9);
}

#[test]
fn ice_cream_melts_and_is_removed() {
    let mut g = new_game(33);
    g.add_joker(JokerInstance::new(0, JokerId::IceCream).with_value(5.0))
        .expect("room");
    select(&mut g);
    let h = g.step(Action::Play(0b1)).expect("play").hand.expect("h");
    // +5 chips this hand, then 5 - 5 = 0 -> destroy_self -> DestroyJoker applied.
    assert!(h.chips >= 5.0 + 5.0 + 2.0);
    assert!(g.state().jokers.is_empty());
}

#[test]
fn supernova_sees_play_count_including_current_hand() {
    let mut g = new_game(34);
    add(&mut g, JokerId::Supernova);
    select(&mut g);
    // High Card mult 1 + Supernova (High Card played once, this hand) 1 = 2.
    let h = g.step(Action::Play(0b1)).expect("play").hand.expect("h");
    assert_eq!(h.mult, 2.0);
    let h = g.step(Action::Play(0b1)).expect("play").hand.expect("h");
    assert_eq!(h.mult, 3.0);
}

#[test]
fn space_joker_level_up_persists() {
    let (mut hit, mut miss) = (false, false);
    for seed in 0..64 {
        let mut g = new_game(seed);
        add(&mut g, JokerId::SpaceJoker);
        select(&mut g);
        let h = g.step(Action::Play(0b1)).expect("play").hand.expect("h");
        let lvl = g.state().hand_levels[HandType::HighCard.index()];
        assert_eq!(lvl, h.level, "the persisted level equals the level used");
        if h.level == 2 {
            hit = true;
        } else {
            miss = true;
        }
        if hit && miss {
            return;
        }
    }
    panic!("Space Joker never both triggered and missed in 64 seeds (1 in 4)");
}

#[test]
fn glass_break_destroys_card_permanently() {
    for seed in 0..64 {
        let mut g = new_game(seed);
        select(&mut g);
        let uid = {
            let s = g.state_mut();
            s.deck.hand[0].enhancement = Enhancement::Glass;
            s.deck.hand[0].uid
        };
        let info = g.step(Action::Play(0b1)).expect("play");
        if info.cards_destroyed == 1 {
            let s = g.state();
            assert_eq!(s.deck.len(), 51);
            assert_eq!(s.stats.cards_destroyed, 1);
            assert!(s.deck.iter_all().all(|c| c.uid != uid));
            assert!(s.check_invariants().is_ok());
            // The deck stays at 51 into the next round.
            beat_blind(&mut g);
            assert_eq!(g.state().deck.draw_pile.len(), 51);
            return;
        }
    }
    panic!("no Glass break in 64 seeds (1 in 4 each)");
}

#[test]
fn selling_jokers_pays_sell_value_and_refreshes_modifiers() {
    let mut g = new_game(35);
    add(&mut g, JokerId::Joker); // cost 2 -> sells for 1
    add(&mut g, JokerId::FourFingers); // cost 7 -> sells for 3
    assert!(g.state().mods.rules.four_fingers);
    assert_eq!(g.state().jokers[1].sell_value, 3);
    select(&mut g);
    let info = g.step(Action::SellJoker(1)).expect("sell in play");
    assert_eq!(info.sold_for, Some(3));
    assert_eq!(g.state().money, 7);
    assert!(!g.state().mods.rules.four_fingers);
    assert_eq!(g.state().stats.jokers_sold, 1);
    beat_blind(&mut g);
    let m = g.state().money;
    assert!(g.legal_actions().contains(&Action::SellJoker(0)));
    g.step(Action::SellJoker(0)).expect("sell in shop");
    assert_eq!(g.state().money, m + 1);
    assert!(g.state().jokers.is_empty());
}

#[test]
fn joker_slots_limit_add_joker() {
    let mut g = new_game(36);
    for _ in 0..5 {
        add(&mut g, JokerId::Joker);
    }
    assert!(g.add_joker(JokerInstance::new(0, JokerId::Joker)).is_none());
    // A Negative joker brings its own slot.
    let neg = JokerInstance::new(0, JokerId::Joker).with_edition(Edition::Negative);
    assert_eq!(g.add_joker(neg), Some(5));
    assert_eq!(g.state().joker_capacity(), 6);
    let uids: Vec<u32> = g.state().jokers.iter().map(|j| j.uid).collect();
    assert_eq!(uids, vec![0, 1, 2, 3, 4, 5]);
}

// ----- side-effect application -----

#[test]
fn side_effects_apply_to_state() {
    let mut g = new_game(40);
    add(&mut g, JokerId::Joker);
    add(&mut g, JokerId::Bull);
    select(&mut g);
    let s = g.state_mut();
    s.jokers[0].stickers.eternal = true;
    let eternal_uid = s.jokers[0].uid;
    let bull_uid = s.jokers[1].uid;
    let card_uid = s.deck.hand[0].uid;
    let template = Card::new(999, Rank::Ace, Suit::Spades);
    let fx = [
        SideEffect::LevelUpHand {
            hand: HandType::Flush,
            levels: 2,
        },
        SideEffect::LevelUpHand {
            hand: HandType::Pair,
            levels: -5,
        },
        SideEffect::DestroyJoker { uid: eternal_uid }, // refused (eternal)
        SideEffect::DestroyJoker { uid: bull_uid },
        SideEffect::DestroyJoker { uid: bull_uid }, // idempotent
        SideEffect::DestroyCard { uid: card_uid },
        SideEffect::AddCard {
            card: template,
            to: Pile::Hand,
        },
        SideEffect::AddSellValue {
            target: SellTarget::AllJokers,
            amount: 3,
        },
        SideEffect::AddHands(2),
        SideEffect::ClearDiscards,
        SideEffect::CreateTag { id: 7 },
        SideEffect::SetRunFlag(RunFlag::GrosMichelExtinct),
        SideEffect::SetRunFlag(RunFlag::GrosMichelExtinct),
        SideEffect::PreventDeath,
        SideEffect::CreateJoker { rarity: None }, // Phase 5: dropped
    ];
    let rep = apply_side_effects(s, &fx);
    assert_eq!(s.hand_levels[HandType::Flush.index()], 3);
    assert_eq!(s.hand_levels[HandType::Pair.index()], 1); // never below 1
    assert_eq!(s.jokers.len(), 1);
    assert_eq!(s.jokers[0].uid, eternal_uid);
    assert_eq!(s.jokers[0].sell_value, 1 + 3);
    assert!(s.deck.iter_all().all(|c| c.uid != card_uid));
    let added = s
        .deck
        .hand
        .iter()
        .find(|c| c.rank == Rank::Ace && c.uid >= 52 && c.uid != 999)
        .expect("added card in hand");
    assert_eq!(added.uid, 52); // fresh uid from the deck
    assert_eq!(s.deck.len(), 52); // -1 destroyed +1 added
    assert_eq!((s.stats.cards_destroyed, s.stats.cards_added), (1, 1));
    assert_eq!((s.hands_left, s.discards_left), (6, 0));
    assert_eq!(s.tags, vec![7]);
    assert_eq!(s.run_flags, vec![RunFlag::GrosMichelExtinct]);
    assert!(rep.prevent_death);
    assert_eq!((rep.destroyed, rep.added), (1, 1));
    assert_eq!(rep.dropped, 2); // eternal refusal + CreateJoker
    assert!(s.check_invariants().is_ok());
}

#[test]
fn copy_joker_needs_room_and_fires_acquire() {
    let mut g = new_game(41);
    add(&mut g, JokerId::Bull);
    let s = g.state_mut();
    let uid = s.jokers[0].uid;
    let rep = apply_side_effects(
        s,
        &[SideEffect::CopyJoker {
            uid,
            strip_negative: true,
        }],
    );
    assert_eq!(rep.dropped, 0);
    assert_eq!(s.jokers.len(), 2);
    assert_ne!(s.jokers[1].uid, uid);
    assert_eq!(s.jokers[1].id, JokerId::Bull);
    // Fill the slots, then a copy is dropped.
    for _ in 0..3 {
        apply_side_effects(
            s,
            &[SideEffect::CopyJoker {
                uid,
                strip_negative: true,
            }],
        );
    }
    assert_eq!(s.jokers.len(), 5);
    let rep = apply_side_effects(
        s,
        &[SideEffect::CopyJoker {
            uid,
            strip_negative: true,
        }],
    );
    assert_eq!(rep.dropped, 1);
    assert_eq!(s.jokers.len(), 5);
}

#[test]
fn added_cards_outside_a_round_go_to_the_draw_pile() {
    let mut g = new_game(42);
    let s = g.state_mut();
    apply_side_effects(
        s,
        &[SideEffect::AddCard {
            card: Card::new(0, Rank::Two, Suit::Hearts),
            to: Pile::Hand,
        }],
    );
    assert_eq!(s.deck.draw_pile.len(), 53);
    assert!(s.deck.hand.is_empty());
    assert!(s.check_invariants().is_ok());
}

// ----- invariants / transitions -----

#[test]
fn invariants_detect_lost_cards_and_stale_modifiers() {
    let mut g = new_game(43);
    g.state_mut().deck.draw_pile.pop();
    assert!(g.state().check_invariants().is_err());

    let mut g = new_game(44);
    g.state_mut()
        .jokers
        .push(JokerInstance::new(0, JokerId::FourFingers));
    g.state_mut().next_joker_uid = 1;
    // Modifier cache not refreshed.
    assert!(g.state().check_invariants().is_err());
    g.state_mut().refresh_modifiers();
    assert!(g.state().check_invariants().is_ok());
}

#[test]
fn phase_transition_table() {
    use Phase::*;
    for (from, to) in [
        (BlindSelect, Play),
        (BlindSelect, BlindSelect),
        (Play, Play),
        (Play, Shop),
        (Play, GameOver),
        (Play, Won),
        (Shop, Shop),
        (Shop, BlindSelect),
    ] {
        assert!(transition_allowed(from, to), "{from:?} -> {to:?}");
    }
    for (from, to) in [
        (BlindSelect, Shop),
        (Shop, Play),
        (Play, BlindSelect),
        (GameOver, BlindSelect),
        (Won, Won),
        (Won, Shop),
    ] {
        assert!(!transition_allowed(from, to), "{from:?} -> {to:?}");
    }
}
