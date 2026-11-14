//! Phase 4.R closing audit: registry integrity (consumables, bosses), `effects_dropped`
//! provenance, and natural coverage of every boss in full games.

use balatro_core::actions::Action;
use balatro_core::agents::{Agent, RandomAgent};
use balatro_core::consumable::{ConsumableInstance, MAX_CONSUMABLE_TARGETS};
use balatro_core::content::jokers::JokerId;
use balatro_core::content::{
    BossId, ConsumableId, boss_def_count, bosses, consumable_def_count, try_boss_def,
    try_consumable_def,
};
use balatro_core::enums::Phase;
use balatro_core::game::Game;
use balatro_core::joker::JokerInstance;
use balatro_core::rng::{GameRng, RngKey};
use balatro_core::state::GameConfig;

// ----- registry integrity ---------------------------------------------------------------------

#[test]
fn every_consumable_id_has_exactly_one_consistent_def() {
    assert_eq!(ConsumableId::COUNT, 52);
    assert_eq!(ConsumableId::PLANETS.len(), 12);
    assert_eq!(ConsumableId::TAROTS.len(), 22);
    assert_eq!(ConsumableId::SPECTRALS.len(), 18);
    for id in ConsumableId::ALL {
        assert_eq!(consumable_def_count(id), 1, "{id:?}");
        let d = try_consumable_def(id).expect("registered");
        assert_eq!(d.id, id);
        assert!(!d.name.is_empty());
        assert!(d.min_targets <= d.max_targets, "{}", d.name);
        assert!(d.max_targets <= MAX_CONSUMABLE_TARGETS, "{}", d.name);
        if d.max_targets == 0 {
            assert_eq!(d.min_targets, 0, "{}", d.name);
        }
        assert_eq!(d.kind(), id.kind(), "{}", d.name);
        if d.max_targets > 0 {
            assert!(d.requires_hand(), "{}", d.name);
        }
    }
}

/// Min antes from docs/mechanics/bosses.md (source S1, balatrowiki.org/w/Blinds_and_Antes;
/// "Any" = 1, Showdown = 8).
const BOSS_MIN_ANTE: [(BossId, i32); 28] = [
    (BossId::TheHook, 1),
    (BossId::TheOx, 6),
    (BossId::TheHouse, 2),
    (BossId::TheWall, 2),
    (BossId::TheWheel, 2),
    (BossId::TheArm, 2),
    (BossId::TheClub, 1),
    (BossId::TheFish, 2),
    (BossId::ThePsychic, 1),
    (BossId::TheGoad, 1),
    (BossId::TheWater, 2),
    (BossId::TheWindow, 1),
    (BossId::TheManacle, 1),
    (BossId::TheEye, 3),
    (BossId::TheMouth, 2),
    (BossId::ThePlant, 4),
    (BossId::TheSerpent, 5),
    (BossId::ThePillar, 1),
    (BossId::TheNeedle, 2),
    (BossId::TheHead, 1),
    (BossId::TheTooth, 3),
    (BossId::TheFlint, 2),
    (BossId::TheMark, 2),
    (BossId::AmberAcorn, 8),
    (BossId::VerdantLeaf, 8),
    (BossId::VioletVessel, 8),
    (BossId::CrimsonHeart, 8),
    (BossId::CeruleanBell, 8),
];

#[test]
fn every_boss_id_has_exactly_one_def_with_the_documented_min_ante() {
    assert_eq!(BossId::COUNT, 28);
    assert_eq!(bosses::DEFS.len(), 28);
    for id in BossId::ALL {
        assert_eq!(boss_def_count(id), 1, "{id:?}");
        let table: Vec<_> = BOSS_MIN_ANTE.iter().filter(|(b, _)| *b == id).collect();
        assert_eq!(table.len(), 1, "{id:?} missing from the table");
        let d = try_boss_def(id).expect("registered");
        assert_eq!(d.min_ante, table[0].1, "{}", d.name);
    }
}

// ----- effects_dropped provenance ------------------------------------------------------------
//
// `stats.effects_dropped` is one aggregate counter: it cannot tell a dropped `CreateJoker`
// (Judgement, Wraith, The Soul, Riff-raff; Phase 5 stub) from a creation refused because a slot
// area was full, an eternal joker that could not be destroyed, or a cascade overflow. So the
// strongest test removes the other causes (huge slot areas, no eternal) and requires every step
// that grows the counter to be explained by a CreateJoker source.

fn is_joker_source(id: ConsumableId) -> bool {
    matches!(
        id,
        ConsumableId::Judgement | ConsumableId::Wraith | ConsumableId::TheSoul
    )
}

fn run_dropped_games(games: u64, allow_riff_raff: bool) -> (u64, u64) {
    let (mut explained, mut total_steps) = (0u64, 0u64);
    for seed in 0..games {
        let mut g = Game::new(seed, GameConfig::default());
        g.state_mut().consumable_slots = 40;
        g.state_mut().joker_slots = 30;
        let mut agent = RandomAgent::with_play_weight(seed, 4.0);
        let mut pick = GameRng::new(seed ^ 0xD20F);
        for _ in 0..pick.range(RngKey::Misc, 0, 6) {
            let j = JokerId::ALL[pick.range(RngKey::Misc, 0, JokerId::COUNT)];
            if j == JokerId::RiffRaff && !allow_riff_raff {
                continue;
            }
            let _ = g.add_joker(JokerInstance::new(0, j));
        }
        let mut steps = 0;
        while !g.is_terminal() && steps < 400 {
            if g.state().phase == Phase::Play {
                if g.state().consumables.len() < 6 && pick.range(RngKey::Misc, 0, 6) == 0 {
                    let id = ConsumableId::ALL[pick.range(RngKey::Misc, 0, ConsumableId::COUNT)];
                    let _ = g.add_consumable(ConsumableInstance::new(id));
                }
                if pick.range(RngKey::Misc, 0, 3) == 0 {
                    let t = g.state().target;
                    g.state_mut().round_score = g.state().round_score.max(t * 0.9);
                }
            }
            let a = agent.act(&g);
            let used = match a {
                Action::UseConsumable { slot, .. } => {
                    g.state().consumables.get(usize::from(slot)).map(|c| c.id)
                }
                _ => None,
            };
            let before = g.state().stats.effects_dropped;
            let riff = g.state().jokers.iter().any(|j| j.id == JokerId::RiffRaff);
            g.step(a).expect("legal");
            let delta = g.state().stats.effects_dropped - before;
            if delta > 0 {
                let by_consumable = used.is_some_and(is_joker_source);
                assert!(
                    by_consumable || riff,
                    "seed {seed}: {delta} dropped by {a:?} ({used:?}) without a CreateJoker source"
                );
                explained += u64::from(delta);
            }
            steps += 1;
            total_steps += 1;
        }
    }
    (explained, total_steps)
}

#[test]
fn effects_dropped_only_come_from_joker_creating_effects() {
    let (explained, steps) = run_dropped_games(1500, false);
    assert!(steps > 10_000);
    // The CreateJoker consumables are in the random pool: the counter must actually move.
    assert!(explained > 0, "no joker-creating consumable was ever used");
}

#[test]
fn effects_dropped_with_riff_raff_in_the_pool_stay_explained() {
    let (_, steps) = run_dropped_games(600, true);
    assert!(steps > 4_000);
}
