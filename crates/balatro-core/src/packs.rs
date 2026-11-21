//! Booster packs: [`PackSpec`] (kind x size), the open pack of the `PackOpen` phase, and content
//! generation. Phase 5.0. Rules: docs/mechanics/packs.md; numbers:
//! `data/shop_weights.rs`. Guide: `docs/design/shop_packs_vouchers_tags.md`.
//!
//! Contents are rolled when the pack is opened, on `RngKey::Packs`. Arcana and Spectral packs
//! also deal a hand (current hand size) so targeted cards can be used; the hand is gathered back
//! into the draw pile when the pack closes. Picking a consumable uses it at once (it never takes
//! a slot); a Standard card is shuffled into the deck; a Buffoon joker needs a free joker slot
//! (Negative needs none). The step functions are in `game_shop.rs`.

use crate::card::Card;
use crate::consumable::{ConsumableKind, random_consumable_excluding};
use crate::content::ids::ConsumableId;
use crate::content::jokers::JokerId;
use crate::data::shop_weights::{
    OMEN_GLOBE_SPECTRAL_CHANCE, SOUL_BLACK_HOLE_CHANCE, STANDARD_ENHANCEMENT_CHANCE,
    STANDARD_SEAL_CHANCE, pack_cards, pack_picks,
};
use crate::deck::Pile;
use crate::enums::{Edition, HandType, Phase};
use crate::hooks::PackKind;
use crate::joker::JokerInstance;
use crate::rng::RngKey;
use crate::shop::{random_playing_card, roll_joker};
use crate::state::GameState;
use serde::{Deserialize, Serialize};

/// Size of a booster pack.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PackSize {
    /// Normal: $4.
    Normal,
    /// Jumbo: $6.
    Jumbo,
    /// Mega: $8, two picks.
    Mega,
}

impl PackSize {
    /// All sizes.
    pub const ALL: [PackSize; 3] = [PackSize::Normal, PackSize::Jumbo, PackSize::Mega];
    /// Number of sizes.
    pub const COUNT: usize = Self::ALL.len();
    /// Stable 0-based index.
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// One of the 15 booster packs: kind x size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PackSpec {
    /// Arcana, Celestial, Spectral, Standard or Buffoon.
    pub kind: PackKind,
    /// Normal, Jumbo or Mega.
    pub size: PackSize,
}

impl PackSpec {
    /// Items shown.
    pub const fn cards(self) -> u8 {
        pack_cards(self.kind, self.size)
    }
    /// Items that may be taken.
    pub const fn picks(self) -> u8 {
        pack_picks(self.size)
    }
    /// Stable index `kind * 3 + size` in `0..15`.
    pub const fn index(self) -> usize {
        self.kind.index() * PackSize::COUNT + self.size.index()
    }
    /// Arcana and Spectral packs deal a hand of playing cards.
    pub const fn deals_hand(self) -> bool {
        matches!(self.kind, PackKind::Arcana | PackKind::Spectral)
    }
}

/// One item of an open pack.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PackItem {
    /// A consumable, used when picked.
    Consumable(ConsumableId),
    /// A playing card, added to the deck when picked.
    Card(Card),
    /// A joker, added to the joker area when picked.
    Joker(JokerInstance),
}

/// Where an open pack returns to when it closes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PackOrigin {
    /// Bought in the shop: back to `Shop`.
    Shop,
    /// Opened by a skip tag: back to `BlindSelect`.
    Tag,
}

/// The pack of the current `PackOpen` phase.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenPack {
    /// Kind and size.
    pub spec: PackSpec,
    /// Items by position; a picked item leaves `None` (positions do not shift).
    pub items: Vec<Option<PackItem>>,
    /// Picks still allowed (Mega starts at 2).
    pub picks_left: u8,
    /// Where to return.
    pub origin: PackOrigin,
}

/// The Planet-of-the-most-played-hand rule (Telescope): the hand played most often this run,
/// ties to the highest tier; `None` if nothing was played yet.
pub fn most_played_hand(plays: &[u32; HandType::COUNT]) -> Option<HandType> {
    let mut best: Option<(u32, HandType)> = None;
    for &h in HandType::ALL.iter().rev() {
        let n = plays[h.index()];
        if n > 0 && best.is_none_or(|(b, _)| n > b) {
            best = Some((n, h));
        }
    }
    best.map(|(_, h)| h)
}

/// Roll the items of a pack (`RngKey::Packs`). No duplicates inside one pack unless Showman.
pub fn generate_items(s: &mut GameState, spec: PackSpec) -> Vec<Option<PackItem>> {
    let key = RngKey::Packs;
    let n = usize::from(spec.cards());
    let allow_dup = s.mods.allow_duplicates;
    let mut items: Vec<Option<PackItem>> = Vec::with_capacity(n);
    let mut taken: Vec<ConsumableId> = Vec::new();
    let mut taken_jokers: Vec<JokerId> = Vec::new();
    for i in 0..n {
        let item = match spec.kind {
            PackKind::Standard => {
                let em = s.shop_mods.edition_mult;
                let c = random_playing_card(
                    &mut s.rng,
                    key,
                    STANDARD_ENHANCEMENT_CHANCE,
                    STANDARD_SEAL_CHANCE,
                    em,
                );
                PackItem::Card(c)
            }
            PackKind::Buffoon => match roll_joker(s, key, None, &taken_jokers) {
                Some(j) => {
                    taken_jokers.push(j.id);
                    PackItem::Joker(j)
                }
                None => break,
            },
            kind => {
                let r = s.rng.float(key);
                let soul_ok =
                    |id: ConsumableId, taken: &[ConsumableId]| allow_dup || !taken.contains(&id);
                let id = match kind {
                    PackKind::Arcana => {
                        if r < SOUL_BLACK_HOLE_CHANCE && soul_ok(ConsumableId::TheSoul, &taken) {
                            ConsumableId::TheSoul
                        } else if s.shop_mods.omen_globe
                            && s.rng.float(key) < OMEN_GLOBE_SPECTRAL_CHANCE
                        {
                            pick(s, ConsumableKind::Spectral, &taken)
                        } else {
                            pick(s, ConsumableKind::Tarot, &taken)
                        }
                    }
                    PackKind::Celestial => {
                        let tele = if i == 0 && s.shop_mods.telescope {
                            most_played_hand(&s.hand_plays_run).map(ConsumableId::planet_for)
                        } else {
                            None
                        };
                        match tele {
                            Some(id) => id,
                            None if r < SOUL_BLACK_HOLE_CHANCE
                                && soul_ok(ConsumableId::BlackHole, &taken) =>
                            {
                                ConsumableId::BlackHole
                            }
                            None => pick(s, ConsumableKind::Planet, &taken),
                        }
                    }
                    _ => {
                        // Spectral: The Soul and Black Hole, 0.3% each.
                        if r < SOUL_BLACK_HOLE_CHANCE && soul_ok(ConsumableId::TheSoul, &taken) {
                            ConsumableId::TheSoul
                        } else if (SOUL_BLACK_HOLE_CHANCE..2.0 * SOUL_BLACK_HOLE_CHANCE)
                            .contains(&r)
                            && soul_ok(ConsumableId::BlackHole, &taken)
                        {
                            ConsumableId::BlackHole
                        } else {
                            pick(s, ConsumableKind::Spectral, &taken)
                        }
                    }
                };
                taken.push(id);
                PackItem::Consumable(id)
            }
        };
        items.push(Some(item));
    }
    items
}

fn pick(s: &mut GameState, kind: ConsumableKind, taken: &[ConsumableId]) -> ConsumableId {
    random_consumable_excluding(
        kind,
        taken,
        &s.hand_plays_run,
        s.mods.allow_duplicates,
        RngKey::Packs,
        &mut s.rng,
    )
}

/// Open a pack: roll its items, deal the hand of an Arcana/Spectral pack (shuffle, then draw to
/// the hand size) and enter `Phase::PackOpen`. Fires no joker event (the game layer does).
pub fn open(s: &mut GameState, spec: PackSpec, origin: PackOrigin) {
    let items = generate_items(s, spec);
    if spec.deals_hand() {
        s.deck.reset_for_round(&mut s.rng);
        let n = s.hand_size() as usize;
        s.deck.draw_to_hand_size(n);
    }
    s.pack = Some(OpenPack {
        spec,
        items,
        picks_left: spec.picks(),
        origin,
    });
    s.phase = Phase::PackOpen;
}

/// A free joker slot exists for `j` (Negative always fits).
pub fn joker_fits(s: &GameState, j: &JokerInstance) -> bool {
    j.edition == Edition::Negative || s.has_joker_room()
}

/// Where a standard-pack card goes: shuffled into the draw pile.
pub const STANDARD_CARD_PILE: Pile = Pile::Draw;
