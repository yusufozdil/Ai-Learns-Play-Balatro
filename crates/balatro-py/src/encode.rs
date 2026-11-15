//! Observation and mask encoding (`docs/design/rl_interface.md` sections 3 and 5).
//!
//! Everything the player cannot see is read through the `observe.rs` accessors of the core
//! (`Card::visible_card`, `GameState::visible_joker`, `draw_pile_histogram`); card/joker uids,
//! the draw order, the RNG and the real `preview_play` are never touched.

use crate::layout::*;
use balatro_core::action_mask::{ActionKind, CardSelect, card_select, structured_mask};
use balatro_core::blinds::BlindKind;
use balatro_core::boss;
use balatro_core::card::Card;
use balatro_core::enums::Phase;
use balatro_core::joker::JokerInstance;
use balatro_core::packs::PackItem;
use balatro_core::shop::ShopCardKind;
use balatro_core::state::GameState;

/// `log10(1 + x) / 10`, `x` clamped to `[0, 1e30]`.
fn lg(x: f64) -> f32 {
    ((1.0 + x.clamp(0.0, 1e30)).log10() / 10.0) as f32
}

/// Signed `ln(1 + |x|) / 5`.
fn sl(x: f64) -> f32 {
    if x == 0.0 || x.is_nan() {
        0.0
    } else {
        (x.signum() * (1.0 + x.abs().min(1e30)).ln() / 5.0) as f32
    }
}

/// `ln(1 + x) / 3` for counts.
fn cc(x: f64) -> f32 {
    ((1.0 + x.max(0.0)).ln() / 3.0) as f32
}

/// Sequential writer into a feature slice.
struct Cur<'a> {
    s: &'a mut [f32],
    i: usize,
}

impl<'a> Cur<'a> {
    fn new(s: &'a mut [f32]) -> Self {
        Cur { s, i: 0 }
    }
    #[inline]
    fn put(&mut self, v: f32) {
        self.s[self.i] = v;
        self.i += 1;
    }
    #[inline]
    fn flag(&mut self, b: bool) {
        self.put(if b { 1.0 } else { 0.0 });
    }
}

/// `x / d` as f32.
fn div(x: f64, d: f64) -> f32 {
    (x / d) as f32
}

fn rank_cat(c: &Card) -> i32 {
    if c.is_stone() {
        (balatro_core::enums::Rank::COUNT + 1) as i32
    } else {
        c.rank.index() as i32 + 1
    }
}

fn suit_cat(c: &Card) -> i32 {
    if c.is_stone() {
        (balatro_core::enums::Suit::COUNT + 1) as i32
    } else {
        c.suit.index() as i32 + 1
    }
}

/// Categorical columns of a visible card (`rank, suit, enhancement, edition, seal`).
fn card_cats(c: &Card, out: &mut [i32]) {
    out[0] = rank_cat(c);
    out[1] = suit_cat(c);
    out[2] = c.enhancement.index() as i32 + 1;
    out[3] = c.edition.index() as i32 + 1;
    out[4] = c.seal.index() as i32 + 1;
}

/// Write the observation of `s` into `row` (every observation buffer is fully rewritten).
pub fn write_obs(s: &GameState, row: &mut Row<'_>) {
    row.global_f.fill(0.0);
    row.global_i.fill(0);
    row.hand_f.fill(0.0);
    row.hand_i.fill(0);
    row.hand_valid.fill(false);
    row.joker_f.fill(0.0);
    row.joker_i.fill(0);
    row.joker_valid.fill(false);
    row.cons_f.fill(0.0);
    row.cons_i.fill(0);
    row.cons_valid.fill(false);
    row.offer_f.fill(0.0);
    row.offer_i.fill(0);
    row.offer_valid.fill(false);
    write_global(s, row);
    write_hand(s, row);
    write_jokers(s, row);
    write_consumables(s, row);
    write_offers(s, row);
}

fn write_global(s: &GameState, row: &mut Row<'_>) {
    let gi = &mut *row.global_i;
    gi[0] = s.phase.index() as i32;
    gi[1] = match s.blind {
        BlindKind::Small => 0,
        BlindKind::Big => 1,
        BlindKind::Boss => 2,
    };
    gi[2] = s.boss.map_or(0, |b| b.index() as i32 + 1);
    gi[3] = s.last_hand.map_or(0, |h| h.index() as i32 + 1);
    gi[4] = if s.boss_in_effect() {
        s.boss_state.hand.map_or(0, |h| h.index() as i32 + 1)
    } else {
        0
    };
    gi[5] = s.pack.as_ref().map_or(0, |p| p.spec.index() as i32 + 1);
    if s.phase == Phase::BlindSelect {
        let slot = match s.blind {
            BlindKind::Small => Some(0),
            BlindKind::Big => Some(1),
            BlindKind::Boss => None,
        };
        if let Some(i) = slot {
            gi[6] = s.blind_tags[i].index() as i32 + 1;
            gi[7] = s.blind_tag_hands[i].map_or(0, |h| h.index() as i32 + 1);
        }
    }

    gi[8] = s.config.deck.index() as i32;
    gi[9] = s.config.stake.index() as i32;

    let finite = s.target.is_finite() && s.target > 0.0;
    let ratio = if finite {
        s.round_score / s.target
    } else {
        0.0
    };
    let mut w = Cur::new(&mut *row.global_f);
    w.put(div(f64::from(s.ante), 8.0));
    w.put(div(f64::from(s.round), 24.0));
    w.put(lg(s.target));
    w.put(lg(s.round_score));
    w.put(ratio.clamp(0.0, 1.0) as f32);
    w.put((ratio.clamp(0.0, 4.0) / 4.0) as f32);
    w.put(sl(f64::from(s.money)));
    w.put(div(f64::from(s.hands_left), 8.0));
    w.put(div(f64::from(s.discards_left), 8.0));
    w.put(div(f64::from(s.hand_size()), 16.0));
    w.put(div(s.deck.hand.len() as f64, 16.0));
    w.put(div(s.jokers.len() as f64, 10.0));
    w.put(div(f64::from(s.joker_capacity()), 10.0));
    w.put(div(s.consumables.len() as f64, 4.0));
    w.put(div(f64::from(s.consumable_capacity()), 4.0));
    w.put(sl(f64::from(s.reroll_price())));
    w.put(div(f64::from(s.interest_cap), 25.0));
    w.put(div(s.deck.draw_pile.len() as f64, 52.0));
    w.put(div(s.deck.len() as f64, 52.0));
    w.put(div(f64::from(s.hands_played_round), 8.0));
    w.put(div(f64::from(s.discards_used_round), 8.0));
    w.flag(s.boss_in_effect());
    w.flag(s.boss.is_some() && s.boss_disabled);
    w.flag(s.jokers_hidden());
    w.flag(s.is_showdown());
    w.flag(s.won);
    w.put(div(
        s.pack.as_ref().map_or(0.0, |p| f64::from(p.picks_left)),
        2.0,
    ));
    for l in s.hand_levels {
        w.put((f64::from(l.max(1)).ln() / 3.0) as f32);
    }
    for n in s.hand_plays_round {
        w.put(cc(f64::from(n)));
    }
    for n in s.hand_plays_run {
        w.put(cc(f64::from(n)));
    }
    let h = s.draw_pile_histogram();
    for suits in &h.rank_suit {
        for &n in suits {
            w.put(f32::from(n) / 4.0);
        }
    }
    w.put(f32::from(h.stone) / 4.0);
    for &n in &h.by_enhancement[1..] {
        w.put(cc(f64::from(n)));
    }
    for &n in &h.by_edition[1..] {
        w.put(cc(f64::from(n)));
    }
    for &n in &h.by_seal[1..] {
        w.put(cc(f64::from(n)));
    }
    let owned = |v: balatro_core::state::VoucherId| s.vouchers.contains(&v);
    for v in balatro_core::state::VoucherId::ALL {
        w.flag(owned(v));
    }
    let mut tags = [0u32; balatro_core::state::TagId::COUNT];
    for t in &s.tags {
        tags[t.index()] += 1;
    }
    for n in tags {
        w.put(n as f32 / 2.0);
    }
    let mut packs = [0u32; PACK_SPECS];
    for p in &s.pending_packs {
        packs[p.index()] += 1;
    }
    for n in packs {
        w.put(n as f32 / 2.0);
    }
    // Run configuration (interface version 2).
    let stake = s.config.stake;
    let ddef = s.config.deck.def();
    w.put(div(stake.index() as f64, 7.0));
    w.flag(stake.no_small_reward());
    w.flag(stake.eternal_jokers());
    w.flag(stake.perishable_jokers());
    w.flag(stake.rental_jokers());
    w.flag(ddef.blind_size_mult > 1.0);
    w.put(div(f64::from(s.base_hands), 8.0));
    w.put(div(f64::from(s.base_discards), 8.0));
    w.flag(ddef.no_interest);
    w.put(div(f64::from(ddef.hand_reward), 2.0));
    w.put(div(f64::from(ddef.discard_reward), 2.0));
    debug_assert_eq!(w.i, GLOBAL_F);
}

fn write_hand(s: &GameState, row: &mut Row<'_>) {
    let nf = HAND_F.len();
    let ni = HAND_I.len();
    let forced = boss::forced_position(s);
    for (i, card) in s.deck.hand.iter().take(HAND_CAP).enumerate() {
        row.hand_valid[i] = true;
        let f = &mut row.hand_f[i * nf..(i + 1) * nf];
        f[0] = i as f32 / 16.0;
        f[5] = if forced == Some(i) { 1.0 } else { 0.0 };
        match card.visible_card() {
            None => f[4] = 1.0,
            Some(c) => {
                f[1] = c.base_chips() as f32 / 11.0;
                f[2] = c.perma_bonus as f32 / 50.0;
                f[3] = if c.debuffed { 1.0 } else { 0.0 };
                card_cats(&c, &mut row.hand_i[i * ni..(i + 1) * ni]);
            }
        }
    }
}

fn joker_flags(j: &JokerInstance, f: &mut [f32]) {
    f[2] = if j.stickers.eternal { 1.0 } else { 0.0 };
    f[3] = if j.stickers.perishable { 1.0 } else { 0.0 };
    f[4] = f32::from(j.stickers.perish_rounds_left) / 5.0;
    f[5] = if j.stickers.rental { 1.0 } else { 0.0 };
}

fn write_jokers(s: &GameState, row: &mut Row<'_>) {
    let nf = JOKER_F.len();
    let ni = JOKER_I.len();
    for i in 0..s.jokers.len().min(JOKER_CAP) {
        row.joker_valid[i] = true;
        let f = &mut row.joker_f[i * nf..(i + 1) * nf];
        f[0] = i as f32 / 16.0;
        let Some(j) = s.visible_joker(i) else {
            f[7] = 1.0; // hidden (Amber Acorn)
            continue;
        };
        f[1] = sl(f64::from(j.sell_value));
        joker_flags(j, f);
        f[6] = if j.debuffed { 1.0 } else { 0.0 };
        f[8] = (j.value / 10.0).clamp(-10.0, 10.0) as f32;
        f[9] = sl(j.value);
        f[10] = f64::from(j.counter.min(100)) as f32 / 10.0;
        f[11] = (f64::from(j.counter).ln_1p() / 5.0) as f32;
        f[12] = (j.aux_byte() as f32) / 255.0;
        let c = &mut row.joker_i[i * ni..(i + 1) * ni];
        c[0] = joker_item(j.id.index());
        c[1] = j.edition.index() as i32 + 1;
        c[2] = j.aux_suit().map_or(0, |x| x.index() as i32 + 1);
        c[3] = j.aux_rank().map_or(0, |x| x.index() as i32 + 1);
        c[4] = j.aux_hand().map_or(0, |x| x.index() as i32 + 1);
    }
}

fn write_consumables(s: &GameState, row: &mut Row<'_>) {
    let nf = CONS_F.len();
    for (i, c) in s.consumables.iter().take(CONS_CAP).enumerate() {
        row.cons_valid[i] = true;
        let f = &mut row.cons_f[i * nf..(i + 1) * nf];
        f[0] = i as f32 / 4.0;
        f[1] = sl(f64::from(c.sell_value));
        f[2] = if c.negative { 1.0 } else { 0.0 };
        row.cons_i[i] = cons_item(c.id.index());
    }
}

/// Shared writer of one offer row.
struct Offer<'r, 'a> {
    row: &'r mut Row<'a>,
    idx: usize,
}

impl Offer<'_, '_> {
    fn begin(&mut self, source: i32, price: f64, afford: bool, free: bool) {
        let i = self.idx;
        self.row.offer_valid[i] = true;
        self.row.offer_i[i * OFFER_I.len()] = source;
        let f = &mut self.row.offer_f[i * OFFER_F.len()..(i + 1) * OFFER_F.len()];
        f[0] = sl(price);
        f[1] = if free { 1.0 } else { 0.0 };
        f[2] = if afford { 1.0 } else { 0.0 };
    }
    fn cats(&mut self) -> &mut [i32] {
        let n = OFFER_I.len();
        &mut self.row.offer_i[self.idx * n..(self.idx + 1) * n]
    }
    fn feats(&mut self) -> &mut [f32] {
        let n = OFFER_F.len();
        &mut self.row.offer_f[self.idx * n..(self.idx + 1) * n]
    }
    fn item(&mut self, item: i32) {
        self.cats()[1] = item;
    }
    fn joker(&mut self, j: &JokerInstance) {
        self.item(joker_item(j.id.index()));
        self.cats()[2] = j.edition.index() as i32 + 1;
        let f = self.feats();
        f[3] = if j.stickers.eternal { 1.0 } else { 0.0 };
        f[4] = if j.stickers.perishable { 1.0 } else { 0.0 };
        f[5] = if j.stickers.rental { 1.0 } else { 0.0 };
        f[6] = sl(f64::from(j.sell_value));
    }
    fn card(&mut self, c: &Card) {
        let cats = self.cats();
        cats[2] = c.edition.index() as i32 + 1;
        cats[3] = rank_cat(c);
        cats[4] = suit_cat(c);
        cats[5] = c.enhancement.index() as i32 + 1;
        cats[6] = c.seal.index() as i32 + 1;
    }
}

fn write_offers(s: &GameState, row: &mut Row<'_>) {
    if s.phase == Phase::Shop {
        for (i, slot) in s.shop.cards.iter().take(SHOP_CARD_CAP).enumerate() {
            let Some(card) = slot else { continue };
            let price = s.shop_card_price(i).unwrap_or(0);
            let mut o = Offer { row, idx: i };
            o.begin(1, f64::from(price), s.money >= price, price == 0);
            match &card.kind {
                ShopCardKind::Joker(j) => o.joker(j),
                ShopCardKind::Consumable(id) => o.item(cons_item(id.index())),
                ShopCardKind::PlayingCard(c) => o.card(c),
            }
        }
        for (i, slot) in s.shop.packs.iter().take(SHOP_PACK_CAP).enumerate() {
            let Some(p) = slot else { continue };
            let price = s.shop_pack_price(i).unwrap_or(0);
            let mut o = Offer {
                row,
                idx: OFFER_PACK0 + i,
            };
            o.begin(2, f64::from(price), s.money >= price, price == 0);
            o.item(pack_item(p.spec.index()));
        }
        for (i, slot) in s.shop.vouchers.iter().take(SHOP_VOUCHER_CAP).enumerate() {
            let Some(v) = slot else { continue };
            let price = s.shop_voucher_price(i).unwrap_or(0);
            let mut o = Offer {
                row,
                idx: OFFER_VOUCHER0 + i,
            };
            o.begin(3, f64::from(price), s.money >= price, price == 0);
            o.item(voucher_item(v.id.index()));
        }
    }
    if let Some(pack) = &s.pack {
        for (i, slot) in pack.items.iter().take(PACK_ITEM_CAP).enumerate() {
            let Some(item) = slot else { continue };
            let mut o = Offer {
                row,
                idx: OFFER_ITEM0 + i,
            };
            o.begin(4, 0.0, true, false);
            match item {
                PackItem::Consumable(id) => o.item(cons_item(id.index())),
                PackItem::Card(c) => o.card(c),
                PackItem::Joker(j) => o.joker(j),
            }
        }
    }
}

/// One card-selection row (`sel_*` buffers).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SelRow {
    /// Hand positions that may be selected.
    pub candidates: u32,
    /// Positions that must be selected.
    pub required: u32,
    /// Fewest cards.
    pub min: u8,
    /// Most cards (0 = the row takes no cards).
    pub max: u8,
}

impl SelRow {
    /// The core's selection rule for this row.
    pub fn select(&self) -> CardSelect {
        CardSelect {
            candidates: self.candidates,
            required: self.required,
            min: self.min,
            max: self.max,
        }
    }
}

/// The masks of one state in compact form: what `write_masks` writes and what the Rust-side
/// samplers draw from, so a sampled action is by construction one the model could build.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MaskView {
    /// Bit `k` = kind `k` is legal (and addressable).
    pub kinds: u32,
    /// Per kind, bit `a` = `arg0 = a` is legal (kinds without `arg0`: bit 0).
    pub arg0: [u32; KINDS],
    /// `MoveJoker` destination candidates (bit = joker slot; `arg0` itself is excluded by the
    /// consumer).
    pub arg1: u32,
    /// Card-selection rows.
    pub sel: [SelRow; NSEL],
}

fn low(n: usize) -> u32 {
    if n >= 32 { u32::MAX } else { (1u32 << n) - 1 }
}

impl MaskView {
    /// Compute the masks of `s` (one pass of `validate` calls, no allocation).
    pub fn compute(s: &GameState) -> MaskView {
        let sm = structured_mask(s);
        let mut v = MaskView::default();
        for kind in ActionKind::ALL {
            let k = kind.index();
            if !sm.allows_kind(kind) {
                continue;
            }
            let set = sm.arg0(kind);
            // Slots below 64 only: every cap is far below that.
            let bits = (set.0[0] & u64::from(low(arg0_cap(k)))) as u32;
            if bits != 0 {
                v.kinds |= 1 << k;
                v.arg0[k] = bits;
            }
        }
        if v.kinds >> ActionKind::MoveJoker.index() & 1 == 1 {
            v.arg1 = low(usize::from(sm.joker_count).min(JOKER_CAP));
        }
        for kind in [ActionKind::Play, ActionKind::Discard] {
            if v.kinds >> kind.index() & 1 == 1
                && let (Some(sel), Some(r)) = (card_select(s, kind, 0), sel_row(kind.index(), 0))
            {
                v.sel[r] = SelRow {
                    candidates: sel.candidates,
                    required: sel.required,
                    min: sel.min,
                    max: sel.max,
                };
            }
        }
        for (kind, cap) in [
            (ActionKind::UseConsumable, CONS_CAP),
            (ActionKind::PickPackItem, PACK_ITEM_CAP),
        ] {
            let mut bits = v.arg0[kind.index()];
            while bits != 0 {
                let a = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                debug_assert!(a < cap);
                if let (Some(sel), Some(r)) =
                    (card_select(s, kind, a as u8), sel_row(kind.index(), a))
                {
                    v.sel[r] = SelRow {
                        candidates: sel.candidates,
                        required: sel.required,
                        min: sel.min,
                        max: sel.max,
                    };
                }
            }
        }
        v
    }
}

/// Write `v` into the mask buffers of `row`.
pub fn write_masks(v: &MaskView, row: &mut Row<'_>) {
    for k in 0..KINDS {
        row.kind_mask[k] = v.kinds >> k & 1 == 1;
        let bits = v.arg0[k];
        let r = &mut row.arg0_mask[k * MAX_ARG..(k + 1) * MAX_ARG];
        for (a, slot) in r.iter_mut().enumerate() {
            *slot = bits >> a & 1 == 1;
        }
    }
    for (a, slot) in row.arg1_mask.iter_mut().enumerate() {
        *slot = v.arg1 >> a & 1 == 1;
    }
    for (i, sel) in v.sel.iter().enumerate() {
        row.sel_candidates[i] = sel.candidates;
        row.sel_required[i] = sel.required;
        row.sel_min[i] = sel.min;
        row.sel_max[i] = sel.max;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::{ConfigPool, Env, RewardConfig, Shared};
    use crate::layout::Owned;
    use balatro_core::content::decks::DeckKind;
    use balatro_core::content::ids::BossId;
    use balatro_core::content::jokers::JokerId;
    use balatro_core::content::stakes::StakeKind;
    use balatro_core::enums::{Enhancement, Rank, Suit};
    use balatro_core::game::Game;
    use balatro_core::joker::JokerInstance;
    use balatro_core::state::GameConfig;

    fn encode(s: &GameState) -> Owned {
        let mut o = Owned::new(1);
        {
            let b = o.bufs();
            // SAFETY: single-threaded, one live row.
            let mut row = unsafe { b.row(0) };
            write_obs(s, &mut row);
            write_masks(&MaskView::compute(s), &mut row);
        }
        o
    }

    fn play_game(seed: u64) -> Game {
        let mut g = Game::new(seed, GameConfig::default());
        g.step(balatro_core::actions::Action::SelectBlind).unwrap();
        assert_eq!(g.state().phase, Phase::Play);
        g
    }

    #[test]
    fn amber_acorn_hides_joker_identity_but_not_the_count() {
        let build = |ids: [JokerId; 3], value: f64| {
            let mut g = play_game(1);
            for (i, id) in ids.into_iter().enumerate() {
                let mut j = JokerInstance::new(10 + i as u32, id).with_value(value);
                j.stickers.eternal = i == 0;
                g.add_joker(j).unwrap();
            }
            let s = g.state_mut();
            s.blind = BlindKind::Boss;
            s.boss = Some(BossId::AmberAcorn);
            s.boss_state.jokers_hidden = true;
            g
        };
        let a = build(
            [JokerId::Joker, JokerId::GreedyJoker, JokerId::LustyJoker],
            3.0,
        );
        let b = build(
            [
                JokerId::WrathfulJoker,
                JokerId::GluttonousJoker,
                JokerId::Joker,
            ],
            9.0,
        );
        assert!(a.state().jokers_hidden());
        let (oa, ob) = (encode(a.state()), encode(b.state()));
        assert_eq!(oa.joker_f, ob.joker_f);
        assert_eq!(oa.joker_i, ob.joker_i);
        assert_eq!(oa.joker_valid, ob.joker_valid);
        assert_eq!(oa.joker_valid.iter().filter(|&&v| v).count(), 3);
        assert_eq!(oa.joker_f[7], 1.0, "hidden flag");
        assert!(oa.joker_i[..5].iter().all(|&c| c == 0));
        // Without Amber Acorn the same two states differ.
        let mut a2 = a.clone();
        a2.state_mut().boss_state.jokers_hidden = false;
        let mut b2 = b.clone();
        b2.state_mut().boss_state.jokers_hidden = false;
        assert_ne!(encode(a2.state()).joker_i, encode(b2.state()).joker_i);
    }

    #[test]
    fn run_configuration_columns() {
        let names = global_f_names();
        let col = |n: &str| names.iter().position(|x| x == n).unwrap();
        let obs = |deck, stake| {
            let g = Game::new(
                4,
                GameConfig {
                    deck,
                    stake,
                    endless: false,
                },
            );
            encode(g.state())
        };
        let o = obs(DeckKind::Red, StakeKind::White);
        let f = |o: &Owned, n: &str| o.global_f[col(n)];
        assert_eq!(&o.global_i[8..10], &[0, 0]);
        assert_eq!(f(&o, "stake_idx"), 0.0);
        for n in [
            "no_small_reward",
            "eternal_on",
            "perishable_on",
            "rental_on",
        ] {
            assert_eq!(f(&o, n), 0.0, "{n}");
        }
        assert_eq!(f(&o, "plasma_blind_x2"), 0.0);
        assert_eq!(f(&o, "base_hands"), 0.5);
        assert_eq!(f(&o, "base_discards"), 0.5);
        assert_eq!(f(&o, "money_per_hand"), 0.5);
        assert_eq!(f(&o, "money_per_discard"), 0.0);
        assert_eq!(f(&o, "no_interest"), 0.0);

        let o = obs(DeckKind::Plasma, StakeKind::Gold);
        assert_eq!(
            &o.global_i[8..10],
            &[
                DeckKind::Plasma.index() as i32,
                StakeKind::Gold.index() as i32
            ]
        );
        assert_eq!(f(&o, "stake_idx"), 1.0);
        for n in [
            "no_small_reward",
            "eternal_on",
            "perishable_on",
            "rental_on",
            "plasma_blind_x2",
        ] {
            assert_eq!(f(&o, n), 1.0, "{n}");
        }
        assert_eq!(f(&o, "base_discards"), 2.0 / 8.0); // 3 base, -1 Blue Stake and above

        let o = obs(DeckKind::Green, StakeKind::Blue);
        assert_eq!(f(&o, "no_interest"), 1.0);
        assert_eq!(f(&o, "money_per_hand"), 1.0);
        assert_eq!(f(&o, "money_per_discard"), 0.5);
        assert_eq!(f(&o, "stake_idx"), 4.0 / 7.0);
        assert_eq!(f(&o, "eternal_on"), 1.0);
        assert_eq!(f(&o, "perishable_on"), 0.0);
        assert_eq!(f(&o, "base_discards"), 2.0 / 8.0);
    }

    #[test]
    fn face_down_cards_are_interchangeable_in_the_observation() {
        let mut g = play_game(2);
        let s = g.state_mut();
        for c in s.deck.hand.iter_mut().skip(1).step_by(2) {
            c.face_down = true;
        }
        let a = encode(g.state());
        // Replace every hidden card with a completely different card (new uid too).
        let s = g.state_mut();
        for (i, c) in s.deck.hand.iter_mut().enumerate().skip(1).step_by(2) {
            *c = Card {
                face_down: true,
                perma_bonus: 17,
                debuffed: true,
                ..Card::new(900 + i as u32, Rank::Ace, Suit::Hearts)
            };
        }
        let b = encode(g.state());
        assert_eq!(a.hand_f, b.hand_f);
        assert_eq!(a.hand_i, b.hand_i);
        assert_eq!(a.global_f, b.global_f);
        assert_eq!(a.sel_candidates, b.sel_candidates);
    }

    #[test]
    fn stone_and_edition_cards_use_their_categories() {
        let mut g = play_game(3);
        let s = g.state_mut();
        s.deck.hand[0].enhancement = Enhancement::Stone;
        s.deck.hand[1].edition = balatro_core::enums::Edition::Polychrome;
        let o = encode(g.state());
        let n = HAND_I.len();
        assert_eq!(o.hand_i[0], 14); // Stone rank
        assert_eq!(o.hand_i[1], 5); // Stone suit
        assert_eq!(o.hand_i[2], Enhancement::Stone.index() as i32 + 1);
        assert_eq!(
            o.hand_i[n + 3],
            balatro_core::enums::Edition::Polychrome.index() as i32 + 1
        );
        assert_eq!(o.hand_f[1], 0.0, "a Stone card has no chip value");
    }

    #[test]
    fn market_and_pack_offers_mirror_the_state() {
        let sh = Shared {
            config: GameConfig::default(),
            configs: ConfigPool::EMPTY,
            max_steps: 100_000,
            base_seed: 11,
            reward: RewardConfig::default(),
        };
        let (mut saw_shop, mut saw_pack) = (false, false);
        for idx in 0..6 {
            let mut e = Env::new(idx, &sh);
            for _ in 0..400 {
                let p = e.teacher_parts(&sh);
                if e.step_parts(p, &sh, true).unwrap().done != 0 {
                    break;
                }
                let s = e.game.state();
                let o = encode(s);
                let no = OFFER_I.len();
                if s.phase == Phase::Shop {
                    saw_shop = true;
                    for (i, c) in s.shop.cards.iter().take(SHOP_CARD_CAP).enumerate() {
                        assert_eq!(o.offer_valid[i], c.is_some());
                        if c.is_some() {
                            assert_eq!(o.offer_i[i * no], 1);
                            let price = s.shop_card_price(i).unwrap();
                            assert_eq!(o.offer_f[i * OFFER_F.len()], sl(f64::from(price)));
                            assert_eq!(o.offer_f[i * OFFER_F.len() + 2] == 1.0, s.money >= price);
                        }
                    }
                    for (i, v) in s.shop.vouchers.iter().take(SHOP_VOUCHER_CAP).enumerate() {
                        let row = OFFER_VOUCHER0 + i;
                        assert_eq!(o.offer_valid[row], v.is_some());
                        if let Some(v) = v {
                            assert_eq!(o.offer_i[row * no + 1], voucher_item(v.id.index()));
                        }
                    }
                    assert!(o.offer_valid[OFFER_ITEM0..].iter().all(|&v| !v));
                } else {
                    assert!(o.offer_valid[..OFFER_ITEM0].iter().all(|&v| !v));
                }
                if let Some(pack) = &s.pack {
                    saw_pack = true;
                    for (i, it) in pack.items.iter().enumerate() {
                        assert_eq!(o.offer_valid[OFFER_ITEM0 + i], it.is_some());
                    }
                    assert_eq!(o.global_i[5], pack.spec.index() as i32 + 1);
                } else {
                    assert_eq!(o.global_i[5], 0);
                }
            }
        }
        assert!(
            saw_shop && saw_pack,
            "the teacher should reach shops and packs"
        );
    }
}
