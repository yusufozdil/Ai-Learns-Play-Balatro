//! `Game` steps of the market, booster packs and boss rerolls (Phase 5.0): `Buy`, `BuyAndUse`,
//! `OpenPack`, `BuyVoucher`, `Reroll`, `MoveJoker`, `PickPackItem`, `SkipPack`, `RerollBoss`.
//! Every method assumes `actions::validate` accepted the action (it is the single gate), so
//! none of them can fail; they pay first, then change the world, then fire joker events.

use crate::consumable::ConsumableInstance;
use crate::data::shop_weights::BOSS_REROLL_COST;
use crate::deck::Pile;
use crate::economy;
use crate::enums::Phase;
use crate::game::{Game, StepInfo, fire_paid, gather_deck};
use crate::hooks::{self, PackKind, SideEffect};
use crate::packs::{self, PackItem, PackOrigin, PackSpec};
use crate::shop::{self, ShopCardKind};
use crate::voucher;

impl Game {
    /// Pay `price` (validated) and record it in `info`.
    fn pay(&mut self, price: i32, info: &mut StepInfo) {
        let s = &mut self.state;
        s.money = s.money.saturating_sub(price);
        info.spent = Some(price);
    }

    /// `Buy(slot)`: joker, consumable (into its slot) or playing card (into the deck).
    pub(crate) fn buy(&mut self, slot: usize, info: &mut StepInfo) {
        let Some(card) = self.state.shop.cards.get_mut(slot).and_then(Option::take) else {
            return; // rejected by `validate`
        };
        let price = shop::card_price(&self.state, &card);
        self.pay(price, info);
        match card.kind {
            ShopCardKind::Joker(mut j) => {
                j.sell_value = economy::sell_value(price, 0);
                let _ = self.add_joker(j);
            }
            ShopCardKind::Consumable(id) => {
                let mut c = ConsumableInstance::new(id);
                c.sell_value = economy::sell_value(price, 0);
                self.state.consumables.push(c);
            }
            ShopCardKind::PlayingCard(c) => {
                self.fx.push(SideEffect::AddCard {
                    card: c,
                    to: Pile::Draw,
                });
                self.resolve_fx();
            }
        }
    }

    /// `BuyAndUse(slot)`: pay, then use the consumable at once (it never takes a slot).
    pub(crate) fn buy_and_use(&mut self, slot: usize, info: &mut StepInfo) {
        let Some(card) = self.state.shop.cards.get_mut(slot).and_then(Option::take) else {
            return;
        };
        let ShopCardKind::Consumable(id) = card.kind else {
            return;
        };
        let price = shop::card_price(&self.state, &card);
        self.pay(price, info);
        let mut c = ConsumableInstance::new(id);
        c.sell_value = economy::sell_value(price, 0);
        self.apply_use(c, 0, info);
    }

    /// `OpenPack(slot)`: pay, open the pack (`PackOpen`), `on_pack_open`.
    pub(crate) fn open_pack_slot(&mut self, slot: usize, info: &mut StepInfo) {
        let Some(pack) = self.state.shop.packs.get_mut(slot).and_then(Option::take) else {
            return;
        };
        let price = shop::pack_price_of(&self.state, &pack);
        self.pay(price, info);
        self.begin_pack(pack.spec, PackOrigin::Shop);
    }

    /// Open `spec` and fire `on_pack_open`.
    fn begin_pack(&mut self, spec: PackSpec, origin: PackOrigin) {
        let s = &mut self.state;
        packs::open(s, spec, origin);
        let kind = spec.kind;
        fire_paid(
            s,
            &mut self.fx,
            |h| h.on_pack_open.is_some(),
            |inp| hooks::fire_pack_open(inp, kind),
        );
        self.resolve_fx();
    }

    /// Open the next queued pack (pack tags), if any. Used after a skip.
    pub(crate) fn open_next_pending_pack(&mut self) {
        if self.state.pending_packs.is_empty() {
            return;
        }
        let spec = self.state.pending_packs.remove(0);
        self.begin_pack(spec, PackOrigin::Tag);
    }

    /// `BuyVoucher(slot)`: pay and redeem.
    pub(crate) fn buy_voucher(&mut self, slot: usize, info: &mut StepInfo) {
        let Some(v) = self
            .state
            .shop
            .vouchers
            .get_mut(slot)
            .and_then(Option::take)
        else {
            return;
        };
        let price = shop::voucher_price(&self.state, v.id);
        self.pay(price, info);
        if v.from_ante {
            self.state.ante_voucher = None;
        }
        voucher::redeem(&mut self.state, v.id);
    }

    /// `Reroll`: a free reroll first (Chaos the Clown), else paid (+$1 each); new card slots;
    /// `on_reroll` for jokers (Flash Card).
    pub(crate) fn reroll(&mut self, info: &mut StepInfo) {
        let price = shop::reroll_price(&self.state);
        let free = shop::has_free_reroll(&self.state);
        self.pay(price, info);
        let s = &mut self.state;
        if free {
            s.shop.free_rerolls_used += 1;
        } else {
            s.shop.rerolls += 1;
        }
        shop::reroll_cards(s);
        fire_paid(
            s,
            &mut self.fx,
            |h| h.on_reroll.is_some(),
            hooks::fire_reroll,
        );
        self.resolve_fx();
    }

    /// `MoveJoker`: remove at `from`, insert at `to`; the modifier cache follows the order.
    pub(crate) fn move_joker(&mut self, from: u8, to: u8) {
        let s = &mut self.state;
        let (from, to) = (usize::from(from), usize::from(to));
        if from >= s.jokers.len() || to >= s.jokers.len() {
            return;
        }
        let j = s.jokers.remove(from);
        s.jokers.insert(to, j);
        s.refresh_modifiers();
    }

    /// `PickPackItem`: take the item (a consumable is used at once, a card joins the deck, a joker
    /// the joker area); the pack closes when the picks are used up or nothing is left.
    pub(crate) fn pick_pack_item(&mut self, index: usize, targets: u16, info: &mut StepInfo) {
        let Some(pack) = self.state.pack.as_mut() else {
            return;
        };
        let Some(item) = pack.items.get_mut(index).and_then(Option::take) else {
            return;
        };
        pack.picks_left = pack.picks_left.saturating_sub(1);
        match item {
            PackItem::Consumable(id) => {
                self.apply_use(ConsumableInstance::new(id), targets, info);
            }
            PackItem::Card(c) => {
                self.fx.push(SideEffect::AddCard {
                    card: c,
                    to: packs::STANDARD_CARD_PILE,
                });
                self.resolve_fx();
            }
            PackItem::Joker(j) => {
                let _ = self.add_joker(j);
            }
        }
        let done = self
            .state
            .pack
            .as_ref()
            .is_none_or(|p| p.picks_left == 0 || p.items.iter().all(Option::is_none));
        if done {
            self.close_pack();
        }
    }

    /// `SkipPack`: `on_pack_skip` (once per skipped pack, Mega after one pick included), close.
    pub(crate) fn skip_pack(&mut self) {
        let kind: PackKind = self
            .state
            .pack
            .as_ref()
            .map_or(PackKind::Standard, |p| p.spec.kind);
        let s = &mut self.state;
        fire_paid(
            s,
            &mut self.fx,
            |h| h.on_pack_skip.is_some(),
            |inp| hooks::fire_pack_skip(inp, kind),
        );
        self.resolve_fx();
        self.close_pack();
    }

    /// Close the open pack: the dealt hand goes back to the draw pile; the next queued pack
    /// opens, otherwise return to the Shop (bought) or BlindSelect (tag).
    fn close_pack(&mut self) {
        let s = &mut self.state;
        let Some(pack) = s.pack.take() else {
            return;
        };
        if pack.spec.deals_hand() {
            gather_deck(s);
        }
        if !s.pending_packs.is_empty() {
            self.open_next_pending_pack();
            return;
        }
        s.phase = match pack.origin {
            PackOrigin::Shop => Phase::Shop,
            PackOrigin::Tag => Phase::BlindSelect,
        };
    }

    /// `RerollBoss`: pay $10 and draw a different boss for this ante.
    pub(crate) fn reroll_boss(&mut self, info: &mut StepInfo) {
        self.pay(BOSS_REROLL_COST, info);
        let s = &mut self.state;
        s.boss_rerolls_ante += 1;
        crate::boss::reroll_boss(s);
    }
}
