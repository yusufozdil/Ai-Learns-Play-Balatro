# Decks

All 15 base decks. Code: `content/decks.rs` (`DeckKind`, `DeckDef` data table, `build_deck`), numbers
in `data/stakes.rs`; the effects are read in `GameState::new`, `Game::finish_round`
(cash-out), `shop::card_type_weights`, `GameState::refresh_target`, `GameState::run_view`.
Items marked `UNVERIFIED` are not stated on a fetched page.

Sources (fetched 2026-11-15):
- D1 https://balatrowiki.org/w/Decks (effect table of all 15 decks)
- D2 https://balatrowiki.org/w/Plasma_Deck
- D3 https://balatrowiki.org/w/Erratic_Deck
- D4 https://balatrowiki.org/w/Blinds_and_Antes (Plasma doubles the requirement "for all stakes")
- D5 https://balatrowiki.org/w/Money (Green Deck payout, also blinds.md §4)

Run-start base (project decision): $4, 4 hands, 3 discards, hand size 8, 5 joker slots, 2 consumable
slots, 52 cards. A deck is a set of deltas on that base.

| Deck | Effect (D1) | Implementation |
|---|---|---|
| Red | +1 discard every round | discards 4 |
| Blue | +1 hand every round | hands 5 |
| Yellow | start with an extra $10 | money 14 |
| Green | end of round: $2 per remaining hand and $1 per remaining discard; no interest | `cash_out_with(.., hand_reward 2, discard_reward 1)` and `no_interest` |
| Black | +1 joker slot, -1 hand every round | slots 6, hands 3 |
| Magic | start with Crystal Ball (voucher) and 2 copies of The Fool | voucher redeemed, 2 consumables; slots 2 + 1 = 3 |
| Nebula | start with Telescope (voucher), -1 consumable slot | slots 1; Telescope's effect lives in `ShopMods::telescope` |
| Ghost | Spectral cards can appear in the shop; start with a Hex card | spectral weight 2 in card slots (`WEIGHT_SPECTRAL_GHOST_DECK`); Hex held |
| Abandoned | start with no face cards | 40 cards (no Jack/Queen/King) |
| Checkered | start with 26 Spades and 26 Hearts | two of each rank in Spades and Hearts |
| Zodiac | start with Tarot Merchant, Planet Merchant and Overstock | three vouchers redeemed |
| Painted | +2 hand size, -1 joker slot | hand size 10, slots 4 |
| Anaglyph | after defeating each Boss Blind, gain a Double Tag | Double Tag pushed on boss cash-out |
| Plasma | chips and mult balanced when scoring; base blind size x2 | `RunView::plasma_deck`, target x2 |
| Erratic | all ranks and suits are randomized | 52 cards, independent random rank and suit |

## Notes per deck

- **Starting vouchers (Magic, Nebula, Zodiac)** are redeemed before the first ante rolls its
  voucher, so the ante voucher can never be one the deck already owns (`voucher::redeem`, the
  normal path). They count as owned (`GameState::vouchers`) and feed `ShopMods`.
- **Magic**: The Fool creates the last Tarot/Planet used; with none used it does nothing (existing
  consumable rule). The two copies fill 2 of the 3 slots.
- **Nebula**: the wiki's "Celestial Packs always contain the Planet for your most played hand" is the
  Telescope voucher itself (D1 shows it with the deck); no deck-specific logic.
- **Ghost**: spectral weight 2 against joker 20 / tarot 4 / planet 4 (D1 says only "may appear";
  the weight 2 is the existing shop.md §2 decision). Hex needs a joker without edition to do
  anything; held from the start, it is usable in a round or a pack once a joker exists.
- **Zodiac**: Tarot Merchant / Planet Merchant raise those shop weights to 9.6 (shop.md §2, an
  existing decision; the wiki text says "2x more often"); Overstock gives 3 card slots from the
  first shop.
- **Green**: the $2 per hand replaces the base $1 (it is not added to it): hands money
  = 2 x hands left, discards money = 1 x discards left, interest 0 (D5, blinds.md §4). Used
  discards earn nothing. Gold cards held and joker payouts still pay.
- **Black / Painted**: Black's extra joker slot is a base slot (Negative editions add more);
  Painted's -1 slot never goes below 0.
- **Anaglyph**: the tag is added after the boss's cash-out and after the Investment Tag; it is
  held like any Double Tag and doubles the next tag gained (`tag::gain`). Also granted at the
  ante-8 boss in endless mode (the run continues); at the win the run ends anyway.
- **Plasma**:
  - Blind size x2 applies to every blind (Small, Big, Boss), on top of the boss's own multiplier
    (D4: "doubles the Ante requirement for all stakes"). D2 does not say which blinds are
    affected; applying it to all three is the reading of "base Blind size" (the factor multiplies
    the ante's base chips, as the game's ante scaling does). `UNVERIFIED` for Showdown bosses; they
    are treated the same.
  - The balancing step is `Scorer` step 7 (`tests/scoring.rs::plasma_deck_balances`,
    `plasma_deck_floors_the_balanced_value`). Verified against the real game: both values become `math.floor((chips + mult) / 2)` (the wiki's "rounds
    half-down"; the old engine kept the unfloored average, off by up to 0.5 each).
    Its place relative to joker effects follows scoring.md step 7 (after Observatory, i.e. after
    all joker/edition effects).
  - `GameState::refresh_target` multiplies by `DeckDef::blind_size_mult`; the heuristic agent's
    boss-pressure estimate does too.
- **Erratic**: each of the 52 cards gets an independent uniform rank and suit (D3: "mathematically
  like sampling with replacement from a standard deck"; the wiki also lists a seed bug that gives
  all 10 of Spades, not modelled). Drawn from `RngKey::Deck` at run creation, so the deck is
  fixed per seed and costs no extra stream. Uids 0..52, so every invariant holds.
- **Abandoned**: `starting_deck_size` is 40, which Erosion, the card-conservation invariant and the
  deck histogram all read.

## Observation
Nothing deck- or stake-specific is in the observation yet (see the Phase 5E report for the proposal).
