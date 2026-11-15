# Stakes

All eight stakes, cumulative: every stake applies the modifiers of the stakes below it. Code:
`content/stakes.rs` (`StakeKind`), numbers in `data/stakes.rs`, chip tables in `data/blind_chips.rs`.
Items marked `UNVERIFIED` are not stated on a fetched page.

Sources (fetched 2026-11-15):
- S1 https://balatrowiki.org/w/Stakes
- S2 https://balatrowiki.org/w/Stickers
- S3 https://balatrowiki.org/w/Eternal
- S4 https://balatrowiki.org/w/Perishable
- S5 https://balatrowiki.org/w/Rental
- S6 https://balatrowiki.org/w/Blinds_and_Antes (chip tables, see also blinds.md §2)
- S7 https://balatrowiki.org/w/Plasma_Deck (blind size; decks.md)

(balatrogame.fandom.com returned HTTP 402 for every fetch, so the wiki above is the only source.)

## 1. Ladder (S1)

| # | Stake | Effect added | Implementation |
|---|---|---|---|
| 1 | White | none | base chip table (S6) |
| 2 | Red | Small Blind gives no reward money | `no_small_reward` (`economy::blind_reward`) |
| 3 | Green | required score scales faster each ante | second chip table (`StakeScaling::Green`) |
| 4 | Black | shop (and pack) jokers can be Eternal | sticker roll, 30% |
| 5 | Blue | -1 discard | `discard_penalty` = 1, applied to the starting discards |
| 6 | Purple | required score scales faster again | third chip table (`StakeScaling::Purple`) |
| 7 | Orange | shop (and pack) jokers can be Perishable | sticker roll, 30% |
| 8 | Gold | shop (and pack) jokers can be Rental | sticker roll, 30% |

- Blind chip tables per scaling level (S6, identical to blinds.md §2): White and Red use the base
  table, Green/Black/Blue the "Green or higher" column, Purple/Orange/Gold the "Purple or higher"
  column (the wiki header says "Green Stake or higher" and "Purple Stake or higher"). Ante 1 is 300
  for all of them; ante 8 is 50,000 / 100,000 / 200,000.
- Blue Stake does NOT change blind size. blinds.md noted a possible conflict ("doubles base blind
  size" under Blue); S1 lists only "-1 Discard" for Blue and S6 attributes the doubling to the
  Plasma Deck alone. Decision: Blue = -1 discard only. (Resolves the `UNVERIFIED` in blinds.md §2.)
- Red Stake: the Small blind pays $0; Big ($4), Boss ($5) and Showdown ($8) are unchanged (S1, S6).
- Blue Stake: the discard is removed from the starting value, so Red Deck + Blue Stake = 3
  discards, Blue Deck + Blue Stake = 2. Vouchers (Wasteful etc.) still add on top. The clamp at 0
  never triggers with the current decks.
- Endless (ante > 8) uses the White formula on every stake (S6 gives no other formula), as before.

## 2. Stickers (S2-S5)

### Where they appear
- Jokers generated for the shop's card slots and for Buffoon Packs (S3: "Jokers in shops or booster
  packs ... the same 30% chance applies to both sources"; S4 likewise).
- Also on the free jokers an Uncommon/Rare Tag adds to the shop (they are shop jokers).
  `UNVERIFIED` by the wiki; in the real game those tags create their joker through the shop path.
- NOT on jokers created by Judgement, Wraith, Riff-Raff, Top-up Tag or The Soul, and never on
  Legendary jokers (S3, S4, S5: "Legendary Jokers can't receive it"). Judgement/Wraith/Riff-Raff
  are `UNVERIFIED` on the wiki (only the legendary exclusion is stated); they create jokers
  directly into the joker area, outside the shop/pack generator.

### Probabilities
- Eternal: 30%, Black Stake and above (S3, S2).
- Perishable: 30%, Orange Stake and above (S4, S2).
- Rental: 30%, Gold Stake (S5, S2); S5 says non-Legendary jokers only.
- Exclusivity: a joker cannot be both Eternal and Perishable (S2, S3, S4). Rental stacks with
  either (S5: "It can stack with Eternal and Perishable").
- S2 derives "40% neither" (100 - 30 - 30) on Orange/Gold and "70% no Rental" on Gold, i.e. one
  roll decides Eternal/Perishable/neither and an independent roll decides Rental. That is exactly
  how `shop::roll_stickers` works: one uniform poll, top 30% Eternal, next 30% Perishable (from
  Orange), plus a separate 30% Rental roll (Gold).
- An incompatible joker still consumes its band: it gets nothing instead of re-rolling. So the
  Eternal rate of the whole pool is a bit below 30%. `UNVERIFIED` (the wiki gives 30% "for random
  jokers" without saying how incompatible ones are handled; this is the simplest reading and
  matches the derived "40% neither").
- Oops! All 6s does not scale these chances (not a listed interaction). `UNVERIFIED`.
- No RNG is consumed below Black Stake, so White/Red runs are bit-identical to before.

### Jokers that cannot carry a sticker
- Cannot be Eternal (S3, 11): Gros Michel, Ice Cream, Cavendish, Luchador, Turtle Bean, Diet Cola,
  Popcorn, Ramen, Seltzer, Mr. Bones, Invisible Joker. (Matches `eternal_compatible: false` in the
  joker registry; asserted by `tests/stakes.rs`.)
- Cannot be Perishable (S4, 18, the scaling jokers that are not retroactive): Ceremonial Dagger,
  Ride the Bus, Runner, Constellation, Green Joker, Red Card, Madness, Square Joker, Vampire,
  Hologram, Rocket, Obelisk, Lucky Cat, Flash Card, Spare Trousers, Castle, Glass Joker, Wee Joker.
  (`perishable_compatible: false` in the registry; the registry was checked against this list by
  the joker audit, `tests/stakes.rs` checks the roll respects the flag.) Popcorn and Turtle Bean
  CAN be Perishable (S4).
- Rental: every non-Legendary joker (S5).

### Effects
- Eternal: cannot be sold or destroyed (S2). Already enforced: `ActionError::EternalJoker`,
  destroy effects skip it (Ankh/Hex/Ceremonial Dagger/Madness).
- Perishable: after 5 rounds the joker is debuffed for good (S2, S4). The countdown ticks when a
  round is won, before the Cash Out screen, so a Golden Joker or Cloud 9 that expires on the fifth
  round pays only four times (S4); implemented by ticking before the joker end-of-round event
  (`tests/stakes.rs::perishable_golden_joker_pays_four_times`). A debuffed joker loses its
  effect and edition (existing `debuffed` handling). A Mr. Bones save also ticks it
  (`UNVERIFIED`, no source; Mr. Bones is the only way a lost round counts as finished). The debuff
  survives Crimson Heart restoring its target.
- Rental: costs $1 to buy regardless of base price, edition or discounts (S5; "the Rental Sticker
  sets the card's price to $1"); sell value $1 (S5); $3 is deducted at the end of every round (S2,
  S5). The wiki says the rent is deducted before the round payout, so it lowers the balance
  interest is computed on; the implementation charges it right before `economy::cash_out`.
  Charged for every Rental joker present when the round ends, including debuffed ones and ones
  that destroy themselves that round (S5: Popcorn, Crimson Heart, Perishable still pay). Money can
  go negative (S5: "you go into debt"). Charged on won rounds only (a lost round ends the run;
  `UNVERIFIED` for a Mr. Bones save: charged, since the run continues).
  - Free items (Coupon/Uncommon/Rare Tag) stay $0; a Rental price of $1 only replaces the paid price.
  - Voucher re-pricing (Clearance Sale/Liquidation) leaves Rental sell values alone.

## 3. Interaction with decks
Stake and deck effects stack independently: e.g. Blue Deck (+1 hand) on Blue Stake = 5 hands and
2 discards; Plasma Deck on Purple Stake = 2 x the third table.
