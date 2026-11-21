# Blinds, antes, round flow

Target: latest stable Balatro, Red Deck, White Stake. Items marked `UNVERIFIED` are not stated by any fetched page.

Sources:
- B1 https://balatrowiki.org/w/Blinds_and_Antes (also reached as /w/Blinds)
- B2 https://balatrowiki.org/w/Money
- B3 https://balatrowiki.org/w/Interest
- B4 https://balatrowiki.org/w/Tags
- B5 https://balatrowiki.org/w/Shop (sell value, reroll)
- B6 https://balatrowiki.org/w/Stakes
- B7 https://balatrowiki.org/w/Decks
- B8 https://balatrowiki.org/w/Mr._Bones
- B9 https://balatrowiki.org/w/Gold_Card (Gold card, Blue Seal at end of round)
- B10 https://balatrowiki.org/w/Gameplay
- B11 https://balatrowiki.org/w/Vouchers
- N project decisions

## 1. Run structure

- Each ante = Small Blind, Big Blind, Boss Blind, in that order (B1).
- A run starts at ante 1 (B1). Ante 0 or lower only via Hieroglyph/Petroglyph (B1, B11); base requirement there is 100 (B1).
- Winning: defeat the ante 8 boss (a Showdown blind). The run is won at that moment; play may continue in Endless Mode from ante 9 (B1).
- Mr. Bones saving the ante 8 boss still counts as a win (B8).
- Showdown (finisher) bosses appear at antes 8, 16, 24, 32 in endless (B1). Generally ante % 8 == 0.
- Boss blinds cannot be skipped; Small and Big can (B1).
- Ante advances when the Boss Blind is defeated (standard; B1 lists the order Small, Big, Boss per ante). The wiki does not spell this sentence out; `UNVERIFIED` but implied.
- Highest reachable ante without glitches: 39 (requirement overflows to infinity there) (B1). Recommended: cap the simulator at ante 39 / treat requirements as f64 and stop episodes earlier.

## 2. Chip requirement ("base") per ante

White Stake (B1):

| Ante | Base | Small x1 | Big x1.5 | Boss x2 |
|---|---|---|---|---|
| <=0 | 100 | 100 | 150 | 200 |
| 1 | 300 | 300 | 450 | 600 |
| 2 | 800 | 800 | 1,200 | 1,600 |
| 3 | 2,000 | 2,000 | 3,000 | 4,000 |
| 4 | 5,000 | 5,000 | 7,500 | 10,000 |
| 5 | 11,000 | 11,000 | 16,500 | 22,000 |
| 6 | 20,000 | 20,000 | 30,000 | 40,000 |
| 7 | 35,000 | 35,000 | 52,500 | 70,000 |
| 8 | 50,000 | 50,000 | 75,000 | 100,000 |

Other stakes (B1): Green and above use a steeper table; Purple steeper still:

| Ante | Green+ | Purple+ |
|---|---|---|
| 1 | 300 | 300 |
| 2 | 900 | 1,000 |
| 3 | 2,600 | 3,200 |
| 4 | 8,000 | 9,000 |
| 5 | 20,000 | 25,000 |
| 6 | 36,000 | 60,000 |
| 7 | 60,000 | 110,000 |
| 8 | 100,000 | 200,000 |

Blue Stake and Plasma Deck: wiki says blind size doubles (B6 for Blue; B1 for Plasma). `UNVERIFIED` conflict: B6 lists "doubles base Blind size" under Blue Stake, while B1 mentions doubling only for Plasma Deck. Out of scope for White Stake; check before implementing Blue Stake.

### Endless (ante >= 9)

Formula as printed on B1, with c = ante - 8 and A8 = 50,000:

    base(ante) = truncate_to_2_significant_digits( A8 * ( 1.6 + (0.75 * c) ^ (1 + 0.2 * c) ) ^ c )

The wiki text is flattened, so the layout above is my reading; I verified it by hand against the published values for ante 9 (115,4xx -> 110,000) and ante 10 (565,8xx -> 560,000), both match the B1 table. Rounding direction: the published figures match truncation (round down) to two significant digits; plain rounding gives the same for these two but direction is `UNVERIFIED` for other antes. Recommended: truncate, and unit-test against the table below.

Published base values (B1), White Stake:
9: 110,000; 10: 560,000; 11: 7,200,000; 12: 300,000,000; 13: 47,000,000,000; 14: 2.9e13; 15: 7.7e16; 16: 8.6e20; 17: 4.2e25; 18: 9.2e30; 19: 9.2e36; 20: 4.3e43; 21: 9.7e50; 22: 1.0e59; 23: 5.8e67; 24: 1.6e77; 25: 2.4e87; 26: 1.9e98; 27: 8.4e109; 28: 2.0e122; 29: 2.7e135; 30: 2.1e149; 31: 9.9e163; 32: 2.7e179; 33: 4.4e195; 34: 4.4e212; 35: 2.8e230; 36: 1.1e249; 37: 2.7e268; 38: 4.5e288; 39: 4.8e309 (overflow).
Values above ante 13 exceed u64/exact f64 integer range; use f64 for requirement and score in endless.

## 3. Blind multipliers and rounding

- Small x1, Big x1.5, Boss x2 of the ante base (B1).
- Exceptions: The Wall x4, Violet Vessel x6, The Needle x1 (B1). Chicot disables boss effects: Wall and Vessel return to x2; the Needle stays x1 (B1).
- Rounding of the requirement: all table values for antes 1-8 are integers with no rounding needed (x1.5 of the listed bases is always whole). For endless, round the base first (2 significant digits), then apply the multiplier; the wiki's derived table (Big at ante 9 = 165,000, Boss = 220,000) matches multiplying the rounded base (B1). Whether the game multiplies before rounding is `UNVERIFIED`; recommended: round base, then multiply.

## 4. Rewards (White Stake)

- Small $3, Big $4, Boss $5, Showdown boss $8 (B1, B2). Red Stake and above: Small pays $0 (B1, B6). Boss reward can differ by blind only for Showdown; all 23 regular bosses pay $5 (B1).
- $1 per unused hand (B2, B10). Unused discards pay nothing in the base game (B2); Green Deck pays $2 per hand and $1 per discard and gives no interest (B2).
- No blind reward if the blind was saved by Mr. Bones; interest still paid; Investment Tag and Anaglyph still apply (B2, B8).
- Cash Out order shown by the wiki (B2): 1) blind reward, 2) unused hands, 3) joker end-of-round money (Golden Joker etc.), 4) Investment Tag ($25 for boss), 5) interest. The wiki gives no finer sequencing.

### Interest

- $1 per $5 held, capped at $5 per round, i.e. no extra above $25 (B2, B3).
- Cap raised to $10 by Seed Money and $20 by Money Tree (B3, B11). To the Moon: +$1 extra per $5 held (so $2 per $5); with both vouchers the maximum is $40 per round (B3).
- Which balance counts: the wiki says Gold cards held in hand and jokers that pay before the end of the round (e.g. Reserved Parking) count; jokers paying at cash out (Golden Joker, Rocket) do not count for that round's interest (B3). That implies interest is computed on the money held BEFORE the cash-out rewards (blind reward and hand money are not part of the base), even though the Cash Out list shows interest last. The wiki never states "interest is computed before reward" literally, so `UNVERIFIED`. Recommended: compute interest from the balance at the moment the round ends (after Gold card payout and any during-round money, before blind reward, hands money, and cash-out joker payouts).
- No interest: Green Deck, The Omelette, Mad World (B2, B3). Omelette also pays no blind reward; Omelette and Mad World pay nothing for extra hands (B2). Not relevant to Red/White.

## 5. Skipping

- Small and Big may be skipped; Boss may not (B1).
- Skipping yields exactly one Tag (B4). 24 tags exist; 15 available at any ante, 9 only from ante 2 (Negative, Standard, Meteor, Buffoon, Handy, Garbage, Ethereal, Top-up, Orbital) (B4). Rare/Negative/Foil/Holographic/Polychrome Tags require discoveries (profile-dependent; for the simulator assume all unlocked or fixed pool, decide later) (B4).
- Each tag is equally likely; repeats allowed, even within one ante (B4). Tags stack, resolve oldest first, some instantly, some at the next shop (B4).
- Effect on ante progression: the skipped blind is simply not played; the next blind in the order becomes selectable (B1 order). The ante still advances only by beating the Boss. Skipping gives no reward money. Wiki does not state this sentence; `UNVERIFIED`, implied.
- When the tag for each blind is decided: the wiki does not say (B4). `UNVERIFIED`. In the game the tag offered is visible on the blind-select screen for both Small and Big, so it is generated when the ante's blind choices are set up (start of ante), not at the skip action. Recommended: generate both tags at ante start from a dedicated RNG stream so skip decisions do not affect other RNG; record this in the seed contract.
- Tag effects (B4): Investment $25 after defeating the boss; Boss rerolls the boss blind; Voucher adds 1 voucher to the next shop; Economy doubles money, capped at +$40; Speed $5 per blind skipped this run; Handy $1 per hand played this run; Garbage $1 per unused discard this run; Juggle +3 hand size next round; D6 next shop rerolls start at $0; Coupon next shop initial cards and packs free; Double copies the next tag chosen (not another Double); Orbital upgrades a poker hand by 3 levels (which hand is not stated, `UNVERIFIED`); Top-up creates up to 2 Common jokers if room. Remaining tags (Uncommon, Rare, Foil, Holographic, Polychrome, Negative, Charm, Meteor, Standard, Buffoon, Ethereal) create the corresponding item/pack; details belong to the shop task.

## 6. Round flow

Sources are thin; items marked `UNVERIFIED` are standard game behavior not stated by the wiki.

1. Blind select: choose play or (Small/Big) skip.
2. Round start: the boss effect that applies at blind start is set (see bosses.md). Deck is shuffled (all 52 cards, minus destroyed/plus added). The player draws up to hand size (8 by default) `UNVERIFIED` (not stated on B10; consistent with N).
3. During the round: play 1-5 cards (a Play uses one hand), or discard 1-5 cards (uses one discard; requires discards > 0). After each play or discard, draw back up to hand size (The Serpent changes this, see bosses.md). `UNVERIFIED` for the limits; standard.
4. Round ends the moment the chip score is >= the requirement after a hand finishes scoring `UNVERIFIED` (wiki: "defeated" then Cash Out). Remaining hands are paid.
5. Loss: hands reach 0 and score < requirement (B10, B1). Exception: Mr. Bones with score >= 25% of the requirement; it is destroyed and the blind counts as won with no reward (B8). Mr. Bones does not save you if you run out of cards (B8).
6. Deck runs out: no reshuffle mid-round. The run is lost if no cards remain to be played (B10). In practice: if the draw pile and hand are empty and hands remain, play ends; the same condition as above applies (score < requirement means loss). `UNVERIFIED` for the exact trigger (empty hand with hands left); recommended: when hand is empty and draw pile empty, end the round as a loss unless score >= requirement.
7. End of round, then Cash Out. Order known from wiki (B2, B3, B9):
   a. Held-in-hand end-of-round effects: Gold cards pay $3 each (Red Seal: $6), Blue Seal makes the Planet for the last played hand type; both only if held in hand at round end (B9). These count toward interest (B3). Joker end-of-round effects such as Egg (+$3 sell value), Gift Card (+$1 sell value to every joker and consumable) also fire at end of round (B5); order among jokers is left to right `UNVERIFIED`.
   b. Cash Out: blind reward, unused hands $1 each, joker cash-out money (Golden Joker $4, Rocket etc.; amounts belong to the joker task), Investment Tag $25 if boss, interest (B2).
   c. All cards return to the deck; the shop opens. The Voucher restocks after the boss is defeated, packs restock on each shop entry (B5).
   d. Next blind select.
8. Hand and discard counts reset each round to run values (Red Deck: 4 hands, 4 discards) plus voucher bonuses: Grabber +1 hand, Nacho Tong +1 more, Wasteful +1 discard, Recyclomancy +1 more, Hieroglyph -1 hand and ante -1, Petroglyph -1 discard and ante -1 (B11). Boss effects that change counts (The Water: 0 discards, The Needle: 1 hand, The Manacle: -1 hand size) end with the blind.

## 7. Starting state (Red Deck, White Stake)

- Red Deck effect: +1 discard every round; no other effect (B7). Wiki pages fetched do not give the generic base values (B7, B10: "not mentioned").
- Values are a project decision (matches the well known game defaults): money $4, hands 4, discards 3 + 1 (Red) = 4, hand size 8, joker slots 5, consumable slots 2, 52-card standard deck. `UNVERIFIED` by the fetched wiki pages; the decks overview does not list them. Recommended: keep these values.
- Money may go negative only with Credit Card (debt up to -$20) (B-Ox page https://balatrowiki.org/w/The_Ox).

## 8. Selling

- Sell value = floor(cost / 2), minimum $1 (B5: "recalculate the buy cost, divide by 2, round down; below $1 becomes $1"). Egg and Gift Card bonuses are added afterward (B5).
- The cost used includes edition surcharge and vouchers discounts: jokers Foil +$2, Holographic +$3, Polychrome +$5, Negative +$5; consumables Negative +$5 only (B5). Mr. Bones listed Buy $5, Sell $2 (B8) is consistent with floor(5/2).
- Egg: +$3 of sell value at the end of each round (B5). Gift Card: +$1 sell value to every joker and consumable at end of round; applied after all other steps and not affected by discount vouchers (B5). Both are flat additions to the stored sell value; they persist.
- Whether the cost in the formula uses the discounted shop price (Clearance Sale) or the base cost: B5 says Gift Card is not affected by discount vouchers; for the base formula it says "recalculate the buy cost", which suggests the discount is included. `UNVERIFIED`. Recommended: use the cost the item would have at purchase with current discounts removed at time of purchase, i.e. store sell_value at purchase time; revisit in the shop task.
- Rental jokers sell for $1 `UNVERIFIED` (standard game, outside White Stake).
