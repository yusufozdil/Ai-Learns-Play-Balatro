# Poker hands: definitions, scoring cards, detection rules

Target: latest stable Balatro (1.0.x). Wiki pages do not state a build number; treat as 1.0.1-era. Items marked `UNVERIFIED` were not confirmed by a fetched wiki page.

Sources (S-codes used below):
- S1 https://balatrowiki.org/w/Poker_Hands
- S2 https://balatrowiki.org/w/Four_Fingers
- S3 https://balatrowiki.org/w/Stone_Card
- S4 https://balatrowiki.org/w/Wild_Card
- S5 https://balatrowiki.org/w/Debuffed
- S6 https://balatrowiki.org/w/Jokers (Shortcut, Splash, Smeared, Four Fingers entries)
- S7 https://balatrowiki.org/w/Smeared_Joker
- S8 https://balatrowiki.org/w/Card_Ranks

## 1. The 12 hands (level 1 values, per-level increments)

Source: S1. Matches the prd.md Faz 1 draft table in every cell (no differences).

| Hand | Chips | Mult | +Chips/lvl | +Mult/lvl | Definition (5 cards unless noted) | Cards that score |
|---|---|---|---|---|---|---|
| High Card | 5 | 1 | 10 | 1 | Nothing else matches | Only the single highest-rank card |
| Pair | 10 | 2 | 15 | 1 | 2 cards same rank | the 2 paired cards |
| Two Pair | 20 | 2 | 20 | 1 | 2 different ranks each paired | the 4 cards |
| Three of a Kind | 30 | 3 | 20 | 2 | 3 cards same rank | the 3 cards |
| Straight | 30 | 4 | 30 | 3 | 5 consecutive ranks | all 5 |
| Flush | 35 | 4 | 15 | 2 | 5 cards same suit | all 5 |
| Full House | 40 | 4 | 25 | 2 | 3 of one rank + 2 of another | all 5 |
| Four of a Kind | 60 | 7 | 30 | 3 | 4 cards same rank | the 4 cards |
| Straight Flush | 100 | 8 | 40 | 4 | Straight and Flush | all 5 |
| Five of a Kind (secret) | 120 | 12 | 35 | 3 | 5 cards same rank | all 5 |
| Flush House (secret) | 140 | 14 | 40 | 4 | Full House where all 5 share a suit | all 5 |
| Flush Five (secret) | 160 | 16 | 50 | 3 | 5 cards same rank AND suit | all 5 |

- Level value: `chips = base + (level-1)*inc_chips`, `mult = base + (level-1)*inc_mult` (S1; level 1 = base). Planets: Pluto, Mercury, Venus, Earth, Mars, Jupiter, Saturn, Neptune, Uranus, Ceres (Flush House), Planet X (Five of a Kind), Eris (Flush Five) (S1; not needed by the simulator for Faz 1).
- Secret hands (Five of a Kind, Flush House, Flush Five) are hidden in Run Info until played once; their Planets only appear after that in a run (S1). This is a UI/shop concern, the evaluator always detects them.
- Royal Flush is only a display name; it is a Straight Flush for all purposes (levels, jokers) (S1). Do not model as a hand type.
- Hands may have fewer than 5 cards (1 to 5 played). Only hands needing 5 cards (Straight, Flush, Full House, Straight Flush, Five-based) require 5 played cards, except with Four Fingers (section 3).

## 2. Priority (which hand type is chosen)

Highest tier wins, regardless of level (S1; example: 9-9-9-9-2 all one suit = Four of a Kind, not Flush). Evaluate in this order and return the first match:

1. Flush Five
2. Flush House
3. Five of a Kind
4. Straight Flush
5. Four of a Kind
6. Full House
7. Flush
8. Straight
9. Three of a Kind
10. Two Pair
11. Pair
12. High Card

Notes:
- Order 1-3: Flush Five and Flush House exceed Five of a Kind (S1 lists them in this order). A hand can never be both Flush House and Five of a Kind, so the relative order of 2 vs 3 is unobservable. Order of 4 vs 5 and 6-8 follows the wiki table order, which is by base value. UNVERIFIED that the game's internal list uses exactly this order, but it is the only order consistent with the S1 example and table.
- Wild cards "result in the highest tier hand type possible" (S4: Wild + Wild Ace of Diamonds + three Ace of Spades = Flush Five, not Five of a Kind).
- Straight Flush vs Flush House: a 5-card hand cannot be both without Four Fingers; with Four Fingers see section 4.

## 3. "Contains" relation (for jokers like Jolly, Zany, Mad, Crazy, Droll, The Duo/Trio/Family/Order/Tribe)

A hand "contains" a sub-pattern if the played cards include it, even when a higher hand type was chosen. Jokers conditioned on "contains X" fire when the played hand type is any row marked below. Derived logic; the Jokers page (S6) confirms "contains" semantics exist but gives no table, so the table is `UNVERIFIED` as a whole, with the specific items below being widely reported:

| Hand played | Contains Pair | Contains Two Pair | Contains Three | Contains Four | Contains Straight | Contains Flush |
|---|---|---|---|---|---|---|
| High Card | no | no | no | no | no | no |
| Pair | yes | no | no | no | no | no |
| Two Pair | yes | yes | no | no | no | no |
| Three of a Kind | yes | no | yes | no | no | no |
| Straight | no (unless duplicates with Four Fingers; compute from cards) | | | | yes | no |
| Flush | compute from cards (a flush may contain pairs) | | | | no | yes |
| Full House | yes | yes | yes | no | no | no |
| Four of a Kind | yes | no (UNVERIFIED) | yes | yes | no | no |
| Straight Flush | compute | | | | yes | yes |
| Five of a Kind | yes | no (UNVERIFIED) | yes | yes | no | no |
| Flush House | yes | yes | yes | no | no | yes |
| Flush Five | yes | no (UNVERIFIED) | yes | yes | no | yes |

Recommended implementation: do not hardcode the table. Compute the `contains` bitset from the played cards themselves:
- Pair: some rank appears >= 2 times. Three: >= 3. Four: >= 4.
- Two Pair: at least two distinct ranks each appearing >= 2 times. (Therefore Four of a Kind of one rank alone does not contain Two Pair; a Full House does.) Whether the game treats 4-of-a-kind as containing Two Pair is `UNVERIFIED`; recommended: no (needs two distinct ranks).
- Full House (as a "contained" pattern, if needed): rank with >=3 and another rank with >=2.
- Straight / Flush: result of the straight/flush detectors (section 4).
- Stone cards contribute nothing to any of these (section 5). Debuffed cards still contribute (section 7).

## 4. Straight, Flush, and joker modifiers

### 4.1 Straight
- 5 cards with consecutive ranks. Ace is high (...Q-K-A) or low (A-2-3-4-5), but not both in one straight; no wrap-around, so K-A-2-3-4 and Q-K-A-2-3 are invalid (S1: "Aces can be high or low, but not both in one straight"; S8: Ace "always score[s] as high" for chips; chip value stays 11).
- Duplicated ranks break a plain 5-card straight (it needs 5 distinct consecutive ranks). With Four Fingers duplicates are allowed alongside a 4-run (S2: 2,3,4,5,5 scores as a Straight with all five cards scoring; 10,9,9,8,7 also).
- Shortcut: a straight may skip one rank between adjacent members (S1/S2/S6; example 10-8-6-5-3). Precisely: sorted distinct ranks, each adjacent difference is 1 or 2. Ace-low use with Shortcut (A-3-5-... ) is `UNVERIFIED` (recommend: allow, treat Ace as rank 1 for the low interpretation only). Gap size > 1 missing rank is never allowed.
- Four Fingers: only 4 cards are required to form a straight (S2/S6). Five cards still work. Multiple copies do not stack, and it is not Blueprint-copyable (S2).

### 4.2 Flush
- All counted cards share a suit. Wild cards count as every suit simultaneously (S4). Stone cards have no suit and never count. Debuffed Wild cards revert to their base suit (S4, S5).
- Four Fingers: 4 cards of one suit suffice (S2). The remaining card (if 5 were played) does not score, except in combined hands (4.4).
- Smeared Joker: Hearts and Diamonds are one suit; Spades and Clubs are one suit (S6, S7). Wild still matches all. Interaction of Smeared with Four Fingers is not covered by S7 but follows from composition (`UNVERIFIED`, recommended: just apply the merged-suit rule inside the 4-card flush test).

### 4.3 Splash
- Every played card scores (S6). Stone cards already always score (S3). Splash does not change hand type or `contains`; it only changes the scoring set to all played cards.

### 4.4 Combined hands with Four Fingers (and general rule)
- Straight Flush = flush detected AND straight detected, not necessarily on the same 4 cards. Wiki example: 2♠ 3♠ 4♦ 5♠ 10♠ is a Straight Flush with Four Fingers (2-3-4-5 straight, 2♠3♠5♠10♠ flush) (S2). Recommended scoring set = union of the straight cards and the flush cards (all 5 in the example). `UNVERIFIED` on scoring set.
- Flush Five with Four Fingers: needs 5 same-rank cards, only 4 need to match suit (S2).
- Flush House with Four Fingers: Full House (always 5 cards) with a 4-card flush (S2: "only four cards need to share a suit").
- Four Fingers does not reduce the card count of Full House, Five of a Kind (S2: "5th card still matters").

## 5. Which played cards score (scoring set)

Default rule (S1): only the cards that form the hand type score; unmatched "kicker" cards are played but do not score.
- High Card: only the highest-rank card (Ace high). Tie among equal ranks: which card scores is `UNVERIFIED` (recommend leftmost of the highest rank; only matters for suit/enhancement effects).
- Pair/Two Pair/Three/Four: only the cards of the matching ranks.
- Straight/Flush/Full House/Straight Flush/Five of a Kind/Flush House/Flush Five: all 5 (S1). With Four Fingers, a 4-card Straight or Flush scores the cards in the 4-card pattern; for straights containing duplicate ranks inside the run, wiki says all five score (S2). Recommended: scoring set = every played card whose rank belongs to the run (so a fifth card unrelated to the run, like 2-3-4-5-K, does not score). `UNVERIFIED`. For a 4-card flush with a fifth off-suit card, the off-suit card does not score (`UNVERIFIED`, derived).
- Stone cards: always in the scoring set, whatever the hand type (S3). They never add to hand type or contains. Under Four Fingers a Stone card can fill the 5th slot of a 4-card Straight/Flush and score (S3 "Marble Joker" note in S2).
- Splash: every played card scores (S6).
- Debuffed cards stay in the scoring set (they count for hand type) but contribute nothing (section 7). Also joker "when scored" effects skip debuffed cards (S6).

Scoring order inside the set: left to right as in the player's hand order (prd.md 3.1; scoring.md).

## 6. Stone cards and hand detection
- No rank, no suit (S3). Excluded from pair/straight/flush detection; excluded from Pareidolia-independent rank/suit jokers.
- A Stone card is ignored when typing the hand: detect the hand type on the non-Stone cards only. If every played card is Stone, the hand type is High Card and all Stone cards score (S3: always score). The all-Stone edge case is `UNVERIFIED`; in a mixed play where Stone cards are the only ones left after the matched cards, they score as extras. High Card with Stone cards present: the highest non-Stone card scores plus all Stones (`UNVERIFIED`).
- A debuffed Stone still scores as part of the hand and keeps no rank/suit (S3, S5).
- If Vampire strips the Stone enhancement mid-hand, the card loses +50 and no longer counts as "stone for hand conditions" but still scores (S3). Edge case; defer to the joker implementer.

## 7. Debuffed cards and hand detection
- A debuffed card keeps its base rank and suit and still counts toward hand type and `contains`, even if all played cards are debuffed (S5).
- Wild debuffed: reverts to its base suit (S4, S5). Stone debuffed: no rank/suit (S5).
- A debuffed card in the scoring set contributes no chips, mult, enhancement, edition or seal effects (S5), and gives no perma/Hiker bonus (S5). Money from Gold Seal = $0 (S4/Gold Seal page https://balatrowiki.org/w/Gold_Seal).
- Boss "hand restriction" blinds (Psychic/Eye/Mouth): hands not meeting the restriction can be played but score nothing and fire no effects (https://balatrowiki.org/w/Blinds_and_Antes).
