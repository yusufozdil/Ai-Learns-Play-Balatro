# Scoring order (one played hand)

Target: latest stable Balatro. Main source: the wiki's Activation Sequence guide. Items marked `UNVERIFIED` are not in any fetched page.

Sources:
- A1 https://balatrowiki.org/w/Activation_Sequence (order of events)
- A2 https://balatrowiki.org/w/Jokers (retrigger and timing notes)
- A3 https://balatrowiki.org/w/Card_Modifiers (enhancements, seals, editions, retrigger exceptions)
- A4 https://balatrowiki.org/w/Glass_Card
- A5 https://balatrowiki.org/w/Gold_Seal
- A6 https://balatrowiki.org/w/The_Flint
- A7 https://balatrowiki.org/w/Debuffed
- A8 https://balatrowiki.org/w/Stone_Card
- A9 https://balatrowiki.org/w/Blinds_and_Antes

## Algorithm

Inputs: played cards in hand order (left to right), held cards (not played) in hand order, jokers left to right, hand levels.

0. Validate the played hand. If a boss restriction (Psychic needs 5 cards, Eye no repeat hand type, Mouth single hand type) is violated, the hand scores 0 and fires no scoring effects (A9). Hand detection runs first (hands.md): hand type, `contains` set, scoring set (Stone always in, Splash = all).
1. Boss effects on base (A1 step 1). The Flint halves base chips and base mult of the hand; rounded UP (halves rounded up per A6, e.g. 5 chips x 1 mult becomes 3 x 1). Rounding rule is `UNVERIFIED`: the page states round up, and no minimum is stated. Recommended: `ceil(x/2)`, and mult never below 1 (ceil already gives >=1 for >=1 base).
2. "On played" joker effects, left to right, before any scoring (A1 step 2): examples Space Joker (1 in 4: level up the played hand type), Midas Mask (played face cards become Gold; the wiki says "when scored"), Vampire (strip enhancements), Ride the Bus style counters. Whether a level-up from Space Joker is applied BEFORE the base chips/mult are read (so the level-up benefits the current hand) is `UNVERIFIED` on the wiki; commonly reported: yes, the new level is used. Recommended: yes. Order of Midas Mask / Vampire vs card chips: they act in the on-played phase, so the card is already Gold/stripped when it scores (`UNVERIFIED` for Midas, wiki card page A3 says "when scored"; recommended: apply before the per-card loop, consistent with A1's placement). Treat this as one category: `before_scoring` hooks.
3. Read base chips and mult from the hand's CURRENT level (after step 2 level-ups and step 1 Flint). Set `chips`, `mult`.
4. For each card in the scoring set, left to right (A1 step 3). One "trigger" does, in order:
   a. Base chips: rank chip value (cards.md), or Stone: no rank chip (Stone gives +50 as enhancement), plus perma bonus chips (Hiker) (A8: Hiker adds directly to that total). Bonus enhancement +30 chips is in the enhancement step of the card modifiers (A1 groups "base chips, including bonus chips"; the ordering between them does not matter, both are additive).
   b. Enhancement: Mult +4; Glass x2; Lucky (1/5: +20 mult, 1/15: +$20, independent rolls); Bonus +30 chips; Stone +50 chips. Wild/Steel/Gold do nothing here.
   c. Seal: Gold Seal pays $3 now (A5). Other seals do nothing in this step.
   d. Edition: Foil +50 chips, Holographic +10 mult, Polychrome x1.5 mult.
   e. "On card scored" jokers, left to right (Scholar, Walkie Talkie, Greedy-family, etc.). Hiker's permanent +5 gain also fires here (it changes the card's perma bonus and the bonus applies on the same trigger per the wiki phrasing `UNVERIFIED`; recommended: increment first, then add the updated chips, since Hiker is described as chips "added directly").
   f. Retriggers: re-run steps a-e for the same card. Red Seal retrigger first, then retriggering jokers left to right (Hack, Sock and Buskin, Dusk, Seltzer, Hanging Chad). Multiple retriggers stack additively: total triggers = 1 + sum of retrigger counts (A1 step 3.4, A3 example: a Red Seal 2-5 card with Hack triggers 3 times).
   - Which parts repeat: everything in a-e, including the on-scored joker effects, the card's own chips, enhancement, edition, and Gold Seal money (A2, A5: "each retrigger adds another $3"). Red Seal itself is not "retriggered by jokers" (A3: it is the source of one retrigger, it does not multiply).
   - Not retriggered / no re-roll: the Glass break roll happens once after scoring, not per trigger (A4). Whether Lucky re-rolls on each retrigger: `UNVERIFIED`, recommended: yes, independent roll per trigger.
   - Debuffed card in the scoring set: skip a through f entirely; joker "on scored" effects also skip it (A2, A7).
   - Hanging Chad: retriggers only the first scored card, 2 extra times (A2).
5. Held-in-hand phase (A1 step 4): for each card in the hand that was not played, left to right:
   a. Enhancement: Steel x1.5 mult (only in-hand modifier).
   b. "On held" jokers, left to right (Baron x1.5 per King held, Shoot the Moon +13 per Queen, Raised Fist +2x lowest rank mult, Blackboard-type conditions are evaluated against the held set).
   c. Retriggers: Red Seal first (A1: Red Seal applies to held cards too), then Mime (and Blueprint/Brainstorm copies of it), left to right; additive. Everything in a-b repeats. Does not repeat edition or base chips (those only trigger when scored) (A2).
   - Debuffed held cards: no enhancement/seal/edition held effects; jokers tied to specific held cards generally do not trigger; hand-type or held-set based jokers (Blackboard) still work (A7).
6. Joker phase, left to right over all jokers (A1 step 5). For each joker J in order:
   a. J's edition Foil (+50 chips) or Holographic (+10 mult) is applied first.
   b. J's own effect (independent jokers: flat +chips, +mult, xmult, hand-type conditionals). Not affected by card retriggers. Blueprint/Brainstorm run the copied joker's effect at their own position (the Brainstorm/Blueprint slot, not the original's slot) (A2).
   c. Joker-on-joker effects (currently only Baseball Card): each uncommon joker is multiplied X1.5 by Baseball Card, applied at that uncommon joker's turn, after its own effect (A1 step 5.3). Debuffed jokers are still counted by Baseball (A2).
   d. J's edition Polychrome (x1.5 mult) last (A1 step 5.4).
   - Debuffed joker: no effect and no edition effects (A2; edition effects for debuffed jokers `UNVERIFIED`, recommended: none).
   - Consequence: all additions made by a joker to the right happen later; ordering is significant because +mult must precede x mult to benefit.
7. Consumable-like final modifiers: Observatory voucher gives x1.5 mult per matching Planet card held in the consumable slots, applied left to right after jokers (A1 step 6). Plasma Deck: chips and mult are balanced (averaged) after that (A1 step 7). Both are late-game; out of scope for Faz 1.
8. Score: `hand_score = chips * mult`. Rounding: the wiki sources do not describe any rounding of fractional values (A1 silent; search found none). `UNVERIFIED`. Recommendation: keep `chips` and `mult` as f64 with no intermediate rounding at any step (x1.5 on odd values yields fractions), and set `hand_score = floor(chips * mult)` once, then add to the round score. Flint is the only case where the wiki states explicit rounding (round up). Provide a single `round_score` function so it can be changed in one place if real-game tests disagree.
9. After scoring (wiki A1 does not give the order; see notes). Recommended order, `UNVERIFIED`:
   a. Add `hand_score` to the round score; decrement hands left.
   b. Glass cards in the scoring set (non-debuffed): each rolls once, 1 in 4 to be destroyed (A4); retriggers do not add rolls. Debuffed Glass cards cannot break (A4). Oops! All 6s doubles to 2 in 4 (A4).
   c. "After hand played" joker hooks (counters, scaling such as Glass Joker gaining from destroyed Glass cards (A4), Ice Cream decay, Vampire gains).
   d. Held-in-hand end-of-round effects are NOT part of this: Gold card $3 is paid at end of round for Gold cards still in hand (A5), not when played. Gold Seal $3 is paid immediately in step 4c on each trigger. Blue Seal (held at end of round) makes a Planet; Purple Seal creates a Tarot when discarded (A3).
   e. Discard and draw back up to hand size.

## Notes on events that are not "per hand"

- Discard: no scoring; seals/jokers react on discard (Purple Seal Tarot; Burnt Joker; debuffed cards' discard effects are nullified, A7).
- End of round: Gold cards in hand pay $3 each; Blue Seal cards in hand create Planet for last-played hand type (A3, A5).
- Probabilities (Lucky, Glass) use the shared `chance()` function so Oops! All 6s applies (prd.md Faz 1 item 3).

## Differences versus prd.md 3.2 draft

1. Card step order: the wiki lists enhancement, then SEAL (Gold Seal), then edition, then on-scored jokers (A1). The draft lists enhancement, edition, then Gold Seal. Only the money timing differs; no score impact. Use the wiki order.
2. Joker phase: the wiki puts Baseball Card's x1.5 BEFORE the target joker's Polychrome, and "independent jokers" sit between Foil/Holo and Polychrome. The draft put Baseball Card after Polychrome. Use the wiki order (6a-6d). Numerically it affects only the order of x multipliers (commutative), so final results are equal unless an additive effect sits in between; still implement the wiki order.
3. Held phase: wiki says Red Seal retriggers held cards (Red Seal first, then Mime). Draft mentions Red Seal for held cards too: consistent.
4. Draft omits Observatory (x1.5 per Planet) and Plasma Deck balancing: add as late steps 7.
5. Flint: the draft says "halves the base". Wiki says rounds up. 
6. Draft step 8 says `floor(chips*mult)`. Wiki is silent. Kept as recommendation, `UNVERIFIED`.
7. Draft lists "Space Joker" as a before-scoring hook: consistent with the wiki's "on played" placement (A1 step 2); it must run before reading base values (draft has step 3 before step 4: consistent).
