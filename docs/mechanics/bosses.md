# Boss blinds

Sources:
- S1 https://balatrowiki.org/w/Blinds_and_Antes (main table: effect, min ante, multiplier, reward)
- S2 https://balatrowiki.org/w/The_Ox
- S3 https://balatrowiki.org/w/The_Hook
- S4 https://balatrowiki.org/w/Verdant_Leaf
- S5 https://balatrowiki.org/w/Cerulean_Bell
- S6 https://balatrowiki.org/w/Crimson_Heart
- S7 https://balatrowiki.org/w/The_Serpent
- S8 https://balatrowiki.org/w/The_Wheel
- S9 https://balatrowiki.org/w/Blinds (same data as S1)
- Others: scoring.md (restriction handling, Flint), `UNVERIFIED` means not on any fetched page.

Convention: "Min ante" Any = ante 1. Reward is $5 for regular, $8 for Showdown. Score = multiplier of the ante base (blinds.md).
Debuffed cards (per cards.md/scoring.md) stay in hand-type detection and add nothing when scored.
A hand that violates a play restriction can still be played but scores nothing and triggers no other effects (S1, scoring.md step 0).

## Regular bosses (23)

| Boss | Effect (S1) | Min ante | Mult | Hook point | State needed |
|---|---|---|---|---|---|
| The Hook | after each played hand, 2 random cards held in hand are discarded | Any | 2 | after play | rng |
| The Ox | playing the (pre-chosen) most-played hand sets money to $0 | 6 | 2 | on play (scoring) | hand play counts, chosen hand |
| The House | first hand drawn face down | 2 | 2 | on draw (first draw) | face_down flag per card |
| The Wall | very large blind | 2 | 4 | on blind select (requirement) | none |
| The Wheel | each drawn card has 1 in 7 chance of being face down | 2 | 2 | on draw | rng, face_down |
| The Arm | playing a hand lowers its level by 1 | 2 | 2 | on play | hand levels |
| The Club | Club cards debuffed | Any | 2 | blind start / suit change | debuff recompute |
| The Fish | cards drawn face down after each hand played | 2 | 2 | on draw after a play | face_down |
| The Psychic | must play exactly 5 cards | Any | 2 | on play validation | none |
| The Goad | Spade cards debuffed | Any | 2 | blind start | debuff |
| The Water | start with 0 discards | 2 | 2 | blind start | discards |
| The Window | Diamond cards debuffed | Any | 2 | blind start | debuff |
| The Manacle | hand size -1 | Any | 2 | blind start | hand size (restored at end) |
| The Eye | no repeat hand types this round | 3 | 2 | on play validation | set of hand types played this round |
| The Mouth | only one hand type may be played this round | 2 | 2 | on play validation | first hand type this round |
| The Plant | face cards debuffed | 4 | 2 | blind start | debuff |
| The Serpent | after Play or Discard, always draw 3 cards | 5 | 2 | on draw | none |
| The Pillar | cards played earlier this ante are debuffed | Any | 2 | on play (mark), debuff recompute | per-card "played this ante" flag |
| The Needle | only 1 hand may be played | 2 | 1 | blind start | hands = 1 |
| The Head | Heart cards debuffed | Any | 2 | blind start | debuff |
| The Tooth | lose $1 per card played | 3 | 2 | on play | money |
| The Flint | base chips and mult halved | 2 | 2 | on scoring (step 1) | none |
| The Mark | face cards drawn face down | 2 | 2 | on draw | face_down |

Source for the whole table: S1 (all Matador columns omitted; Matador is a joker task item: it pays $8 when the boss effect triggers; for The Ox the boss ability resolves first, then Matador pays, S2).

Ante 1 pool (min ante Any): The Hook, The Club, The Psychic, The Goad, The Window, The Manacle, The Pillar, The Head (S1 says "eight bosses").

## Showdown / finisher bosses (5), min ante 8, reward $8 (S1)

| Boss | Effect | Mult | Hook point | State |
|---|---|---|---|---|
| Amber Acorn | flips and shuffles all jokers | 2 | blind start (jokers face down and shuffled) | joker order, face_down flag on jokers |
| Verdant Leaf | all cards debuffed until one joker is sold | 2 | blind start; cleared on joker sale | debuff flag |
| Violet Vessel | very large blind | 6 | requirement | none |
| Crimson Heart | one random joker disabled every hand | 2 | on play | rng, disabled joker |
| Cerulean Bell | forces 1 card to always be selected | 2 | on draw / after play and discard | forced card id |

## Implementation notes and edge cases

- General: a disabled boss (Chicot, or selling Luchador) removes its effect; Wall and Vessel become x2; Needle stays x1 (S1). Luchador disables only the current boss; Chicot all bosses (S1). Selling Luchador during Verdant Leaf disables the blind before Leaf disables itself (S4).
- The Hook (S3): discards 2 random held cards after each play; if only one is held it is discarded; for zero held nothing happens (inferred, S3 hints at it). Timing vs refill: `UNVERIFIED` (wiki does not say). Recommended: discard from held cards after the play is scored, then refill to hand size. These discards trigger discard effects (Purple Seal, Mail-In Rebate, Hit the Road, Castle) but do not count as discards used for Delayed Gratification (S3).
- The Ox (S2): the "most played hand" is determined when the previous boss is defeated, i.e. at the start of the ante, and is not updated during Small or Big; reroll into Ox also uses start-of-ante data. Ties: random choice (flagged [verify] by the wiki, `UNVERIFIED`). Triggers on the scoring hand type (a Straight Flush does not trigger on Flush). Sets money to exactly $0, which can raise negative money (Credit Card).
- The Arm: level decreases by 1 when a hand is played. Minimum level 1 and whether the played hand scores at the lowered level are `UNVERIFIED` on the wiki. Recommended: lower first (floor level 1), then score at the new level, and never go below 1.
- The House: only the first hand drawn is face down (S1). Face down cards are hidden from the agent's observation but can be played normally and score normally (nothing in S1 says otherwise; `UNVERIFIED`). Recommended: observation shows face-down cards as unknown rank/suit; rules otherwise unchanged. All cards are revealed when the blind ends.
- The Wheel (S8): independent 1/7 roll per drawn card. Roll at draw time is `UNVERIFIED`; recommended. Uses the shared chance() probability? Oops! All 6s interaction is `UNVERIFIED`; recommended: it does not apply (the wiki does not list Wheel among affected effects). Mark this for the joker task.
- The Fish: cards drawn after each hand played are face down (not the opening draw), S1. After discards `UNVERIFIED`; recommended: only after plays, as worded.
- The Mark: face cards are face down when drawn (S1).
- The Psychic: invalid unless exactly 5 cards played (scoring.md). The hand uses a hand but scores 0.
- The Eye / The Mouth: a hand violating the rule scores 0. Eye: any previously played hand type this round is forbidden. Mouth: after the first hand, only that type is allowed. Repeats are checked against the scoring hand type (`UNVERIFIED`; recommended: hand_type, not contains).
- The Manacle: hand size -1 for the round; the extra card is not discarded, and hand size is restored after the round (`UNVERIFIED`, standard behaviour).
- The Water: 0 discards at start of round. The Needle: only 1 hand (hands set to 1).
- The Tooth: lose $1 per card played, applied when the hand is played; whether it applies before scoring (so Mr. Bones/gold cannot offset) and whether money may go negative are `UNVERIFIED`. Recommended: deduct $1 per played card at play time before scoring; money may go negative only when a debt-allowing joker (Credit Card) is present, otherwise clamp at $0.
- The Serpent (S7): after a Play or Discard, always draw 3 cards, regardless of how many were played/discarded. Effect on the first draw of the round and hand-size overflow is `UNVERIFIED`; recommended: initial draw to hand size as usual, then after each play or discard draw exactly 3 (hand may exceed or fall below hand size; never exceed deck).
- The Pillar: cards that were played earlier in this ante (including in the Small and Big blinds of the same ante) are debuffed. Track a per-card flag cleared when the ante changes. Applies only while the Pillar is active; card identity persists across rounds.
- The Plant: face cards (J, Q, K) debuffed, Pareidolia makes all cards faces (joker task).
- Suit debuffs (Club, Goad, Window, Head): use effective suit (Wild, Smeared; Wild is debuffed by all four suit bosses `UNVERIFIED`; recommended).
- Amber Acorn: jokers are flipped face down and shuffled at blind start. Hidden to agent; order really changes. Effect on Blueprint/Brainstorm: positions change.
- Verdant Leaf (S4): all cards debuffed until a joker is sold; selling any joker ends it. Not required to sell.
- Crimson Heart (S6): picks one random joker each hand, never the same as previous unless it is the only joker; previous joker restored when the next hand is played; perishable-debuffed jokers ineligible; the debuff stays through the winning hand until the next shop, so end-of-round jokers can be disabled. Swap point relative to scoring is `UNVERIFIED`; recommended: choose the disabled joker before scoring of each hand begins (first hand included).
- Cerulean Bell (S5): a random card in hand is forced to be selected; if no force-selected card remains after a play or discard, pick a new random one. Discarding it is allowed. Initial pick time `UNVERIFIED`; recommended: after the opening draw. Action masks must always include the forced card.
- The Flint: halve base chips and base mult, round up (notes.md §4.1).
- The Wall/Violet Vessel: requirement multiplier only (x4, x6).

## Selection rules

- Each boss has a minimum ante (above). Regular bosses are used for antes not divisible by 8; Showdown bosses at antes 8, 16, 24, 32 (S1: rerolling at any multiple of 8 gives a Showdown, other antes a regular boss).
- Eligible set = bosses whose min ante <= current ante, from the right pool (regular or showdown). Among these only the ones with the fewest appearances so far are selectable; a boss does not repeat until all others (in eligible pool) have appeared, by defeat or by being rerolled (S1). Choice among the tied candidates is random (`UNVERIFIED` as to the RNG; recommended: uniform from the tied list, in a boss-specific RNG stream).
- Appearance counts persist across antes; when exhausted the game resets counts. Whether counts are kept per pool (regular vs showdown) and whether a boss newly eligible later is treated as 0 appearances are `UNVERIFIED`; recommended: a single counter map, regular and showdown pools filtered by ante, reset-by-min logic is automatic.
- Rerolling (noted only): Director's Cut rerolls once per ante for $10; Retcon unlimited at $10 each (unlock: discovering 25 blinds); Boss Tag also rerolls; rerolled bosses count as appeared (S1, https://balatrowiki.org/w/Tags).
- The boss is decided when the ante's blinds are set up (shown at blind select) `UNVERIFIED`; recommended: decide at ante start.
