# Loss analysis of the reference policy (Phase 10E, 2026-11-15)

Question: where and why does `runs/e2e_long_fix/latest.pt` (raw deterministic: 7.51 ante / 68.6% win on the 1000-game test set,
Red Deck / White Stake) lose, and which of the candidate changes (a) reward beyond ante 8 / score margin, (b) higher stakes,
(c) shop-side features (fair joker score preview), (d) bigger model / auxiliary losses, (e) decision-time search does the
evidence support?

## 0. Data and tools

* Recording: `scripts/audit_agent.py --mode det --checkpoint runs/e2e_long_fix/latest.pt` (CPU inference; reproduces 7.51 / 68.5%, one
  game differs from the GPU run) and `--mode teacher`; 1000 test games each, every action replayed in the Rust `Game` with full
  context. Pickles: `runs/analysis/ref_det.pkl`, `runs/analysis/teacher.pkl` (git-ignored). `audit_agent.py` got additive event fields
  (blind, target, boss, hands/discards left, round score, joker list, hand cards, shop leftovers at `EndShop`) and a larger final-state
  summary; nothing existing changed.
* Analysis: `scripts/analyze_losses.py` (`report` = all tables below as markdown, `narrate <pkl> <game> --rounds K` = step-by-step log,
  `losses --max-ante A`). Raw tables: `runs/analysis/report_all.md`, numbers: `runs/analysis/stats.json`.
* Ablation: `runs/analysis/ablate_vouchers.py` (evaluation-time voucher ban, section 3).
* Definitions. *Death round* = the last round of a lost game. `ratio` = score / target of that round. A round "not beaten but run
  continues" is a **Mr. Bones save** (the joker saves at >= 25% of the target), not a death. **The `ante` label is not monotonic**: Hieroglyph
  and Petroglyph (-1 ante each) are bought in 38% of the model's games, so ante 7/8 rounds can be played twice and a showdown boss
  shows up at "ante 7" in 44 games (all 44 own Hieroglyph or Petroglyph). xMult/scaling joker sets in `analyze_losses.py` are
  hand-written lists (xMult includes conditional ones such as Acrobat, Card Sharp, The Duo).
* All win-rate-by-feature tables are **observational** (survivor and selection bias: winners live longer); the one causal check we ran is in section 3.

## 1. Where the model dies

Deaths / rounds entered, by ante and blind (315 losses; ante histogram 1:8 2:14 3:9 4:10 5:26 6:49 7:87 8:797):

| ante | Small | Big | Boss |
|---|---|---|---|
| 1 | 0/1107 | 1/1110 | 7/1060 (0.7%) |
| 2 | 1/1017 | 5/1034 | 8/1055 (0.8%) |
| 3 | 4/1029 | 0/1046 | 5/1047 (0.5%) |
| 4 | 0/1013 | 3/1023 | 7/1020 (0.7%) |
| 5 | 1/1018 | 4/1020 | 21/1016 (2.1%) |
| 6 | 1/979 | 7/982 | 41/975 (4.2%) |
| 7 | 1/928 | 3/928 | 83/926 (9.0%) |
| 8 | 3/800 | 3/795 | 106/791 (13.4%) |

* 88% of the losses (278/315) are on **boss blinds**; 87% (274) are at ante >= 5; ante 7-8 bosses alone are 189 (60%). Small/Big blinds
  are 37 deaths (12%), 15 of them with the ratio >= 0.9.
* Teacher for comparison: boss death rates 2.5% / 3.5% / 3.5% / 9% / 21% / 25% / 34% / 35% (ante 1..8), Big blind at ante 6-7 8-10%.
  The model's gain over the teacher is a roughly **3x lower boss death rate at every ante >= 4**.
* The model's deaths are *late and sudden*: only 41 losses (4.1%) happen in ante 1-4.

### Which bosses (death rate = deaths / encounters; Mr. Bones saves listed separately)

| boss | enc | deaths | MrBones saves | death rate | teacher rate |
|---|---|---|---|---|---|
| **Violet Vessel** (x6 base = 300k at ante 8) | 126 | 52 | 22 | **41.3%** | 65.4% |
| **The Eye** (no repeated hand type) | 228 | 38 | 9 | **16.7%** | 12.7% |
| Crimson Heart | 183 | 26 | 14 | 14.2% | 26.4% |
| Amber Acorn | 166 | 15 | 10 | 9.0% | 15.7% |
| The Needle (1 hand) | 268 | 24 | 7 | 9.0% | 16.2% |
| Verdant Leaf | 177 | 15 | 1 | 8.5% | 38.2% |
| The Wall (2x target) | 284 | 15 | 14 | 5.3% | 23.8% |
| The Water (0 discards) | 305 | 14 | 3 | 4.6% | 14.4% |
| Cerulean Bell | 183 | 6 | 2 | 3.3% | 20.0% |
| 19 other bosses | ~5970 | 73 | | 0.0-3.1% | 4-42% |

Cause of death (lost round), 315 losses: other boss ante >= 5 64 (20%), other showdown boss 62 (20%), **Violet Vessel 52 (17%)**,
**The Eye 38 (12%)**, Big blind 26 (8%), Needle 24 (8%), other boss ante <= 4 16 (5%), Wall 15 (5%), Small blind 11 (3%), The Mouth 7 (2%).
At the ante-8 final boss the win rate given the boss is Cerulean 97%, Verdant 91%, Amber 90%, Crimson 85%, **Vessel 57%** (70 wins / 52 deaths).

**Zero-score plays (a clear policy error).** Under The Eye a repeated hand type scores 0, under The Mouth a hand type different
from the first. The model plays them: The Eye 69 of 228 rounds contain >= 1 zero-score play (111 plays); **survival 53.6% with, 96.2% without**.
32 of the 38 Eye deaths and 6 of the 7 Mouth deaths contain such a play (teacher: 4 Eye rounds, 22 Mouth rounds). The model plays Pair
79% of the time at ante >= 5 (teacher 27%), so after the first Pair it has no second type ready, although `hand_plays_round` is in the
observation. Counterfactual bound: if the 69 Eye + 9 Mouth rounds survived at the no-blunder rate, ~35 deaths = **+3.5 win points**.

## 2. How close are the losses

Final score / target in the death round, quantiles 10/25/50/75/90: 0.50 / 0.65 / 0.82 / 0.93 / 0.97.

| ratio | <0.25 | 0.25-0.5 | 0.5-0.75 | 0.75-0.9 | 0.9-1.0 |
|---|---|---|---|---|---|
| losses | 4 | 27 | 94 | 93 | 96 |

Upper bound on the win-rate gain of a uniform score boost (death round only, later rounds ignored): x1.1 -> 90 losses flip (+9.0 pts),
x1.2 -> 147 (+14.7), x1.5 -> 229 (+22.9), x2 -> 284 (+28.4). Teacher: 197 / 332 / 564 / 716, i.e. the model's *margin distribution
is much tighter*: the remaining losses are mostly failures by 10-35%.

State at death:

| group | n | median ratio | discards left | discards used | money | jokers | has xMult | deck size |
|---|---|---|---|---|---|---|---|---|
| ante 1-4 | 41 | 0.84 | 0.78 | 2.5 | 13.1 | 3.8 | 29% | 52.3 |
| ante 5-7 | 162 | 0.85 | 1.62 | 1.95 | 19.4 | 5.1 | 63% | 52.6 |
| ante 8 | 112 | 0.77 | 2.62 | 1.33 | 15.8 | 5.05 | 83% | 53.3 |

* Joker slots are full (5.1 jokers; 41/315 losses with a free slot, 21 of them in ante <= 4). Consumable left over: 0-7%.
  The deck is essentially unmodified (52.3-53.3 cards; `cards_added` 1.2, `cards_destroyed` 0.4 per game): **no deck building** (teacher
  adds 1.3-2.5 and destroys 0.7-1.2 cards).
* 151 of the 315 losses die with >= $10 unspent (mean $13-19), 201 with >= 1 discard left, 75 with all (>= 4) discards left
  (48 of them at ratio >= 0.75). The model rarely discards: it plays the first Pair of an 8-card hand (survived rounds: 83% with zero
  discards at ante >= 5; lost rounds 35%). Since a leveled Pair is almost always available, unused discards are not obviously a mistake.
* Main hand type: Pair in 84% of the losses and 91% of the wins; Pair level 8.1 in wins vs 5.3 in losses (all antes; within ante-8
  reachers the win rate rises only from 80% at level <= 4 to 94-96% at level >= 13, noisy).

## 3. What separates wins from losses

State at the first round of each ante (losses = games that reach the ante):

| ante | win: money | jokers | xMult | loss: money | jokers | xMult |
|---|---|---|---|---|---|---|
| 3 | 22.2 | 4.28 | 0.30 | 21.4 | 4.30 | 0.32 |
| 5 | 48.9 | 5.06 | 0.59 | 43.4 | 5.02 | 0.50 |
| 6 | 58.5 | 5.14 | 0.68 | 49.1 | 5.08 | 0.59 |
| 7 | 63.2 | 5.19 | 0.76 | 46.4 | 5.12 | 0.71 |
| 8 | 67.1 | 5.23 | 0.81 | 44.4 | 5.12 | 0.75 |

* Up to ante 3-4 the future winners and losers look **identical** (money, joker count, xMult share). The gap opens at ante 5 and is
  small even at ante 8. Joker *count* never differs (slots are full from ante 4 on); only the xMult share (-5 to -9 points) and the
  money differ.
* Win rate at the start of ante 5 (959 games, base 71.4%) by number of xMult jokers: 0 -> 67.3% (n=422), 1 -> 73.8%, 2 -> 77.3%, 3 -> 78.6%; with a
  scaling joker 73.3% vs 62.5% without; Mr. Bones 78.8% (n=99).
* Joker lift (same table, n >= 40): **Constellation +16.4, Space Joker +15.7, The Duo +13.1, Golden Joker +9.6, Green Joker +8.6**
  (Pair-friendly or scaling) against **Odd Todd -17.3, Gros Michel -12.5, Half Joker -10.6, Raised Fist -5.3, Blue Joker -3.1**. The
  model holds the weak ones often (Half Joker 12% at ante 5). Final joker sets: wins are richer in Supernova (57% vs 43%), Banner,
  Green Joker (40% vs 25%), Card Sharp (25% vs 14%), Ramen; losses in Raised Fist, Blue Joker, Half Joker, Swashbuckler.
* Vouchers (before ante 5, observational): Director's Cut +10.4 pts (n=118), Telescope +9.2, Overstock +9.5, Nacho Tong +12.4, Grabber
  +4.9; Hieroglyph -7.4 (n=256). **Causal check**: banning Hieroglyph and Petroglyph at evaluation time (BuyVoucher masked,
  guard 2): 68.7% win / 7.47 ante vs 68.5% / 7.51 unbanned (paired +0.2 +- 1.2 pts). The association with Hieroglyph is confounded,
  it neither helps nor hurts. Director's Cut is used as intended (167 `RerollBoss`, 54 of them away from Violet Vessel); only
  5 of 128 Vessel encounters at the last shop had a reroll voucher.

## 4. Shop behaviour

Per game, summed over the shops of the ante (win n=685 / loss n=315 at ante 1, fewer later):

| ante | win rerolls | win packs | win money at EndShop | loss rerolls | loss packs | loss money at EndShop |
|---|---|---|---|---|---|---|
| 3 | 1.68 | 1.47 | 26.5 | 1.65 | 1.50 | 25.2 |
| 5 | 2.37 | 1.89 | 52.0 | 2.57 | 2.49 | 44.1 |
| 6 | 2.32 | 1.85 | 60.7 | 3.48 | 2.81 | 46.4 |
| 7 | 2.34 | 2.02 | 64.6 | 4.00 | 2.92 | 40.4 |
| 8 | 2.29 | 1.77 | 68.2 | 5.10 | 2.90 | 31.0 |

* The model **hoards**: $50-68 at the end of ante 5-8 shops in winning runs, while interest stops at $25 (cap $5) and the five joker slots
  are full. The teacher spends everything (EndShop $1-7, $0-10 at ante start), reroll 1-1.7 per ante. Losing runs reroll more at
  ante 6-8 (3.5-5.1) and spend more ($31-46 left), which is the signature of searching for a rescue.
* Obvious xMult jokers are not missed: only 2.2% (win) / 2.6% (loss) of EndShops leave an affordable xMult joker on the shelf with a
  free slot (`sec_unbought`; free slot = fewer than 5 jokers, approximate). The limit is not recognising xMult, it is what the shop
  offers and the ability to judge a *replacement* (sell a weak joker, buy a better one; 0.5-1.0 sells per ante).
* Blinds: the model skips 0.12 blinds per game (teacher 3.7-6.0); only 121 skips in 1000 games (Buffoon 78, Polychrome 43 tags).

## 5. Early losses (ante <= 4, 41 games, 4.1%)

Blinds: 27 boss, 9 Big, 5 Small (Needle 4, Eye 3, Water 3, Pillar 3, Mouth 2, Wall 2, Manacle 2, Flint 2). Ratio >= 0.9: 14, 0.75-0.9: 12,
< 0.75: 15; <= 2 jokers at death: 8; >= $10 unspent: 16; >= 3 discards unused: 8. Classification from the logs (`scripts/analyze_losses.py losses
--max-ante 4`, narratives in section 7):

| class | games | examples |
|---|---|---|
| zero-score play in the death round (Eye/Mouth) | 5 | 256 (The Mouth at ante 4: HighCard first, then 5 Pairs scoring 0, with $69 unspent), 111, 643, 664, 717 |
| near miss (ratio >= 0.9), rest normal | 12 | 26 (1989/2000), 53 (Needle 4900/5000, one hand), 618 (3993/4000), 538 (448/450) |
| thin early build (<= 2 jokers at death, ratio < 0.9) | 6 | 678 (0 jokers at the ante-1 boss, $3), 323, 447, 704, 743, 907 |
| other: build and round simply too weak (ratio 0.5-0.9) | 18 | 245, 341, 465, 939 |

(The classes are exclusive, assigned in this order.)

Mostly **variance and a slow build**, not systematic errors; the ante-1 deaths (8) have 0-3 jokers and $4-13. The Mouth/Eye
cases (5) and a few unused-discard rounds (618: Hieroglyph, 3 hands, 4 discards unused, ends 7 chips short) are mistakes.

## 6. Margin in the wins

Beaten rounds of winning runs (685 wins), score at the moment the round ends / target:

| round | median ratio | p10 | p90 | share < 1.2 | hands used | hands left | won on the last hand |
|---|---|---|---|---|---|---|---|
| ante 6 boss | 1.27 | 1.05 | 1.80 | 36% | 2.64 | 1.38 | 20% |
| ante 7 boss | 1.26 | 1.04 | 1.79 | 40% | 2.88 | 1.15 | 28% |
| ante 8 small | 1.38 | 1.07 | 1.98 | 29% | 2.10 | 2.09 | 1% |
| ante 8 big | 1.30 | 1.06 | 1.88 | 36% | 2.55 | 1.65 | 12% |
| **ante 8 boss** | **1.19** | **1.01** | **1.60** | **51%** | **3.30** | **0.90** | **42.5%** |

Final boss in the wins: Violet Vessel (300k target) won on the last hand in 74% of the wins (median 1.06, 3.8 hands used), Crimson Heart 53% /
1.16, Amber Acorn 39% / 1.17, Verdant Leaf 34% / 1.22, Cerulean Bell 31% / 1.25. A final-round ratio (the round stops when the target is
reached) overstates the narrowness only by the size of the last hand (median best hand / target 0.33-0.56 at the final boss), so
the more telling numbers are the shares won on the last hand (42.5% overall, 74% at Vessel) and hands left (0.9 of 4).
17.5% of all beaten rounds at ante >= 6 end below 1.1x; a winning run has on average 1.5 such rounds. Wins are **narrow at the final boss
and comfortable on small/big blinds** (median 1.3-1.4, 2+ hands left). Teacher wins look the same (median 1.19 at the final boss).

## 7. Ten losing games, step by step (`narrate`)

1. **Game 678, ante 1 boss, 0 jokers, $3.** Bought Burglar (+3 hands, no discards) in shop 1, then rerolled twice and sold it for $3 at the
   boss shop; faced The Pillar with no joker (0.62 of 600). Slow start with $4-9 to spend: bad luck plus a questionable Burglar purchase.
2. **Game 256, ante 4 The Mouth, $69 unspent, 6 jokers.** First play HighCard with 4 cards (304 points) fixed the hand type, the next five
   Pairs scored 0 (10000 target, 304 total). The prior Pair hands scored 8100. **Clear play blunder.**
3. **Game 53, ante 4 The Needle (one hand), 4900 / 5000.** One discard then a Two Pair for 4900. Near miss (0.98), $1 left after buying Ice Cream and
   Clever Joker. Variance.
4. **Game 618, ante 3 The Flint, 3993 / 4000.** Hieroglyph owned (3 hands). Flush 1562, Pair 1254 with 4 discards unused, then a discard
   instead of a play, last hand Three of a Kind 1177 -> 7 chips short. $32 unspent. Mild mistake (the Pair play could have used discards).
5. **Game 63, ante 7 The Eye, 15776 / 70000.** Pair 15776, then **two more Pairs for 0** (3 hands, 3 discards unused). $7 left after two
   rerolls; clear blunder.
6. **Games 9, 64, ante 7 The Eye (0.93, 0.80).** Handled by playing HighCard / Full House / Two Pair as the second and third hand, losing by
   5-14k; a build that cannot produce three good different hand types.
7. **Game 41, ante 7 The Needle, 11316 / 35000** (Pair lvl 3, jokers Supernova, Spare Trousers, Scary Face, Card Sharp, Ramen) and **game 164 (29495 / 35000, Pair lvl 4, 4 discards unused,
   Hieroglyph).** One-hand boss needs one big hand; the Pair-spam build is weak at one hand when under-leveled.
8. **Game 174, ante 8 Violet Vessel, 100k / 300k.** Pair level 19 but Sly Joker, Half Joker, Blue Joker, Space Joker, Raised Fist, Grabber (5
   hands): 20-25k per hand, $0 after the last shop. The earlier rounds were already at ratios 1.00-1.25. A power deficit, not an
   action error. Games 32 (160k), 76 (170k, 3 hands), 138 (201k) look the same: 3-4 Pair hands of 30-100k against 300k.
9. **Game 18, ante 8 Crimson Heart, 79.8k / 100k, $85 unspent.** Entered the last shop with $140, spent $55 on 6 rerolls and two Wheel of Fortune, bought Card Sharp, but the six jokers are
   Green Joker, Delayed Gratification, Raised Fist, Abstract Joker, Stuntman, Card Sharp (low power) and kept $85. Money hoard that could
   not be converted into power.
10. **Game 16, ante 8 Verdant Leaf, 96860 / 100000.** Four Pairs for 23.8k-24.7k (Half Joker + Photograph), **4 discards unused**, 3.2% short.
    Near miss on a joker set without xMult.

Pattern: late deaths are mostly **a Pair-only engine that stops 10-30% short of an ante-8 boss**, a small group is a play blunder under an
Eye/Mouth boss.

## 8. Implications

Evidence summary: (i) deaths are late and at bosses (88% boss, 87% ante >= 5), (ii) most are a power shortfall, not a tactical error: 4 hands
of 20-30k against a 100-300k target, 60% of the losses at ratio >= 0.75 and the model hoards $45-68 with full slots, (iii) ~14% of the losses
contain a zero-score play (Eye/Mouth), (iv) win rate depends on build quality (xMult count, joker identity), the future winners are not
distinguishable until ante 4, (v) Hieroglyph/Petroglyph purchases and the discard habit are neutral/unclear.

Ranking of the candidate changes (a-e) by evidence:

1. **(c) shop-side build valuation (fair score preview of a joker / replacement)** is the best supported. The bottleneck is build power at
   ante 5-8, the model has the money but does not convert it (hoard $50-68, 2.3 rerolls per ante, 0.5-1 sells), obvious xMult jokers are
   already taken (2.2% misses), so the missing piece is judging *which joker raises the Pair score by how much* and what to replace.
   The joker lifts (+16 to -17 pts) show large differences in value that the policy only partially tracks. Expected upside: the 62 other
   showdown + 52 Vessel + 64 other late-boss losses are the pool (up to ~+10 points; unproven).
2. **(a) margin / score-based reward, cheap to try.** Wins and losses differ by tens of percent in the score (median ante-8 boss win 1.19x, 42%
   won on the last hand, 147 losses within a x1.2 score factor, 60% of losses at ratio >= 0.75), so a dense `log(score/target)` or the
   existing `shaping` term (progress, default 0) gives a gradient exactly where the sparse win/loss signal is thinnest. Endless beyond ante 8 adds
   little in the near term: the model already struggles to clear ante 8 at 300k (Vessel), so ante 9+ would mostly add ~1-2 more rounds of the same
   signal; use the margin term instead of endless. Risks: overkill-chasing; keep the win reward dominant.
3. **(d) auxiliary losses, not a bigger model.** The zero-score blunders show a prediction gap that is cheap to supervise (aux head:
   "this selection scores 0 / hand type of the selection / expected score") and doubles as input for (c). There is no evidence that capacity
   is limiting (memory ablation neutral at 24M; deaths are build power, not representation). Alternatively fix the blunder structurally:
   mask or penalise a play that the engine knows scores 0 (Eye repeat, Mouth mismatch); bound **+3.5 pts**.
4. **(e) decision-time search** has weak support as a general tool: tactical play errors are rare (zero-score plays aside; discards are rarely
   the problem), the shop has a long horizon and hidden draw order, and the cost is high. A narrow one-ply lookahead on the current hand's
   score would fix Eye/Mouth; broader search is not indicated by these losses.
5. **(b) higher stakes** is not supported by any loss cause: the policy is not saturated (31% losses) and every stake raises the same bottleneck
   (target size), so it would just lower the win rate before the fixable causes (zero-score plays, build conversion) are removed. Revisit
   once the win rate is high and seeds repeat.

Other levers visible in the data: Director's Cut / Retcon are valued highly (+10 pts, Vessel is the single worst boss: 41% death,
59% without Mr. Bones) but only 5 of 128 Vessel encounters had a reroll voucher in hand, deck building is unused, blind skipping is almost
unused (teacher 4-6 per game; the freeze ablation attributed -29 points to the teacher's blind decisions), and Hieroglyph/Petroglyph are
neutral under the ban test.

## 9. Caveats

* Observational comparisons (joker lift, vouchers, money) are confounded by survival: winners see more shops. Treat as hypotheses; the
  Hieroglyph ban is the only intervention.
* Recording is one deterministic run per game; sample-mode behaviour (65.9% win) differs.
* The upper bounds in section 2 (score boost) ignore later rounds and are not additive with the zero-score estimate.
* xMult / scaling sets and the "affordable" test (price ~ 2 x sell value) are approximations. A free slot counts as < 5 jokers (negative-edition jokers ignored).
* CPU recording reproduces 68.5% vs 68.6% (GPU) on the test set, so GPU/CPU float differences flip about one game.
