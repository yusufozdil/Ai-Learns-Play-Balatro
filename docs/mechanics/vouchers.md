# Vouchers

Source for everything: https://balatrowiki.org/w/Vouchers (VO), shop page https://balatrowiki.org/w/Shop (SH). All cost $10 (upgrades were $20 before 1.0.0h) [VO].

## Shop rules
- One voucher per ante; restocks after each Boss Blind (or via Voucher Tag) [VO]. Permanent for the run [VO].
- Only base vouchers can appear at first; an upgraded voucher needs its base owned in that run [VO]. Same voucher cannot be bought twice [VO].
- Each eligible voucher is equally likely [VO]. If none are eligible, Blank appears [VO].
- The "unlock" conditions listed below (spend $2,500, reroll 100 times, etc.) are profile-unlock requirements of the real game; the wiki lists them as prerequisite. For the simulator assume ALL vouchers are unlocked (only the base-owned rule applies). UNVERIFIED reading of the wiki; recommended because the RL agent otherwise would never see upgrades, and the original gates them via profile unlocks, not per-run.

## Table
| # | Name | Effect | Needs (base) | Unlock condition (profile) |
|---|---|---|---|---|
| 1 | Overstock | +1 shop card slot | - | - |
| 2 | Overstock Plus | +1 more card slot; fills the empty slot on buy | Overstock | spend $2,500 in shop total |
| 3 | Clearance Sale | all cards and packs 25% off | - | - |
| 4 | Liquidation | 50% off; also lowers sell value of owned Jokers | Clearance Sale | redeem 10 vouchers in a run |
| 5 | Hone | Foil/Holo/Poly 2x as likely | - | - |
| 6 | Glow Up | Foil/Holo/Poly 4x as likely | Hone | 5+ edition Jokers (Foil/Holo/Poly/Negative) |
| 7 | Reroll Surplus | rerolls cost $2 less | - | - |
| 8 | Reroll Glut | rerolls $2 less again | Reroll Surplus | reroll 100 times total |
| 9 | Crystal Ball | +1 consumable slot | - | - |
| 10 | Omen Globe | Spectrals may appear in Arcana packs | Crystal Ball | use 25 Tarots from packs |
| 11 | Telescope | Celestial packs always hold the Planet of most played hand | - | - |
| 12 | Observatory | each Planet held in consumable area gives x1.5 Mult for its hand | Telescope | use 25 Planets from packs |
| 13 | Grabber | +1 hand per round | - | - |
| 14 | Nacho Tong | +1 more hand per round | Grabber | play 2,500 cards |
| 15 | Wasteful | +1 discard per round | - | - |
| 16 | Recyclomancy | +1 more discard | Wasteful | discard 2,500 cards |
| 17 | Tarot Merchant | Tarot weight 9.6 (wiki says "2x") | - | - |
| 18 | Tarot Tycoon | Tarot weight 32 ("4x") | Tarot Merchant | buy 50 Tarots from shop |
| 19 | Planet Merchant | Planets 2x (likely weight 9.6) | - | - |
| 20 | Planet Tycoon | Planets 4x (likely 32) | Planet Merchant | buy 50 Planets from shop |
| 21 | Seed Money | interest cap raised to $10 per round | - | - |
| 22 | Money Tree | cap $20 | Seed Money | max interest 10 rounds in a row |
| 23 | Blank | nothing | - | - |
| 24 | Antimatter | +1 Joker slot | Blank | redeem Blank 10 times |
| 25 | Magic Trick | playing cards can be bought in shop | - | - |
| 26 | Illusion | shop playing cards may have enhancement/edition/seal | Magic Trick | buy 20 playing cards from shop |
| 27 | Hieroglyph | -1 ante, -1 hand per round | - | - |
| 28 | Petroglyph | -1 ante, -1 discard per round | Hieroglyph | reach ante 12 |
| 29 | Director's Cut | reroll Boss Blind once per ante for $10 | - | - |
| 30 | Retcon | reroll Boss Blind unlimited, $10 each | Director's Cut | discover 25 blinds |
| 31 | Paint Brush | +1 hand size | - | - |
| 32 | Palette | +1 hand size | Paint Brush | reduce hand size to 5 |
All rows from [VO]. Interest cap: base $5/round, so Seed Money/Money Tree mean interest on up to $50/$100/$200 held; check the economy doc.

## Implementation notes
- Hieroglyph/Petroglyph reduce the current ante by 1 immediately (min ante 1 [mem] UNVERIFIED); hands/discards reduction is permanent per round.
- Reroll Surplus + Glut combined: start cost $1 (5 - 2 - 2) [VO, SH]; floor $0.
- Liquidation/Clearance discounts apply to the sell value of Jokers (Liquidation per VO).
- Illusion seals are bugged in v1.0.1o [VO] (see shop.md).
- Antimatter, Crystal Ball, Paint Brush effects apply immediately and persist.
