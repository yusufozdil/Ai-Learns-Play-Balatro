# Real-game bridge (Phase 11 / B1): the policy plays real Balatro through BalatroBot

Goal: let the trained simulator policy play the **real** game and measure rule parity (the
simulator's score of every played hand vs the real chips). Code: `python/balatro_rl/bridge/`,
scripts `scripts/bridge_play.py` / `scripts/bridge_report.py`, Lua patch `bridge/`, tests
`tests_py/test_bridge.py`. This document: architecture (1), the Lua patch (2), the state mapping
(3), the id tables (4), actions (5), game flow (6), hidden information (7), score parity (8),
BalatroBot problems (9), what cannot be mapped (10), results (11), how to run (12).

## 1. Architecture

```
                 BalatroBot JSON-RPC (HTTP, 127.0.0.1:12346)
 real Balatro  <------------------------------------------------>  client.py
 (Proton)                                                              |
   |  gamestate + `raw` / `bridge` blocks (Lua patch)                  | gamestate (dict)
   v                                                                   v
                                                               state.py: translate()
                                                                       |  engine state JSON
                                                                       v
                                         balatro_rl._engine.Game.from_state()  (the simulator)
                                                |  observation(), masks()          |  step_parts(Play)
                                                v                                  v
                                       ModelPolicy (recurrent)             parity.py: predicted chips
                                                |  engine action [kind, arg0, arg1, card_mask]
                                                v
                                       actions.py: plan_action() -> API calls  ----> client.py
```

The design direction of the task is kept: **all encoding is done by the simulator.** The bridge never
re-implements the observation; it only builds an engine state that equals the real one in
everything the player can see (`Game.from_state`, the format of `Game.export_state()`,
`docs/design/rl_interface.md` section 13), asks the engine for the observation and masks, runs the
model, and turns the chosen engine action into BalatroBot calls. The same translated state is also
used for the score prediction (`parity.predict_play`).

Files:

| file | role |
|---|---|
| `client.py` | stdlib JSON-RPC client; per-call timeout (`BridgeTimeout` = the known hangs), `ApiError` with the BalatroBot error name |
| `tables.py` | key tables (jokers, consumables, vouchers, tags, bosses, hand types, cards, packs) |
| `engine_tables.py` | engine facts measured on the engine itself: per-joker default `value`/`counter`, per-joker `mods`, per-voucher `shop_mods` |
| `state.py` | `translate(gamestate) -> Translation`, `build_game(tr)` |
| `actions.py` | `plan_action(action, tr) -> [Call]`, `decode_action` |
| `parity.py` | `predict_play`, `compare`, `real_hand_type` |
| `runner.py` | `Runner`, `ModelPolicy`, `RandomPolicy`, JSONL log, known-bug blocks |
| `bridge/lua/` | the Lua patch (section 2) |

## 2. The Lua patch (`bridge/`, version 3)

Stock BalatroBot 1.5.2 gives joker counters only as effect *text* and exposes none of the
round-target cards (Idol / Ancient / Castle / Mail-In), held tags, the boss state, the open pack's
kind, the full deck or the discard pile, and has no Director's Cut / Retcon endpoint. The patch is
minimal and separately versioned:

* `bridge/lua/bridge_ext.lua` (new file, `VERSION = 3`): `card_raw(card)` (the raw `card.ability`
  table, seal, edition, base rank/suit, debuff, facing, centre key, costs) and `game_raw()`
  (selected `G.GAME` fields: `current_round` incl. `idol_card` / `ancient_card` / `castle_card` /
  `mail_card` / `free_rerolls` / `reroll_cost_increase`, `round_resets`, held `tags`, the `blind` object, `orbital_choices`,
  consumable usage, `pool_flags`, `bosses_used`, `last_hand_played`, `starting_deck_size`, the open
  booster's key, `pack_choices`, `interest_cap`, ...), `augment(state)` adds `bridge`, `discard` and
  `playing_cards` to the gamestate.
* `bridge/lua/endpoints/reroll_boss.lua` (new endpoint `reroll_boss`): calls the game's own
  `G.FUNCS.reroll_boss` (Director's Cut once per ante, Retcon unlimited, $10) and waits until the
  boss pane is rebuilt.
* `bridge/lua/endpoints/continue_endless.lua` (new endpoint `continue_endless`, only in
  `ROUND_EVAL` after a win): waits for the win screen and presses its "Endless" button
  (`G.FUNCS.exit_overlay_menu`); `cash_out` follows. BalatroBot has no such call.
* `bridge/lua/balatrobot-1.5.2.patch`: a small diff of three stock files (`balatrobot.lua`: load the
  extension and register the endpoints; `src/lua/utils/gamestate.lua`: add `raw` to every card and
  call `augment`; `src/lua/endpoints/sell.lua`: allow `sell` while a booster pack is open, the stock
  endpoint only accepts SELECTING_HAND / SHOP although the game allows it and the policy often wants
  to sell a joker to make room for a Buffoon pick). Both hooks are wrapped in `pcall`: an error in the extension can never break the
  stock gamestate (it is reported as `bridge_error`).
* `bridge/apply_lua_patch.sh [--revert]` installs pristine 1.5.2 (from the staging directory) + the patch + the new
  files into the installed mod, after backing up the originals once to
  `<mod>/.bridge_backup/orig/`. `bridge/make_patch.py` regenerates the patch from a pristine tree.
  `bridge/restart_game.sh` restarts Balatro (no `--fast`).

## 3. State mapping (BalatroBot gamestate -> engine state JSON)

`state.translate` starts from a fresh engine state of the same deck / stake (`Game(1, deck, stake)`,
which supplies the RNG streams and every constant field) and overwrites:

| engine field | source in the gamestate | note |
|---|---|---|
| `config.deck/stake` | `deck`, `stake` | `RED` -> `Red` ... (tables.py) |
| `phase` | `state` | `BLIND_SELECT`->BlindSelect, `SELECTING_HAND`->Play, `SHOP`->Shop, `SMODS_BOOSTER_OPENED`->PackOpen, `GAME_OVER`->GameOver (section 6) |
| `ante`, `round` | `ante_num`, `round_num` | |
| `blind` | `blinds.{small,big,boss}.status` | Play: `CURRENT`; otherwise the blind with status `SELECT` |
| `target` | Play: `bridge.blind.chips`; else `blinds[k].score` | includes Plasma x2, Needle, Wall, ... |
| `blind_tags`, `blind_tag_hands` | `blinds.*.tag_name`; `bridge.orbital_choices[ante]` | tag names -> `TAG_NAME_TO_ID` |
| `boss`, `boss_disabled`, `boss_seen` | `bridge.round_resets.blind_choices.Boss`; `bridge.blind.disabled`; `bridge.bosses_used` | |
| `boss_state` | Ox: `current_round.most_played_poker_hand`; Needle/Water `hands_sub`/`discards_sub`; Bell: the card with `ability.forced_selection`; Crimson Heart: the debuffed joker; Amber Acorn: jokers `facing=back` | `draws` is approximated (`1 + hands + discards`), `triggered` from `bridge.blind` |
| `money`, `interest_cap` | `money`, `bridge.interest_cap / 5` | real cap is in dollars of money, engine in dollars of interest |
| `hands_left`, `discards_left`, `hands_played_round`, `discards_used_round`, `round_score` | `round.*` (`round.chips`) | between rounds these are stale in the real game exactly as in the engine (verified, section 8) |
| `last_hand` | `bridge.last_hand_played` | |
| `jokers[]` | `jokers.cards[]` (centre key, `raw.ability`, `raw.edition`, `cost.sell`) | state fields below |
| `joker_slots`, `consumable_slots` | `jokers.limit - #Negative`, `consumables.limit - #Negative` | |
| `mods` | folded from the jokers | `engine_tables.fold_joker_mods` (measured per joker on the engine; differential-tested) |
| `consumables[]`, `consumable_use_counts`, `last_tarot_planet`, `stats.run.*` | `consumables.cards`, `bridge.consumeable_usage[_total]`, `bridge.last_tarot_planet`, `bridge.skips` | |
| `deck.hand` | `hand.cards`, sorted by the engine rule (rank desc, suit S/H/C/D, Stone last) | `Translation.hand_real_index` maps engine position -> real index |
| `deck.draw_pile` | `cards.cards` (BalatroBot's real draw pile, in its order) | hidden information, section 7 |
| `deck.discard_pile` | `discard` + every `playing_cards` entry not in hand / draw pile | |
| `starting_deck_size`, `stats.cards_added/destroyed` | `bridge.starting_deck_size`, `len(playing_cards)` | invariant: deck size = start + added - destroyed |
| `hand_levels`, `hand_plays_run`, `hand_plays_round` | `hands[name].level/played/played_this_round` | |
| `base_hands/discards`, `round_hand_size`, `base_hand_size` | `bridge.round_resets.hands/discards`, `temp_handsize`, `bridge.hand_limit - mods - temp - Manacle` | |
| `vouchers`, `shop_mods`, `ante_voucher` | `bridge.used_vouchers`; folded (`engine_tables.fold_shop_mods`); `current_round.voucher[1]` | |
| `tags` | `bridge.tags[].key` | |
| `shop` | `shop` / `packs` / `vouchers` areas; `round_resets.reroll_cost` (base), `current_round.reroll_cost_increase`, `current_round.free_rerolls` | `reroll_base` = real base cost, `rerolls` = increments, `free_rerolls_used = Chaos - free_left`; price of every slot is **measured**: `Runner._price_check` compares the engine's prices with BalatroBot's `cost.buy` every Shop |
| `pack` | `pack` area, `bridge.booster.key`, `bridge.pack_choices` | `origin` is `Tag` iff the previous action was SkipBlind |
| `run_flags` | `bridge.pool_flags.gros_michel_extinct` | |

**Joker internal state** (`state.joker_state`, `joker_aux`): from the raw `card.ability`:

| engine `value` | jokers | real field |
|---|---|---|
| +mult | Ceremonial Dagger, Ride the Bus, Green Joker, Red Card, Flash Card, Popcorn, Spare Trousers | `ability.mult` |
| xmult | Constellation, Madness, Vampire, Hologram, Obelisk, Lucky Cat, Ramen, Campfire, Glass Joker, Hit the Road, Canio | `ability.x_mult` |
| +chips | Runner, Ice Cream, Square Joker, Castle, Wee Joker | `ability.extra.chips` |
| other | Rocket `extra.dollars`; Turtle Bean `extra.h_size`; Yorick `x_mult` + `counter = yorick_discards`; Seltzer `counter = extra` (hands left); Invisible `counter = invis_rounds`; Loyalty Card `counter = hands_played_at_create`, `aux_byte = 1` | |

| engine `aux` lane | jokers | real source |
|---|---|---|
| suit | Ancient Joker, Castle, The Idol | `current_round.ancient_card.suit`, `castle_card.suit`, `idol_card.suit` |
| rank | The Idol, Mail-In Rebate | `idol_card.id`, `mail_card.id` |
| hand | To Do List | `ability.to_do_poker_hand` |

Jokers with no state in the engine (Throwback, Fortune Teller, Satellite, Card Sharp, Supernova, ...)
read run statistics, which are filled from `bridge.skips`, `consumeable_usage_total`, `hands`.
Sell values come from `cost.sell` (Egg, Gift Card, ... already included). Stickers: `ability.eternal`,
`perishable` + `perish_tally`, `rental`. Editions: `edition.type` (`foil`, `holo`, `polychrome`, `negative`).

## 4. Id tables

All tables are in `python/balatro_rl/bridge/tables.py` (the source of truth, completeness-tested in
`tests_py/test_bridge.py`); they are in the engine's declaration order, so the position is the
engine index. The joker order of the engine equals the real game's centre order (checked
against `docs/mechanics/jokers.csv` and the game's `j_*` list).

Notable irregular keys: `j_gluttenous_joker` (typo in the game), `j_8_ball`, `j_ceremonial`,
`j_space` (Space Joker), `j_trousers`, `j_ring_master` (Showman), `j_caino` (Canio), `j_selzer`,
`j_mail`, `j_sock_and_buskin`, `j_stencil`, `j_golden`, `j_ticket` (Golden Ticket), `j_oops`,
`c_heirophant`, `c_deja_vu`, `c_soul`, `v_overstock_norm`, `tag_skip` (= engine `Speed`),
`tag_d_six`, `tag_holo`, `bl_final_bell/leaf/vessel/acorn/heart` (Showdown bosses).

Playing cards: BalatroBot `raw.base.value` (`2`..`10`, `Jack`, `Queen`, `King`, `Ace`) and `suit`
(`Hearts`...) map to the engine's names; the enhancement comes from the centre key (`m_bonus`,
`m_mult`, `m_wild`, `m_glass`, `m_steel`, `m_stone`, `m_gold`, `m_lucky`; `c_base` = none), the
seal from `card.seal` (`Gold`/`Red`/`Blue`/`Purple`), the edition from `edition.type`, bonus chips
from `ability.perma_bonus`, debuff from `debuff`, face-down from `facing == "back"`.
BalatroBot's own `modifier.enhancement` field is *localised text* (#194) and is not used.

**Jokers (150, `j_*` center key → `JokerId`; engine index = real center order)**

`j_joker`→`Joker`, `j_greedy_joker`→`GreedyJoker`, `j_lusty_joker`→`LustyJoker`, `j_wrathful_joker`→`WrathfulJoker`, `j_gluttenous_joker`→`GluttonousJoker`, `j_jolly`→`JollyJoker`, `j_zany`→`ZanyJoker`, `j_mad`→`MadJoker`, `j_crazy`→`CrazyJoker`, `j_droll`→`DrollJoker`, `j_sly`→`SlyJoker`, `j_wily`→`WilyJoker`, `j_clever`→`CleverJoker`, `j_devious`→`DeviousJoker`, `j_crafty`→`CraftyJoker`, `j_half`→`HalfJoker`, `j_stencil`→`JokerStencil`, `j_four_fingers`→`FourFingers`, `j_mime`→`Mime`, `j_credit_card`→`CreditCard`, `j_ceremonial`→`CeremonialDagger`, `j_banner`→`Banner`, `j_mystic_summit`→`MysticSummit`, `j_marble`→`MarbleJoker`, `j_loyalty_card`→`LoyaltyCard`, `j_8_ball`→`EightBall`, `j_misprint`→`Misprint`, `j_dusk`→`Dusk`, `j_raised_fist`→`RaisedFist`, `j_chaos`→`ChaosTheClown`, `j_fibonacci`→`Fibonacci`, `j_steel_joker`→`SteelJoker`, `j_scary_face`→`ScaryFace`, `j_abstract`→`AbstractJoker`, `j_delayed_grat`→`DelayedGratification`, `j_hack`→`Hack`, `j_pareidolia`→`Pareidolia`, `j_gros_michel`→`GrosMichel`, `j_even_steven`→`EvenSteven`, `j_odd_todd`→`OddTodd`, `j_scholar`→`Scholar`, `j_business`→`BusinessCard`, `j_supernova`→`Supernova`, `j_ride_the_bus`→`RideTheBus`, `j_space`→`SpaceJoker`, `j_egg`→`Egg`, `j_burglar`→`Burglar`, `j_blackboard`→`Blackboard`, `j_runner`→`Runner`, `j_ice_cream`→`IceCream`, `j_dna`→`DNA`, `j_splash`→`Splash`, `j_blue_joker`→`BlueJoker`, `j_sixth_sense`→`SixthSense`, `j_constellation`→`Constellation`, `j_hiker`→`Hiker`, `j_faceless`→`FacelessJoker`, `j_green_joker`→`GreenJoker`, `j_superposition`→`Superposition`, `j_todo_list`→`ToDoList`, `j_cavendish`→`Cavendish`, `j_card_sharp`→`CardSharp`, `j_red_card`→`RedCard`, `j_madness`→`Madness`, `j_square`→`SquareJoker`, `j_seance`→`Seance`, `j_riff_raff`→`RiffRaff`, `j_vampire`→`Vampire`, `j_shortcut`→`Shortcut`, `j_hologram`→`Hologram`, `j_vagabond`→`Vagabond`, `j_baron`→`Baron`, `j_cloud_9`→`Cloud9`, `j_rocket`→`Rocket`, `j_obelisk`→`Obelisk`, `j_midas_mask`→`MidasMask`, `j_luchador`→`Luchador`, `j_photograph`→`Photograph`, `j_gift`→`GiftCard`, `j_turtle_bean`→`TurtleBean`, `j_erosion`→`Erosion`, `j_reserved_parking`→`ReservedParking`, `j_mail`→`MailInRebate`, `j_to_the_moon`→`ToTheMoon`, `j_hallucination`→`Hallucination`, `j_fortune_teller`→`FortuneTeller`, `j_juggler`→`Juggler`, `j_drunkard`→`Drunkard`, `j_stone`→`StoneJoker`, `j_golden`→`GoldenJoker`, `j_lucky_cat`→`LuckyCat`, `j_baseball`→`BaseballCard`, `j_bull`→`Bull`, `j_diet_cola`→`DietCola`, `j_trading`→`TradingCard`, `j_flash`→`FlashCard`, `j_popcorn`→`Popcorn`, `j_trousers`→`SpareTrousers`, `j_ancient`→`AncientJoker`, `j_ramen`→`Ramen`, `j_walkie_talkie`→`WalkieTalkie`, `j_selzer`→`Seltzer`, `j_castle`→`Castle`, `j_smiley`→`SmileyFace`, `j_campfire`→`Campfire`, `j_ticket`→`GoldenTicket`, `j_mr_bones`→`MrBones`, `j_acrobat`→`Acrobat`, `j_sock_and_buskin`→`SockAndBuskin`, `j_swashbuckler`→`Swashbuckler`, `j_troubadour`→`Troubadour`, `j_certificate`→`Certificate`, `j_smeared`→`SmearedJoker`, `j_throwback`→`Throwback`, `j_hanging_chad`→`HangingChad`, `j_rough_gem`→`RoughGem`, `j_bloodstone`→`Bloodstone`, `j_arrowhead`→`Arrowhead`, `j_onyx_agate`→`OnyxAgate`, `j_glass`→`GlassJoker`, `j_ring_master`→`Showman`, `j_flower_pot`→`FlowerPot`, `j_blueprint`→`Blueprint`, `j_wee`→`WeeJoker`, `j_merry_andy`→`MerryAndy`, `j_oops`→`OopsAll6s`, `j_idol`→`TheIdol`, `j_seeing_double`→`SeeingDouble`, `j_matador`→`Matador`, `j_hit_the_road`→`HitTheRoad`, `j_duo`→`TheDuo`, `j_trio`→`TheTrio`, `j_family`→`TheFamily`, `j_order`→`TheOrder`, `j_tribe`→`TheTribe`, `j_stuntman`→`Stuntman`, `j_invisible`→`InvisibleJoker`, `j_brainstorm`→`Brainstorm`, `j_satellite`→`Satellite`, `j_shoot_the_moon`→`ShootTheMoon`, `j_drivers_license`→`DriversLicense`, `j_cartomancer`→`Cartomancer`, `j_astronomer`→`Astronomer`, `j_burnt`→`BurntJoker`, `j_bootstraps`→`Bootstraps`, `j_caino`→`Canio`, `j_triboulet`→`Triboulet`, `j_yorick`→`Yorick`, `j_chicot`→`Chicot`, `j_perkeo`→`Perkeo`

**Consumables (52: 12 planets, 22 tarots, 18 spectrals)**

`c_pluto`→`Pluto`, `c_mercury`→`Mercury`, `c_uranus`→`Uranus`, `c_venus`→`Venus`, `c_saturn`→`Saturn`, `c_jupiter`→`Jupiter`, `c_earth`→`Earth`, `c_mars`→`Mars`, `c_neptune`→`Neptune`, `c_planet_x`→`PlanetX`, `c_ceres`→`Ceres`, `c_eris`→`Eris`, `c_fool`→`TheFool`, `c_magician`→`TheMagician`, `c_high_priestess`→`TheHighPriestess`, `c_empress`→`TheEmpress`, `c_emperor`→`TheEmperor`, `c_heirophant`→`TheHierophant`, `c_lovers`→`TheLovers`, `c_chariot`→`TheChariot`, `c_justice`→`Justice`, `c_hermit`→`TheHermit`, `c_wheel_of_fortune`→`TheWheelOfFortune`, `c_strength`→`Strength`, `c_hanged_man`→`TheHangedMan`, `c_death`→`Death`, `c_temperance`→`Temperance`, `c_devil`→`TheDevil`, `c_tower`→`TheTower`, `c_star`→`TheStar`, `c_moon`→`TheMoon`, `c_sun`→`TheSun`, `c_judgement`→`Judgement`, `c_world`→`TheWorld`, `c_familiar`→`Familiar`, `c_grim`→`Grim`, `c_incantation`→`Incantation`, `c_talisman`→`Talisman`, `c_aura`→`Aura`, `c_wraith`→`Wraith`, `c_sigil`→`Sigil`, `c_ouija`→`Ouija`, `c_ectoplasm`→`Ectoplasm`, `c_immolate`→`Immolate`, `c_ankh`→`Ankh`, `c_deja_vu`→`DejaVu`, `c_hex`→`Hex`, `c_trance`→`Trance`, `c_medium`→`Medium`, `c_cryptid`→`Cryptid`, `c_soul`→`TheSoul`, `c_black_hole`→`BlackHole`

**Vouchers (32)**

`v_overstock_norm`→`Overstock`, `v_overstock_plus`→`OverstockPlus`, `v_clearance_sale`→`ClearanceSale`, `v_liquidation`→`Liquidation`, `v_hone`→`Hone`, `v_glow_up`→`GlowUp`, `v_reroll_surplus`→`RerollSurplus`, `v_reroll_glut`→`RerollGlut`, `v_crystal_ball`→`CrystalBall`, `v_omen_globe`→`OmenGlobe`, `v_telescope`→`Telescope`, `v_observatory`→`Observatory`, `v_grabber`→`Grabber`, `v_nacho_tong`→`NachoTong`, `v_wasteful`→`Wasteful`, `v_recyclomancy`→`Recyclomancy`, `v_tarot_merchant`→`TarotMerchant`, `v_tarot_tycoon`→`TarotTycoon`, `v_planet_merchant`→`PlanetMerchant`, `v_planet_tycoon`→`PlanetTycoon`, `v_seed_money`→`SeedMoney`, `v_money_tree`→`MoneyTree`, `v_blank`→`Blank`, `v_antimatter`→`Antimatter`, `v_magic_trick`→`MagicTrick`, `v_illusion`→`Illusion`, `v_hieroglyph`→`Hieroglyph`, `v_petroglyph`→`Petroglyph`, `v_directors_cut`→`DirectorsCut`, `v_retcon`→`Retcon`, `v_paint_brush`→`PaintBrush`, `v_palette`→`Palette`

**Tags (24; the real game's `tag_skip` is the engine's Speed tag)**

`tag_uncommon`→`Uncommon`, `tag_rare`→`Rare`, `tag_negative`→`Negative`, `tag_foil`→`Foil`, `tag_holo`→`Holographic`, `tag_polychrome`→`Polychrome`, `tag_investment`→`Investment`, `tag_voucher`→`Voucher`, `tag_boss`→`Boss`, `tag_standard`→`Standard`, `tag_charm`→`Charm`, `tag_meteor`→`Meteor`, `tag_buffoon`→`Buffoon`, `tag_handy`→`Handy`, `tag_garbage`→`Garbage`, `tag_ethereal`→`Ethereal`, `tag_coupon`→`Coupon`, `tag_double`→`Double`, `tag_juggle`→`Juggle`, `tag_d_six`→`D6`, `tag_top_up`→`TopUp`, `tag_skip`→`Speed`, `tag_orbital`→`Orbital`, `tag_economy`→`Economy`

**Boss blinds (28)**

`bl_hook`→`TheHook`, `bl_ox`→`TheOx`, `bl_house`→`TheHouse`, `bl_wall`→`TheWall`, `bl_wheel`→`TheWheel`, `bl_arm`→`TheArm`, `bl_club`→`TheClub`, `bl_fish`→`TheFish`, `bl_psychic`→`ThePsychic`, `bl_goad`→`TheGoad`, `bl_water`→`TheWater`, `bl_window`→`TheWindow`, `bl_manacle`→`TheManacle`, `bl_eye`→`TheEye`, `bl_mouth`→`TheMouth`, `bl_plant`→`ThePlant`, `bl_serpent`→`TheSerpent`, `bl_pillar`→`ThePillar`, `bl_needle`→`TheNeedle`, `bl_head`→`TheHead`, `bl_tooth`→`TheTooth`, `bl_flint`→`TheFlint`, `bl_mark`→`TheMark`, `bl_final_acorn`→`AmberAcorn`, `bl_final_leaf`→`VerdantLeaf`, `bl_final_vessel`→`VioletVessel`, `bl_final_heart`→`CrimsonHeart`, `bl_final_bell`→`CeruleanBell`

**Hand types (BalatroBot `hands` key → `HandType`)**

`High Card`→`HighCard`, `Pair`→`Pair`, `Two Pair`→`TwoPair`, `Three of a Kind`→`ThreeOfAKind`, `Straight`→`Straight`, `Flush`→`Flush`, `Full House`→`FullHouse`, `Four of a Kind`→`FourOfAKind`, `Straight Flush`→`StraightFlush`, `Five of a Kind`→`FiveOfAKind`, `Flush House`→`FlushHouse`, `Flush Five`→`FlushFive`



## 5. Actions (engine -> BalatroBot)

`actions.plan_action` (table also in the module docstring):

| engine kind | calls |
|---|---|
| 0 SelectBlind | `select` |
| 1 SkipBlind | `skip` |
| 2 Play | [`rearrange hand`] `play cards` |
| 3 Discard | `discard cards` |
| 4 SellJoker | `sell joker` |
| 5 EndShop | `next_round` |
| 6 UseConsumable | [`rearrange hand`] `use consumable [cards]` |
| 7 SellConsumable | `sell consumable` |
| 8 MoveJoker(from,to) | `rearrange jokers` with the permutation that removes `from` and inserts at `to` |
| 9 Buy | `buy card` |
| 10 BuyAndUse | `buy card`, then `use consumable = last slot` |
| 11 OpenPack | `buy pack` |
| 12 BuyVoucher | `buy voucher` |
| 13 Reroll | `reroll` |
| 14 PickPackItem | [`rearrange hand`] `pack card [targets]` |
| 15 SkipPack | `pack skip` |
| 16 RerollBoss | `reroll_boss` (Lua patch) |

* **Hand order.** The engine hand is rank-sorted, BalatroBot keeps its own order, and the played
  cards score left to right. Before every action that depends on card order (Play; consumables
  and pack picks with targets) the bridge sends `rearrange hand` with `Translation.hand_real_index`
  when the real order differs, after which engine position = BalatroBot index. Discard maps
  positions to real indices without rearranging.
* **Round end.** The engine pays out and opens the shop inside the winning Play; the real game
  stops at `ROUND_EVAL`. The runner sends `cash_out` as soon as it sees it and continues from the Shop.
* **Rejected actions.** Any `ApiError` is logged (`reject`), the action is masked
  (`kind` or `(kind, arg0)`) and the policy is queried again on the same state; after 12 rejects in
  one state the game is abandoned (`error`).
* **BuyAndUse** needs a free consumable slot in BalatroBot (`buy` refuses otherwise); the
  rejection path handles it.

## 6. Game flow

| BalatroBot `state` | bridge |
|---|---|
| `MENU` | `start {deck, stake, seed}` (the runner first calls `menu` if a run is open) |
| `BLIND_SELECT` | translate -> BlindSelect |
| `SELECTING_HAND` | translate -> Play |
| `ROUND_EVAL` | `cash_out` (not an engine phase) |
| `SHOP` | translate -> Shop |
| `SMODS_BOOSTER_OPENED` | translate -> PackOpen |
| `HAND_PLAYED`, `DRAW_TO_HAND`, `NEW_ROUND`, `*_PACK`, `PLAY_TAROT` or `bridge.controller_locked` | transition: poll every 0.4 s up to 90 s |
| `GAME_OVER` | loss; the next game starts with `menu` |
| `won == true` (ante-8 boss beaten) | win; without `--endless` the run is abandoned. With `--endless` the runner presses `continue_endless`, `cash_out`s and keeps playing with `endless=True` engine states until the run is lost or the boss of `--endless-max-ante` (default 12) is beaten (`ante_num > cap`); the log has `won_ante` and `endless_ante` (highest ante reached), the outcome stays `win` |

## 7. Hidden information

Only visible information has to match, and the observation (built by the engine) never contains the
rest (`rl_interface.md` section 9): the draw pile keeps BalatroBot's real order (the observation
only holds its composition), the RNG streams come from a fresh template, face-down cards keep their
real identity in the state but are encoded as hidden by the engine's observation, and jokers
under Amber Acorn are hidden the same way. The translated state is **not** reseeded, so the
score prediction of a hand that contains face-down cards uses the true cards. A predicted hand
whose score depends on RNG (Lucky, Misprint, Space Joker, ...) is detected by playing it on 4
copies that differ only in the scoring RNG streams (`parity.SCORING_STREAMS`).

## 8. Score parity

For every Play, before it is sent, `parity.predict_play` plays the same positions on copies of the
translated state and reads `round_score` after - before. All copies equal -> *deterministic*, the
real chips gained (`round.chips` after - before; `G.GAME.chips` survives the round end) must be
identical (`exact`). Otherwise the verdict is `in_range` if the real value lies in the min..max
of the copies. The log (`play` events) also keeps the hand cards (rank, suit, enhancement,
edition, seal), jokers with state, boss, vouchers, hand levels, predicted and real hand type.
`scripts/bridge_report.py` aggregates it (mismatch rate by joker / enhancement / edition / seal /
boss).

Engine conventions verified against the real game on the way: between rounds the engine keeps the
finished round's `hands_left`, `discards_left`, `round_score`, `hands_played_round`; the real game does the
same (`G.GAME.current_round`, `G.GAME.chips`), so these pass through unchanged. Shop reroll
removes tag jokers too in both. Real prices equal the engine's (2034 checked shop states, section 11).

## 9. BalatroBot problems and how the bridge handles them

| issue | effect | handling |
|---|---|---|
| #234 `--fast` corrupts the deck order | silent | never used (`restart_game.sh` starts it plainly) |
| #195 selling an Invisible Joker never returns | hang | `known_bug_blocks`: `SellJoker` of that slot is masked |
| #199 Black Hole in a Celestial pack | hang | the pick of that item is masked |
| #233 skipping into a tag pack (Standard / Charm / Meteor / Buffoon / Ethereal) | hang | `SkipBlind` masked when the offered tag is a pack tag |
| #198 thin decks | hang | not handled (not reached on Red/White so far) |
| #194 `modifier.enhancement` is localised text | wrong enhancement | not used; the centre key from the patch's `raw` is used |
| `sell` only in SELECTING_HAND / SHOP, `use` only in SELECTING_HAND / SHOP | the engine allows both while a pack is open; 3260 rejected `SellJoker` in one game | `sell` patched (section 2); `use` of a held consumable in a pack is masked (`known_bug_blocks`) |
| `buy` refuses a joker / consumable when the slots are full | engine allows Negative / BuyAndUse | rejection path (logged with context) |
| `use` of a consumable "cannot be used at this time" while the game is still animating | rejected action | retried up to 3x after 1.5 s |
| `continue_endless` + `cash_out` after the ante-8 win crashes the game | endless mode only | experimental, see section 11 |
| any unanswered call | `BridgeTimeout` | the game is recorded as `hang`; the runner waits for the game to answer `health` before the next game |
| `Lua` empty tables are JSON `[]` even when they are maps | e.g. `used_vouchers: []` | `state._dict` / `_list` accept both |

## 10. What cannot be mapped, and known gaps

* **Draw order / RNG**: not mirrored (non-goal, `prd.md` section 1); the policy never sees them.
* **Orbital tag hand, held tags**: only through the patch (`orbital_choices`, `tags`).
* **Pack origin** (`Shop` vs `Tag`) is inferred from the previous action; it only matters for
  skipping rules.
* **`boss_state.draws`** is approximated; `triggered` comes from the blind object; boss effects that
  keep per-round counters beyond those (The Wheel's draws, The Hook's discards) are not
  reconstructed mid-round, they only matter for the *prediction* of later hands.
* **Endless mode**: off by default (the run stops at the ante-8 win, which is what the win rate measures); `--endless` continues through the patch's `continue_endless` endpoint (section 6).
* **Challenge / seeded runs**: only deck and stake of `start`.
* **Stakes above White** translate (stickers, `perish_tally`, rental) but have not been run against the real game.
* **Joker `Throwback`/`Fortune Teller`/... retroactive stats** use the run counters; `Rocket`/`Satellite`
  use the same.
* **`Show` text only fields** (effect strings) are ignored.
* Engine side: nothing had to be changed; `Game.from_state` accepted every translated state
  (the derived caches `mods` / `shop_mods` are recomputed in Python and verified against the engine;
  a `Game.refresh_caches()` would make `engine_tables.py` unnecessary).

## 11. Results (2026-11-17/18, `runs/st2_v2_s1/latest.pt`, Red / White, deterministic policy)

Logs: `runs/bridge/main1.jsonl` (games 0-9, seeds `BRG000`..`BRG009`), `main2.jsonl` (10-17), `main3.jsonl`
(18-25); `dbg_*`, `endless*` are debugging / endless runs. Games are 6-9 minutes of wall clock each
(about 180 decisions, 45 played hands). Seeds are `seed_for("BRG", i)`.

| batch | code state | games | wins | notes |
|---|---|---|---|---|
| main1 | `sell` refused in packs (BalatroBot limitation), 4-sample prediction | 10 | 4 / 9 completed | game 7 looped 3260 rejected `SellJoker` in an open pack (`max_decisions`); 20 rejects in game 0 |
| main2 | `sell` allowed in packs (patch), use-retry | 8 | 8 / 8 | |
| main3 | + well-spread prediction seeds, mismatch dumps | 8 | 4 / 8 | |
| all | | 26 | 16 / 25 completed = 64 % (main2+main3: 12 / 16 = 75 %) | simulator reference: White 74.2 % (10k holdout); n = 16 gives a standard error of about 11 points, so no significant gap |

No game ended in a BalatroBot hang in the non-endless runs (the three known hang bugs are masked by
`known_bug_blocks`, which fired in 4-8 states per game). Rejected actions of the real game: `BuyAndUse`
with full consumable slots and `Buy` of a joker with full joker slots (BalatroBot's endpoint pre-check
refuses both although the engine allows them, probably Negative-edition shop cards; the rejection path
masks them), a few `use` calls the game refuses while still animating (retried after 1.5 s).
Engine and real **shop prices agreed in all 2034 price-checked shop states of main1-3** (`price_check`; zero `price_mismatch`).

### Score parity (1117 played hands in main1-3)

| | hands | share |
|---|---|---|
| exact (engine prediction == real chips) | 948 | 84.9 % |
| in range (stochastic hand, real inside the engine's min..max) | 145 | 13.0 % |
| `stochastic_miss` | 17 | 1.5 % |
| `mismatch` (deterministic prediction differs) | 7 | 0.6 % |
| deterministic hands exact | 855 / 862 | 99.2 % |
| hand type agrees | 1117 / 1117 | 100 % |

main3 (all fixes in): 344 hands, 317 exact, 25 in range, 2 `stochastic_miss`, **0 deterministic
mismatches in 284 deterministic hands**. The two main3 `stochastic_miss` were reproduced offline
(`scripts/bridge_replay_mismatch.py`) and are in range once the prediction seeds are well spread.

Classification of every non-exact deterministic or out-of-range hand:

| class | n | cause | status |
|---|---|---|---|
| **bridge** (transport precision) | 4 (+1 chip each) | BalatroBot's JSON encoder keeps 14 significant digits: Constellation `x_mult` 1.2000000000000002 arrives as 1.2, the final `floor` then differs by 1. All four had Constellation (1.2, 1.2, 1.6, 2.4) | fixed: `state._snap` replays the accumulation (Constellation, Vampire, Obelisk, Ramen); 0 such cases in main3 |
| **bridge** (prediction sampling) | 17 (`stochastic_miss`) | all with Misprint (+0..+23 mult) or Space Joker; the 4 / 64 reseeded copies did not reach the extremes. Root cause: `GameRng::new(seed)` of *consecutive small* seeds gives correlated streams (256 consecutive seeds never produced Misprint's extreme rolls) | fixed: golden-ratio seed stride (`parity._mix`) + 256 copies for stochastic hands; the 2 hands with dumps are in range (real 4824 within 1876..4958, real 1824 within 152..1900). The 15 hands of main1/main2 have no dump; their pattern (real just outside a 4-sample range, diff = exactly one mult step times chips) matches |
| **BalatroBot** | 0 hand mismatches | see section 9 | |
| **suspected simulator** (unresolved) | 2 + 2 | see below | no dumps existed for main1/main2; not seen again in main3 (284 deterministic hands) |

Suspected simulator issues (evidence, all from `main2.jsonl`, game 14 / 15; the log records cards, held
cards were not logged yet):

1. **Ramen, -1 chip** (g15 round a3: pred 772, real 771, jokers Raised Fist + Ramen + Walkie Talkie +
   Burglar; g15 a8: pred 22842, real 22841, Ramen + Abstract + Supernova + Crafty + Banner). Both
   deterministic, real is one lower. Ramen's `x_mult` decrements by 0.01 per discarded card in the game
   in double precision (`1.9900000000000002`...) while the engine rounds it to hundredths after
   every card (`ramen_discard`), and the translation hands over the *reconstructed* real value, so the
   most likely cause is the engine's rounding of Ramen / the final `floor` differing at the 1e-12 level.
   Repro: `bridge_play.py --games 1 --first-game 15` plays the same seed; or
   `scripts/bridge_joker_sweep.py --only j_ramen` after discarding.
2. **The Hook + Raised Fist** (g14 a7: pred 41310 [39690..41310], real 36450; g15 a1: pred 2732
   deterministic, real 2494; Raised Fist in both). In the real game The Hook discards 2 random cards
   *held in hand* before scoring (`Blind:press_play`, an event queued after the played cards moved), then
   Raised Fist (`temp_ID >= id`, i.e. the **rightmost** lowest card, nominal x2) looks at the remaining
   held cards. The engine also discards before scoring (`hook_before_play`). Real being *lower* than any
   engine outcome cannot come from fewer held cards (the lowest rank can only rise), so the real game
   probably also lets the Hook pick from cards that are already part of the play, or scores Raised
   Fist on a different card. Not verified; needs a dump of a Hook hand (`.mismatch/` now exists) or a
   targeted scenario (the boss cannot be forced through the API).

`bridge_report.py` lists these hands (`== mismatches`), grouped by joker / boss / enhancement.

### Per-joker sweep (`scripts/bridge_joker_sweep.py`)

One hand per joker in a fresh run (`add` after the blind is selected): the first 47 jokers of the
table (Joker .. Burglar, `runs/bridge/sweep1.jsonl`) gave 94 plays, all `exact` or `in_range`
(Misprint, Space Joker); Ramen, Raised Fist and Constellation were also exact in the smoke run. The
sweep is slow (about 2 jokers per minute) and was stopped there for time; the rest is open.

### Endless mode (`--endless`, EXPERIMENTAL, default off)

After the ante-8 win the game shows the win screen (paused game, "Endless" button =
`G.FUNCS.exit_overlay_menu`), BalatroBot has no call for it, so the patch adds `continue_endless`.
Status: works in short scripted scenarios (`set ante 8`, `set chips`, play one card, with and without
end-of-round joker rows) and in one organic game (seed `BRG026`: won, continued 3 rounds, lost at the
next boss, `endless_ante` logged). In organic game `BRG027` (win at the ante-8 boss) the
following `cash_out` **crashes the game** (`functions/common_events.lua:1148: attempt to index field
'round_eval' (a nil value)`, reproduced twice, Lua crash screen; the API then times out). A second
variant of the endpoint that waited for the cash-out button and an idle event queue made
`continue_endless` itself time out (the game was alive, in `ROUND_EVAL`, and a plain `cash_out` ten
minutes later worked). Best hypothesis: the win overlay pauses the game; after it is dismissed the
end-of-round rows (`add_round_eval_row` events) are still being added for many seconds (many held
cards / jokers) and `G.FUNCS.cash_out` removes `G.round_eval` under them. The runner therefore sleeps
`endless_wait` (25 s) after `continue_endless`; this last variant is **untested**. Default runs never
use any of it.



## 12. How to run

```
bridge/apply_lua_patch.sh            # once; backs up the installed originals
bridge/restart_game.sh               # (re)starts Balatro with BalatroBot, no --fast
uv run --no-sync python scripts/bridge_play.py --checkpoint /abs/path/latest.pt --games 3 \
    --deck RED --stake WHITE --seed-prefix BRG --log runs/bridge/run.jsonl
uv run --no-sync python scripts/bridge_report.py runs/bridge/run.jsonl
```

`--policy random` plays random legal actions (bridge debugging, no model needed);
`scripts/bridge_replay_mismatch.py DUMP` reproduces a mismatching hand offline from `<log>.mismatch/`;
`scripts/bridge_joker_sweep.py` plays one hand per joker (`add` endpoint) for per-joker parity;
`--fixture-dir DIR` saves a few gzipped gamestates per phase (the recorded fixtures of
`tests_py/fixtures/bridge/` come from there).
