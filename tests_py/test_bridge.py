"""Real-game bridge (Phase 11 / B1) tests that run WITHOUT the game.

* mapping completeness: every engine id has a BalatroBot key and vice versa (checked against
  the engine's own serde names via ``from_state`` / ``start_joker``),
* derived caches (joker ``mods``, voucher ``shop_mods``) equal the engine's over many states,
* state translation on recorded real gamestates (``tests_py/fixtures/bridge/*.json.gz``),
* action translation (engine action -> API calls).
"""

from __future__ import annotations

import copy
import csv
import gzip
import json
from pathlib import Path

import pytest
from balatro_rl import _engine
from balatro_rl.bridge import actions as A
from balatro_rl.bridge import engine_tables as et
from balatro_rl.bridge import parity
from balatro_rl.bridge import tables as T
from balatro_rl.bridge.client import ApiError, Client
from balatro_rl.bridge.runner import seed_for
from balatro_rl.bridge.state import (
    TranslateError,
    build_game,
    translate,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = sorted((Path(__file__).parent / "fixtures" / "bridge").glob("*.json.gz"))


def load_fixture(path: Path) -> dict:
    with gzip.open(path, "rt") as f:
        return json.load(f)


def teacher_states(seed: int, steps: int = 400):
    g = _engine.Game(seed)
    for _ in range(steps):
        if g.is_terminal():
            break
        yield json.loads(g.export_state()), g
        g.step_parts(*g.teacher_parts())


# --------------------------------------------------------------------------- mappings


def test_table_sizes():
    assert len(T.JOKERS) == 150
    assert len(T.CONSUMABLES) == 52
    assert len(T.VOUCHERS) == 32
    assert len(T.TAGS) == 24
    assert len(T.BOSSES) == 28
    assert len(T.HAND_TYPES) == 12
    for table in (T.JOKERS, T.CONSUMABLES, T.VOUCHERS, T.TAGS, T.BOSSES, T.HAND_TYPES):
        keys = [k for k, _ in table]
        ids = [v for _, v in table]
        assert len(set(keys)) == len(keys) and len(set(ids)) == len(ids)


def test_joker_table_matches_the_mechanics_csv():
    with open(ROOT / "docs/mechanics/jokers.csv") as f:
        rows = list(csv.DictReader(f))
    assert [r["id"] for r in rows] == list(T.JOKER_IDS)
    for k, _ in T.JOKERS:
        assert k.startswith("j_")


def test_every_joker_id_exists_in_the_engine():
    for jid in T.JOKER_IDS:  # serde rejects unknown variants
        s = json.loads(_engine.Game(1, start_joker=jid).export_state())
        assert s["jokers"][0]["id"] == jid


def test_ids_are_valid_engine_names_and_in_engine_order():
    base = json.loads(_engine.Game(7).export_state())
    # consumables / vouchers / tags / bosses / hand types: serde accepts exactly the names
    s = copy.deepcopy(base)
    s["consumables"] = [{"id": c, "negative": False, "sell_value": 1} for c in T.CONSUMABLE_IDS[:2]]
    s["vouchers"] = []
    s["tags"] = list(T.TAG_IDS)
    s["boss"] = T.BOSS_IDS[3]
    s["last_hand"] = T.HAND_IDS[5]
    s["last_tarot_planet"] = T.CONSUMABLE_IDS[10]
    _engine.Game.from_state(json.dumps(s))
    for cid in T.CONSUMABLE_IDS:
        s["consumables"] = [{"id": cid, "negative": False, "sell_value": 1}]
        s["consumables"] *= 1
        s["consumable_slots"] = 2
        _engine.Game.from_state(json.dumps(s))
    for bid in T.BOSS_IDS:
        s["boss"] = bid
        _engine.Game.from_state(json.dumps(s))
    for h in T.HAND_IDS:
        s["last_hand"] = h
        _engine.Game.from_state(json.dumps(s))
    # order: the engine indexes boss_seen / hand_plays_run / consumable_use_counts by id index
    hit_boss = hit_hand = hit_cons = 0
    for seed in range(12):
        for st, _ in teacher_states(seed, 300):
            if st["boss"] is not None:
                assert st["boss_seen"][T.BOSS_IDS.index(st["boss"])] >= 1
                hit_boss += 1
            if st["last_hand"]:
                assert st["hand_plays_run"][T.HAND_IDS.index(st["last_hand"])] >= 1
                hit_hand += 1
            ltp = st["last_tarot_planet"]
            if ltp:
                assert st["consumable_use_counts"][T.CONSUMABLE_IDS.index(ltp)] >= 1
                hit_cons += 1
    assert hit_boss and hit_hand and hit_cons


def test_voucher_order_matches_engine_requires():
    # the engine's rule: odd index = upgrade requiring the previous (base) voucher
    facts = et.voucher_facts()  # asserts every voucher can be bought with its base owned
    assert set(facts) == set(T.VOUCHER_IDS)
    assert facts["OverstockPlus"]["card_slots"] == 2
    assert facts["Hone"]["edition_mult"] == 2 and facts["GlowUp"]["edition_mult"] == 4


def test_pack_specs():
    assert T.pack_spec("p_buffoon_jumbo_1") == {"kind": "Buffoon", "size": "Jumbo"}
    assert T.pack_spec("p_arcana_mega_2") == {"kind": "Arcana", "size": "Mega"}
    assert T.pack_spec("p_standard_normal_4") == {"kind": "Standard", "size": "Normal"}
    with pytest.raises(KeyError):
        T.pack_spec("p_nonsense_1")


def test_tag_and_boss_names():
    assert T.TAG_NAME_TO_ID["Skip Tag"] == "Speed"
    assert set(T.TAG_NAME_TO_ID.values()) == set(T.TAG_IDS)
    assert set(T.BOSS_NAME_TO_ID.values()) == set(T.BOSS_IDS)


def test_seed_for():
    assert seed_for("brg", 7) == "BRG007"
    assert len(seed_for("LONGPREFIX", 3)) <= 8


# --------------------------------------------------------------------------- derived caches


def test_joker_mods_fold_equals_engine():
    checked = 0
    for seed in range(40, 70):
        for st, _ in teacher_states(seed, 350):
            assert et.fold_joker_mods(st["jokers"]) == st["mods"]
            checked += bool(st["jokers"])
    assert checked > 50


def test_voucher_fold_equals_engine():
    seen = 0
    for seed in range(80, 130):
        for st, _ in teacher_states(seed, 500):
            assert et.fold_shop_mods(st["vouchers"]) == st["shop_mods"]
            seen += bool(st["vouchers"])
    assert seen > 20


def test_every_joker_mods_roundtrip_through_from_state():
    # all 150 jokers as the only joker: derived mods must satisfy the engine's invariant
    base = json.loads(_engine.Game(5).export_state())
    for jid in T.JOKER_IDS:
        j = json.loads(_engine.Game(1, start_joker=jid).export_state())["jokers"][0]
        s = copy.deepcopy(base)
        s["jokers"] = [j]
        s["next_joker_uid"] = 1
        s["mods"] = et.fold_joker_mods(s["jokers"])
        _engine.Game.from_state(json.dumps(s))


# --------------------------------------------------------------------------- actions


def fake_tr(n_jokers=3, hand_order=None):
    from balatro_rl.bridge.state import Translation

    order = hand_order or list(range(8))
    return Translation(
        state={"jokers": [{} for _ in range(n_jokers)]},
        phase="Play",
        hand_real_index=order,
        hand_needs_rearrange=order != sorted(order),
    )


def test_move_joker_permutation_matches_engine_semantics():
    for n in range(2, 6):
        for frm in range(n):
            for to in range(n):
                jokers = list(range(n))
                j = jokers.pop(frm)
                jokers.insert(to, j)
                perm = A.joker_move_permutation(n, frm, to)
                assert [list(range(n))[p] for p in perm] == jokers


def test_plan_simple_actions():
    tr = fake_tr()
    p = A.plan_action
    assert [(c.method, c.params) for c in p([A.SELECT_BLIND, 0, 0, 0], tr)] == [("select", {})]
    assert [(c.method) for c in p([A.SKIP_BLIND, 0, 0, 0], tr)] == ["skip"]
    assert [(c.method, c.params) for c in p([A.SELL_JOKER, 2, 0, 0], tr)] == [
        ("sell", {"joker": 2})
    ]
    assert [(c.method, c.params) for c in p([A.SELL_CONSUMABLE, 1, 0, 0], tr)] == [
        ("sell", {"consumable": 1})
    ]
    assert [c.method for c in p([A.END_SHOP, 0, 0, 0], tr)] == ["next_round"]
    assert [(c.method, c.params) for c in p([A.BUY, 1, 0, 0], tr)] == [("buy", {"card": 1})]
    assert [(c.method, c.params) for c in p([A.OPEN_PACK, 0, 0, 0], tr)] == [("buy", {"pack": 0})]
    assert [(c.method, c.params) for c in p([A.BUY_VOUCHER, 0, 0, 0], tr)] == [
        ("buy", {"voucher": 0})
    ]
    assert [c.method for c in p([A.REROLL, 0, 0, 0], tr)] == ["reroll"]
    assert [c.method for c in p([A.REROLL_BOSS, 0, 0, 0], tr)] == ["reroll_boss"]
    assert [(c.method, c.params) for c in p([A.SKIP_PACK, 0, 0, 0], tr)] == [
        ("pack", {"skip": True})
    ]
    bau = p([A.BUY_AND_USE, 1, 0, 0], tr)
    assert [(c.method, c.tag) for c in bau] == [("buy", ""), ("use", "use_new_consumable")]
    mv = p([A.MOVE_JOKER, 0, 2, 0], tr)
    assert mv[0].method == "rearrange" and mv[0].params == {"jokers": [1, 2, 0]}


def test_plan_hand_actions_rearrange_to_engine_order():
    # engine position p lives at real index order[p]; Play first rearranges the real hand
    order = [2, 0, 1, 3, 4, 5, 6, 7]
    tr = fake_tr(hand_order=order)
    calls = A.plan_action([A.PLAY, 0, 0, 0b101], tr)
    assert [c.method for c in calls] == ["rearrange", "play"]
    assert calls[0].params == {"hand": order}
    assert calls[1].params == {"cards": [0, 2]}  # positions after the rearrange
    # Discard needs no rearrange: it maps positions to real indices
    calls = A.plan_action([A.DISCARD, 0, 0, 0b101], tr)
    assert [(c.method, c.params) for c in calls] == [("discard", {"cards": [2, 1]})]
    # a consumable with targets and a pack pick with targets
    calls = A.plan_action([A.USE_CONSUMABLE, 1, 0, 0b11], tr)
    assert [c.method for c in calls] == ["rearrange", "use"]
    assert calls[1].params == {"consumable": 1, "cards": [0, 1]}
    calls = A.plan_action([A.PICK_PACK_ITEM, 2, 0, 0b1], tr)
    assert calls[-1].method == "pack" and calls[-1].params == {"card": 2, "targets": [0]}
    # no rearrange when the hand is already in engine order
    calls = A.plan_action([A.PLAY, 0, 0, 0b11], fake_tr())
    assert [c.method for c in calls] == ["play"]


def test_decode_action_covers_all_kinds():
    assert len(A.KIND_NAMES) == 17
    for k in range(17):
        assert A.decode_action([k, 0, 0, 0])["kind"] == A.KIND_NAMES[k]
    assert A.decode_action([A.PLAY, 0, 0, 0b1001])["cards"] == [0, 3]


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        A.plan_action([99, 0, 0, 0], fake_tr())


# --------------------------------------------------------------------------- client


def test_client_error_names():
    e = ApiError("play", {"cards": [0]}, -32003, "nope", "NOT_ALLOWED")
    assert e.name == "NOT_ALLOWED" and "NOT_ALLOWED" in str(e)
    assert Client("http://127.0.0.1:1", timeout=0.2).health() is False


# --------------------------------------------------------------------------- state translation


@pytest.mark.skipif(not FIXTURES, reason="no recorded gamestates")
@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_fixture_translates_and_builds(path):
    gs = load_fixture(path)
    tr = translate(gs)
    assert tr.phase in ("BlindSelect", "Play", "Shop", "PackOpen", "GameOver")
    g = build_game(tr)
    obs, masks = g.observation(), g.masks()
    assert masks["kind_mask"].any()
    # engine state facts equal the real ones
    s = tr.state
    assert s["money"] == gs["money"]
    assert s["ante"] == gs["ante_num"]
    assert len(s["jokers"]) == len(gs["jokers"]["cards"])
    assert len(s["consumables"]) == len(gs["consumables"]["cards"])
    assert sorted(tr.hand_real_index) == list(range(len(gs["hand"]["cards"])))
    assert len(s["deck"]["draw_pile"]) == len(gs["cards"]["cards"])
    assert len(s["deck"]["hand"]) == len(gs["hand"]["cards"])
    # observation: the hand the model sees is the real hand
    n_hand = int(obs["hand_valid"].sum())
    assert n_hand == len(gs["hand"]["cards"])
    assert int(obs["joker_valid"].sum()) == len(gs["jokers"]["cards"])
    # every legal action of the engine can be planned into API calls
    for kind, a0, a1, mask in g.legal_parts()[:200]:
        calls = A.plan_action([kind, a0, a1, mask], tr)
        assert calls and all(c.method for c in calls)


@pytest.mark.skipif(not FIXTURES, reason="no recorded gamestates")
@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_fixture_engine_can_step_a_legal_action(path):
    tr = translate(load_fixture(path))
    g = build_game(tr)
    legal = g.legal_parts()
    assert legal
    for act in legal[:5]:
        c = g.clone()
        _, _, invalid = c.step_parts(*act)
        assert not invalid


@pytest.mark.skipif(not FIXTURES, reason="no recorded gamestates")
def test_fixture_shop_prices_match_the_real_ones():
    import math

    checked = 0
    for path in FIXTURES:
        gs = load_fixture(path)
        if gs["state"] != "SHOP":
            continue
        g = build_game(translate(gs))
        obs = g.observation()
        for base, area in ((0, "shop"), (8, "packs"), (12, "vouchers")):
            for i, c in enumerate((gs.get(area) or {}).get("cards") or []):
                v = float(obs["offer_f"][base + i][0])
                price = round(math.copysign(math.expm1(5 * abs(v)), v))
                assert price == c["cost"]["buy"], (path.name, area, i, c["key"])
                checked += 1
    assert checked


@pytest.mark.skipif(not FIXTURES, reason="no recorded gamestates")
def test_fixture_play_prediction_runs():
    n = 0
    for path in FIXTURES:
        gs = load_fixture(path)
        if gs["state"] != "SELECTING_HAND":
            continue
        tr = translate(gs)
        g = build_game(tr)
        plays = [p for p in g.legal_parts() if p[0] == A.PLAY]
        pr = parity.predict_play(tr.state, plays[0][3])
        assert not pr["invalid"] and pr["pred"] >= 0
        assert pr["hand_type"] in T.HAND_IDS
        n += 1
    assert n


def test_translate_refuses_gamestate_without_bridge_block():
    gs = {"state": "SHOP", "deck": "RED", "stake": "WHITE"}
    with pytest.raises(TranslateError):
        translate(gs)
    with pytest.raises(TranslateError):
        translate({"state": "ROUND_EVAL"})


def test_compare_verdicts():
    det = {"invalid": False, "pred": 100.0, "min": 100.0, "max": 100.0, "deterministic": True}
    assert parity.compare(det, 100.0)["status"] == "exact"
    assert parity.compare(det, 99.0)["status"] == "mismatch"
    sto = {"invalid": False, "pred": 100.0, "min": 100.0, "max": 160.0, "deterministic": False}
    assert parity.compare(sto, 130.0)["status"] == "in_range"
    assert parity.compare(sto, 200.0)["status"] == "stochastic_miss"
    assert parity.compare({"invalid": True}, 1.0)["status"] == "pred_invalid"


# ---------------------------------------------------------------------- joker state, bug blocks


def test_snap_rebuilds_the_games_float_accumulation():
    from balatro_rl.bridge.state import _snap

    assert _snap(1.2, 1.0, 0.1) == 1.0 + 0.1 + 0.1  # 1.2000000000000002, not 1.2
    assert _snap(1.0, 1.0, 0.1) == 1.0
    assert _snap(1.99, 2.0, -0.01) == 2.0 - 0.01
    assert _snap(1.234, 1.0, 0.1) == 1.234  # not on the grid: untouched


def test_joker_state_mapping():
    from balatro_rl.bridge.state import joker_state

    assert joker_state("RideTheBus", {"mult": 3}, 0.0, 0) == (3.0, 0)
    assert joker_state("Runner", {"extra": {"chips": 30, "chip_mod": 15}}, 0.0, 0) == (30.0, 0)
    assert joker_state("Seltzer", {"extra": 7}, 0.0, 10) == (0.0, 7)
    assert joker_state("InvisibleJoker", {"invis_rounds": 2}, 0.0, 0) == (0.0, 2)
    assert joker_state("LoyaltyCard", {"hands_played_at_create": 9}, 0.0, 0) == (0.0, 9)
    assert joker_state("Hologram", {"x_mult": 1.5}, 1.0, 0) == (1.5, 0)
    assert joker_state("Joker", {"mult": 99}, 0.0, 0) == (0.0, 0)  # stateless: defaults


def test_known_bug_blocks():
    from balatro_rl.bridge.runner import known_bug_blocks
    from balatro_rl.bridge.state import Translation

    state = {
        "jokers": [{"id": "Joker"}, {"id": "InvisibleJoker"}],
        "consumables": [{}],
        "blind": "Small",
        "blind_tags": ["Charm", "Economy"],
        "pack": None,
    }
    tr = Translation(state=state, phase="BlindSelect")
    b = known_bug_blocks(tr)
    assert b[(A.SELL_JOKER, 1)] == "balatrobot#195"
    assert b[(A.SKIP_BLIND, -1)] == "balatrobot#233"
    tr.state["blind"] = "Big"  # the Economy tag is fine
    assert (A.SKIP_BLIND, -1) not in known_bug_blocks(tr)
    items = [{"Consumable": "Mars"}, {"Consumable": "BlackHole"}]
    pack = {"spec": {"kind": "Celestial"}, "items": items}
    tr = Translation(state={**state, "pack": pack}, phase="PackOpen")
    b = known_bug_blocks(tr)
    assert b[(A.PICK_PACK_ITEM, 1)] == "balatrobot#199"
    assert (A.PICK_PACK_ITEM, 0) not in b
    assert (A.USE_CONSUMABLE, 0) in b


def test_prediction_variants_cover_the_misprint_support():
    # regression: consecutive small reseeds gave correlated streams and a truncated range
    found = None
    for st, g in teacher_states(60, 600):
        if st["phase"] == "Play" and st["jokers"] and st["blind"] != "Boss":
            found = (copy.deepcopy(st), g.legal_parts())
            break
    if found is None:
        pytest.skip("no Play state with jokers found")
    st, legal = found
    misprint = json.loads(_engine.Game(1, start_joker="Misprint").export_state())["jokers"][0]
    st["jokers"][0] = misprint
    st["mods"] = et.fold_joker_mods(st["jokers"])
    mask = next(p[3] for p in legal if p[0] == A.PLAY and bin(p[3]).count("1") >= 2)
    pr = parity.predict_play(st, mask)
    assert not pr["deterministic"] and pr["max"] > pr["min"]


# --------------------------------------------------------------------------- real-game float parity


def _play_state(seed: int = 3) -> dict:
    """An engine state in the Play phase with a full hand (a template to overwrite)."""
    g = _engine.Game(seed)
    for _ in range(600):
        st = json.loads(g.export_state())
        if st["phase"] == "Play" and len(st["deck"]["hand"]) == 8:
            return st
        g.step_parts(*g.teacher_parts())
    raise AssertionError("no Play state reached")


def _card(rank: str, suit: str) -> tuple[str, str]:
    return rank, suit


def _joker(uid: int, jid: str, value: float = 0.0, edition: str = "Base", debuffed=False) -> dict:
    return {
        "uid": uid, "id": jid, "edition": edition,
        "stickers": {"eternal": False, "perishable": False, "perish_rounds_left": 0,
                     "rental": False},
        "debuffed": debuffed, "sell_value": 2, "value": value, "counter": 0, "aux": 0,
    }  # fmt: skip


def _ramen(discarded: int) -> float:
    """Ramen after ``discarded`` cards, accumulated like the game (plain double subtraction)."""
    v = 2.0
    for _ in range(discarded):
        v -= 0.01
    return v


def _fresh(st: dict, hand: list[tuple[str, str]], jokers: list[dict]) -> dict:
    st = json.loads(json.dumps(st))
    # overwrite the first cards of the template hand (uids stay: the deck invariants hold)
    for c, (rank, suit) in zip(st["deck"]["hand"], hand, strict=True):
        c.update(rank=rank, suit=suit, enhancement="None", edition="Base", seal="None",
                 perma_bonus=0, debuffed=False, face_down=False)  # fmt: skip
    st["jokers"] = jokers
    st["next_joker_uid"] = 100
    st["boss_disabled"] = False
    st["blind"] = "Small"  # no boss in effect
    st["round_score"] = 0.0
    st["target"] = 1e12
    st["hands_left"], st["discards_left"] = 4, 4
    st["hand_levels"] = [1] * 12
    st["hand_plays_run"] = [0] * 12
    st["hand_plays_round"] = [0] * 12
    return st


def test_real_game_ramen_ante3_scores_771():
    """Bridge game 15 ante 3: real 771 (the engine said 772). Pair of fives, Raised Fist on a held
    9 (+18), Ramen 1.93: the game computes mult as ``m + m * (x - 1)`` (SMODS), 20 + 20 * 0.93 =
    38.599999999999994, so 20 chips score 771.99.. -> 771."""
    st = _fresh(
        _play_state(),
        [_card("King", "Clubs"), _card("Jack", "Diamonds"), _card("Six", "Spades"),
         _card("Five", "Hearts"), _card("Five", "Clubs"), _card("Nine", "Spades"),
         _card("Queen", "Hearts"), _card("King", "Hearts")],
        [_joker(1, "RaisedFist"), _joker(2, "Ramen", _ramen(7)), _joker(3, "WalkieTalkie"),
         _joker(4, "Burglar")],
    )  # fmt: skip
    pr = parity.predict_play(st, 0b11111)
    assert pr["deterministic"] and pr["hand_type"] == "Pair"
    assert pr["pred"] == 771.0


def test_real_game_ramen_ante8_scores_22841():
    """Bridge game 15 ante 8: real 22841 (the engine said 22842). Pair L11, two tens, Abstract
    +15, Supernova debuffed (The Crimson Heart), Crafty Joker Polychrome, Banner (4 discards left
    = +120 chips), Ramen 1.88: 300 x ((12 + 15) x 1.5 x 1.88) = 22841.99.. -> 22841."""
    st = _fresh(
        _play_state(),
        [_card("Ten", "Clubs"), _card("Ten", "Diamonds"), _card("Nine", "Hearts"),
         _card("Eight", "Clubs"), _card("Five", "Hearts"), _card("Two", "Spades"),
         _card("Two", "Hearts"), _card("Three", "Hearts")],
        [_joker(1, "AbstractJoker"), _joker(2, "Supernova", debuffed=True),
         _joker(3, "CraftyJoker", edition="Polychrome"), _joker(4, "Banner"),
         _joker(5, "Ramen", _ramen(12))],
    )  # fmt: skip
    st["hand_levels"][1] = 11  # Pair
    pr = parity.predict_play(st, 0b11111)
    assert pr["deterministic"] and pr["hand_type"] == "Pair"
    assert pr["pred"] == 22841.0
