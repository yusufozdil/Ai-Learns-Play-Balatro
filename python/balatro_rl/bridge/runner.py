"""Play real Balatro through BalatroBot with a policy of the simulator.

Loop (one iteration = one decision of the policy)::

    gamestate --translate--> engine state --from_state--> Game --obs/masks--> policy
        ^                                                                       |
        +------------- API calls (actions.plan_action) <------ engine action ---+

* ``ROUND_EVAL`` is cashed out immediately (the engine does this inside the winning Play).
* Before every Play the engine predicts the chips of the hand (``parity.predict_play``); the real
  chips gained are read from the response and both go to the JSONL log.
* An action the real game rejects (``ApiError``) is masked out and the policy is asked again;
  timeouts are the known BalatroBot hangs: the game is abandoned and recorded as ``hang``.

Log: one JSON object per line, ``ev`` in ``game_start`` / ``decision`` / ``play`` / ``reject`` /
``warn`` / ``game_end``.
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import actions as A
from . import parity
from . import tables as T
from .client import ApiError, BridgeError, BridgeTimeout, Client
from .state import (
    TRANSIENT_STATES,
    TranslateError,
    Translation,
    build_game,
    strip_game_text,
    translate,
)

MAX_REJECTS_PER_STATE = 12
MAX_REJECTS_PER_GAME = 300
MAX_DECISIONS = 4000
WIN_ANTE = 8  # the ante whose boss ends a normal run (G.GAME.win_ante)


# --------------------------------------------------------------------------- policies


class RandomPolicy:
    """Uniformly random legal engine action (debugging the bridge without a model)."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def reset(self) -> None:
        pass

    def act(self, game, blocked: set[tuple[int, int]]) -> tuple[list[int], dict]:
        for _ in range(64):
            kind, a0, a1, mask = game.random_parts()
            if (kind, a0) not in blocked and (kind, -1) not in blocked:
                return [kind, a0, a1, mask], {}
        legal = [p for p in game.legal_parts() if (p[0], p[1]) not in blocked]
        p = legal[0] if legal else game.legal_parts()[0]
        return [p[0], p[1], p[2], p[3]], {}


class ModelPolicy:
    """The trained team model, deterministic, memory carried across the game."""

    name = "model"

    def __init__(self, checkpoint: str, device: str = "cpu", deterministic: bool = True):
        import torch

        from balatro_rl.model import load_checkpoint

        self.torch = torch
        self.device = device
        self.model, _ = load_checkpoint(checkpoint, device)
        self.model.eval()
        self.deterministic = deterministic
        self.gen = torch.Generator(device=device).manual_seed(0)
        self.hidden = None
        self.first = True

    def reset(self) -> None:
        self.hidden = self.model.init_hidden(1)
        self.first = True

    def act(self, game, blocked: set[tuple[int, int]]) -> tuple[list[int], dict]:
        torch = self.torch
        obs = {k: v[None] for k, v in game.observation().items()}
        masks = {k: np.array(v)[None] for k, v in game.masks().items()}
        km = masks["kind_mask"]
        a0m = masks["arg0_mask"]
        for kind, a0 in blocked:
            if a0 < 0:
                km[0, kind] = False
            else:
                a0m[0, kind, a0] = False
                if not a0m[0, kind].any():
                    km[0, kind] = False
        if not km.any():  # everything was rejected: let the engine mask decide again
            km[...] = np.array(game.masks()["kind_mask"])[None]
        f = torch.tensor([self.first], dtype=torch.bool, device=self.device)
        a, _, _, v, h = self.model.act(
            obs, masks, self.hidden, self.deterministic, first=f, generator=self.gen
        )
        self.hidden = h
        self.first = False
        return [int(x) for x in a[0].cpu().numpy()], {"value": float(v[0])}


# --------------------------------------------------------------------------- the runner


@dataclass
class GameResult:
    game: int
    seed: str
    outcome: str  # win | loss | hang | error | max_decisions
    ante: int = 0
    round: int = 0
    decisions: int = 0
    rejects: int = 0
    seconds: float = 0.0
    plays: int = 0
    exact: int = 0
    mismatch: int = 0
    note: str = ""
    won_ante: int | None = None  # ante at which the run was won (ante-8 boss beaten)
    endless_ante: int | None = None  # highest ante reached in endless mode (--endless only)
    # how the endless continuation of a won run ended (--endless only): loss | cap | crash | hang
    # | max_decisions | error; `crash` = the game died / the API broke, `hang` = no answer
    endless_end: str | None = None
    extra: dict = field(default_factory=dict)


class Runner:
    def __init__(
        self,
        client: Client,
        policy,
        log_path: str | Path,
        *,
        predict: bool = True,
        verbose: bool = True,
        settle_timeout: float = 90.0,
        fixture_dir: str | Path | None = None,
        fixture_limit: int = 3,
        endless: bool = False,
        endless_max_ante: int = 12,
        endless_wait: float = 2.0,
        cash_out_timeout: float = 90.0,
        state_dump_dir: str | Path | None = None,
    ):
        self.state_dump_dir = Path(state_dump_dir) if state_dump_dir else None
        self.cash_out_timeout = cash_out_timeout
        self.endless = endless
        self.endless_max_ante = endless_max_ante
        self.endless_wait = endless_wait
        self._continued = False
        self._expect_money: int | None = None
        self.c = client
        self.policy = policy
        self.predict = predict
        self.verbose = verbose
        self.settle_timeout = settle_timeout
        self.fixture_dir = Path(fixture_dir) if fixture_dir else None
        self.fixture_limit = fixture_limit
        self._fixture_count: dict[str, int] = {}
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "a", buffering=1)  # noqa: SIM115 - closed in close()

    # -- logging -----------------------------------------------------------------------

    def log(self, ev: str, **kw: Any) -> None:
        rec = {"ev": ev, "t": round(time.time(), 3), **kw}
        self._log.write(json.dumps(rec, default=str) + "\n")

    def close(self) -> None:
        self._log.close()

    def _maybe_fixture(self, gs: dict) -> None:
        """Save a gzipped copy of a few gamestates per kind (test fixtures)."""
        if self.fixture_dir is None:
            return
        kind = gs["state"]
        pack = (gs.get("bridge") or {}).get("booster") or {}
        if pack.get("key"):
            kind += "_" + pack["key"].rsplit("_", 1)[0]
        n = self._fixture_count.get(kind, 0)
        if n >= self.fixture_limit:
            return
        self._fixture_count[kind] = n + 1
        self.fixture_dir.mkdir(parents=True, exist_ok=True)
        import gzip

        with gzip.open(self.fixture_dir / f"{kind.lower()}_{n}.json.gz", "wt") as f:
            json.dump(strip_game_text(gs), f)

    def _maybe_dump_state(self, gs: dict, res: GameResult, tag: str = "") -> None:
        """``--dump-states``: every gamestate the policy decides on, for offline audits."""
        if self.state_dump_dir is None:
            return
        import gzip

        self.state_dump_dir.mkdir(parents=True, exist_ok=True)
        name = f"g{res.game}_d{res.decisions + 1:04d}_{tag or gs['state'].lower()}.json.gz"
        with gzip.open(self.state_dump_dir / name, "wt") as f:
            json.dump(strip_game_text(gs), f)

    # -- state handling ----------------------------------------------------------------

    def settle(self, gs: dict, res: GameResult | None = None) -> dict:
        """Wait out transitions; cash out ``ROUND_EVAL``. Returns a translatable or final state.

        ``res`` (optional) gets ``won_ante`` / ``endless_ante`` the moment a win is seen, so that
        whatever happens during the endless continuation cannot lose the win.
        """
        t0 = time.time()
        while True:
            st = gs["state"]
            if res is not None:
                res.ante = max(res.ante, int(gs.get("ante_num", 0)))
                note_win(res, gs)
            if st in ("GAME_OVER", "MENU") or (gs.get("won") and not self.endless):
                return gs
            if st == "ROUND_EVAL":
                if gs.get("won") and not self._continued:
                    # win screen: press "Endless" (bridge patch endpoint), then cash out
                    gs = self.c.call("continue_endless", timeout=120)
                    self._continued = True
                    time.sleep(self.endless_wait)
                    gs = self.c.gamestate()
                    continue
                gs = self.wait_cash_out_ready(gs)
                if gs["state"] != "ROUND_EVAL":
                    continue
                if res is not None:
                    self._maybe_dump_state(gs, res, tag="roundeval")
                before_money = int(gs.get("money", 0))
                rows = (gs.get("bridge") or {}).get("eval_rows")
                gs = self.c.call("cash_out")
                self._check_payout(res, gs, before_money, rows)
                continue
            if st in TRANSIENT_STATES or (gs.get("bridge") or {}).get("controller_locked"):
                if time.time() - t0 > self.settle_timeout:
                    raise BridgeTimeout(f"stuck in transient state {st}")
                time.sleep(0.4)
                gs = self.c.gamestate()
                continue
            return gs

    def _check_payout(self, res, gs: dict, money_before: int, rows) -> None:
        """The engine pays the round out inside the winning Play: compare with the real cash-out."""
        expect, self._expect_money = self._expect_money, None
        if res is None or expect is None or gs.get("state") != "SHOP":
            return
        res.extra["payout_checks"] = res.extra.get("payout_checks", 0) + 1
        real = int(gs.get("money", 0))
        if real != expect:
            res.extra["payout_mismatch"] = res.extra.get("payout_mismatch", 0) + 1
            self.log("warn", game=res.game, msg="payout_mismatch", engine=expect, real=real,
                     money_before=money_before, rows=rows)  # fmt: skip

    def wait_cash_out_ready(self, gs: dict) -> dict:
        """Poll until the end-of-round rows are built (``bridge.cash_out_ready``).

        Stock ``play`` returns early once ``G.GAME.won`` is set, i.e. for EVERY round of an
        endless run (and after the win screen); a ``cash_out`` sent while rows are still being
        added crashes the game (docs/design/bridge.md, Endless mode). Normal rounds are ready
        when ``play`` returns. A patch older than version 4 has no flag: no waiting.
        """
        t0 = time.time()
        while True:
            b = gs.get("bridge") or {}
            if gs["state"] != "ROUND_EVAL" or b.get("cash_out_ready", True):
                return gs
            if time.time() - t0 > self.cash_out_timeout:
                raise BridgeTimeout("round-eval rows never finished (cash_out not ready)")
            time.sleep(0.5)
            gs = self.c.gamestate()

    def go_to_menu(self) -> dict:
        gs = self.c.gamestate()
        if gs["state"] != "MENU":
            gs = self.c.call("menu", timeout=60)
        return gs

    # -- one game ----------------------------------------------------------------------

    def play_game(self, game_idx: int, deck: str, stake: str, seed: str) -> GameResult:
        t0 = time.time()
        res = GameResult(game=game_idx, seed=seed, outcome="error")
        self.policy.reset()
        self._continued = False
        try:
            self.go_to_menu()
            gs = self.c.call("start", deck=deck, stake=stake, seed=seed, timeout=120)
            patch = (gs.get("bridge") or {}).get("patch_version")
            self.log("game_start", game=game_idx, seed=seed, deck=deck, stake=stake,
                     policy=self.policy.name, patch=patch)  # fmt: skip
            self._loop(gs, res)
        except BridgeTimeout as e:
            res.outcome = "hang"
            res.note = str(e)
            res.extra["fail"] = "hang"
        except BridgeError as e:
            res.outcome = "error"
            res.note = str(e)
            res.extra["fail"] = "crash"
        except Exception as e:  # noqa: BLE001 - never lose the log of a long run
            res.outcome = "error"
            res.note = f"{type(e).__name__}: {e}"
            self.log("warn", game=game_idx, msg="exception", trace=traceback.format_exc())
        finalize_result(res, self.endless)
        res.seconds = time.time() - t0
        self.log("game_end", **res.__dict__)
        if self.verbose:
            print(
                f"[game {game_idx}] {res.outcome} ante={res.ante} decisions={res.decisions} "
                f"plays={res.plays} exact={res.exact} mismatch={res.mismatch} "
                f"rejects={res.rejects} {res.seconds:.0f}s {res.note}",
                flush=True,
            )
        return res

    def _loop(self, gs: dict, res: GameResult) -> None:
        blocked: set[tuple[int, int]] = set()
        last_sig = None
        pack_origin = "Shop"
        prev_action: list[int] | None = None
        while res.decisions < MAX_DECISIONS:
            gs = self.settle(gs, res)
            res.ante = max(res.ante, int(gs.get("ante_num", 0)))
            res.round = int(gs.get("round_num", 0))
            if gs["state"] == "GAME_OVER":
                res.extra["final_won_flag"] = repr(gs.get("won"))
            note_win(res, gs)
            if gs["state"] == "GAME_OVER":
                res.outcome = "win" if res.won_ante is not None else "loss"
                if self.endless and res.won_ante is not None:
                    res.endless_ante = res.ante
                    res.endless_end = "loss"
                return
            if gs.get("won") and not self.endless:
                res.outcome = "win"
                return
            if self.endless and gs.get("won") and res.ante > self.endless_max_ante:
                res.outcome = "win"
                res.endless_ante = res.ante
                res.endless_end = "cap"
                return
            if gs["state"] == "MENU":
                res.outcome = "error"
                res.note = "unexpected MENU"
                return
            self._maybe_fixture(gs)
            self._maybe_dump_state(gs, res)
            # pack origin: a pack opened right after a skip comes from a tag
            if gs["state"] == "SMODS_BOOSTER_OPENED" and prev_action is not None:
                if prev_action[0] == A.SKIP_BLIND:
                    pack_origin = "Tag"
                elif prev_action[0] == A.OPEN_PACK:
                    pack_origin = "Shop"
            try:
                tr = translate(
                    gs, pack_origin=pack_origin, endless=bool(self.endless and gs.get("won"))
                )
                game = build_game(tr)
            except (TranslateError, ValueError, KeyError) as e:
                self.log(
                    "warn", game=res.game, msg="translate failed", err=str(e), state=gs["state"]
                )
                res.outcome = "error"
                res.note = f"translate failed: {e}"
                self._dump_state(gs, res, "translate_fail")
                return
            if gs["state"] == "SHOP":
                self._price_check(game, gs, res)
            sig = (gs["state"], gs["money"], gs.get("round_num"), len(json.dumps(gs.get("round"))))
            sig = (sig, json.dumps(tr.state["jokers"], sort_keys=True), gs["ante_num"],
                   [c.get("id") for c in (gs.get("hand") or {}).get("cards", [])],
                   [c.get("id") for c in (gs.get("shop") or {}).get("cards", [])],
                   [c.get("id") for c in (gs.get("pack") or {}).get("cards", [])])  # fmt: skip
            if sig != last_sig:
                blocked = set()
                last_sig = sig
            bug = known_bug_blocks(tr)
            if bug:
                res.extra["bug_blocks"] = res.extra.get("bug_blocks", 0) + 1
            action, info = self.policy.act(game, blocked | set(bug))
            if bug:
                info = {**info, "bug_blocked": sorted(set(bug.values()))}
            res.decisions += 1
            ok, gs_new = self._execute(action, tr, gs, res, info)
            if ok:
                prev_action = action
                gs = gs_new
                if res.decisions % 50 == 0 and self.verbose:
                    print(f"  game {res.game}: decision {res.decisions} ante {res.ante} "
                          f"state {gs['state']}", flush=True)  # fmt: skip
            else:
                res.rejects += 1
                if res.rejects > MAX_REJECTS_PER_GAME:
                    res.outcome = "error"
                    res.note = "too many rejected actions in the game"
                    return
                kind, a0 = action[0], action[1]
                has_arg = kind in (A.SELL_JOKER, A.USE_CONSUMABLE, A.SELL_CONSUMABLE, A.MOVE_JOKER,
                                   A.BUY, A.BUY_AND_USE, A.OPEN_PACK, A.BUY_VOUCHER,
                                   A.PICK_PACK_ITEM)  # fmt: skip
                blocked.add((kind, a0 if has_arg else -1))
                if len(blocked) > MAX_REJECTS_PER_STATE:
                    res.outcome = "error"
                    res.note = "too many rejected actions in one state"
                    self._dump_state(gs, res, "reject_loop")
                    return
                gs = self.c.gamestate()
        res.outcome = "max_decisions"

    def _dump_mismatch(self, before: dict, mask: int, res: GameResult) -> None:
        """Keep the exact pre-play gamestate of a mismatching hand for offline reproduction."""
        import gzip

        d = self.log_path.with_suffix(".mismatch")
        d.mkdir(exist_ok=True)
        with gzip.open(d / f"g{res.game}_s{res.decisions}.json.gz", "wt") as f:
            json.dump({"mask": mask, "gamestate": strip_game_text(before)}, f)

    def _price_check(self, game, gs: dict, res: GameResult) -> None:
        """Engine prices (from the observation) vs the prices BalatroBot reports."""
        import math

        obs = game.observation()
        offer_f = obs["offer_f"]
        valid = obs["offer_valid"]
        mism = []
        for base, area in ((0, "shop"), (8, "packs"), (12, "vouchers")):
            for i, c in enumerate(_cards(gs, area)):
                row = base + i
                if row >= len(valid) or not valid[row]:
                    continue
                v = float(offer_f[row][0])
                price = round(math.copysign(math.expm1(5 * abs(v)), v))
                real = int((c.get("cost") or {}).get("buy", -1))
                if price != real:
                    mism.append(
                        {"area": area, "slot": i, "key": c["key"], "engine": price, "real": real}
                    )
        gv = float(obs["global_f"][15])
        eng_reroll = round(math.copysign(math.expm1(5 * abs(gv)), gv))
        real_reroll = int((gs.get("round") or {}).get("reroll_cost", -1))
        if eng_reroll != real_reroll:
            mism.append({"area": "reroll", "engine": eng_reroll, "real": real_reroll})
        res.extra["price_checks"] = res.extra.get("price_checks", 0) + 1
        if mism:
            res.extra["price_mismatch"] = res.extra.get("price_mismatch", 0) + 1
            self.log("warn", game=res.game, msg="price_mismatch", items=mism,
                     vouchers=gs.get("used_vouchers"))  # fmt: skip

    def _dump_state(self, gs: dict, res: GameResult, tag: str) -> None:
        p = self.log_path.with_suffix(f".{tag}.g{res.game}.json")
        p.write_text(json.dumps(gs))

    # -- executing an action -----------------------------------------------------------

    def _execute(
        self, action: list[int], tr: Translation, gs: dict, res: GameResult, info: dict
    ) -> tuple[bool, dict]:
        calls = A.plan_action(action, tr)
        pred = None
        if action[0] == A.PLAY and self.predict:
            try:
                pred = parity.predict_play(tr.state, int(action[3]))
            except Exception as e:  # noqa: BLE001
                pred = {"invalid": True, "error": str(e)}
        t0 = time.time()
        new = gs
        done_calls = []
        try:
            for call in calls:
                params = dict(call.params)
                if call.tag == "use_new_consumable":
                    params["consumable"] = len(new["consumables"]["cards"]) - 1
                done_calls.append(Call_str(call.method, params))
                for attempt in range(4):
                    try:
                        new = self.c.call(call.method, **params)
                        break
                    except ApiError as e:
                        # the game is still animating (controller locked): wait and retry
                        if attempt < 3 and "cannot be used at this time" in e.message:
                            res.extra["use_retries"] = res.extra.get("use_retries", 0) + 1
                            time.sleep(1.5)
                            continue
                        raise
        except ApiError as e:
            self.log("reject", game=res.game, step=res.decisions, state=gs["state"],
                     action=A.decode_action(action), calls=done_calls, err=str(e),
                     money=gs["money"], context=_reject_context(action, tr))  # fmt: skip
            if self.verbose:
                print(f"  reject {A.decode_action(action)}: {e}", flush=True)
            return False, gs
        dt = time.time() - t0
        self.log(
            "decision", game=res.game, step=res.decisions, state=gs["state"], ante=gs["ante_num"],
            round=gs["round_num"], money=gs["money"], action=A.decode_action(action),
            calls=done_calls, api_s=round(dt, 3), **info,
        )  # fmt: skip
        if pred is not None:
            self._log_play(action, tr, gs, new, pred, res)
            self._expect_money = (
                pred.get("post_money") if pred.get("post_phase") == "Shop" else None
            )
        return True, new

    def _log_play(self, action, tr, before: dict, after: dict, pred: dict, res: GameResult) -> None:
        real_gain = float((after.get("round") or {}).get("chips", 0) or 0) - float(
            (before.get("round") or {}).get("chips", 0) or 0
        )
        # a won/lost round may reset nothing, but never trust a negative delta
        verdict = parity.compare(pred, real_gain)
        mask = int(action[3])
        hand = tr.state["deck"]["hand"]
        played = [hand[i] for i in range(len(hand)) if (mask >> i) & 1]
        res.plays += 1
        held = [_card_str(c) for i, c in enumerate(hand) if not (mask >> i) & 1]
        if verdict["status"] not in ("exact", "in_range"):
            self._dump_mismatch(before, mask, res)
        if verdict["status"] in ("exact", "in_range"):
            res.exact += 1
        else:
            res.mismatch += 1
        self.log(
            "play", game=res.game, step=res.decisions, ante=before["ante_num"],
            round=before["round_num"], blind=tr.state["blind"], boss=tr.state["boss"],
            boss_disabled=tr.state["boss_disabled"],
            cards=[_card_str(c) for c in played],
            jokers=[
                {"id": j["id"], "ed": j["edition"], "v": j["value"], "c": j["counter"]}
                for j in tr.state["jokers"]
            ],
            held=held,
            real=real_gain, real_hand=parity.real_hand_type(before, after),
            hand_levels=tr.state["hand_levels"], pred=pred, verdict=verdict,
            round_score_before=tr.state["round_score"], target=tr.state["target"],
            money=tr.state["money"], vouchers=tr.state["vouchers"],
        )  # fmt: skip
        if verdict["status"] not in ("exact", "in_range") and self.verbose:
            print(f"  PARITY {verdict['status']}: pred={pred.get('pred')} real={real_gain} "
                  f"hand={pred.get('hand_type')}/{parity.real_hand_type(before, after)} "
                  f"cards={[f'{c['rank']}{c['suit'][0]}' for c in played]} "
                  f"jokers={[j['id'] for j in tr.state['jokers']]}", flush=True)  # fmt: skip


PACK_TAGS = ("Standard", "Charm", "Meteor", "Buffoon", "Ethereal")


def known_bug_blocks(tr: Translation) -> dict[tuple[int, int], str]:
    """Actions that trigger known BalatroBot hangs (docs/design/bridge.md section 9)."""
    out: dict[tuple[int, int], str] = {}
    s = tr.state
    for slot, j in enumerate(s["jokers"]):
        if j["id"] == "InvisibleJoker":  # #195: selling it never returns
            out[(A.SELL_JOKER, slot)] = "balatrobot#195"
    if tr.phase == "PackOpen":  # `use` of a held consumable needs SELECTING_HAND or SHOP
        for slot in range(len(s["consumables"])):
            out[(A.USE_CONSUMABLE, slot)] = "balatrobot:use-in-pack"
    if tr.phase == "BlindSelect":  # `use` needs SELECTING_HAND or SHOP, the engine also allows this
        for slot in range(len(s["consumables"])):
            out[(A.USE_CONSUMABLE, slot)] = "balatrobot:use-in-blind-select"
    if tr.phase == "BlindSelect" and s["blind"] in ("Small", "Big"):
        tag = s["blind_tags"][0 if s["blind"] == "Small" else 1]
        if tag in PACK_TAGS:  # #233: skipping into a tag pack never returns
            out[(A.SKIP_BLIND, -1)] = "balatrobot#233"
    if tr.phase == "PackOpen" and s["pack"] and s["pack"]["spec"]["kind"] == "Celestial":
        for i, it in enumerate(s["pack"]["items"]):
            if it and it.get("Consumable") == "BlackHole":  # #199
                out[(A.PICK_PACK_ITEM, i)] = "balatrobot#199"
    return out


def _reject_context(action, tr: Translation) -> dict:
    """What the engine thought it was doing when the real game said no (for the report)."""
    s = tr.state
    kind, a0 = int(action[0]), int(action[1])
    ctx: dict = {"jokers": len(s["jokers"]), "joker_slots": s["joker_slots"]}
    ctx["consumables"] = len(s["consumables"])
    ctx["consumable_slots"] = s["consumable_slots"]
    cards = s["shop"]["cards"]
    if kind in (A.BUY, A.BUY_AND_USE) and a0 < len(cards):
        item = cards[a0]["kind"]
        if "Joker" in item:
            ctx["item"] = {"joker": item["Joker"]["id"], "edition": item["Joker"]["edition"]}
        else:
            ctx["item"] = item
    return ctx


def _card_str(c: dict) -> str:
    """``AceS+Glass/Foil#Red``: rank+suit, +enhancement, /edition, #seal."""
    s = f"{c['rank']}{c['suit'][0]}"
    if c["enhancement"] != "None":
        s += f"+{c['enhancement']}"
    if c["edition"] != "Base":
        s += f"/{c['edition']}"
    if c["seal"] != "None":
        s += f"#{c['seal']}"
    if c.get("debuffed"):
        s += "!"
    return s


def _cards(gs: dict, area: str) -> list[dict]:
    a = gs.get(area) or {}
    return a.get("cards") or []


def Call_str(method: str, params: dict) -> str:  # noqa: N802
    return f"{method}({params})" if params else method


def note_win(res: GameResult, gs: dict) -> None:
    """Remember the win (ante-8 boss beaten) as soon as the gamestate shows ``won``."""
    if gs.get("won") and res.won_ante is None and gs.get("state") not in ("GAME_OVER", "MENU"):
        # `ante_num` already reads 9 on the win screen (the ante counter advances when the boss is
        # beaten), so the win ante is the game's win_ante, not the number read here
        res.won_ante = WIN_ANTE
        res.ante = max(res.ante, int(gs.get("ante_num", 0)))


def finalize_result(res: GameResult, endless: bool = True) -> None:
    """A game whose ante-8 boss was beaten is a ``win`` whatever happened afterwards.

    Called at the end of every game. If the endless continuation then crashed, hung, hit a
    reject loop, ... the outcome is turned back into ``win`` and ``endless_end`` / ``note`` keep
    the evidence; ``endless_ante`` is the highest ante reached so far (the win ante if the run
    never continued). Games that were never won keep their ``hang`` / ``error`` outcome.
    """
    if res.won_ante is None:
        return
    if res.outcome != "win":
        end = str(res.extra.get("fail") or res.outcome)
        res.note = f"after the win ({end}): {res.note}".strip()
        res.outcome = "win"
        if endless:
            res.endless_end = end
    if endless:
        if res.endless_ante is None:
            res.endless_ante = max(res.ante, res.won_ante)
        if res.endless_end is None:
            res.endless_end = "cap"


def seed_for(prefix: str, index: int) -> str:
    """BalatroBot seed string of game ``index``: ``<prefix><index>`` upper-cased, alphanumeric."""
    s = "".join(ch for ch in f"{prefix}{index:03d}".upper() if ch.isalnum())
    return s[:8]


__all__ = ["GameResult", "ModelPolicy", "RandomPolicy", "Runner", "seed_for", "T"]
