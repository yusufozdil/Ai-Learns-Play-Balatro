"""Outcome / recovery / report logic of the real-game runner, without the game (mock client)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from balatro_rl.bridge import recovery
from balatro_rl.bridge.client import BridgeError, BridgeTimeout
from balatro_rl.bridge.runner import GameResult, Runner, finalize_result, note_win

ROOT = Path(__file__).resolve().parents[1]


def _report_module():
    spec = importlib.util.spec_from_file_location(
        "bridge_report", ROOT / "scripts/bridge_report.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bridge_report"] = mod
    spec.loader.exec_module(mod)
    return mod


class FakePolicy:
    name = "fake"

    def reset(self):
        pass

    def act(self, game, blocked):  # pragma: no cover - never reached in these tests
        raise AssertionError("policy must not be asked")


class FakeClient:
    """Scripted JSON-RPC client: `script` maps method -> result dict or exception."""

    def __init__(self, script: dict):
        self.script = script
        self.calls: list[str] = []

    def call(self, method, timeout=None, **params):
        self.calls.append(method)
        r = self.script[method]
        if isinstance(r, list):  # consecutive answers
            r = r.pop(0) if len(r) > 1 else r[0]
        if isinstance(r, Exception):
            raise r
        return r

    def gamestate(self):
        return self.call("gamestate")

    def health(self):
        return True


def _won_eval(ante=8):
    return {"state": "ROUND_EVAL", "won": True, "ante_num": ante, "round_num": 24, "money": 9}


def _runner(tmp_path, script, endless=True):
    return Runner(
        FakeClient(script), FakePolicy(), tmp_path / "log.jsonl", verbose=False, endless=endless,
        endless_wait=0.0,
    )  # fmt: skip


def _ends(tmp_path):
    recs = [json.loads(x) for x in (tmp_path / "log.jsonl").read_text().splitlines()]
    return [r for r in recs if r["ev"] == "game_end"]


def test_note_win_records_once_and_ignores_game_over():
    res = GameResult(game=0, seed="X", outcome="error")
    note_win(res, {"state": "GAME_OVER", "won": True, "ante_num": 9})
    assert res.won_ante is None
    note_win(res, _won_eval(8))
    note_win(res, _won_eval(9))
    assert res.won_ante == 8 and res.ante == 8


def test_finalize_turns_a_broken_endless_continuation_back_into_a_win():
    for fail, end in (("hang", "hang"), ("crash", "crash")):
        res = GameResult(game=1, seed="X", outcome="hang", ante=8, won_ante=8, note="boom")
        res.extra["fail"] = fail
        finalize_result(res, endless=True)
        assert res.outcome == "win" and res.endless_end == end
        assert res.endless_ante == 8 and "boom" in res.note


def test_finalize_keeps_failures_before_the_win():
    res = GameResult(game=2, seed="X", outcome="hang", ante=5)
    finalize_result(res, endless=True)
    assert res.outcome == "hang" and res.endless_end is None and res.endless_ante is None


def test_finalize_non_endless_sets_no_endless_fields():
    res = GameResult(game=3, seed="X", outcome="win", ante=8, won_ante=8)
    finalize_result(res, endless=False)
    assert res.endless_ante is None and res.endless_end is None


def test_finalize_keeps_highest_endless_ante_and_error_kinds():
    res = GameResult(game=4, seed="X", outcome="error", ante=10, won_ante=8, note="translate")
    finalize_result(res, endless=True)
    assert (res.outcome, res.endless_end, res.endless_ante) == ("win", "error", 10)


def test_play_game_win_survives_continue_endless_timeout(tmp_path):
    r = _runner(
        tmp_path,
        {
            "gamestate": {"state": "MENU"},
            "start": _won_eval(),
            "continue_endless": BridgeTimeout("continue_endless({}) timed out after 120s"),
        },
    )
    res = r.play_game(27, "RED", "WHITE", "BRG027")
    r.close()
    assert res.outcome == "win" and res.won_ante == 8
    assert res.endless_end == "hang" and res.endless_ante == 8
    (end,) = _ends(tmp_path)
    assert end["outcome"] == "win" and end["endless_end"] == "hang"


def test_play_game_win_survives_cash_out_crash(tmp_path):
    r = _runner(
        tmp_path,
        {
            "gamestate": [{"state": "MENU"}, _won_eval(9)],
            "start": _won_eval(),
            "continue_endless": _won_eval(9),
            "cash_out": BridgeError("cash_out({}) connection failed: Remote end closed"),
        },
    )
    res = r.play_game(27, "RED", "WHITE", "BRG027")
    r.close()
    assert res.outcome == "win" and res.endless_end == "crash"
    assert res.endless_ante == 9  # highest ante seen before the crash


def test_play_game_hang_before_the_win_stays_a_hang(tmp_path):
    r = _runner(
        tmp_path,
        {
            "gamestate": {"state": "MENU"},
            "start": {
                "state": "ROUND_EVAL",
                "won": False,
                "ante_num": 3,
                "round_num": 7,
                "money": 1,
            },  # fmt: skip
            "cash_out": BridgeTimeout("cash_out timed out"),
        },
    )
    res = r.play_game(1, "RED", "WHITE", "BRG001")
    r.close()
    assert res.outcome == "hang" and res.won_ante is None and res.endless_end is None


def test_non_endless_win_stops_at_the_win(tmp_path):
    r = _runner(tmp_path, {"gamestate": {"state": "MENU"}, "start": _won_eval()}, endless=False)
    res = r.play_game(0, "RED", "WHITE", "BRG000")
    r.close()
    assert res.outcome == "win" and res.endless_ante is None


def _ends_log(rows):
    return [{"ev": "game_end", **r} for r in rows]


def test_report_endless_stats_counts_wins_and_breakdowns():
    rep = _report_module()
    ends = [
        {"outcome": "win", "won_ante": 8, "endless_ante": 10, "endless_end": "loss"},
        {"outcome": "win", "won_ante": 8, "endless_ante": 12, "endless_end": "cap"},
        {"outcome": "win", "won_ante": 8, "endless_ante": 8, "endless_end": "crash"},
        {"outcome": "win", "won_ante": 8},  # non-endless win: not in the endless stats
        {"outcome": "loss", "ante": 6},
        {"outcome": "hang", "ante": 4},
    ]
    st = rep.endless_stats(ends)
    assert st["runs"] == 3 and st["antes"] == [8, 10, 12]
    assert st["ends"] == {"loss": 1, "cap": 1, "crash": 1}
    assert st["beyond_win"] == 2 and st["ante_hist"] == {8: 1, 10: 1, 12: 1}
    assert rep.endless_stats([{"outcome": "loss", "ante": 3}]) is None


def test_report_script_prints_wins_and_excluded_games(tmp_path, capsys, monkeypatch):
    rep = _report_module()
    log = tmp_path / "r.jsonl"
    rows = _ends_log(
        [
            dict(game=0, seed="A000", outcome="win", ante=10, round=1, decisions=1, rejects=0,
                 seconds=1, plays=0, exact=0, mismatch=0, note="", won_ante=8, endless_ante=10,
                 endless_end="loss"),
            dict(game=1, seed="A001", outcome="win", ante=8, round=1, decisions=1, rejects=0,
                 seconds=1, plays=0, exact=0, mismatch=0, note="after the win (crash): x",
                 won_ante=8, endless_ante=8, endless_end="crash"),
            dict(game=2, seed="A002", outcome="loss", ante=5, round=1, decisions=1, rejects=0,
                 seconds=1, plays=0, exact=0, mismatch=0, note=""),
            dict(game=3, seed="A003", outcome="hang", ante=4, round=1, decisions=1, rejects=0,
                 seconds=1, plays=0, exact=0, mismatch=0, note="t"),
        ]
    )  # fmt: skip
    rows.append({"ev": "restart", "ok": True, "reason": "x"})
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    monkeypatch.setattr(sys, "argv", ["bridge_report", str(log)])
    rep.main()
    out = capsys.readouterr().out
    assert "completed games: 3  win rate: 66.7%" in out
    assert "excluded from the win rate" in out and "A003" in out
    assert "endless ends: {'loss': 1, 'crash': 1}" in out
    assert "game restarts: 1 (1 ok" in out


def test_finished_seeds_ignores_unfinished_and_garbage(tmp_path):
    log = tmp_path / "l.jsonl"
    log.write_text(
        '{"ev": "game_start", "seed": "S1"}\n{"ev": "game_end", "seed": "S0", "game": 0}\n'
        '{"ev": "game_end", "seed": "S2", "game": 2}\n{"ev": "game_en'
    )
    assert recovery.finished_seeds(log) == {"S0", "S2"}
    assert recovery.finished_seeds(tmp_path / "missing.jsonl") == set()


class _Proc:
    stdout = "game up"
    stderr = ""


def test_restart_game_retries_until_healthy():
    class C:
        n = 0

        def wait_healthy(self, s):
            self.n += 1
            return self.n >= 3

    runs = []
    info = recovery.restart_game(C(), attempts=3, run=lambda *a, **k: runs.append(a) or _Proc(),
                                 sleep=lambda s: None)  # fmt: skip
    assert info["ok"] and info["attempts"] == 3 and len(runs) == 3


def test_restart_game_gives_up_and_survives_script_errors():
    class C:
        def wait_healthy(self, s):
            return False

    def boom(*a, **k):
        raise OSError("no script")

    info = recovery.restart_game(C(), attempts=2, run=boom, sleep=lambda s: None)
    assert not info["ok"] and info["attempts"] == 2 and "no script" in info["detail"]


def test_mouth_locked_hand_is_reconstructed_from_only_hand():
    """Real game (ETS002, final200): The Mouth had locked `Pair`, the policy played Two Pair and
    scored 0; the translation used to forget `blind.only_hand` and predicted 934."""
    import gzip

    from balatro_rl.bridge import parity
    from balatro_rl.bridge.state import translate

    path = ROOT / "tests_py/fixtures/bridge_mismatch/mouth_locked_hand.json.gz"
    with gzip.open(path, "rt") as f:
        d = json.load(f)
    tr = translate(d["gamestate"])
    assert tr.state["boss"] == "TheMouth"
    assert tr.state["boss_state"]["hand"] == "Pair"
    pred = parity.predict_play(tr.state, d["mask"])
    assert pred["hand_type"] == "TwoPair" and pred["pred"] == 0.0
