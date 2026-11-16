"""Log parsing (dashboard) and queue bookkeeping (gpu_queue); stdlib only."""
import importlib.util
import json
import math
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dash = _load("dashboard")
gq = _load("gpu_queue")


def _upd(step, wall, **kw):
    return {"type": "update", "update": step // 100, "global_step": step, "wall_seconds": wall, **kw}


def test_logcache_incremental_partial_and_bad_lines(tmp_path):
    log = tmp_path / "log.jsonl"
    cache = dash.LogCache()
    log.write_text(json.dumps(_upd(100, 1.0, ep_win=0.1)) + "\nnot json\n" + json.dumps(_upd(200, 2.0))[:20])
    st = cache.get(log)
    assert [u["global_step"] for u in st["updates"]] == [100]  # partial last line ignored
    with open(log, "a") as f:
        f.write(json.dumps(_upd(200, 2.0))[20:] + "\n")
        f.write(json.dumps({"type": "eval", "global_step": 200, "det_win_rate": 0.4, "new_best": True}) + "\n")
    st = cache.get(log)
    assert [u["global_step"] for u in st["updates"]] == [100, 200]
    assert st["evals"][0]["det_win_rate"] == 0.4
    log.write_text(json.dumps(_upd(5, 0.1)) + "\n")  # rewritten shorter -> reset
    assert [u["global_step"] for u in cache.get(log)["updates"]] == [5]
    assert dash.num(float("nan")) is None


def test_eta_and_downsample():
    ups = [_upd(1000 * i, 10.0 * i) for i in range(1, 21)]
    info = dash.eta_info(ups, 100000)
    assert math.isclose(info["steps_per_sec"], 100.0)
    assert math.isclose(info["eta_seconds"], (100000 - 20000) / 100.0)
    assert 0 < info["progress"] < 1
    ds = dash.downsample(list(range(1000)), 100)
    assert len(ds) == 100 and ds[-1] == 999
    assert dash.eta_info([], 10) == {}


def test_summarize_eval_formats():
    one = dash.App.summarize_eval("x", {"games": 100, "win_rate": 0.5, "mean_ante": 6.0, "truncated": 0.0})
    assert math.isclose(one[0]["se_pct"], 5.0) and one[0]["win_pct"] == 50.0
    multi = dash.App.summarize_eval("y", {"det": {"games": 400, "win_rate": 0.25, "name": "d"}, "sample": {"games": 0},
                                          "checkpoint": "c.pt", "deck": "red", "stake": "gold"})
    assert {r["entry"] for r in multi} == {"det", "sample"}
    assert multi[0]["set"] == "red/gold"
    assert dash.App.summarize_eval("z", [1, 2]) == []


def test_app_state_robust_and_confined(tmp_path):
    runs = tmp_path / "runs"
    (runs / "r1").mkdir(parents=True)
    (runs / "r1" / "config.json").write_text(json.dumps({"total_steps": 1000, "decks": "red"}))
    (runs / "r1" / "log.jsonl").write_text(json.dumps(_upd(500, 5.0, ep_win=0.2)) + "\n{broken\n")
    (runs / "eval").mkdir()
    (runs / "eval" / "bad.json").write_text("{nope")
    (runs / "badcfg").mkdir()
    (runs / "badcfg" / "config.json").write_text("[[[")
    app = dash.App(tmp_path)
    assert (runs / "dashboard" / "roadmap.json").exists()
    st = app.state()
    assert st["default_run"] in ("r1", "badcfg") and st["now"] is not None  # fresh log counts as active
    assert app.run_summary("r1")["progress"] == 0.5
    assert app.results() == []
    assert app.run_dir("../etc") is None and app.run_dir("r1/../..") is None
    assert app.run_series("nope", 100) is None
    json.dumps(app.run_series("badcfg", 100), allow_nan=False)


def test_queue_add_numbering_and_run(tmp_path):
    q = gq.Queue(tmp_path)
    p1 = q.add("ok", "echo hi", str(tmp_path), branch="b", commit="abc")
    p2 = q.add("bad", "exit 3", str(tmp_path), branch="b", commit="abc", kind="bench")
    assert p1.name == "001_ok.json" and p2.name == "002_bad.json"
    try:
        q.add("bad name", "x", str(tmp_path), branch="b", commit="c")
        assert False
    except ValueError:
        pass
    r = gq.Runner(q)
    for p in sorted((q.qdir / "pending").glob("*.json")):
        r.run_job(p)
    done, failed = q.list_state("done"), q.list_state("failed")
    assert [j["id"] for j in done] == ["001_ok"] and done[0]["exit_code"] == 0
    assert [j["id"] for j in failed] == ["002_bad"] and failed[0]["exit_code"] == 3
    assert "hi" in (q.qdir / "logs" / "001_ok.log").read_text()
    assert not list((q.qdir / "running").glob("*.json"))
    q.write_status()
    st = json.loads((q.qdir / "status.json").read_text())
    assert len(st["finished"]) == 2 and st["pending"] == []
    assert q.next_number() == 3


def test_queue_orphan_recovery_and_bad_job(tmp_path):
    q = gq.Queue(tmp_path)
    (q.qdir / "running" / "001_x.json").write_text(json.dumps({"name": "x", "started": "t"}))
    (q.qdir / "pending" / "002_junk.json").write_text("{not json")
    r = gq.Runner(q)
    r.recover_orphans()
    r.run_job(q.qdir / "pending" / "002_junk.json")
    failed = {j["id"]: j for j in q.list_state("failed")}
    assert failed["001_x"]["error"].startswith("orphaned")
    assert failed["002_junk"]["error"].startswith("bad job file")
