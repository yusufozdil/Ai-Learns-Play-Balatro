"""Error hall (docs/design/error_hall.md): pool files, resets from the pool in the trainer,
default-off identity, probability composition. CPU only."""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from balatro_rl import _engine  # noqa: E402
from balatro_rl.analysis import hall  # noqa: E402
from balatro_rl.train.ppo import DECK_COL, STAKE_COL, STAKE_NAMES, PPOConfig, Trainer  # noqa: E402

TINY_MODEL = {
    "d_model": 32, "layers": 1, "heads": 2, "ff_mult": 2, "dropout": 0.0,
    "memory": "gru", "mem_dim": 32,
}  # fmt: skip


def _shop_state(seed: int, stake: str = "white", deck: str = "red") -> str:
    g = _engine.Game(seed, deck, stake)
    for _ in range(3000):
        if json.loads(g.state_json())["phase"] == "Shop":
            return g.export_state()
        g.step_parts(*g.teacher_parts())
    raise AssertionError("no shop reached")


def _mistake(cat_played="SellJoker Joker A", cat_better="SellJoker Joker B", delta=0.1) -> dict:
    return {
        "played": cat_played, "better": cat_better, "played_kind": "SellJoker",
        "better_kind": "SellJoker", "phase": "Shop", "ante": 3, "delta": delta, "se": 0.02,
        "q_played": 0.3, "q_better": 0.3 + delta, "step": 17, "pwin_before": 0.5,
    }  # fmt: skip


def _write_pool(path, stakes=("white", "green"), per_stake=2, **mistake_kw) -> list[dict]:
    recs = []
    for k, stake in enumerate(stakes):
        for j in range(per_stake):
            st = _shop_state(10 * k + j + 1, stake)
            recs.append(hall.pool_record(st, _mistake(**mistake_kw), game=j, outcome="Loss"))
    hall.write_pool(path, recs)
    return recs


def _cfg(tmp_path, **kw) -> PPOConfig:
    base = {
        "out_dir": str(tmp_path / "run"), "num_envs": 16, "rollout_len": 16, "chunk": 8,
        "group": 8, "threads": 1, "max_episode_steps": 300, "model": TINY_MODEL, "epochs": 1,
        "eval_every": 0, "save_every": 1000, "decks": "red", "stakes": "white,green",
        "total_steps": 2 * 16 * 16,
    }  # fmt: skip
    base.update(kw)
    return PPOConfig(**base)


# ------------------------------------------------------------------------- pool files


def test_pool_round_trip_append_and_filters(tmp_path):
    path = tmp_path / "pool.jsonl"
    recs = _write_pool(path)
    back = hall.read_pool(path)
    assert [r["state"] for r in back] == [r["state"] for r in recs]
    r = back[0]
    assert r["category"] == "SellJoker->SellJoker" and r["stake"] == "white" and r["deck"] == "red"
    assert r["dq"] == pytest.approx(0.1) and r["ante"] == 3 and r["played"] == "SellJoker Joker A"
    # the stored state is a valid, importable run
    assert not _engine.Game.from_state(r["state"]).is_terminal()
    # append, comma list of files, min dq filter
    more = hall.pool_record(_shop_state(99), _mistake(delta=0.5), game=9)
    hall.write_pool(path, [more], append=True)
    assert len(hall.read_pool(path)) == len(recs) + 1
    other = tmp_path / "b.jsonl"
    hall.write_pool(other, [more])
    assert len(hall.read_pool(f"{path},{other}")) == len(recs) + 2
    assert len(hall.read_pool(path, min_dq=0.3)) == 1
    s = hall.summarize(hall.read_pool(path))
    assert s["n"] == len(recs) + 1 and s["category"]["SellJoker->SellJoker"] == len(recs) + 1
    # malformed files are rejected loudly
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"v": 99, "state": "{}"}) + "\n")
    with pytest.raises(ValueError):
        hall.read_pool(bad)
    with pytest.raises(ValueError):
        hall.write_pool(tmp_path / "x.jsonl", [{"state": "{}"}])


def test_category_names():
    assert hall.category_of("SellJoker Joker A", "SellJoker Joker B") == "SellJoker->SellJoker"
    assert hall.category_of("Buy Joker X", "SellJoker Joker B") == "BuyJoker->SellJoker"
    assert hall.category_of("SelectBlind (Boss blind, target 1)", "RerollBoss") == (
        "SelectBlind->RerollBoss"
    )


# ------------------------------------------------------------------------- composition


def test_probabilities_compose_and_sum_is_asserted(tmp_path):
    pool = tmp_path / "p.jsonl"
    _write_pool(pool, per_stake=1)
    PPOConfig(scenario_prob=0.2, boss_hall=0.5, error_hall=0.3, error_hall_pool=str(pool))
    with pytest.raises(AssertionError):
        PPOConfig(scenario_prob=0.2, boss_hall=0.5, error_hall=0.4, error_hall_pool=str(pool))
    with pytest.raises(AssertionError):
        PPOConfig(error_hall=1.5, error_hall_pool=str(pool))
    with pytest.raises(AssertionError):
        PPOConfig(error_hall=0.1)  # no pool
    # conditional probability on a natural reset: e / (1 - s - h), so the share of ALL starts is e
    t = Trainer(
        _cfg(tmp_path, scenario_prob=0.1, boss_hall=0.2, error_hall=0.3, error_hall_pool=str(pool)),
        echo=lambda *_: None, device="cpu",
    )  # fmt: skip
    assert t.err_q == pytest.approx(0.3 / 0.7)
    t2 = Trainer(
        _cfg(tmp_path / "b", error_hall=0.1, error_hall_pool=str(pool)),
        echo=lambda *_: None, device="cpu",
    )  # fmt: skip
    assert t2.err_q == pytest.approx(0.1)


def test_measured_shares_of_episode_starts(tmp_path):
    """Scenario 0.2 + error hall 0.3: about 20% / 30% / 50% of the episode starts."""
    pool = tmp_path / "p.jsonl"
    _write_pool(pool)
    cfg = _cfg(
        tmp_path, num_envs=64, rollout_len=64, chunk=32, group=32, max_episode_steps=8,
        scenario_prob=0.2, error_hall=0.3, error_hall_pool=str(pool),
    )  # fmt: skip
    t = Trainer(cfg, echo=lambda *_: None, device="cpu")
    fin = []
    for _ in range(3):
        fin += t.collect()["finished"]
    n = len(fin)
    assert n > 1000
    scn = np.mean([r["scenario"] for r in fin])
    err = np.mean([r["errhall"] for r in fin])
    assert abs(scn - 0.2) < 0.05, scn
    assert abs(err - 0.3) < 0.05, err
    assert not any(r["scenario"] and r["errhall"] for r in fin)


# ------------------------------------------------------------------------- reset from the pool


def test_resets_import_pool_states_with_episode_boundary(tmp_path):
    pool = tmp_path / "p.jsonl"
    recs = _write_pool(pool, per_stake=2)
    cats = {r["category"] for r in recs}
    cfg = _cfg(tmp_path, error_hall=1.0, error_hall_pool=str(pool), max_episode_steps=12)
    t = Trainer(cfg, echo=lambda *_: None, device="cpu")
    calls = []
    orig = t.env.import_state

    def spy(i, state, reseed=None, streams=None):
        calls.append((i, reseed))
        return orig(i, state, reseed=reseed, streams=streams)

    t.env.import_state = spy
    col = t.collect()
    fin = col["finished"]
    assert calls and all(rs is not None for _, rs in calls)  # always reseeded: fresh futures
    assert len(set(rs for _, rs in calls)) == len(calls)
    # every episode after the first of an env starts from the hall (share 1.0), at a Shop state;
    # the step it starts at is an episode boundary: GAE cut ("first") and a Shop observation
    first, phase = t.buf["first"], t.buf["phase"]
    starts = [(s, e) for s in range(1, cfg.rollout_len) for e in range(cfg.num_envs) if first[s, e]]
    assert starts
    assert all(int(phase[s, e]) == 2 for s, e in starts), "hall episodes start at the Shop"
    # records: imported episodes carry the category, nothing else is tagged
    errs = [r for r in fin if r["errhall"]]
    assert errs and all(r["imported"] and r["errcat"] in cats for r in errs)
    assert all(not r["errhall"] for r in fin if not r["imported"])
    assert not any(r["scenario"] or r["hall"] for r in fin)
    log = t.errhall_log()
    assert log["errhall_started"] == len(calls) and log["errhall_pool"] == len(recs)
    assert sum(v for k, v in log.items() if k.startswith("errhall_started/")) == len(calls)
    assert t.errhall_log()["errhall_started"] == 0  # counters are per update
    st = Trainer.finished_stats(fin)
    assert st["errhall_episodes"] == len(errs) and "errhall_win" in st
    assert any(k.startswith("errhall_n/") for k in st)
    # natural statistics exclude the hall episodes
    assert st.get("episodes", 0) == len([r for r in fin if not r["errhall"]])


def test_reset_matches_stake_of_the_new_episode(tmp_path):
    """A pool that only has green states: white episodes cannot take a state (stake mismatch
    -> stays natural, counted as fallback), green ones do."""
    pool = tmp_path / "p.jsonl"
    _write_pool(pool, stakes=("green",), per_stake=2)
    cfg = _cfg(
        tmp_path, num_envs=32, group=16, error_hall=1.0, error_hall_pool=str(pool),
        max_episode_steps=10,
    )  # fmt: skip
    t = Trainer(cfg, echo=lambda *_: None, device="cpu")
    seen = []
    orig = t.env.import_state

    def spy(i, state, reseed=None, streams=None):
        out = orig(i, state, reseed=reseed, streams=streams)
        seen.append(STAKE_NAMES[int(t.obs["global_i"][i, STAKE_COL])])
        return out

    t.env.import_state = spy
    fin = t.collect()["finished"]
    assert seen and set(seen) == {"green"}
    assert t.err_fallback > 0
    assert all(r["stake"] == "green" for r in fin if r["errhall"])
    assert int(t.obs["global_i"][0, DECK_COL]) == 0  # red deck


# ------------------------------------------------------------------------- default off


def test_default_off_is_inert(tmp_path):
    """No error hall: nothing is built, the env is never imported into, no errhall log keys, and
    the (ignored) hall fields do not change the run: identical weights."""
    pool = tmp_path / "p.jsonl"
    _write_pool(pool, per_stake=1)

    def run(name, **kw):
        t = Trainer(_cfg(tmp_path / name, **kw), echo=lambda *_: None, device="cpu")
        assert not t.errhall_on and not hasattr(t, "err_pool")

        def boom(*_a, **_k):
            raise AssertionError("import_state called with the error hall off")

        t.env.import_state = boom
        t.train()
        recs = [json.loads(x) for x in (t.out / "log.jsonl").read_text().splitlines()]
        assert not any(k.startswith("errhall") for r in recs for k in r)
        return [p.detach().clone() for p in t.model.parameters()]

    a = run("a")
    b = run("b", error_hall=0.0, error_hall_pool=str(pool), error_hall_min_dq=0.2)
    for x, y in zip(a, b, strict=True):
        assert torch.equal(x, y)


def test_error_hall_training_smoke_with_v2_flags(tmp_path):
    """The hall composes with the v2 flags (scenario, boss hall, per-stake advantage
    normalisation, stake curriculum) for a couple of updates."""
    pool = tmp_path / "p.jsonl"
    _write_pool(pool)
    cfg = _cfg(
        tmp_path, adv_norm="per_config", adv_norm_min_rows=1, stake_curriculum="auto",
        stake_curriculum_min_eps=1, scenario_prob=0.1, boss_hall=0.1, boss_hall_min_ante=1,
        error_hall=0.3, error_hall_pool=str(pool), max_episode_steps=40,
    )  # fmt: skip
    t = Trainer(cfg, echo=lambda *_: None, device="cpu")
    t.train()
    recs = [json.loads(x) for x in (t.out / "log.jsonl").read_text().splitlines()]
    ups = [r for r in recs if r["type"] == "update"]
    assert len(ups) == 2
    for r in ups:
        assert "errhall_started" in r and "errhall_pool" in r and "hall_pool" in r
        assert np.isfinite(r["vloss"]) and np.isfinite(r["pg"])
    assert sum(r["errhall_started"] for r in ups) > 0


# ------------------------------------------------------------------------- hold-out and probe


def test_holdout_split_is_by_game_and_deterministic(tmp_path):
    recs = []
    for g in range(60):
        for _ in range(2):  # two mistakes per game
            recs.append(hall.pool_record(_shop_state(1), _mistake(), game=g, seed_base=7))
    path = tmp_path / "p.jsonl"
    hall.write_pool(path, recs)
    tr = hall.read_pool(path, holdout=0.25, part="train")
    ho = hall.read_pool(path, holdout=0.25, part="holdout")
    assert len(tr) + len(ho) == len(recs) == len(hall.read_pool(path, holdout=0.25, part="all"))
    assert {r["game"] for r in tr}.isdisjoint({r["game"] for r in ho})  # games never straddle
    assert 0.1 < len(ho) / len(recs) < 0.45
    assert hall.read_pool(path, holdout=0.0, part="holdout") == []
    again = hall.read_pool(path, holdout=0.25, part="holdout")
    assert [r["game"] for r in ho] == [r["game"] for r in again]
    # the trainer sees only the training part
    t = Trainer(
        _cfg(tmp_path, error_hall=0.1, error_hall_pool=str(path), error_hall_holdout=0.25),
        echo=lambda *_: None, device="cpu",
    )  # fmt: skip
    assert len(t.err_pool) == len(tr)


def test_probe_reports_repeat_and_fix(tmp_path):
    import importlib.util
    from pathlib import Path

    from balatro_rl.analysis.features import PolicyRunner
    from balatro_rl.model import ModelConfig, TeamModel

    path = Path(__file__).resolve().parent.parent / "scripts" / "errhall_probe.py"
    spec = importlib.util.spec_from_file_location("errhall_probe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    torch.manual_seed(0)
    runner = PolicyRunner(TeamModel(ModelConfig(**TINY_MODEL)).eval())
    states = [_shop_state(s) for s in (1, 2, 3)]
    end_shop = [5, 0, 0, 0]
    recs = [
        hall.pool_record(
            st, {**_mistake(), "played_action": end_shop, "better_action": [13, 0, 0, 0]}, game=i
        )
        for i, st in enumerate(states)
    ]
    rows = mod.probe(runner, recs, batch=2)  # also exercises the batching
    assert len(rows) == 3 and all(0 <= r["p_played"] <= 1 for r in rows)
    # declare the policy's own greedy action as the "played" one: repeat must be True
    env = mod.BalatroVecEnv(3, seed=1)
    env.reset()
    for i, st in enumerate(states):
        env.import_state(i, st)
    act, _, _, _ = runner.run(env.obs, env.masks, runner.init_hidden(3), True)
    own = [
        {**r, "played_action": a, "better_action": a}
        for r, a in zip(recs, act.cpu().tolist(), strict=True)
    ]
    rows = mod.probe(runner, own)
    assert all(r["repeat"] and r["fix"] for r in rows)
    assert "ALL" in mod.table(rows, 3)


# ------------------------------------------------------------------------- mining pipeline


def test_mining_pipeline_to_pool_with_tiny_model(tmp_path):
    """errmap on a random tiny model (everything flagged: threshold -1) -> pool records with the
    states before the decisions -> a file -> importable states with the right actions."""
    from balatro_rl.analysis import errmap, pwin
    from balatro_rl.analysis.features import PHASE_BLIND, PHASE_SHOP, PolicyRunner
    from balatro_rl.model import ModelConfig, TeamModel

    torch.manual_seed(0)
    model = TeamModel(ModelConfig(**TINY_MODEL)).eval()
    runner = PolicyRunner(model)
    head = pwin.PWinHead(runner.feat_dim).eval()
    kw = {"decks": ["red", "blue"], "stakes": ["white", "green"]}
    base = 4242
    acts, codes = errmap.record_games(runner, base, 3, env_kwargs=kw)
    cfg = errmap.Config(
        m_screen=2, m_confirm=3, n_confirm=2, slots=16, min_alt_prob=0.0,
        threshold=-1.0, z_min=-9.0, phases=(PHASE_BLIND, PHASE_SHOP),
    )  # fmt: skip
    decisions, info = errmap.run_trunk(
        runner, head, [0, 1, 2], base, dict(enumerate(acts)), cfg, env_kwargs=kw
    )
    pool = errmap.RolloutPool(runner, head, cfg)
    rng = np.random.default_rng(0)
    # a random tiny model loses within 7 steps (every future is a 0): feed random leaf values so
    # that the screen / confirm stages have differences to flag
    pool.run = lambda jobs: (rng.random(len(jobs)), {"env_steps": 0, "jobs": len(jobs)})
    errmap.screen_and_confirm(pool, decisions, cfg)
    recs = []
    for g in (0, 1, 2):
        res = errmap.finalize_game(g, decisions[g], info[g], cfg)
        by_step = {d.step: d.state for d in decisions[g]}
        for m in res["mistakes"]:
            recs.append(hall.pool_record(by_step[m["step"]], m, game=g, seed_base=base))
    assert recs, "with threshold -1 every confirmed decision is flagged"
    path = tmp_path / "mined.jsonl"
    hall.write_pool(path, recs)
    back = hall.read_pool(path, min_dq=-1.0)  # threshold -1 also flags negative dq
    assert len(back) == len(recs)
    for r in back:
        assert r["deck"] in ("red", "blue") and r["stake"] in ("white", "green")
        assert len(r["played_action"]) == 4 and len(r["better_action"]) == 4
        g = _engine.Game.from_state(r["state"])  # the better action is legal in the saved state
        assert tuple(r["better_action"]) in {tuple(a) for a in g.legal_parts()}
