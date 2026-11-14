"""Team model: shapes, mask conformity, evaluate == act, canonical card order (brute force),
memory reset on done, gradient flow, CPU / CUDA. Skipped when torch is not installed."""

import itertools
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from balatro_rl import LAYOUT, BalatroVecEnv, Game  # noqa: E402
from balatro_rl.model import HEAD_NAMES, ModelConfig, TeamModel  # noqa: E402
from balatro_rl.model.cards import canonical_step, pack, unpack  # noqa: E402
from balatro_rl.select import conforms, sel_row  # noqa: E402
from test_select import random_rule, valid_sets  # noqa: E402  (tests_py is on sys.path)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
SMALL = ModelConfig(d_model=32, layers=1, heads=2, mem_dim=32)


def make_model(config=SMALL, seed=0, device="cpu"):
    torch.manual_seed(seed)
    return TeamModel(config).to(device).eval()


def rollout_states(steps=60, num_envs=24, seed=3, teacher=True):
    """A list of (obs, masks) numpy snapshots of a teacher rollout (mixed phases)."""
    env = BalatroVecEnv(num_envs, seed=seed, num_threads=1)
    obs, masks = env.reset()
    snaps = []
    for _ in range(steps):
        snaps.append(
            ({k: v.copy() for k, v in obs.items()}, {k: v.copy() for k, v in masks.items()})
        )
        a = env.teacher_actions() if teacher else env.random_actions()
        obs, masks, *_ = env.step(a)
    return snaps


def cat_snaps(snaps):
    obs = {k: np.concatenate([s[0][k] for s in snaps]) for k in snaps[0][0]}
    masks = {k: np.concatenate([s[1][k] for s in snaps]) for k in snaps[0][1]}
    return obs, masks


# --------------------------------------------------------------------------- canonical order


def reachable_canonical(cand, req, mn, mx, h=12):
    """DFS over the canonical rule; asserts no dead end and that every legal set is reached by
    exactly one chain. Returns the set of stops and the number of chains per set."""
    cand_t = unpack(torch.tensor([cand]), h)
    req_t = unpack(torch.tensor([req]), h)
    mn_t, mx_t = torch.tensor([mn]), torch.tensor([mx])
    chains: dict[int, int] = {}
    stack = [(0, -1)]
    while stack:
        p, last = stack.pop()
        part = unpack(torch.tensor([p]), h)
        nxt, stop = canonical_step(cand_t, req_t, mn_t, mx_t, part, torch.tensor([last]))
        nxt_pos = [i for i in range(h) if nxt[0, i]]
        assert nxt_pos or stop[0], f"dead end at {p:b} for {cand:b}/{req:b}/{mn}/{mx}"
        if stop[0]:
            chains[p] = chains.get(p, 0) + 1
        for i in nxt_pos:
            stack.append((p | 1 << i, i))
    return chains


def test_canonical_rule_brute_force():
    rnd = random.Random(11)
    for _ in range(250):
        cand, req, mn, mx = random_rule(rnd)
        chains = reachable_canonical(cand, req, mn, mx)
        want = valid_sets(cand, req, mn, mx)
        assert set(chains) == want
        assert all(c == 1 for c in chains.values()), "a set is reachable by several chains"


def test_canonical_rule_several_required_and_high_minimum():
    rnd = random.Random(12)
    for _ in range(250):
        n = rnd.randint(2, 9)
        pos = rnd.sample(range(12), n)
        cand = sum(1 << p for p in pos)
        mx = rnd.randint(2, min(5, n))
        mn = rnd.randint(1, mx)  # minimums above 1 (Death needs 2 cards, ...)
        req = sum(1 << p for p in rnd.sample(pos, rnd.randint(0, mx)))
        chains = reachable_canonical(cand, req, mn, mx)
        assert set(chains) == valid_sets(cand, req, mn, mx)
        assert all(c == 1 for c in chains.values())


def test_pack_unpack_roundtrip():
    x = torch.tensor([0, 1, 0b1011, (1 << 31) | 5, 0xFFFFFFFF])
    assert torch.equal(pack(unpack(x, 32)), x)


# --------------------------------------------------------------------------- model API


def test_shapes_and_dtypes():
    snaps = rollout_states(steps=8)
    obs, masks = snaps[-1]
    m = make_model()
    h = m.init_hidden(24)
    a, lp, ent, v, h2 = m.act(obs, masks, h)
    assert a.shape == (24, 4) and a.dtype == torch.int64
    assert lp.shape == ent.shape == v.shape == (24,)
    assert h2.shape == (24, SMALL.mem_dim)
    assert torch.isfinite(lp).all() and torch.isfinite(v).all() and (ent >= 0).all()
    assert (lp <= 1e-6).all()
    assert len(HEAD_NAMES) == 4 and len(m.heads) == 4
    # memory = none: same API, empty hidden
    m0 = make_model(ModelConfig(d_model=32, layers=1, heads=2, memory="none"))
    h0 = m0.init_hidden(24)
    assert h0.shape == (24, 0)
    a0, _, _, _, h0b = m0.act(obs, masks, h0)
    assert a0.shape == (24, 4) and h0b.shape == (24, 0)
    lp0, ent0, v0 = m0.evaluate(obs, masks, h0, a0)
    assert torch.isfinite(lp0).all()


@pytest.mark.parametrize("deterministic", [False, True])
def test_actions_conform_to_masks_in_the_env(deterministic):
    """Thousands of model actions (random init, sampled and greedy): zero invalid in the env and
    every action expressible through the masks."""
    n = 64
    env = BalatroVecEnv(n, seed=5, num_threads=2, max_episode_steps=400)
    obs, masks = env.reset()
    m = make_model(seed=2)
    h = m.init_hidden(n)
    first = torch.ones(n, dtype=torch.bool)
    invalid = steps = 0
    kinds_seen = set()
    for _ in range(80 if deterministic else 120):
        a, _, _, _, h = m.act(obs, masks, h, deterministic=deterministic, first=first)
        an = a.numpy()
        for i in range(0, n, 3):
            assert conforms(masks, i, an[i])
        kinds_seen |= set(an[:, 0].tolist())
        obs, masks, _, done, inv = env.step(an)
        first = torch.from_numpy(done != 0)
        invalid += int(inv.sum())
        steps += n
    assert steps >= 5000 and invalid == 0
    assert len(kinds_seen) >= (3 if deterministic else 8)


def test_evaluate_matches_act_logprob_over_a_rollout():
    m = make_model(seed=1)
    n = 16
    env = BalatroVecEnv(n, seed=9, num_threads=1, max_episode_steps=300)
    obs, masks = env.reset()
    h = m.init_hidden(n)
    gen = torch.Generator().manual_seed(0)
    worst = 0.0
    seen_phases = set()
    for _t in range(150):
        first = torch.rand(n, generator=gen) < 0.05
        h_in = h
        a, lp, ent, v, h = m.act(obs, masks, h_in, first=first, generator=gen)
        lp2, ent2, v2 = m.evaluate(obs, masks, h_in * (~first)[:, None], a)
        worst = max(worst, (lp - lp2).abs().max().item(), (ent - ent2).abs().max().item())
        assert torch.allclose(v, v2, atol=1e-5)
        seen_phases |= set(obs["global_i"][:, 0].tolist())
        obs, masks, *_ = env.step(env.teacher_actions())  # the teacher drives (all phases)
    assert worst < 1e-5, worst
    assert {0, 1, 2, 3}.issubset(seen_phases)


def test_sequence_evaluate_matches_stepwise_act_and_resets_memory():
    m = make_model(seed=4)
    t_len, n = 10, 8
    env = BalatroVecEnv(n, seed=21, num_threads=1)
    obs, masks = env.reset()
    gen = torch.Generator().manual_seed(1)
    h0 = torch.randn(n, SMALL.mem_dim, generator=gen)
    h = h0
    snaps, acts, lps, firsts = [], [], [], []
    for _t in range(t_len):
        first = torch.rand(n, generator=gen) < 0.2
        snaps.append(
            ({k: v.copy() for k, v in obs.items()}, {k: v.copy() for k, v in masks.items()})
        )
        a, lp, _, _, h = m.act(obs, masks, h, first=first, generator=gen)
        acts.append(a)
        lps.append(lp)
        firsts.append(first)
        obs, masks, *_ = env.step(a.numpy())
    o, mk = cat_snaps(snaps)  # time-major (T * n)
    a_all = torch.cat(acts)
    first_all = torch.stack(firsts)
    lp_seq, ent_seq, v_seq = m.evaluate(o, mk, h0, a_all, first=first_all)
    assert (lp_seq - torch.cat(lps)).abs().max() < 1e-5
    r = m.evaluate_full(o, mk, h0, a_all, first=first_all)
    assert torch.allclose(r["hidden"], h, atol=1e-6)
    assert r["ok"].all()


def test_memory_resets_on_done():
    m = make_model(seed=6)
    snaps = rollout_states(steps=3, num_envs=12)
    obs, masks = snaps[-1]
    rng = torch.Generator().manual_seed(0)
    h = torch.randn(12, SMALL.mem_dim, generator=rng)
    first = torch.zeros(12, dtype=torch.bool)
    first[::2] = True
    _, _, _, v1, hn1 = m.act(obs, masks, h, deterministic=True, first=first)
    _, _, _, v0, hn0 = m.act(obs, masks, torch.zeros_like(h), deterministic=True)
    _, _, _, v2, hn2 = m.act(obs, masks, h, deterministic=True)
    f = first
    assert torch.allclose(hn1[f], hn0[f], atol=1e-6) and torch.allclose(v1[f], v0[f], atol=1e-6)
    assert torch.allclose(hn1[~f], hn2[~f], atol=1e-6)
    assert not torch.allclose(hn0[~f], hn2[~f], atol=1e-3)  # memory does influence the output


def test_gradient_flows_to_every_component():
    snaps = rollout_states(steps=80, num_envs=32, teacher=True)
    obs, masks = cat_snaps(snaps[::2])
    m = make_model(ModelConfig(d_model=32, layers=2, heads=2, mem_dim=32), seed=3)
    m.train()
    h = m.init_hidden(obs["global_f"].shape[0])
    a, *_ = m.act(obs, masks, h)
    phases = set(obs["global_i"][:, 0].tolist())
    lp, ent, v = m.evaluate(obs, masks, h, a)
    loss = -(lp.mean()) + 0.5 * v.pow(2).mean() - 0.01 * ent.mean()
    loss.backward()

    def g(mod):
        return sum(p.grad.abs().sum().item() for p in mod.parameters() if p.grad is not None)

    assert g(m.encoder) > 0 and g(m.cell) > 0 and g(m.critic) > 0 and g(m.ctx) > 0
    for pid, head_name in enumerate(HEAD_NAMES):
        if pid in phases:
            head = m.heads[pid]
            assert g(head.kind_mlp) > 0 and g(head.q0) > 0, head_name
    assert 1 in phases and g(m.heads[1].card_q) > 0 and g(m.heads[1].stop) > 0
    for p in m.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


@pytest.mark.skipif("cuda" not in DEVICES, reason="no CUDA")
def test_cpu_and_cuda_agree():
    snaps = rollout_states(steps=30, num_envs=16)
    obs, masks = cat_snaps(snaps[::3])
    n = obs["global_f"].shape[0]
    cpu = make_model(seed=8)
    gpu = make_model(seed=8, device="cuda")
    gpu.load_state_dict(cpu.state_dict())
    h = torch.randn(n, SMALL.mem_dim, generator=torch.Generator().manual_seed(0))
    a, lp, ent, v, hn = cpu.act(obs, masks, h, deterministic=True)
    lp_g, ent_g, v_g = gpu.evaluate(obs, masks, h.cuda(), a.cuda())
    assert torch.allclose(lp, lp_g.cpu(), atol=2e-3, rtol=1e-3)
    assert torch.allclose(v, v_g.cpu(), atol=2e-3, rtol=1e-3)
    ag, *_ = gpu.act(obs, masks, h.cuda(), deterministic=True)
    assert (ag.cpu() == a).float().mean() > 0.97
    # sampling on the GPU is mask-conforming
    gen = torch.Generator(device="cuda").manual_seed(0)
    ag, *_ = gpu.act(obs, masks, h.cuda(), generator=gen)
    an = ag.cpu().numpy()
    assert all(conforms(masks, i, an[i]) for i in range(n))


def test_generator_makes_sampling_reproducible():
    snaps = rollout_states(steps=6, num_envs=16)
    obs, masks = snaps[-1]
    m = make_model()
    h = m.init_hidden(16)
    a1, *_ = m.act(obs, masks, h, generator=torch.Generator().manual_seed(5))
    a2, *_ = m.act(obs, masks, h, generator=torch.Generator().manual_seed(5))
    a3, *_ = m.act(obs, masks, h, generator=torch.Generator().manual_seed(6))
    assert torch.equal(a1, a2)
    assert not torch.equal(a1, a3)


# --------------------------------------------------------------------------- set probabilities


def _row_action(row):
    cons = LAYOUT["caps"]["cons"]
    if row == 0:
        return 2, 0
    if row == 1:
        return 3, 0
    if row < 2 + cons:
        return 6, row - 2
    return 14, row - 2 - cons


def test_card_set_probabilities_sum_to_one_and_cover_every_set():
    """For real states (Play/Discard, targeted consumables, pack picks, Cerulean's forced card)
    the canonical chain probabilities of all legal sets sum to 1, each set is representable and
    no illegal set has probability mass (random-init, non-uniform logits)."""
    m = make_model(seed=13)
    rnd = random.Random(0)
    checked_rows = set()
    n_checked = 0
    for seed in range(10):
        g = Game(seed=seed, max_episode_steps=10**9)
        for step in range(220):
            if g.is_terminal():
                break
            obs = {k: v[None] for k, v in g.observation().items()}
            masks = {k: v[None] for k, v in g.masks().items()}
            if step % 3 == 0:
                if masks["kind_mask"][0, 2] and rnd.random() < 0.3:
                    g.debug_force_card(rnd.randrange(4))
                    obs = {k: v[None] for k, v in g.observation().items()}
                    masks = {k: v[None] for k, v in g.masks().items()}
                for row in range(15):
                    mx = int(masks["sel_max"][0, row])
                    if mx == 0:
                        continue
                    cand = int(masks["sel_candidates"][0, row])
                    if bin(cand).count("1") > 11:
                        continue
                    req = int(masks["sel_required"][0, row])
                    sets = sorted(valid_sets(cand, req, int(masks["sel_min"][0, row]), mx))
                    kind, arg0 = _row_action(row)
                    assert sel_row(kind, arg0) == row
                    acts = torch.tensor([[kind, arg0, 0, s] for s in sets])
                    k = len(sets)
                    ob = {key: np.repeat(v, k, axis=0) for key, v in obs.items()}
                    mk = {key: np.repeat(v, k, axis=0) for key, v in masks.items()}
                    r = m.evaluate_full(ob, mk, m.init_hidden(k), acts)
                    assert r["ok"].all()
                    # lp_cards of every legal set: sums to one over the sets
                    total = r["lp_cards"].exp().sum().item()
                    assert abs(total - 1.0) < 1e-4, (row, total, k)
                    checked_rows.add((kind, bool(req)))
                    n_checked += 1
                    # a set outside the legal sets is not expressible
                    if sets:
                        bad_mask = (cand ^ 0xFFFFF) & 0xFFFFF
                        if bad_mask:
                            bad = torch.tensor([[kind, arg0, 0, bad_mask & -bad_mask]])
                            rb = m.evaluate_full(
                                {key: v[:1] for key, v in ob.items()},
                                {key: v[:1] for key, v in mk.items()},
                                m.init_hidden(1),
                                bad,
                            )
                            assert not rb["ok"].any()
            parts = g.teacher_parts() if rnd.random() < 0.7 else g.random_parts()
            g.step_parts(*parts)
    assert n_checked > 100
    assert (2, False) in checked_rows and (3, False) in checked_rows
    assert any(k in (6, 14) for k, _ in checked_rows)


def test_default_size_model_parameters():
    m = make_model(ModelConfig())
    counts = m.num_parameters()
    assert 1_000_000 < counts["total"] < 5_000_000
    assert counts["total"] == sum(v for k, v in counts.items() if k != "total")


def test_brute_force_helper_sanity():
    # the itertools-based oracle itself: C(4,2) sets
    assert len(valid_sets(0b1111, 0, 2, 2)) == len(list(itertools.combinations(range(4), 2)))
