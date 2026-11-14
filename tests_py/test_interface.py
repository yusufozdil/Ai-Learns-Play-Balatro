"""The RL interface: layout vs the design document, observation/mask invariants, pointer
targets, mask-conforming policies, teacher actions, hidden-information rules."""

import re
from pathlib import Path

import numpy as np
from balatro_rl import INTERFACE_VERSION, LAYOUT, BalatroVecEnv, Game
from balatro_rl.select import conforms, sample_actions, sel_row, unpack_bits

DOC = Path(__file__).resolve().parents[1] / "docs" / "design" / "rl_interface.md"
NAMES = LAYOUT["feature_names"]
CAPS = LAYOUT["caps"]
GI = {n: i for i, n in enumerate(NAMES["global_i"])}
GF = {n: i for i, n in enumerate(NAMES["global_f"])}
HF = {n: i for i, n in enumerate(NAMES["hand_f"])}
HI = {n: i for i, n in enumerate(NAMES["hand_i"])}


# --------------------------------------------------------------------------- document


def test_design_document_matches_the_layout():
    text = DOC.read_text()
    assert f"INTERFACE_VERSION = {INTERFACE_VERSION}" in text
    for name, _dtype, _shape in LAYOUT["buffers"]:
        assert f"`{name}`" in text, f"buffer {name} is not documented"
    for group in ("global_f", "global_i", "hand_f", "hand_i", "joker_f", "joker_i"):
        for feat in NAMES[group]:
            base = feat.split(".")[0]
            assert f"`{base}`" in text, f"{group}.{feat} is not documented"
    for group in ("cons_f", "cons_i", "offer_f", "offer_i"):
        for feat in NAMES[group]:
            assert re.search(rf"\b{re.escape(feat)}\b", text), f"{group}.{feat} is not documented"
    # Capacities appear as documented numbers.
    for key, want in (("hand", 32), ("joker", 20), ("cons", 8), ("offer", 21)):
        assert CAPS[key] == want
    assert "(203,)" in text and "MAX_ARG = 20" in text
    # Tables of shapes in the document equal the layout.
    for name, _dtype, shape in LAYOUT["buffers"]:
        if len(shape) == 0:
            continue
        shape_txt = "(" + ", ".join(map(str, shape)) + ("," if len(shape) == 1 else "") + ")"
        row = next((ln for ln in text.splitlines() if ln.startswith(f"| `{name}` |")), None)
        assert row is not None and shape_txt in row, f"{name}: {shape_txt} not in document table"


def test_interface_version_exposed_everywhere():
    assert INTERFACE_VERSION == LAYOUT["version"] == 1


# --------------------------------------------------------------------------- invariants


def collect(env_factory, steps, teacher_every=0):
    """Yield (obs, masks) after the reset and every step of a rollout."""
    env = env_factory()
    obs, masks = env.reset()
    yield env, obs, masks
    for t in range(steps):
        a = env.teacher_actions() if teacher_every and t % teacher_every else env.random_actions()
        obs, masks, *_ = env.step(a)
        yield env, obs, masks


def check_obs(obs):
    vocab = LAYOUT["vocab"]
    for name, arr in obs.items():
        if arr.dtype == np.float32:
            assert np.isfinite(arr).all(), name
            assert np.abs(arr).max() < 50, name
    for name in ("global_i", "hand_i", "joker_i", "cons_i", "offer_i"):
        v = np.array(vocab[name])
        arr = obs[name]
        assert (arr >= 0).all() and (arr < v).all(), name
    # Padding rows are all zero.
    for tok in ("hand", "joker", "cons", "offer"):
        invalid = ~obs[f"{tok}_valid"]
        assert not obs[f"{tok}_f"][invalid].any(), tok
        assert not obs[f"{tok}_i"][invalid].any(), tok


def check_masks(masks, obs):
    km, a0, a1 = masks["kind_mask"], masks["arg0_mask"], masks["arg1_mask"]
    assert km.any(axis=1).all()
    # A kind is allowed iff its arg0 row is non-empty.
    assert np.array_equal(a0.any(axis=2), km)
    # MoveJoker destinations only with MoveJoker; jokers visible.
    assert not a1[~km[:, 8]].any()
    # Pointer targets: every legal arg0 points at a valid token (of the right source).
    jv, cv, ov = obs["joker_valid"], obs["cons_valid"], obs["offer_valid"]
    src = obs["offer_i"][:, :, 0]
    for k, ptr in enumerate(LAYOUT["arg0_pointer"]):
        if ptr is None:
            continue
        tok, off, _cap = ptr
        for i, a in zip(*np.nonzero(a0[:, k]), strict=True):
            if tok == "joker":
                assert jv[i, a]
            elif tok == "cons":
                assert cv[i, a]
            else:
                assert ov[i, off + a]
                want = {9: 1, 10: 1, 11: 2, 12: 3, 14: 4}[k]
                assert src[i, off + a] == want
    # MoveJoker: destinations are valid jokers.
    assert not (a1 & ~jv).any()
    # Card rows: candidates within the hand, required within candidates, min<=max<=cap.
    cand, req = masks["sel_candidates"], masks["sel_required"]
    mn, mx = masks["sel_min"], masks["sel_max"]
    hv = unpack_bits(cand) & ~obs["hand_valid"][:, None, :]
    assert not hv.any(), "candidate outside the hand"
    assert not (req & ~cand).any()
    assert ((mx == 0) | ((mn >= 1) & (mn <= mx) & (mx <= 5))).all()
    # Play / Discard rows exist exactly when the kind is allowed.
    assert np.array_equal(mx[:, 0] > 0, km[:, 2]) and np.array_equal(mx[:, 1] > 0, km[:, 3])
    # The forced-card flag equals the required bit of the Play row.
    forced = obs["hand_f"][:, :, HF["forced"]] > 0
    assert np.array_equal(unpack_bits(req[:, 0]), forced)
    # Hand bookkeeping.
    n_hand = np.round(obs["global_f"][:, GF["hand_count"]] * 16)
    assert np.array_equal(obs["hand_valid"].sum(axis=1), n_hand.astype(int))
    # Draw-pile histogram adds up to the pile size.
    h = NAMES["global_f"]
    lo = h.index("draw_rank_suit.0")
    cards = obs["global_f"][:, lo : lo + 53].sum(axis=1) * 4
    assert np.allclose(cards, obs["global_f"][:, GF["draw_pile"]] * 52, atol=1e-3)


def test_observation_and_mask_invariants_on_random_and_teacher_play():
    seen_phases = set()
    for _env, obs, masks in collect(
        lambda: BalatroVecEnv(48, seed=4, num_threads=4, max_episode_steps=800), 500
    ):
        check_obs(obs)
        check_masks(masks, obs)
        seen_phases |= set(np.unique(obs["global_i"][:, GI["phase"]]))
    # Teacher play reaches shops, packs and deeper antes.
    for _env, obs, masks in collect(
        lambda: BalatroVecEnv(16, seed=9, num_threads=4, max_episode_steps=2000),
        150,
        teacher_every=1,
    ):
        check_obs(obs)
        check_masks(masks, obs)
        seen_phases |= set(np.unique(obs["global_i"][:, GI["phase"]]))
    assert {0, 1, 2, 3} <= seen_phases, seen_phases


# --------------------------------------------------------------------------- policies


def test_rust_random_policy_conforms_to_masks_and_is_never_invalid():
    env = BalatroVecEnv(64, seed=1, num_threads=4, max_episode_steps=1500)
    env.reset()
    invalid = 0
    for _ in range(600):
        acts = env.random_actions().copy()
        for i in range(0, 64, 7):  # spot check against the Python view of the masks
            assert conforms(env.masks, i, acts[i]), (i, acts[i])
        env.step(acts)
        invalid += int(env.invalid.sum())
    assert invalid == 0
    assert all(f[6] == 0 for f in env.drain_finished())


def test_python_reference_sampler_is_never_invalid():
    env = BalatroVecEnv(24, seed=2, num_threads=2, max_episode_steps=1500)
    env.reset()
    rng = np.random.default_rng(0)
    for _ in range(300):
        acts = sample_actions(env.masks, rng)
        env.step(acts)
        assert env.invalid.sum() == 0


def test_masked_random_policy_covers_all_phase_kinds():
    # Every kind the masks allow is sampled eventually (no kind is silently unreachable).
    env = BalatroVecEnv(64, seed=5, num_threads=4, max_episode_steps=3000)
    env.reset()
    kinds = set()
    for t in range(2500):
        a = env.teacher_actions() if t % 3 == 0 else env.random_actions()
        kinds |= set(np.unique(a[:, 0]))
        env.step(a)
    assert {0, 1, 2, 3, 4, 5, 9, 11, 13, 14, 15} <= kinds, kinds


def test_teacher_actions_are_valid_conforming_and_better_than_random():
    def run(use_teacher):
        env = BalatroVecEnv(24, seed=3, num_threads=4, max_episode_steps=4000)
        env.reset()
        for _ in range(120 if use_teacher else 400):
            a = env.teacher_actions() if use_teacher else env.random_actions()
            if use_teacher:
                for i in range(24):
                    assert conforms(env.masks, i, a[i]), (i, a[i])
            env.step(a)
            assert env.invalid.sum() == 0
        return env.drain_finished(), env

    teacher_eps, env = run(True)
    random_eps, _ = run(False)
    # Teacher envs are mid-run when we stop: compare progress (blinds beaten + live ante).
    progress = [e[5] for e in teacher_eps] + [
        round(float(a) * 8) for a in env.obs["global_f"][:, GF["ante"]]
    ]
    assert np.mean(progress) > np.mean([e[5] for e in random_eps]) + 0.5


def test_teacher_agent_is_reset_with_the_episode():
    # Two identical envs: one gets teacher_actions called early on, then both must still agree
    # after an auto-reset (the agent state of a finished episode must not leak into the next).
    def go(call_teacher_first):
        env = BalatroVecEnv(4, seed=6, num_threads=1, max_episode_steps=6)
        env.reset()
        if call_teacher_first:
            env.teacher_actions()
        log = []
        for _ in range(8):
            a = env.random_actions().copy()
            env.step(a)
            log.append(env.obs["global_f"].copy())
        # After >= 6 steps every env auto-reset (truncation): fresh episode.
        return env.teacher_actions().copy(), log

    a, _ = go(False)
    b, _ = go(True)
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------- hidden information


def play_game(seed=0):
    g = Game(seed=seed)
    g.step_parts(0, 0, 0, 0)  # SelectBlind -> Play with a dealt hand
    assert g.masks()["kind_mask"][2]
    return g


def all_arrays(g):
    return {**g.observation(), **g.masks()}


def test_face_down_identity_does_not_leak():
    n_changed = 0
    for seed in range(6):
        g = play_game(seed)
        hand = int(g.observation()["hand_valid"].sum())
        down = 0b10110101 & ((1 << hand) - 1)
        before = g.state_json()
        visible = all_arrays(g)
        g.debug_face_down(down)
        ref = all_arrays(g)
        # Face-down rows: flagged, every categorical hidden (0), no features.
        fd = unpack_bits(np.array(down))  # (32,) bool, hand positions flagged face down
        up = ~fd & ref["hand_valid"]
        assert (ref["hand_f"][fd, HF["face_down"]] == 1).all()
        assert not ref["hand_i"][fd].any()
        assert not ref["hand_f"][fd][:, [HF["chips"], HF["perma_bonus"], HF["debuffed"]]].any()
        assert np.array_equal(ref["hand_i"][up], visible["hand_i"][up])  # visible cards unchanged
        for s in range(8):
            g.debug_scramble_hidden(1000 * seed + s)
            now = all_arrays(g)
            for k in ref:
                assert np.array_equal(ref[k], now[k]), (seed, s, k)
            n_changed += g.state_json() != before
        # The real state did change (hidden identities / draw order / RNG), the view did not.
        assert g.state_json() != before
    assert n_changed > 0


def test_state_json_never_reaches_the_observation():
    # Two runs of the same seed are identical, a different seed differs: the observation is a
    # function of the visible state, not of uids or ordering of hidden things.
    a, b = play_game(1), play_game(1)
    ra, rb = all_arrays(a), all_arrays(b)
    assert all(np.array_equal(ra[k], rb[k]) for k in ra)
    a.debug_scramble_hidden(5)  # nothing is face down: only the draw order and the RNG change
    for k, v in all_arrays(b).items():
        assert np.array_equal(v, all_arrays(a)[k]), k


def test_cerulean_bell_forced_card_is_encoded():
    g = play_game(2)
    assert g.debug_force_card(3)
    obs, m = g.observation(), g.masks()
    forced = obs["hand_f"][:, HF["forced"]] > 0
    assert forced.sum() == 1 and forced[3]
    assert int(m["sel_required"][0]) == 1 << 3 == int(m["sel_required"][1])
    # A play without the forced card is rejected as a no-op; with it, accepted.
    _, _, invalid = g.step_parts(2, 0, 0, 0b111)
    assert invalid
    _, _, invalid = g.step_parts(2, 0, 0, 0b1011)
    assert not invalid


def test_hand_tokens_describe_the_dealt_hand():
    g = play_game(3)
    obs = g.observation()
    hv = obs["hand_valid"]
    assert hv.sum() == 8 and hv[:8].all()
    hi = obs["hand_i"][:8]
    assert (hi[:, HI["rank"]] >= 1).all() and (hi[:, HI["rank"]] <= 13).all()
    assert (hi[:, HI["suit"]] >= 1).all() and (hi[:, HI["suit"]] <= 4).all()
    assert (hi[:, HI["enhancement"]] == 1).all() and (hi[:, HI["edition"]] == 1).all()
    # The Red deck is rank-sorted by the engine: the hand arrives sorted.
    ranks = hi[:, HI["rank"]]
    assert (np.diff(ranks) <= 0).all() or (np.diff(ranks) >= 0).all()
    # Position feature.
    assert np.allclose(obs["hand_f"][:8, HF["pos"]], np.arange(8) / 16)


def test_sel_row_helper_matches_layout():
    assert sel_row(2, 0) == 0 and sel_row(3, 0) == 1
    assert sel_row(6, 0) == 2 and sel_row(14, 0) == 2 + CAPS["cons"]
    assert sel_row(4, 0) is None and sel_row(14, CAPS["pack_items"]) is None
