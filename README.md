# Ai-Learns-Play-Balatro
A reinforcement learning agent that plays a **Balatro-style poker roguelike** from start to
finish (blind selection, playing hands, shop, booster packs), trained entirely in a fast,
headless Rust re-implementation of the game's rules.

The agent is a single network organised as a *council*: a shared "game reader" encoder, a
recurrent strategy memory carried across phases, and one expert head per decision phase. All
experts optimise **one shared reward**, so they cooperate rather than compete.

| Phase          | Expert       | Decides                                              |
|----------------|--------------|------------------------------------------------------|
| Blind select   | Blind expert | play, skip (take the tag) or reroll the boss         |
| Play           | Hand expert  | which cards to play / discard, consumables, joker order |
| Shop           | Shop expert  | buy, sell, reroll, open packs, use consumables, leave |
| Pack open      | Pack expert  | which card to take from a booster pack               |

## Results

Best model: `final200` (PPO, 200M environment steps, ~2M parameters). Evaluated on a fixed
**10,000-game holdout** set of seeds never used in training, Red Deck, deterministic policy,
a run counts as won after beating the Ante 8 boss:

| Stake  | White | Red  | Green | Black | Blue | Purple | Orange | Gold |
|--------|------:|-----:|------:|------:|-----:|-------:|-------:|-----:|
| Win %  | **84.6** | 77.0 | 46.1 | 38.7 | 36.1 | 12.6 | 7.1 | 2.9 |

In endless mode (capped at Ante 12) the won White-stake runs reach Ante 10.2 on average, and
27% of them get past Ante 11. For reference, the hand-written heuristic agent built into the
simulator wins 19.5% of Red / White runs.

## The simulator

`crates/balatro-core` is a deterministic, seedable rules engine with full base-game content:

- all 150 jokers, 52 consumables (Tarot, Planet, Spectral), 28 boss blinds, 32 vouchers,
  24 tags, 15 booster packs, card enhancements / editions / seals, all 15 decks and 8 stakes;
- shop, economy, interest, rerolls, skips, endless mode;
- exact scoring order (retriggers, copy jokers, held-in-hand effects) and per-hand score traces.

Rules follow the community Balatro Wiki (sources are cited per rule in
[`docs/mechanics`](docs/mechanics)) and were checked hand by hand against the real game through
the bridge below: on deterministic hands the simulator's score matched the game's score exactly
in 4009 of 4010 hands.

Throughput (Ryzen 5 7600, observations and action masks included): about 2.6M steps/s on one
core and 8M steps/s with the batched `VecEnv` on 12 threads, so training is bound by the
network, not the simulator.

## Repository layout

```
crates/balatro-core   Rust rules engine, baseline agents, CLI tools (no Python dependency)
crates/balatro-py     PyO3 bindings -> Python module `balatro_rl._engine` (Env, VecEnv)
python/balatro_rl     Model (encoder, memory, heads), PPO / BC training, evaluation, bridge
scripts/              Training, evaluation, real-game bridge and monitoring entry points
bridge/               BalatroBot patch for driving the real game (optional)
docs/design           Design notes: RL interface, model, training, bridge, hooks
docs/mechanics        Rule references with sources
```

## Getting started

Requirements: Rust (stable, via rustup), [uv](https://github.com/astral-sh/uv), Python 3.12.

```bash
make setup         # create .venv, install dev deps, build the Rust extension
make setup-train   # add PyTorch (CUDA 12.8 wheels) for model / training code
make test          # cargo tests + pytest
make lint          # rustfmt, clippy (-D warnings), ruff
```

Play a game in the terminal, or let the built-in agents play:

```bash
cargo run --release -p balatro-core --bin play_cli
cargo run --release -p balatro-core --bin eval_agents
```

### Evaluate the trained model

Download `final200.pt` from the [Releases](../../releases) page, then:

```bash
uv run --no-sync python scripts/evaluate.py --checkpoint final200.pt --set holdout --stake white
```

### Train

```bash
uv run --no-sync python scripts/train_ppo.py --help
uv run --no-sync python scripts/reproduce_final200.py --dry-run   # the whole chain, as commands
```

`final200` was the last stage of a chain of nine runs (behaviour cloning from the heuristic
agent, then PPO stages on growing deck / stake sets). The recorded configuration of every
stage is in [`configs/final200/`](configs/final200), and `scripts/reproduce_final200.py` runs
the whole chain (about 17 hours on one 16 GB GPU, resumable; `--smoke` is a 35-second test).
The stage table, wall times and the caveats (a rerun is statistically similar, not
bit-identical) are in [`docs/design/training.md`](docs/design/training.md#17-reproducing-final200).

### Real game (optional)

`python/balatro_rl/bridge` plays the real game through
[BalatroBot](https://github.com/coder/balatrobot) 1.5.2 (Steamodded 1.0.0-beta-1503a,
Lovely), translating the game state into the simulator's observation and checking every
played hand's score against the simulator. It requires your own copy of the game. See
[`docs/design/bridge.md`](docs/design/bridge.md) for setup and `scripts/bridge_play.py --help`.

## Scope and legal note

This is an unofficial fan research project. It is not affiliated with or endorsed by
LocalThunk or Playstack. Balatro is a trademark of its respective owners.

The repository contains **no game assets, game code or game text**; the simulator is a
headless training environment, not a playable game, and you need to own the game to use the
real-game bridge.

## License

[MIT](LICENSE). `bridge/lua` contains a patch for and small extensions of
[BalatroBot](https://github.com/coder/balatrobot) (MIT, Copyright (c) 2025 Coder).
