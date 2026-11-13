# council-of-jokers

A reinforcement learning project that teaches a **team of specialist agents** to play a
Balatro-style poker roguelike.

Instead of one monolithic policy, the agent is a *council*: a shared "game reader" encoder,
a strategy memory carried across phases, and one expert head per decision phase:

| Phase          | Expert            | Decides                                            |
|----------------|-------------------|----------------------------------------------------|
| Blind select   | Blind expert      | play or skip the blind                             |
| Play           | Hand expert       | which cards to play / discard, consumable use      |
| Shop           | Shop expert       | buy, sell, reroll, open packs, leave               |
| Pack open      | Pack expert       | which card to take from a booster pack             |

All experts optimize **one shared reward**, so they cooperate rather than compete.

## Status

Early development. The first milestone is a fast, deterministic, headless rules simulator.
See [`prd.md`](prd.md) for the full phased plan and [`notes.md`](notes.md) for engineering
notes (both in Turkish).

## Architecture

```
crates/balatro-core   Rust rules engine (no Python dependency)
crates/balatro-py     PyO3 bindings -> Python module `balatro_rl._engine`
python/balatro_rl     Env wrappers, agents, training code (PyTorch)
docs/mechanics        Rule references with sources
```

- **Simulator:** Rust, chosen for throughput (~40x faster than pure Python in an early
  benchmark), cheap state cloning and exhaustive `match` checks over all content.
- **Training:** Python + PyTorch, PPO with phase-masked expert heads.

## Getting started

Requirements: Rust (stable, via rustup), [uv](https://github.com/astral-sh/uv), Python 3.12.

```bash
make setup     # create .venv, install dev deps, build the Rust extension
make test      # cargo tests + pytest
make lint      # rustfmt, clippy (-D warnings), ruff
make bench     # criterion benchmarks
```

## Scope and legal note

This repository contains **no game assets, game code or game text**. Mechanics are modeled
from public descriptions for research purposes. The simulator is a headless training
environment, not a playable game. Balatro is a trademark of its respective owners; this
project is unofficial and not affiliated with them.
