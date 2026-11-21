# BalatroRL task runner. Run `source ~/.cargo/env` first if cargo is not on PATH.
SHELL := /bin/bash
export PATH := $(HOME)/.cargo/bin:$(PATH)

.PHONY: setup setup-train test test-rs test-py lint fmt build-py bench bc-smoke ppo-e2e ppo-s1 eval-ckpt

# `uv run --no-sync` so uv never rebuilds/uninstalls the maturin-installed extension.
UVR := uv run --no-sync

setup:
	uv sync --no-install-project --inexact
	$(UVR) maturin develop --release --quiet

# PyTorch (CUDA 12.8 wheels, ~6.6 GB with the NVIDIA libs) for the model / training code.
setup-train:
	uv sync --no-install-project --inexact --group train
	$(UVR) maturin develop --release --quiet

# Behaviour-cloning smoke test (collect teacher data, train, play games); see docs/design/model.md.
bc-smoke:
	$(UVR) python scripts/bc_smoke.py --stage all

# PPO (docs/design/training.md). Runs go to runs/<name>/ (gitignored); override NAME / STEPS.
NAME ?= e2e
STEPS ?= 30e6
ppo-e2e:  # end-to-end PPO from the BC checkpoint with a decaying KL penalty to the BC policy
	$(UVR) python -u scripts/train_ppo.py --name $(NAME) --init runs/bc/model.pt --lr 1e-4 \
	    --kl-coef 0.3 --kl-anneal-updates 150 --critic-warmup 2 --total-steps $(STEPS)

ppo-s1:  # from scratch, curriculum stage 1 (model plays PLAY, teacher the rest)
	$(UVR) python -u scripts/train_ppo.py --name $(NAME) --curriculum play --total-steps $(STEPS)

eval-ckpt:  # 1000-game test set; usage: make eval-ckpt CKPT=runs/e2e/best.pt
	$(UVR) python scripts/evaluate.py --checkpoint $(CKPT)

test: test-rs test-py

test-rs:
	cargo test --workspace --quiet

test-py: build-py
	$(UVR) pytest -q

build-py:
	$(UVR) maturin develop --release --quiet

lint:
	cargo fmt --all -- --check
	cargo clippy --workspace --all-targets --quiet -- -D warnings
	$(UVR) ruff check python tests_py scripts
	$(UVR) ruff format --check python tests_py scripts

fmt:
	cargo fmt --all
	$(UVR) ruff format python tests_py scripts
	$(UVR) ruff check --fix python tests_py scripts

bench:
	cargo bench -p balatro-core
