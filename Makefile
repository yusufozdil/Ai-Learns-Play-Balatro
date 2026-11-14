# BalatroRL task runner. Run `source ~/.cargo/env` first if cargo is not on PATH.
SHELL := /bin/bash
export PATH := $(HOME)/.cargo/bin:$(PATH)

.PHONY: setup setup-train test test-rs test-py lint fmt build-py bench snapshot bc-smoke

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

# Phase-end archive (no git yet). Usage: make snapshot PHASE=1
snapshot:
	@test -n "$(PHASE)" || (echo "usage: make snapshot PHASE=<n>" && exit 1)
	tar --exclude=./target --exclude=./.venv --exclude=./snapshots --exclude='__pycache__' \
	    --exclude='*.so' -czf snapshots/phase-$(PHASE)-$$(date +%Y%m%d-%H%M).tar.gz .
	@ls -lh snapshots | tail -1
