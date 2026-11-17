#!/bin/bash
# G4 speed matrix: one exclusive GPU job (run it under the queue / `flock runs/.gpu.lock`).
# Every config = 1 warm-up + 3 measured real updates (1024 envs x 128 steps, BC init, KL 0.3),
# <= ~3 min each (compile configs add ~20-40 s of warm-up). Results: one JSON line per config in
# $OUT (default runs/bench_g4.jsonl), console log on stdout.
#   usage: scripts/bench_matrix.sh [OUT.jsonl]
set -u
export PATH=$HOME/.cargo/bin:$PATH
OUT=${1:-runs/bench_g4.jsonl}
BASE="--init runs/bc/model.pt --kl-coef 0.3 --updates 3 --warmup 1 --bench-out $OUT"
run() { echo "=== $*"; timeout 400 uv run --no-sync python scripts/bench_update.py $BASE "$@" 2>&1 | grep -v -i warn | tail -6; }
run --profile                                            # baseline + per-section breakdown
run                                                      # baseline, no profiler overhead
run --update-dtype bf16                                  # bf16 autocast update
run --update-dtype bf16 --profile                        # where the rest of the time goes
run --update-dtype bf16 --recompute-old-logp             # exact first-minibatch ratio
run --compile                                            # encoder torch.compile (fp32)
run --update-dtype bf16 --compile
run --epochs 1
run --update-dtype bf16 --epochs 1
run --update-dtype bf16 --num-envs 2048                  # same minibatch size, 2x rows/update
run --update-dtype bf16 --num-envs 2048 --epochs 1
run --update-dtype bf16 --compile --num-envs 2048 --epochs 1
