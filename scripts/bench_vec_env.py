"""VecEnv throughput: env steps/s with a Python loop (`random_actions` + `step` per iteration),
no NN. Usage: ``uv run --no-sync python scripts/bench_vec_env.py [--threads 1 12]
[--num-envs 64 256 1024] [--seconds 3]``.
"""

from __future__ import annotations

import argparse
import time

from balatro_rl import BalatroVecEnv


def bench(num_envs: int, threads: int, seconds: float, seed: int = 0) -> tuple[float, float]:
    """Returns (steps/s, mean done episodes/s)."""
    env = BalatroVecEnv(num_envs, seed=seed, num_threads=threads)
    env.reset()
    for _ in range(50):  # warm-up (thread pool spin-up, caches)
        env.step(env.random_actions())
    env.drain_finished()
    iters = 0
    t0 = time.perf_counter()
    deadline = t0 + seconds
    while time.perf_counter() < deadline:
        for _ in range(20):
            env.step(env.random_actions())
        iters += 20
    dt = time.perf_counter() - t0
    episodes = len(env.drain_finished())
    return iters * num_envs / dt, episodes / dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 12])
    ap.add_argument("--num-envs", type=int, nargs="+", default=[64, 256, 1024])
    ap.add_argument("--seconds", type=float, default=3.0)
    args = ap.parse_args()
    print(f"{'num_envs':>9} {'threads':>8} {'steps/s':>12} {'episodes/s':>11}")
    for t in args.threads:
        for n in args.num_envs:
            sps, eps = bench(n, t, args.seconds)
            print(f"{n:>9} {t:>8} {sps:>12,.0f} {eps:>11,.0f}")


if __name__ == "__main__":
    main()
