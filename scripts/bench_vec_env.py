"""VecEnv throughput with a Python loop and no NN: env steps/s of ``random_actions`` +
``step`` (structured actions, full token observation and masks written every step), and the
speed of ``teacher_actions`` (heuristic-agent actions/s).

Usage: ``uv run --no-sync python scripts/bench_vec_env.py [--threads 1 12]
[--num-envs 256 1024] [--seconds 3] [--teacher-seconds 5]``.
"""

from __future__ import annotations

import argparse
import time

from balatro_rl import BalatroVecEnv


def bench(num_envs: int, threads: int, seconds: float, seed: int = 0) -> tuple[float, float, float]:
    """Returns (steps/s, finished episodes/s, invalid actions per step)."""
    env = BalatroVecEnv(num_envs, seed=seed, num_threads=threads)
    env.reset()
    for _ in range(50):  # warm-up (thread pool spin-up, caches)
        env.step(env.random_actions())
    env.drain_finished()
    iters = 0
    invalid = 0
    t0 = time.perf_counter()
    deadline = t0 + seconds
    while time.perf_counter() < deadline:
        for _ in range(20):
            env.step(env.random_actions())
            invalid += int(env.invalid.sum())
        iters += 20
    dt = time.perf_counter() - t0
    episodes = len(env.drain_finished())
    return iters * num_envs / dt, episodes / dt, invalid / (iters * num_envs)


def bench_teacher(
    num_envs: int, threads: int, seconds: float, seed: int = 0
) -> tuple[float, float]:
    """Returns (teacher actions/s, env steps/s) of a loop ``teacher_actions`` + ``step``."""
    env = BalatroVecEnv(num_envs, seed=seed, num_threads=threads)
    env.reset()
    t_act = 0.0
    t_step = 0.0
    iters = 0
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        t0 = time.perf_counter()
        acts = env.teacher_actions()
        t1 = time.perf_counter()
        env.step(acts)
        t2 = time.perf_counter()
        t_act += t1 - t0
        t_step += t2 - t1
        iters += 1
    return iters * num_envs / t_act, iters * num_envs / (t_act + t_step)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 12])
    ap.add_argument("--num-envs", type=int, nargs="+", default=[256, 1024])
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--teacher-seconds", type=float, default=5.0)
    args = ap.parse_args()
    print("random structured policy (random_actions + step, obs + masks written):")
    print(f"{'num_envs':>9} {'threads':>8} {'steps/s':>12} {'episodes/s':>11} {'invalid/step':>13}")
    for t in args.threads:
        for n in args.num_envs:
            sps, eps, inv = bench(n, t, args.seconds)
            print(f"{n:>9} {t:>8} {sps:>12,.0f} {eps:>11,.0f} {inv:>13.6f}")
    print("teacher policy (teacher_actions, then step):")
    print(f"{'num_envs':>9} {'threads':>8} {'teacher act/s':>14} {'loop steps/s':>13}")
    for t in args.threads:
        for n in args.num_envs:
            aps, sps = bench_teacher(n, t, args.teacher_seconds)
            print(f"{n:>9} {t:>8} {aps:>14,.0f} {sps:>13,.0f}")


if __name__ == "__main__":
    main()
