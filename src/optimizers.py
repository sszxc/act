"""Derivative-free optimizers over FiLM theta: ARS, CMA-ES (each with a batched-rollout variant),
and the manual one-at-a-time grid sweep (--method sweep). Each run_* function drives its own
loop, logs a .jsonl trace, and checkpoints reward_curve.png/curves.npz after every round so a
killed run still leaves usable partial results."""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from src.logging_utils import _save_progress_checkpoint


def _handle_optimizer_interrupt(tag: str, e: BaseException, it: int, n_completed: int) -> None:
    """Print a clear message when a run_* optimizer loop is interrupted (Ctrl+C or an
    unhandled exception mid-rollout). The caller still returns its partial history/best-so-far
    normally; main() decides whether to exit cleanly (KeyboardInterrupt) or re-raise (a real
    bug) after saving plots/checkpoints for whatever completed so far.
    """
    kind = "KeyboardInterrupt (Ctrl+C)" if isinstance(e, KeyboardInterrupt) else f"{type(e).__name__}: {e}"
    print(f"[{tag}] interrupted at iter {it} ({kind}); saving partial results ({n_completed} rounds completed)")


def run_ars(
    fitness_fn,
    theta0: np.ndarray,
    *,
    n_iters: int,
    n_pairs: int,
    sigma: float,
    alpha: float,
    seed: int,
    log_path: Path,
):
    rng = np.random.default_rng(seed)
    theta = theta0.astype(np.float64).copy()
    dim = theta.size
    out_dir = log_path.parent
    history_best = []
    history_iter_max = []
    best_so_far = -np.inf
    best_theta = theta.copy()

    t_start = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as flog:
        flog.write(
            f"# ARS dim={dim} sigma={sigma} alpha={alpha} n_pairs={n_pairs} "
            f"start_time={datetime.now().isoformat()}\n"
        )

    interrupted_exc: BaseException | None = None
    it = -1
    try:
        for it in range(n_iters):
            t0 = time.perf_counter()
            print(f"[ARS] iter {it}/{n_iters-1} starting; will eval ≈ {2*n_pairs + 1} rollouts")
            grad = np.zeros(dim, dtype=np.float64)
            iter_best = -np.inf
            snap = theta.copy()
            for _ in range(n_pairs):
                eps = rng.standard_normal(dim)
                t_p = theta + sigma * eps
                t_m = theta - sigma * eps
                r_plus = fitness_fn(t_p)
                r_minus = fitness_fn(t_m)
                if r_plus > iter_best:
                    iter_best = r_plus
                    snap = t_p.copy()
                if r_minus > iter_best:
                    iter_best = r_minus
                    snap = t_m.copy()
                grad += (r_plus - r_minus) / (2.0 * sigma) * eps
            grad /= max(n_pairs, 1)
            theta = theta + alpha * grad
            r_end = fitness_fn(theta)
            if r_end > iter_best:
                iter_best = r_end
                snap = theta.copy()
            if iter_best > best_so_far:
                best_so_far = float(iter_best)
                best_theta = snap.copy()
            history_best.append(best_so_far)
            history_iter_max.append(float(iter_best))
            elapsed_now = time.perf_counter() - t_start
            with open(log_path, "a", encoding="utf-8") as flog:
                flog.write(
                    json.dumps(
                        {
                            "iter": it,
                            "best_so_far": best_so_far,
                            "iter_best": iter_best,
                            "elapsed_sec": round(elapsed_now, 3),
                        }
                    )
                    + "\n"
                )
            _save_progress_checkpoint(
                out_dir,
                "ars_curves.npz",
                {"best_so_far": np.array(history_best), "iter_max": np.array(history_iter_max)},
                np.array(history_best),
                "best_so_far",
                np.array(history_iter_max),
                "iter_max",
                elapsed_now,
            )
            dt = time.perf_counter() - t0
            print(f"[ARS] iter {it}: iter_best={iter_best:.4f} best_so_far={best_so_far:.4f} wall={dt:.2f}s")
    except (KeyboardInterrupt, Exception) as e:
        interrupted_exc = e
        _handle_optimizer_interrupt("ARS", e, it, len(history_best))

    elapsed_sec = time.perf_counter() - t_start
    return best_theta, np.array(history_best), np.array(history_iter_max), interrupted_exc, elapsed_sec


def run_ars_batched(
    fitness_batch_fn,
    theta0: np.ndarray,
    *,
    n_iters: int,
    n_pairs: int,
    sigma: float,
    alpha: float,
    seed: int,
    log_path: Path,
    batch_size: int,
):
    """
    ARS with batched fitness over multiple candidates (single-GPU batched forward).
    fitness_batch_fn: (N,dim)->(N,) episode_return
    """
    rng = np.random.default_rng(seed)
    theta = theta0.astype(np.float64).copy()
    dim = theta.size
    out_dir = log_path.parent
    best_so_far = -np.inf
    best_theta = theta.copy()
    history_best = []
    history_iter_best = []

    t_start = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as flog:
        flog.write(
            f"# ARS(batched) dim={dim} sigma={sigma} alpha={alpha} n_pairs={n_pairs} batch_size={batch_size} "
            f"start_time={datetime.now().isoformat()}\n"
        )

    interrupted_exc: BaseException | None = None
    it = -1
    try:
        for it in range(n_iters):
            t0 = time.perf_counter()
            eps_list = [rng.standard_normal(dim) for _ in range(n_pairs)]
            cand = []
            for eps in eps_list:
                cand.append(theta + sigma * eps)
                cand.append(theta - sigma * eps)
            # evaluate all perturbations
            print(f"[ARS(batched)] iter {it}/{n_iters-1} evaluating {len(cand)} candidates (batch_size={batch_size})")
            rewards = []
            for i in range(0, len(cand), batch_size):
                tb = np.stack(cand[i : i + batch_size], axis=0)
                rewards.extend(list(map(float, fitness_batch_fn(tb))))
            rewards = np.asarray(rewards, dtype=np.float64)

            iter_best = float(np.max(rewards))
            best_idx = int(np.argmax(rewards))
            snap = cand[best_idx].copy()

            # grad estimate
            grad = np.zeros(dim, dtype=np.float64)
            for k, eps in enumerate(eps_list):
                r_plus = rewards[2 * k]
                r_minus = rewards[2 * k + 1]
                grad += (r_plus - r_minus) / (2.0 * sigma) * eps
            grad /= max(n_pairs, 1)

            theta = theta + alpha * grad
            r_end = float(fitness_batch_fn(theta.reshape(1, -1))[0])
            if r_end > iter_best:
                iter_best = r_end
                snap = theta.copy()

            if iter_best > best_so_far:
                best_so_far = iter_best
                best_theta = snap.copy()

            history_best.append(best_so_far)
            history_iter_best.append(iter_best)
            elapsed_now = time.perf_counter() - t_start
            with open(log_path, "a", encoding="utf-8") as flog:
                flog.write(
                    json.dumps(
                        {
                            "iter": it,
                            "best_so_far": best_so_far,
                            "iter_best": iter_best,
                            "elapsed_sec": round(elapsed_now, 3),
                        }
                    )
                    + "\n"
                )
            _save_progress_checkpoint(
                out_dir,
                "ars_curves.npz",
                {"best_so_far": np.array(history_best), "iter_max": np.array(history_iter_best)},
                np.array(history_best),
                "best_so_far",
                np.array(history_iter_best),
                "iter_max",
                elapsed_now,
            )
            dt = time.perf_counter() - t0
            r_mean = float(np.mean(rewards)) if rewards.size else float("nan")
            r_std = float(np.std(rewards)) if rewards.size else float("nan")
            print(
                f"[ARS(batched)] iter {it}: iter_best={iter_best:.4f} best_so_far={best_so_far:.4f} "
                f"mean±std={r_mean:.4f}±{r_std:.4f} wall={dt:.2f}s"
            )
    except (KeyboardInterrupt, Exception) as e:
        interrupted_exc = e
        _handle_optimizer_interrupt("ARS(batched)", e, it, len(history_best))

    elapsed_sec = time.perf_counter() - t_start
    return best_theta, np.asarray(history_best), np.asarray(history_iter_best), interrupted_exc, elapsed_sec


def run_cma(
    fitness_fn,
    theta0: np.ndarray,
    *,
    sigma0: float,
    maxiter: int,
    popsize: int | None,
    seed: int,
    log_path: Path,
):
    import cma

    x0 = theta0.astype(np.float64).copy()
    opts: dict = {
        "seed": seed,
        "maxiter": maxiter,
        "verb_disp": 1,
    }
    if popsize is not None:
        opts["popsize"] = popsize
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    out_dir = log_path.parent
    history_best = []
    history_gen_max = []
    best_so_far = -np.inf
    best_x = x0.copy()

    t_start = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as flog:
        flog.write(
            f"# CMA-ES dim={x0.size} sigma0={sigma0} maxiter={maxiter} popsize={getattr(es, 'popsize', None)} "
            f"start_time={datetime.now().isoformat()}\n"
        )

    interrupted_exc: BaseException | None = None
    gen = -1
    try:
        while not es.stop():
            t0 = time.perf_counter()
            xs = es.ask()
            gen = int(es.countiter)
            print(f"[CMA] gen {gen}/{maxiter-1} evaluating pop={len(xs)} (waiting for rollouts)")
            rewards = [fitness_fn(np.asarray(x, dtype=np.float64)) for x in xs]
            r_arr = np.asarray(list(map(float, rewards)), dtype=np.float64)
            es.tell(xs, [-float(r) for r in rewards])
            gen_max = float(np.max(r_arr)) if r_arr.size else float("nan")
            ib = int(np.argmax(r_arr)) if r_arr.size else 0
            if rewards[ib] > best_so_far:
                best_so_far = float(rewards[ib])
                best_x = np.asarray(xs[ib], dtype=np.float64).copy()
            history_best.append(best_so_far)
            history_gen_max.append(gen_max)
            elapsed_now = time.perf_counter() - t_start
            with open(log_path, "a", encoding="utf-8") as flog:
                flog.write(
                    json.dumps(
                        {
                            "generation": gen,
                            "params": np.asarray(xs[ib], dtype=np.float64).tolist(),
                            "gen_max": gen_max,
                            "best_so_far": best_so_far,
                            "elapsed_sec": round(elapsed_now, 3),
                        }
                    )
                    + "\n"
                )
            _save_progress_checkpoint(
                out_dir,
                "cma_curves.npz",
                {"best_so_far": np.array(history_best), "gen_max": np.array(history_gen_max)},
                np.array(history_best),
                "best_so_far",
                np.array(history_gen_max),
                "gen_max",
                elapsed_now,
            )
            dt = time.perf_counter() - t0
            r_mean = float(np.mean(r_arr)) if r_arr.size else float("nan")
            r_std = float(np.std(r_arr)) if r_arr.size else float("nan")
            print(
                f"[CMA] gen {gen}: gen_max={gen_max:.4f} best_so_far={best_so_far:.4f} "
                f"mean±std={r_mean:.4f}±{r_std:.4f} wall={dt:.2f}s"
            )
    except (KeyboardInterrupt, Exception) as e:
        interrupted_exc = e
        _handle_optimizer_interrupt("CMA", e, gen, len(history_best))

    elapsed_sec = time.perf_counter() - t_start
    return best_x, np.array(history_best), np.array(history_gen_max), interrupted_exc, elapsed_sec


def run_cma_batched(
    fitness_batch_fn,
    theta0: np.ndarray,
    *,
    sigma0: float,
    maxiter: int,
    popsize: int | None,
    seed: int,
    log_path: Path,
    batch_size: int,
):
    import cma

    x0 = theta0.astype(np.float64).copy()
    opts: dict = {"seed": seed, "maxiter": maxiter, "verb_disp": 1}
    if popsize is not None:
        opts["popsize"] = popsize
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    out_dir = log_path.parent
    best_so_far = -np.inf
    best_x = x0.copy()
    history_best = []
    history_gen_max = []

    t_start = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as flog:
        flog.write(
            f"# CMA-ES(batched) dim={x0.size} sigma0={sigma0} maxiter={maxiter} popsize={getattr(es,'popsize',None)} "
            f"batch_size={batch_size} start_time={datetime.now().isoformat()}\n"
        )

    interrupted_exc: BaseException | None = None
    gen = -1
    try:
        while not es.stop():
            t0 = time.perf_counter()
            xs = es.ask()
            rewards = []
            gen = int(es.countiter)
            print(f"[CMA(batched)] gen {gen}/{maxiter-1} evaluating pop={len(xs)} (batch_size={batch_size})")
            for i in range(0, len(xs), batch_size):
                tb = np.stack([np.asarray(x, dtype=np.float64) for x in xs[i : i + batch_size]], axis=0)
                rewards.extend(list(map(float, fitness_batch_fn(tb))))
            rewards = np.asarray(rewards, dtype=np.float64)
            es.tell(xs, list((-rewards).astype(float)))

            gen_max = float(np.max(rewards))
            ib = int(np.argmax(rewards))
            if rewards[ib] > best_so_far:
                best_so_far = float(rewards[ib])
                best_x = np.asarray(xs[ib], dtype=np.float64).copy()
            history_best.append(best_so_far)
            history_gen_max.append(gen_max)
            elapsed_now = time.perf_counter() - t_start
            with open(log_path, "a", encoding="utf-8") as flog:
                flog.write(
                    json.dumps(
                        {
                            "generation": gen,
                            "params": np.asarray(xs[ib], dtype=np.float64).tolist(),
                            "gen_max": gen_max,
                            "best_so_far": best_so_far,
                            "elapsed_sec": round(elapsed_now, 3),
                        }
                    )
                    + "\n"
                )
            _save_progress_checkpoint(
                out_dir,
                "cma_curves.npz",
                {"best_so_far": np.array(history_best), "gen_max": np.array(history_gen_max)},
                np.array(history_best),
                "best_so_far",
                np.array(history_gen_max),
                "gen_max",
                elapsed_now,
            )
            dt = time.perf_counter() - t0
            r_mean = float(np.mean(rewards)) if rewards.size else float("nan")
            r_std = float(np.std(rewards)) if rewards.size else float("nan")
            print(
                f"[CMA(batched)] gen {gen}: gen_max={gen_max:.4f} best_so_far={best_so_far:.4f} "
                f"mean±std={r_mean:.4f}±{r_std:.4f} wall={dt:.2f}s"
            )
    except (KeyboardInterrupt, Exception) as e:
        interrupted_exc = e
        _handle_optimizer_interrupt("CMA(batched)", e, gen, len(history_best))

    elapsed_sec = time.perf_counter() - t_start
    return best_x, np.asarray(history_best), np.asarray(history_gen_max), interrupted_exc, elapsed_sec


def run_sweep(
    fitness_fn,
    theta_base: np.ndarray,
    *,
    dim_names: list[str],
    sweep_values: np.ndarray,
    log_path: Path,
):
    """
    One-at-a-time manual grid sweep (--method sweep): for each name in dim_names (in order),
    set theta[dim] to each of sweep_values while every other dim stays at theta_base, and call
    fitness_fn(theta, dim_name, value) -> reward. Not an optimizer — just drives the grid and
    logs/checkpoints progress the same way run_ars/run_cma do, so it gets the same
    reward_curve.png / interrupt handling for free. fitness_fn is responsible for its own side
    effects (saving video/trajectory) per point.
    """
    out_dir = log_path.parent
    history_best = []
    history_point = []
    best_so_far = -np.inf
    best_theta = theta_base.copy()

    grid = [(i, float(v)) for i in range(len(dim_names)) for v in sweep_values]
    t_start = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as flog:
        flog.write(
            f"# Sweep dim={theta_base.size} n_dims={len(dim_names)} n_values={len(sweep_values)} "
            f"n_points={len(grid)} start_time={datetime.now().isoformat()}\n"
        )

    interrupted_exc: BaseException | None = None
    it = -1
    try:
        for it, (dim_idx, value) in enumerate(grid):
            t0 = time.perf_counter()
            theta = theta_base.copy()
            theta[dim_idx] = value
            dim_name = dim_names[dim_idx]
            print(f"[sweep] point {it}/{len(grid)-1}: {dim_name}={value:g} starting")
            reward = float(fitness_fn(theta, dim_name, value))
            if reward > best_so_far:
                best_so_far = reward
                best_theta = theta.copy()
            history_best.append(best_so_far)
            history_point.append(reward)
            elapsed_now = time.perf_counter() - t_start
            with open(log_path, "a", encoding="utf-8") as flog:
                flog.write(
                    json.dumps(
                        {
                            "point": it,
                            "dim_name": dim_name,
                            "value": value,
                            "reward": reward,
                            "best_so_far": best_so_far,
                            "elapsed_sec": round(elapsed_now, 3),
                        }
                    )
                    + "\n"
                )
            _save_progress_checkpoint(
                out_dir,
                "sweep_curves.npz",
                {"best_so_far": np.array(history_best), "point_reward": np.array(history_point)},
                np.array(history_best),
                "best_so_far",
                np.array(history_point),
                "point_reward",
                elapsed_now,
            )
            dt = time.perf_counter() - t0
            print(
                f"[sweep] point {it}: {dim_name}={value:g} reward={reward:.4f} "
                f"best_so_far={best_so_far:.4f} wall={dt:.2f}s"
            )
    except (KeyboardInterrupt, Exception) as e:
        interrupted_exc = e
        _handle_optimizer_interrupt("sweep", e, it, len(history_best))

    elapsed_sec = time.perf_counter() - t_start
    return best_theta, np.array(history_best), np.array(history_point), interrupted_exc, elapsed_sec
