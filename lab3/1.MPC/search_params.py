"""Optuna TPE search for MPC Q/R/N/niter on the digital twin.

Usage:
  python search_params.py --hours 2
  python search_params.py --hours 2 --resume
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
LAB3 = ROOT.parent
if str(LAB3) not in sys.path:
    sys.path.insert(0, str(LAB3))

from search_common import (  # noqa: E402
    add_common_args,
    quiet_stdio,
    record_trial_row,
    run_timed_study,
    update_best_if_improved,
)
from rip_mpc_sim import (  # noqa: E402
    ControllerConfig,
    MPCConfig,
    NoiseConfig,
    initial_state_from_name,
    result_metrics,
    run_simulation,
)

# Course / zip-aligned defaults (trial 0 baseline).
BASELINE = {
    "horizon": 8,
    "q_alpha": 80.0,
    "q_alpha_dot": 2.0,
    "r_input": 0.001,
    "pgd_iterations": 16,
}


def mpc_score(metrics: dict) -> float:
    """Higher is better: capture + long stable hold + small angle error."""
    capture = metrics.get("capture_time", math.nan)
    stable_dur = float(metrics.get("stable_duration", 0.0) or 0.0)
    alpha_mean = metrics.get("alpha_abs_mean", math.nan)
    captured = math.isfinite(float(capture)) and float(capture) >= 0.0
    if not captured or not math.isfinite(float(alpha_mean)):
        return -1e6 + stable_dur
    # Prefer fast capture and small |alpha|, with stable duration bonus.
    return (
        100.0
        - 2.0 * float(capture)
        - 50.0 * float(alpha_mean)
        + 0.5 * stable_dur
    )


def evaluate_params(params: dict, seed: int = 0) -> tuple[float, dict]:
    mpc = MPCConfig(
        horizon=int(params["horizon"]),
        q_theta=1.0,
        q_theta_dot=0.05,
        q_alpha=float(params["q_alpha"]),
        q_alpha_dot=float(params["q_alpha_dot"]),
        r_input=float(params["r_input"]),
        pgd_iterations=int(params["pgd_iterations"]),
    )
    mpc.validate()
    ctrl = ControllerConfig()
    noise = NoiseConfig(enabled=False)
    result = run_simulation(
        mpc,
        ctrl,
        duration=10.0,
        dt=0.005,
        initial_state=initial_state_from_name("downward"),
        noise_config=noise,
        rng=np.random.default_rng(seed),
    )
    metrics = result_metrics(result)
    return mpc_score(metrics), metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="MPC Optuna TPE search")
    add_common_args(parser)
    parser.set_defaults(hours=2.0)
    args = parser.parse_args()

    root = ROOT
    # Paths mirror SearchState layout (study also creates SearchState).
    logs_dir = root / "search_logs"
    csv_path = root / "search_results.csv"
    best_json = root / "BEST.json"
    best_run = root / "best_run"
    logs_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial) -> float:
        params = {
            "horizon": int(trial.suggest_int("horizon", 6, 14)),
            "q_alpha": float(trial.suggest_float("q_alpha", 20.0, 120.0, log=True)),
            "q_alpha_dot": float(trial.suggest_float("q_alpha_dot", 0.5, 6.0, log=True)),
            "r_input": float(trial.suggest_float("r_input", 1e-4, 5e-3, log=True)),
            "pgd_iterations": int(trial.suggest_int("pgd_iterations", 8, 32)),
        }
        log_path = logs_dir / f"trial_{trial.number:04d}.log"
        try:
            with quiet_stdio(log_path, args.verbose):
                score, metrics = evaluate_params(params, seed=trial.number)
        except Exception as exc:
            record_trial_row(csv_path, trial, float("nan"), failed=True)
            print(f"FAIL trial={trial.number} err={exc}", flush=True)
            raise

        trial_dir = logs_dir / f"trial_{trial.number:04d}_artifacts"
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
        (trial_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        record_trial_row(csv_path, trial, score, run_dir=str(trial_dir))
        update_best_if_improved(
            best_json=best_json,
            best_run=best_run,
            trial=trial,
            score=score,
            run_dir=trial_dir,
            extra={"metrics": metrics},
        )
        print(f"OK trial={trial.number} score={score:.4f} params={params}", flush=True)
        return score

    # Cap MPC trials so hours don't create unbounded hundreds unless asked.
    n_trials = args.trials if args.trials is not None else 250
    run_timed_study(
        root=root,
        study_name="mpc_tpe",
        hours=args.hours,
        resume=args.resume,
        n_trials=n_trials,
        baseline_params=BASELINE,
        objective=objective,
        direction="maximize",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
