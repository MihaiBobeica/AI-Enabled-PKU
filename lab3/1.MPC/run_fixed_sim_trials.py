"""Run 10 fixed-setting MPC simulation trials for Report-3 tables.

Reads frozen_mpc_params.json (from apply_best.py) or BEST.json.
Writes CSVs under data/mpc/sim/.

Usage:
  python run_fixed_sim_trials.py
  python run_fixed_sim_trials.py --trials 10
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from rip_mpc_sim import (
    ESTIMATOR_DIFFERENTIAL,
    ControllerConfig,
    MPCConfig,
    NoiseConfig,
    initial_state_from_name,
    result_metrics,
    run_simulation,
)


def normalize_params(p: dict) -> dict:
    return {
        "horizon": int(p["horizon"]),
        "q_theta": float(p.get("q_theta", 1.0)),
        "q_theta_dot": float(p.get("q_theta_dot", 0.05)),
        "q_alpha": float(p["q_alpha"]),
        "q_alpha_dot": float(p["q_alpha_dot"]),
        "r_input": float(p["r_input"]),
        "pgd_iterations": int(p["pgd_iterations"]),
        "estimator": str(p.get("estimator", ESTIMATOR_DIFFERENTIAL)),
        "velocity_lpf": float(p.get("velocity_lpf", 0.25)),
    }


def load_params(root: Path) -> dict:
    frozen = root / "frozen_mpc_params.json"
    best = root / "BEST.json"
    if frozen.exists():
        return normalize_params(json.loads(frozen.read_text(encoding="utf-8")))
    if best.exists():
        return normalize_params(json.loads(best.read_text(encoding="utf-8"))["params"])
    raise SystemExit("Need frozen_mpc_params.json or BEST.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--protocol", type=Path, default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    lab3 = root.parent
    protocol_path = args.protocol or (lab3 / "docs" / "common_test_protocol.json")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.exists() else {}
    seeds = list(protocol.get("simulation_seed_list", list(range(args.trials))))[: args.trials]

    params = load_params(root)
    out_dir = lab3 / "submission" / "data" / "mpc" / "sim"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "MPC_SIM_summary.csv"
    fields = [
        "trial", "seed", "success", "capture_time", "alpha_abs_mean",
        "alpha_abs_std", "pwm_abs_mean", "max_abs_theta", "stable_duration",
    ]
    rows = []
    mpc = MPCConfig(**params)
    ctrl = ControllerConfig(pwm_limit=float(protocol.get("pwm_safety_limit", 150)))

    for i, seed in enumerate(seeds, start=1):
        result = run_simulation(
            mpc,
            ctrl,
            duration=float(protocol.get("test_duration_s", args.duration)),
            noise_config=NoiseConfig(enabled=False),
            initial_state=initial_state_from_name("downward"),
            rng=np.random.default_rng(int(seed)),
        )
        m = result_metrics(result)
        success = math.isfinite(float(m.get("capture_time", math.nan))) and float(m.get("stable_duration", 0)) > 0
        row = {
            "trial": i,
            "seed": seed,
            "success": int(success),
            "capture_time": m.get("capture_time"),
            "alpha_abs_mean": m.get("alpha_abs_mean"),
            "alpha_abs_std": m.get("alpha_abs_std"),
            "pwm_abs_mean": m.get("pwm_abs_mean"),
            "max_abs_theta": m.get("max_abs_theta"),
            "stable_duration": m.get("stable_duration"),
        }
        rows.append(row)
        trial_csv = out_dir / f"MPC_SIM_R{i:02d}.csv"
        with trial_csv.open("w", encoding="utf-8", newline="") as handle:
            w = csv.writer(handle)
            w.writerow(["t", "theta", "alpha", "pwm"])
            for t, th, al, pw in zip(result.time, result.theta, result.alpha, result.pwm):
                w.writerow([float(t), float(th), float(al), float(pw)])
        print(f"MPC_SIM_R{i:02d} success={success} capture={m.get('capture_time')}", flush=True)

    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
