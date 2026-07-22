"""Apply BEST.json MPC parameters and optionally re-run a confirmation sim.

Usage:
  python apply_best.py
  python apply_best.py --confirm-sim
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rip_mpc_sim import (
    ControllerConfig,
    MPCConfig,
    NoiseConfig,
    initial_state_from_name,
    result_metrics,
    run_simulation,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--best", type=Path, default=Path("BEST.json"))
    parser.add_argument("--confirm-sim", action="store_true")
    args = parser.parse_args()
    if not args.best.exists():
        raise SystemExit(f"Missing {args.best}. Run search_params.py first.")
    data = json.loads(args.best.read_text(encoding="utf-8"))
    params = data["params"]
    frozen = {
        "horizon": int(params["horizon"]),
        "q_theta": 1.0,
        "q_theta_dot": 0.05,
        "q_alpha": float(params["q_alpha"]),
        "q_alpha_dot": float(params["q_alpha_dot"]),
        "r_input": float(params["r_input"]),
        "pgd_iterations": int(params["pgd_iterations"]),
    }
    Path("frozen_mpc_params.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print("Froze MPC params -> frozen_mpc_params.json")
    print(json.dumps(frozen, indent=2))
    if args.confirm_sim:
        mpc = MPCConfig(**frozen)
        result = run_simulation(
            mpc,
            ControllerConfig(),
            duration=10.0,
            noise_config=NoiseConfig(enabled=False),
            initial_state=initial_state_from_name("downward"),
            rng=np.random.default_rng(0),
        )
        print(json.dumps(result_metrics(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
