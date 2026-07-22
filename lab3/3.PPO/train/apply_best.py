"""Apply BEST.json hyperparameters into config for a full overnight PPO train.

Usage:
  python apply_best.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

KL_MAP = {0: None, 1: 0.035, 2: 0.02, 3: 0.05}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--best", type=Path, default=Path("BEST.json"))
    args = parser.parse_args()
    if not args.best.exists():
        raise SystemExit(f"Missing {args.best}. Run search_params.py first.")
    data = json.loads(args.best.read_text(encoding="utf-8"))
    params = data.get("params", {})
    kl_code = int(params.get("target_kl_code", 1))
    overrides = {
        "learning_rate": float(params.get("learning_rate", 3e-4)),
        "ent_coef": float(params.get("ent_coef", 0.002)),
        "target_kl": KL_MAP.get(kl_code, 0.035),
        "total_timesteps": 2_000_000,
        "n_envs": 16,
    }
    print("BEST score:", data.get("score"))
    print(json.dumps(overrides, indent=2))
    Path("best_config_overrides.json").write_text(
        json.dumps({"PPO": overrides}, indent=2), encoding="utf-8"
    )
    print("Wrote best_config_overrides.json — merge into config.PPO then: python run.py train")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
