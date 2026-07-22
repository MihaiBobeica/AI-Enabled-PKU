"""Apply BEST.json hyperparameters into config for a full overnight TD3 train.

Usage:
  python apply_best.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--best", type=Path, default=Path("BEST.json"))
    args = parser.parse_args()
    if not args.best.exists():
        raise SystemExit(f"Missing {args.best}. Run search_params.py first.")
    data = json.loads(args.best.read_text(encoding="utf-8"))
    params = data.get("params", {})
    overrides = {
        "action_noise_sigma": float(params.get("action_noise_sigma", 0.1)),
        "critic_learning_rate": float(params.get("critic_learning_rate", 1e-3)),
        "batch_size": int(params.get("batch_size", 128)),
        "total_timesteps": 5_000_000,
    }
    print("BEST score:", data.get("score"))
    print(json.dumps(overrides, indent=2))
    Path("best_config_overrides.json").write_text(
        json.dumps({"TD3": overrides}, indent=2), encoding="utf-8"
    )
    print("Wrote best_config_overrides.json — merge into config.TD3 then: python run.py --worker train")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
