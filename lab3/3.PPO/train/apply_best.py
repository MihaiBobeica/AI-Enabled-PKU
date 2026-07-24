"""Apply BEST / BEST_long hyperparameters into best_config_overrides.json.

Usage:
  python apply_best.py
  python apply_best.py --best BEST_long.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

KL_MAP = {0: None, 1: 0.035, 2: 0.02, 3: 0.05}

# Defaults matching short-Optuna trial 175 when long study freezes these.
FROZEN_LR = 0.0001755834889508841
FROZEN_ENT = 0.0006163092363741561
FROZEN_KL = 0.05


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--best", type=Path, default=Path("BEST.json"))
    args = parser.parse_args()
    if not args.best.exists():
        raise SystemExit(f"Missing {args.best}. Run search_params.py / search_params_long.py first.")
    data = json.loads(args.best.read_text(encoding="utf-8"))
    params = data.get("params", {})
    extra = data.get("extra", {}) or {}
    frozen = extra.get("frozen", {}) if isinstance(extra, dict) else {}

    is_long = "n_epochs" in params or "dr_curriculum_fraction" in params or "lr_schedule" in params

    if is_long:
        ppo = {
            "learning_rate": float(frozen.get("learning_rate", FROZEN_LR)),
            "ent_coef": float(frozen.get("ent_coef", FROZEN_ENT)),
            "target_kl": float(frozen.get("target_kl", FROZEN_KL)),
            "n_epochs": int(params.get("n_epochs", 10)),
            "clip_range": float(params.get("clip_range", 0.2)),
            "log_std_init": float(params.get("log_std_init", -0.7)),
            "lr_schedule": str(params.get("lr_schedule", "constant")),
            "total_timesteps": 2_000_000,
            "n_envs": 16,
        }
        dr = {
            "dr_curriculum_fraction": float(params.get("dr_curriculum_fraction", 0.2)),
        }
        payload = {"PPO": ppo, "DOMAIN_RANDOMIZATION": dr}
    else:
        kl_code = int(params.get("target_kl_code", 1))
        target_kl = params.get("target_kl", KL_MAP.get(kl_code, 0.035))
        if "target_kl" in params and params["target_kl"] is not None:
            target_kl = float(params["target_kl"])
        elif kl_code in KL_MAP:
            target_kl = KL_MAP[kl_code]
        ppo = {
            "learning_rate": float(params.get("learning_rate", 3e-4)),
            "ent_coef": float(params.get("ent_coef", 0.002)),
            "target_kl": target_kl,
            "total_timesteps": 2_000_000,
            "n_envs": 16,
        }
        payload = {"PPO": ppo}

    print("BEST score:", data.get("score"))
    print(json.dumps(payload, indent=2))
    Path("best_config_overrides.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print("Wrote best_config_overrides.json — set OVERNIGHT_APPLY_BEST=1 then: python run.py train")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
