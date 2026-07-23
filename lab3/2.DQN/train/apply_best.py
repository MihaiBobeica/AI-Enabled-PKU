"""Apply BEST.json hyperparameters into config.py for a full overnight train.

Usage (from this folder):
  python apply_best.py
  python apply_best.py --best BEST.json
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
    print("BEST score:", data.get("score"))
    print("BEST trial:", data.get("trial"))
    print("Params to apply for full overnight train:")
    print(json.dumps(params, indent=2))
    print()
    print("Edit config.py DQN dict accordingly, then:")
    print("  python run.py --worker train")
    print("Suggested mappings:")
    print("  use_double_dqn      -> DQN['use_double_dqn'] = bool(...)")
    print("  learning_rate       -> DQN['learning_rate']")
    print("  exploration_decay   -> DQN['exploration_decay']")
    print("Keep total_timesteps=5_000_000 and the DR stage schedule from the course zip.")
    # Optionally write a ready-to-merge snippet.
    snippet = Path("best_config_overrides.json")
    snippet.write_text(json.dumps({"DQN": {
        "use_double_dqn": bool(params.get("use_double_dqn", 0)),
        "learning_rate": float(params.get("learning_rate", 5e-4)),
        "exploration_decay": float(params.get("exploration_decay", 4e5)),
    }}, indent=2), encoding="utf-8")
    print(f"Wrote {snippet.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
