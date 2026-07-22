"""Single entry point for DQN training and testing."""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
THIRD_PARTY = PROJECT_ROOT / "third_party"
LOCAL_SB3 = PROJECT_ROOT / "stable_baselines3"

# Bundled packages take precedence; torch/numpy/matplotlib come from the active venv.
for path in (PROJECT_ROOT, LOCAL_SB3, THIRD_PARTY):
    value = str(path)
    if path.exists():
        while value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)


def parse_args():
    parser = argparse.ArgumentParser(description="Furuta DQN workflow")
    parser.add_argument("--worker", choices=("panel", "train", "smoke", "test"), default="panel")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--variant", choices=("current", "history", "no_history"), default="current")
    parser.add_argument("--smoke", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def print_dependency_paths() -> None:
    import cloudpickle
    import gymnasium
    import stable_baselines3
    print(f"[LOCAL DEP] python       = {sys.executable}")
    print(f"[LOCAL DEP] gymnasium    = {Path(gymnasium.__file__).resolve()}")
    print(f"[LOCAL DEP] cloudpickle  = {Path(cloudpickle.__file__).resolve()}")
    print(f"[LOCAL DEP] SB3          = {Path(stable_baselines3.__file__).resolve()} (bundled; custom DQN loop is used)")


def main() -> int:
    args = parse_args()
    try:
        print_dependency_paths()
        if args.worker == "panel":
            from panel import TrainingPanel
            TrainingPanel().mainloop()
            return 0
        if args.worker in {"train", "smoke"}:
            from train_worker import run_training
            summary = run_training(smoke=args.worker == "smoke")
            if args.worker == "smoke":
                from test_worker import run_test
                run_test(summary["run_dir"], variant="current", smoke=True)
            return 0
        if args.worker == "test":
            if not args.run_dir:
                raise ValueError("--run-dir is required")
            from test_worker import run_test
            run_test(args.run_dir, variant="current", smoke=bool(args.smoke))
            return 0
        raise RuntimeError(args.worker)
    except Exception as exc:
        print("[PANEL_JSON] " + json.dumps({"event": "stage_error", "stage": args.worker, "error": repr(exc)}, ensure_ascii=False), flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
