"""One-command overnight for DQN: Optuna then full train (wall-clock capped).

  python overnight.py
  python overnight.py --hours 3 --train-hours 6
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="DQN: Optuna then timed full train")
    parser.add_argument("--hours", type=float, default=3.0, help="Optuna wall-clock hours")
    parser.add_argument("--train-hours", type=float, default=6.0, help="Full-train wall-clock hours")
    parser.add_argument("--resume-search", action="store_true")
    args = parser.parse_args()

    search_cmd = [sys.executable, str(ROOT / "search_params.py"), "--hours", str(args.hours)]
    if args.resume_search:
        search_cmd.append("--resume")
    print(f"[OVERNIGHT] Optuna {args.hours}h …", flush=True)
    subprocess.check_call(search_cmd, cwd=str(ROOT))

    print("[OVERNIGHT] apply BEST.json → best_config_overrides.json", flush=True)
    subprocess.check_call([sys.executable, str(ROOT / "apply_best.py")], cwd=str(ROOT))

    env = os.environ.copy()
    env["OVERNIGHT_APPLY_BEST"] = "1"
    train_cmd = [sys.executable, str(ROOT / "run.py"), "--worker", "train"]
    print(f"[OVERNIGHT] full train up to {args.train_hours}h …", flush=True)
    try:
        completed = subprocess.run(
            train_cmd,
            cwd=str(ROOT),
            env=env,
            timeout=max(60.0, float(args.train_hours) * 3600.0),
        )
        print(f"[OVERNIGHT] train exited code={completed.returncode}", flush=True)
        return int(completed.returncode)
    except subprocess.TimeoutExpired:
        print(
            f"[OVERNIGHT] train wall-clock {args.train_hours}h reached — "
            "stopped. Use latest best_/checkpoint under runs/.",
            flush=True,
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
