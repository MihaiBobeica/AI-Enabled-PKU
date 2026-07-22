"""Main-machine sleep runner: PPO full train, then DQN full train.

Runs one after the other so they do not fight for the same CPU.
Designed so you can start this, sleep, and wake to finished (or nearly finished) models.

Usage:
  python overnight_main_machine.py --hours 1
  python overnight_main_machine.py --skip-search
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

LAB3 = Path(__file__).resolve().parent
PPO = LAB3 / "3.PPO" / "train"
DQN = LAB3 / "2.DQN" / "train"


def run(cwd: Path, args: list[str]) -> None:
    cmd = [sys.executable, "overnight.py", *args]
    print(f"\n===== {' '.join(cmd)}  (cwd={cwd}) =====\n", flush=True)
    subprocess.check_call(cmd, cwd=str(cwd))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hours",
        type=float,
        default=1.0,
        help="Optuna hours per method before its full train (keep small to finish trains)",
    )
    parser.add_argument(
        "--skip-search",
        action="store_true",
        help="Skip Optuna; train with current config / existing BEST overrides",
    )
    args = parser.parse_args()

    common = ["--skip-search"] if args.skip_search else ["--hours", str(args.hours)]

    # PPO first (2M, needed for Part IV), then DQN (5M).
    run(PPO, common)
    run(DQN, common)
    print("\n[OVERNIGHT] main machine finished PPO + DQN\n", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
