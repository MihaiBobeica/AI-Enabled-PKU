"""Optuna Longer → apply BEST_long → full 2M train → distill.

Search wall-clock is capped. The final full train is never time-killed.

Usage:
  python overnight_long.py --search-hours 6
  python overnight_long.py --search-hours 6 --skip-distill
  python overnight_long.py --search-hours 0 --skip-search   # apply + train + distill only
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _latest_full_train_run() -> Path | None:
    runs = ROOT / "runs"
    if not runs.is_dir():
        return None
    candidates = [
        p
        for p in runs.glob("ppo_sb3_sim2real_balance_*")
        if p.is_dir() and (p / "best_model" / "best_model.zip").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> int:
    parser = argparse.ArgumentParser(description="PPO Optuna Longer overnight pipeline")
    parser.add_argument("--search-hours", type=float, default=6.0, help="Optuna Longer wall-clock hours")
    parser.add_argument("--skip-search", action="store_true", help="Skip Optuna; use existing BEST_long.json")
    parser.add_argument("--skip-train", action="store_true", help="Skip full 2M train")
    parser.add_argument("--skip-distill", action="store_true", help="Skip DAgger distill")
    parser.add_argument("--resume-search", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    best_long = ROOT / "BEST_long.json"

    if not args.skip_search:
        search_cmd = [
            sys.executable,
            str(ROOT / "search_params_long.py"),
            "--hours",
            str(args.search_hours),
            "--threads",
            str(args.threads),
        ]
        if args.resume_search:
            search_cmd.append("--resume")
        print(f"[OVERNIGHT_LONG] Optuna Longer {args.search_hours}h ...", flush=True)
        subprocess.check_call(search_cmd, cwd=str(ROOT))
    else:
        print("[OVERNIGHT_LONG] skip search", flush=True)

    if not best_long.exists():
        raise SystemExit(f"Missing {best_long}; search did not produce a BEST trial.")

    print("[OVERNIGHT_LONG] apply BEST_long.json -> best_config_overrides.json", flush=True)
    subprocess.check_call(
        [sys.executable, str(ROOT / "apply_best.py"), "--best", str(best_long)],
        cwd=str(ROOT),
    )

    train_run_dir: Path | None = None
    if not args.skip_train:
        env = os.environ.copy()
        env["OVERNIGHT_APPLY_BEST"] = "1"
        print("[OVERNIGHT_LONG] full 2M train (no wall-clock kill) ...", flush=True)
        completed = subprocess.run(
            [sys.executable, str(ROOT / "run.py"), "train"],
            cwd=str(ROOT),
            env=env,
        )
        if completed.returncode != 0:
            raise SystemExit(f"Full train failed with code {completed.returncode}")
        train_run_dir = _latest_full_train_run()
        if train_run_dir is None:
            raise SystemExit("Full train finished but no run dir with best_model found.")
        (ROOT / "last_full_train_run.txt").write_text(str(train_run_dir.resolve()), encoding="utf-8")
        print(f"[OVERNIGHT_LONG] full train run_dir={train_run_dir}", flush=True)
    else:
        print("[OVERNIGHT_LONG] skip full train", flush=True)
        marker = ROOT / "last_full_train_run.txt"
        if marker.exists():
            train_run_dir = Path(marker.read_text(encoding="utf-8").strip())

    if not args.skip_distill:
        if train_run_dir is None or not train_run_dir.is_dir():
            raise SystemExit("No teacher run_dir for distill.")
        print(f"[OVERNIGHT_LONG] distill teacher_dir={train_run_dir} ...", flush=True)
        distill = subprocess.run(
            [
                sys.executable,
                str(ROOT / "distill_student64_deploy.py"),
                "--teacher-dir",
                str(train_run_dir),
                "--no-gui",
            ],
            cwd=str(ROOT),
        )
        if distill.returncode != 0:
            raise SystemExit(f"Distill failed with code {distill.returncode}")
        print("[OVERNIGHT_LONG] distill done", flush=True)
    else:
        print("[OVERNIGHT_LONG] skip distill", flush=True)

    print("[OVERNIGHT_LONG] pipeline complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
