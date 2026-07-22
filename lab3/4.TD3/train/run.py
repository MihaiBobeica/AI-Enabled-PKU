"""Single user-facing entry point for TD3.

Run without arguments to open the training panel:
    python run.py

Worker arguments are used internally by the panel and remain available for
smoke testing and automation.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
for local in (
    PROJECT_ROOT / "third_party",
    PROJECT_ROOT / "stable_baselines3",
    PROJECT_ROOT / "gymnasium",
):
    if local.exists():
        sys.path.insert(0, str(local))


def _current(value: str) -> str:
    return "current" if value in {"current", "both", "history", "no_history"} else value


def parse_args():
    parser = argparse.ArgumentParser(description="RIP TD3; no arguments opens the training panel")
    parser.add_argument(
        "--worker",
        choices=("panel", "train", "smoke", "distill", "test"),
        default="panel",
    )
    parser.add_argument("--run-dir", default="")
    parser.add_argument(
        "--target",
        choices=("current", "both", "history", "no_history"),
        default="current",
    )
    parser.add_argument(
        "--variant",
        choices=("current", "both", "history", "no_history"),
        default="current",
    )
    parser.add_argument("--smoke", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.worker == "panel":
            from panel import TrainingPanel

            TrainingPanel().mainloop()
            return 0
        if args.worker == "train":
            from train_worker import run_training

            run_training(smoke=False)
            return 0
        if args.worker == "smoke":
            from train_worker import run_training
            from distill_worker import run_distillation
            from test_worker import run_test

            train = run_training(smoke=True)
            run_dir = train["run_dir"]
            run_distillation(run_dir=run_dir, target="current", smoke=True)
            run_test(run_dir=run_dir, variant="current", smoke=True)
            print(
                "[SMOKE_OK] "
                + json.dumps({"run_dir": run_dir}, ensure_ascii=False),
                flush=True,
            )
            return 0
        if args.worker == "distill":
            if not args.run_dir:
                raise ValueError("--run-dir is required")
            from distill_worker import run_distillation

            run_distillation(
                run_dir=args.run_dir,
                target=_current(args.target),
                smoke=args.smoke,
            )
            return 0
        if args.worker == "test":
            if not args.run_dir:
                raise ValueError("--run-dir is required")
            from test_worker import run_test

            run_test(
                run_dir=args.run_dir,
                variant=_current(args.variant),
                smoke=args.smoke,
            )
            return 0
        raise RuntimeError(args.worker)
    except Exception as exc:
        print(
            "[PANEL_JSON] "
            + json.dumps(
                {"event": "stage_error", "stage": args.worker, "error": repr(exc)},
                ensure_ascii=False,
            ),
            flush=True,
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
