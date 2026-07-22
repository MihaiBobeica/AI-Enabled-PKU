"""Optuna TPE search for TD3 (action noise / critic LR / batch size).

Usage:
  python search_params.py --hours 8
  python search_params.py --hours 8 --resume
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LAB3 = ROOT.parent.parent
if str(LAB3) not in sys.path:
    sys.path.insert(0, str(LAB3))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from search_common import (  # noqa: E402
    add_common_args,
    quiet_stdio,
    record_trial_row,
    run_timed_study,
    update_best_if_improved,
)

BASELINE = {
    "action_noise_sigma": 0.1,
    "critic_learning_rate": 1e-3,
    "batch_size": 128,
}

DEFAULT_BUDGET = 75_000


def _scale_smoke(budget: int) -> None:
    import config

    nominal = max(budget // 2, budget - max(budget // 4, 1))
    rest = budget - nominal
    config.SMOKE["total_timesteps"] = int(budget)
    config.SMOKE["stage_names"] = ("smoke_nominal", "smoke_randomization_0.10")
    config.SMOKE["stage_levels"] = (0.0, 0.10)
    config.SMOKE["stage_steps"] = (int(nominal), int(rest))
    config.SMOKE["stage_replay_warmup_steps"] = min(2000, max(16, budget // 40))
    config.SMOKE["learning_starts"] = min(1000, max(32, budget // 20))
    config.SMOKE["buffer_size"] = max(10_000, min(200_000, budget * 2))
    config.SMOKE["batch_size"] = 64  # overridden per-trial after suggest if needed
    config.SMOKE["train_freq"] = 1
    config.SMOKE["gradient_steps"] = 1
    config.SMOKE["eval_freq"] = max(budget // 5, 512)
    config.SMOKE["n_eval_episodes"] = 3
    config.SMOKE["max_eval_policy_steps"] = 500
    config.SMOKE["checkpoint_freq"] = max(budget // 2, 1024)


def _score_from_run(run_dir: Path, summary: dict) -> tuple[float, dict]:
    metrics = {}
    for name in ("best_randomized_model", "best_nominal_model"):
        p = run_dir / name / "best_metrics.json"
        if p.exists():
            metrics = json.loads(p.read_text(encoding="utf-8"))
            break
    if not metrics:
        # Fall back to summary nested fields if present.
        metrics = summary.get("best_randomized") or summary.get("best_nominal") or {}
    success = float(metrics.get("stable_success_rate", 0.0) or 0.0)
    capture = float(metrics.get("capture_rate", 0.0) or 0.0)
    control = float(metrics.get("control_score", metrics.get("mean_reward", -1e6)) or -1e6)
    score = 100.0 * success + 10.0 * capture + 0.001 * control
    return score, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="TD3 Optuna TPE search")
    add_common_args(parser)
    args = parser.parse_args()
    budget = int(args.budget or DEFAULT_BUDGET)

    root = ROOT
    logs_dir = root / "search_logs"
    csv_path = root / "search_results.csv"
    best_json = root / "BEST.json"
    best_run = root / "best_run"
    logs_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial) -> float:
        import config
        from train_worker import run_training

        snap = {
            "TD3": copy.deepcopy(config.TD3),
            "SMOKE": copy.deepcopy(config.SMOKE),
            "RUN": copy.deepcopy(config.RUN),
            "EVAL": copy.deepcopy(config.EVAL),
        }
        try:
            sigma = float(trial.suggest_float("action_noise_sigma", 0.04, 0.12))
            critic_lr = float(trial.suggest_float("critic_learning_rate", 3e-4, 2e-3, log=True))
            batch = int(trial.suggest_categorical("batch_size", [64, 128, 256]))

            config.TD3["action_noise_sigma"] = sigma
            config.TD3["critic_learning_rate"] = critic_lr
            config.TD3["batch_size"] = batch
            config.RUN["device"] = "cpu"
            config.RUN["torch_num_threads"] = int(args.threads)
            config.RUN["seed"] = 42 + int(trial.number)
            config.RUN["experiment_prefix"] = f"search_td3_t{trial.number:04d}"
            _scale_smoke(budget)
            config.SMOKE["batch_size"] = min(batch, max(32, budget // 10))

            log_path = logs_dir / f"trial_{trial.number:04d}.log"
            with quiet_stdio(log_path, args.verbose):
                summary = run_training(smoke=True)

            run_dir = Path(summary["run_dir"])
            score, metrics = _score_from_run(run_dir, summary)
            record_trial_row(csv_path, trial, score, run_dir=str(run_dir))
            update_best_if_improved(
                best_json=best_json,
                best_run=best_run,
                trial=trial,
                score=score,
                run_dir=run_dir,
                extra={"metrics": metrics, "budget": budget},
            )
            print(
                f"OK trial={trial.number} score={score:.4f} "
                f"sigma={sigma:.3g} critic_lr={critic_lr:.3g} batch={batch}",
                flush=True,
            )
            trial.report(score, step=budget)
            return score
        except Exception as exc:
            record_trial_row(csv_path, trial, float("nan"), failed=True)
            print(f"FAIL trial={trial.number} err={exc}", flush=True)
            raise
        finally:
            config.TD3.clear(); config.TD3.update(snap["TD3"])
            config.SMOKE.clear(); config.SMOKE.update(snap["SMOKE"])
            config.RUN.clear(); config.RUN.update(snap["RUN"])
            config.EVAL.clear(); config.EVAL.update(snap["EVAL"])

    run_timed_study(
        root=root,
        study_name="td3_tpe",
        hours=args.hours,
        resume=args.resume,
        n_trials=args.trials,
        baseline_params=BASELINE,
        objective=objective,
        direction="maximize",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
