"""Optuna TPE search for DQN (Double DQN / LR / epsilon decay).

Usage:
  python search_params.py --hours 8
  python search_params.py --hours 8 --resume
"""
from __future__ import annotations

import argparse
import copy
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

# Course zip baseline (vanilla DQN).
BASELINE = {
    "use_double_dqn": 0,  # categorical as int for Optuna enqueue
    "learning_rate": 5e-4,
    "exploration_decay": 400000.0,
}

DEFAULT_BUDGET = 40_000


def _scale_smoke(budget: int) -> None:
    import config

    nominal = max(budget // 2, budget - max(budget // 4, 1))
    rest = budget - nominal
    config.SMOKE["total_timesteps"] = int(budget)
    config.SMOKE["stage_levels"] = (0.0, 0.1)
    config.SMOKE["stage_steps"] = (int(nominal), int(rest))
    config.SMOKE["eval_freq"] = max(budget // 4, 256)
    config.SMOKE["n_eval_episodes"] = 2
    config.SMOKE["max_eval_policy_steps"] = 400
    config.SMOKE["checkpoint_freq"] = max(budget // 2, 512)
    config.SMOKE["learning_starts"] = min(500, max(32, budget // 20))
    config.SMOKE["batch_size"] = 64
    config.SMOKE["buffer_size"] = max(5000, min(30000, budget))
    config.SMOKE["stage_replay_warmup_steps"] = min(2000, max(16, budget // 20))


def _score_from_summary(summary: dict) -> tuple[float, dict]:
    metrics = summary.get("best_randomized") or summary.get("best_nominal") or {}
    success = float(metrics.get("stable_success_rate", 0.0) or 0.0)
    reward = float(metrics.get("mean_reward_per_step", -1e6) or -1e6)
    capture = float(metrics.get("capture_rate", 0.0) or 0.0)
    # Lexicographic-ish scalar for TPE.
    score = 100.0 * success + 10.0 * capture + reward
    return score, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="DQN Optuna TPE search")
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

        # Snapshot mutable dicts so trials do not leak into each other.
        snap = {
            "DQN": copy.deepcopy(config.DQN),
            "SMOKE": copy.deepcopy(config.SMOKE),
            "RUN": copy.deepcopy(config.RUN),
            "EVAL": copy.deepcopy(config.EVAL),
        }
        try:
            use_ddqn = bool(trial.suggest_categorical("use_double_dqn", [0, 1]))
            lr = float(trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True))
            decay = float(trial.suggest_float("exploration_decay", 2e5, 6e5, log=True))

            config.DQN["use_double_dqn"] = use_ddqn
            config.DQN["learning_rate"] = lr
            config.DQN["exploration_decay"] = decay
            config.RUN["device"] = "cpu"
            config.RUN["torch_num_threads"] = int(args.threads)
            config.RUN["seed"] = 42 + int(trial.number)
            config.RUN["experiment_prefix"] = f"search_dqn_t{trial.number:04d}"
            _scale_smoke(budget)

            log_path = logs_dir / f"trial_{trial.number:04d}.log"
            with quiet_stdio(log_path, args.verbose):
                summary = run_training(smoke=True)

            run_dir = Path(summary["run_dir"])
            score, metrics = _score_from_summary(summary)
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
                f"ddqn={use_ddqn} lr={lr:.3g} decay={decay:.3g}",
                flush=True,
            )
            trial.report(score, step=budget)
            return score
        except Exception as exc:
            record_trial_row(csv_path, trial, float("nan"), failed=True)
            print(f"FAIL trial={trial.number} err={exc}", flush=True)
            raise
        finally:
            config.DQN.clear(); config.DQN.update(snap["DQN"])
            config.SMOKE.clear(); config.SMOKE.update(snap["SMOKE"])
            config.RUN.clear(); config.RUN.update(snap["RUN"])
            config.EVAL.clear(); config.EVAL.update(snap["EVAL"])

    # Encode baseline categoricals as Optuna expects for enqueue.
    baseline = {
        "use_double_dqn": BASELINE["use_double_dqn"],
        "learning_rate": BASELINE["learning_rate"],
        "exploration_decay": BASELINE["exploration_decay"],
    }
    run_timed_study(
        root=root,
        study_name="dqn_tpe",
        hours=args.hours,
        resume=args.resume,
        n_trials=args.trials,
        baseline_params=baseline,
        objective=objective,
        direction="maximize",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
