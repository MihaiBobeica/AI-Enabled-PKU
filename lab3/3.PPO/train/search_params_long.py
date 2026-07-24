"""Optuna Longer search for PPO (Hyperband, late J with PWM).

Frozen from prior short Optuna (trial 175):
  learning_rate, ent_coef, target_kl

Searched:
  n_epochs, clip_range, log_std_init, lr_schedule, dr_curriculum_fraction

Usage:
  python search_params_long.py --hours 6
  python search_params_long.py --hours 6 --resume
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import optuna
from optuna.pruners import HyperbandPruner

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

# Prior short-Optuna winner (frozen).
FROZEN_LR = 0.0001755834889508841
FROZEN_ENT = 0.0006163092363741561
FROZEN_KL = 0.05

DEFAULT_BUDGET = 2_000_000
MIN_RESOURCE = 500_000

BASELINE = {
    "n_epochs": 10,
    "clip_range": 0.2,
    "log_std_init": -0.7,
    "lr_schedule": "constant",
    "dr_curriculum_fraction": 0.2,
}


def _score_eval_csv_j(run_dir: Path, budget_hint: int | None = None) -> tuple[float, dict]:
    """Late-window J = 100*success + 0.01*S - 25*pwm_norm (randomized rows)."""
    import pandas as pd

    path = run_dir / "eval_logs" / "eval_metrics.csv"
    if not path.exists():
        return -1e6, {}
    df = pd.read_csv(path)
    if df.empty:
        return -1e6, {}
    rnd = df[df["eval_type"] == "randomized"] if "eval_type" in df.columns else df
    if rnd.empty:
        rnd = df
    max_step = float(rnd["timesteps"].max()) if "timesteps" in rnd.columns else float(budget_hint or 0)
    cutoff = max_step * 0.8
    late = rnd[rnd["timesteps"] >= cutoff] if "timesteps" in rnd.columns else rnd
    if late.empty:
        late = rnd.tail(2)
    success = float(late["success_rate"].mean()) if "success_rate" in late.columns else 0.0
    score_col = float(late["score"].mean()) if "score" in late.columns else 0.0
    pwm = float(late["mean_abs_pwm_norm"].mean()) if "mean_abs_pwm_norm" in late.columns else 1.0
    if pwm != pwm:  # NaN
        pwm = 1.0
    j = 100.0 * success + 0.01 * score_col - 25.0 * pwm
    return j, {
        "late_success": success,
        "late_score": score_col,
        "late_pwm": pwm,
        "max_timesteps": max_step,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="PPO Optuna Longer (Hyperband + J/PWM)")
    add_common_args(parser)
    args = parser.parse_args()
    budget = int(args.budget or DEFAULT_BUDGET)

    root = ROOT
    tag = "long"
    logs_dir = root / f"search_logs_{tag}"
    csv_path = root / f"search_results_{tag}.csv"
    best_json = root / f"BEST_{tag}.json"
    best_run = root / f"best_run_{tag}"
    logs_dir.mkdir(parents=True, exist_ok=True)

    pruner = HyperbandPruner(
        min_resource=MIN_RESOURCE,
        max_resource=budget,
        reduction_factor=2,
    )

    def objective(trial: optuna.Trial) -> float:
        import config
        import run as run_mod
        from run import train

        snap = {
            "PPO": copy.deepcopy(config.PPO),
            "EVAL": copy.deepcopy(config.EVAL),
            "RUN": copy.deepcopy(config.RUN),
            "DOMAIN_RANDOMIZATION": copy.deepcopy(config.DOMAIN_RANDOMIZATION),
            "PANEL": copy.deepcopy(getattr(config, "PANEL", {})),
        }
        run_mod._OPTUNA_TRIAL = trial
        run_mod._OPTUNA_RANK_BY_J = True
        try:
            n_epochs = int(trial.suggest_categorical("n_epochs", [5, 8, 10]))
            clip_range = float(trial.suggest_categorical("clip_range", [0.1, 0.15, 0.2]))
            log_std_init = float(trial.suggest_categorical("log_std_init", [-0.9, -0.7, -0.4]))
            lr_schedule = str(trial.suggest_categorical("lr_schedule", ["constant", "linear"]))
            dr_frac = float(trial.suggest_categorical("dr_curriculum_fraction", [0.2, 0.35, 0.5]))

            # Frozen HPs from short Optuna winner.
            config.PPO["learning_rate"] = FROZEN_LR
            config.PPO["ent_coef"] = FROZEN_ENT
            config.PPO["target_kl"] = FROZEN_KL
            config.PPO["n_epochs"] = n_epochs
            config.PPO["clip_range"] = clip_range
            config.PPO["log_std_init"] = log_std_init
            config.PPO["lr_schedule"] = lr_schedule
            config.PPO["total_timesteps"] = budget
            config.PPO["n_envs"] = 8
            config.PPO["verbose"] = 0
            config.PPO["progress_bar"] = False

            config.DOMAIN_RANDOMIZATION["dr_curriculum_fraction"] = dr_frac

            # Sparse eval aligned with Hyperband rungs (~250k).
            config.EVAL["eval_freq"] = 250_000
            config.EVAL["n_eval_episodes"] = 4
            config.EVAL["max_eval_policy_steps"] = 1500
            config.EVAL["checkpoint_freq"] = max(budget // 2, 500_000)

            config.RUN["device"] = "cpu"
            config.RUN["torch_num_threads"] = int(args.threads)
            config.RUN["seed"] = 42 + int(trial.number)
            config.RUN["experiment_name"] = f"search_ppo_long_t{trial.number:04d}_{config._now_str()}"
            if hasattr(config, "PANEL"):
                config.PANEL["print_train_episodes"] = False

            run_dir = Path(config.run_dir())
            log_path = logs_dir / f"trial_{trial.number:04d}.log"
            try:
                with quiet_stdio(log_path, args.verbose):
                    train()
            except optuna.TrialPruned:
                score, metrics = _score_eval_csv_j(run_dir, budget_hint=budget)
                record_trial_row(csv_path, trial, score, run_dir=str(run_dir), pruned=True)
                print(
                    f"PRUNED trial={trial.number} J={score:.4f} "
                    f"epochs={n_epochs} clip={clip_range} log_std={log_std_init} "
                    f"lr_sched={lr_schedule} dr_frac={dr_frac}",
                    flush=True,
                )
                raise

            score, metrics = _score_eval_csv_j(run_dir, budget_hint=budget)
            record_trial_row(csv_path, trial, score, run_dir=str(run_dir), pruned=False)
            update_best_if_improved(
                best_json=best_json,
                best_run=best_run,
                trial=trial,
                score=score,
                run_dir=run_dir,
                extra={
                    "metrics": metrics,
                    "budget": budget,
                    "frozen": {"learning_rate": FROZEN_LR, "ent_coef": FROZEN_ENT, "target_kl": FROZEN_KL},
                },
            )
            print(
                f"OK trial={trial.number} J={score:.4f} "
                f"epochs={n_epochs} clip={clip_range} log_std={log_std_init} "
                f"lr_sched={lr_schedule} dr_frac={dr_frac} "
                f"late_succ={metrics.get('late_success', float('nan')):.3f} "
                f"late_pwm={metrics.get('late_pwm', float('nan')):.3f}",
                flush=True,
            )
            return score
        except optuna.TrialPruned:
            raise
        except Exception as exc:
            record_trial_row(csv_path, trial, float("nan"), failed=True)
            print(f"FAIL trial={trial.number} err={exc}", flush=True)
            raise
        finally:
            run_mod._OPTUNA_TRIAL = None
            run_mod._OPTUNA_RANK_BY_J = False
            config.PPO.clear()
            config.PPO.update(snap["PPO"])
            config.EVAL.clear()
            config.EVAL.update(snap["EVAL"])
            config.RUN.clear()
            config.RUN.update(snap["RUN"])
            config.DOMAIN_RANDOMIZATION.clear()
            config.DOMAIN_RANDOMIZATION.update(snap["DOMAIN_RANDOMIZATION"])
            if hasattr(config, "PANEL") and snap["PANEL"]:
                config.PANEL.clear()
                config.PANEL.update(snap["PANEL"])

    run_timed_study(
        root=root,
        study_name="ppo_tpe_long",
        hours=args.hours,
        resume=args.resume,
        n_trials=args.trials,
        baseline_params=BASELINE,
        objective=objective,
        direction="maximize",
        tag=tag,
        pruner=pruner,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
