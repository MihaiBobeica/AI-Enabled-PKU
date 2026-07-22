"""Optuna TPE search for Bonus-2 PPO single-policy swing-up.

Isolated artifacts (never collide with hybrid PPO search):
  BEST_bonus2.json, best_run_bonus2/, optuna_study_bonus2.db,
  search_results_bonus2.csv, figures_bonus2/, search_logs_bonus2/

Usage:
  python search_params_bonus2.py --hours 8
  python search_params_bonus2.py --hours 8 --resume
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

# Force run.py to bind config_bonus2 on import (search CLI has no --config otherwise).
if "--config" not in sys.argv:
    sys.argv[1:1] = ["--config", "config_bonus2"]

from search_common import (  # noqa: E402
    add_common_args,
    quiet_stdio,
    record_trial_row,
    run_timed_study,
    update_best_if_improved,
)

BASELINE = {
    "learning_rate": 3e-4,
    "ent_coef": 0.002,
    "target_kl_code": 1,  # 0=off, 1=0.035, 2=0.02, 3=0.05
}

KL_MAP = {0: None, 1: 0.035, 2: 0.02, 3: 0.05}
DEFAULT_BUDGET = 100_000
TAG = "bonus2"


def _score_eval_csv(run_dir: Path) -> tuple[float, dict]:
    import pandas as pd

    path = run_dir / "eval_logs" / "eval_metrics.csv"
    if not path.exists():
        return -1e6, {}
    df = pd.read_csv(path)
    if df.empty:
        return -1e6, {}
    rnd = df[df["eval_type"] == "randomized"] if "eval_type" in df.columns else df
    use = rnd if not rnd.empty else df
    success = float(use["success_rate"].max()) if "success_rate" in use.columns else 0.0
    score_col = float(use["score"].max()) if "score" in use.columns else 0.0
    reward = float(use["reward_per_step"].max()) if "reward_per_step" in use.columns else 0.0
    score = 100.0 * success + 0.01 * score_col + reward
    return score, {
        "success_rate": success,
        "eval_score": score_col,
        "reward_per_step": reward,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Bonus-2 PPO Optuna TPE search")
    add_common_args(parser)
    parser.add_argument(
        "--config",
        default="config_bonus2",
        help="Must remain config_bonus2 for this search",
    )
    args = parser.parse_args()
    if str(args.config).replace(".py", "") != "config_bonus2":
        raise SystemExit("search_params_bonus2.py requires --config config_bonus2")
    budget = int(args.budget or DEFAULT_BUDGET)

    root = ROOT
    logs_dir = root / f"search_logs_{TAG}"
    csv_path = root / f"search_results_{TAG}.csv"
    best_json = root / f"BEST_{TAG}.json"
    best_run = root / f"best_run_{TAG}"
    logs_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial) -> float:
        from run import train
        import config  # bound to config_bonus2 via run.py sys.modules patch

        snap = {
            "PPO": copy.deepcopy(config.PPO),
            "EVAL": copy.deepcopy(config.EVAL),
            "RUN": copy.deepcopy(config.RUN),
            "PANEL": copy.deepcopy(getattr(config, "PANEL", {})),
        }
        try:
            lr = float(trial.suggest_float("learning_rate", 1e-4, 5e-4, log=True))
            ent = float(trial.suggest_float("ent_coef", 1e-4, 0.02, log=True))
            kl_code = int(trial.suggest_categorical("target_kl_code", [0, 1, 2, 3]))
            target_kl = KL_MAP[kl_code]

            config.PPO["learning_rate"] = lr
            config.PPO["ent_coef"] = ent
            config.PPO["target_kl"] = target_kl
            config.PPO["total_timesteps"] = budget
            config.PPO["n_envs"] = 8
            config.PPO["verbose"] = 0
            config.PPO["progress_bar"] = False
            config.EVAL["eval_freq"] = max(budget // 5, 2048)
            config.EVAL["n_eval_episodes"] = 4
            config.EVAL["max_eval_policy_steps"] = 2000
            config.EVAL["checkpoint_freq"] = max(budget // 2, 5000)
            config.RUN["device"] = "cpu"
            config.RUN["torch_num_threads"] = int(args.threads)
            config.RUN["seed"] = 42 + int(trial.number)
            config.RUN["experiment_name"] = f"search_ppo_bonus2_t{trial.number:04d}_{config._now_str()}"
            if hasattr(config, "PANEL"):
                config.PANEL["print_train_episodes"] = False

            run_dir = Path(config.run_dir())
            log_path = logs_dir / f"trial_{trial.number:04d}.log"
            with quiet_stdio(log_path, args.verbose):
                train()

            score, metrics = _score_eval_csv(run_dir)
            record_trial_row(csv_path, trial, score, run_dir=str(run_dir))
            update_best_if_improved(
                best_json=best_json,
                best_run=best_run,
                trial=trial,
                score=score,
                run_dir=run_dir,
                extra={"metrics": metrics, "budget": budget, "target_kl": target_kl, "track": "bonus2"},
            )
            print(
                f"OK trial={trial.number} score={score:.4f} "
                f"lr={lr:.3g} ent={ent:.3g} kl={target_kl}",
                flush=True,
            )
            trial.report(score, step=budget)
            return score
        except Exception as exc:
            record_trial_row(csv_path, trial, float("nan"), failed=True)
            print(f"FAIL trial={trial.number} err={exc}", flush=True)
            raise
        finally:
            config.PPO.clear(); config.PPO.update(snap["PPO"])
            config.EVAL.clear(); config.EVAL.update(snap["EVAL"])
            config.RUN.clear(); config.RUN.update(snap["RUN"])
            if hasattr(config, "PANEL") and snap["PANEL"]:
                config.PANEL.clear(); config.PANEL.update(snap["PANEL"])

    run_timed_study(
        root=root,
        study_name="ppo_bonus2_tpe",
        hours=args.hours,
        resume=args.resume,
        n_trials=args.trials,
        baseline_params=BASELINE,
        objective=objective,
        direction="maximize",
        tag=TAG,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
