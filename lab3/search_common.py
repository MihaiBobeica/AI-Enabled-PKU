"""Shared Optuna search helpers for Lab3 method folders.

Used by each method's search_params.py. Keeps stdout quiet, persists study
state, and refreshes best_run/ after every improved trial.
"""
from __future__ import annotations

import csv
import json
import signal
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Optuna is required. Install with: pip install optuna\n"
        f"Import error: {exc}"
    ) from exc


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


class SearchState:
    """Wall-clock deadline + artifact paths for one method search."""

    def __init__(self, root: Path, hours: float, resume: bool, *, tag: str = "") -> None:
        self.root = root
        self.hours = float(hours)
        self.tag = str(tag or "")
        suffix = f"_{self.tag}" if self.tag else ""
        self.state_path = root / f"search_state{suffix}.json"
        self.db_path = root / f"optuna_study{suffix}.db"
        self.csv_path = root / f"search_results{suffix}.csv"
        self.best_json = root / f"BEST{suffix}.json"
        self.best_run = root / f"best_run{suffix}"
        self.logs_dir = ensure_dir(root / f"search_logs{suffix}")
        self.figures_dir = ensure_dir(root / f"figures{suffix}")

        if resume and self.state_path.exists():
            prev = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.deadline = time.time() + self.hours * 3600.0
            self.started_at = str(prev.get("started_at", utc_now_iso()))
        else:
            self.deadline = time.time() + self.hours * 3600.0
            self.started_at = utc_now_iso()
        self.save()

    def save(self) -> None:
        payload = {
            "started_at": self.started_at,
            "updated_at": utc_now_iso(),
            "hours": self.hours,
            "deadline_unix": self.deadline,
            "deadline_iso": datetime.fromtimestamp(
                self.deadline, tz=timezone.utc
            ).isoformat(),
        }
        self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def time_up(self) -> bool:
        return time.time() >= self.deadline

    def seconds_left(self) -> float:
        return max(0.0, self.deadline - time.time())


@contextmanager
def quiet_stdio(log_path: Path, verbose: bool):
    if verbose:
        yield
        return
    ensure_dir(log_path.parent)
    with log_path.open("w", encoding="utf-8") as handle:
        with redirect_stdout(handle), redirect_stderr(handle):
            yield


def append_csv_row(path: Path, fieldnames: Sequence[str], row: Dict[str, Any]) -> None:
    new_file = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_best_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def refresh_best_run(
    best_run: Path,
    source: Optional[Path],
    extra_files: Optional[Dict[str, Any]] = None,
) -> None:
    """Replace best_run/ with a fresh copy of the winning artifacts."""
    import shutil

    if best_run.exists():
        shutil.rmtree(best_run, ignore_errors=True)
    ensure_dir(best_run)
    if source is not None and source.exists():
        if source.is_dir():
            shutil.copytree(source, best_run / "run", dirs_exist_ok=True)
        else:
            shutil.copy2(source, best_run / source.name)
    if extra_files:
        (best_run / "trial_meta.json").write_text(
            json.dumps(extra_files, indent=2), encoding="utf-8"
        )


def load_best_score(best_json: Path) -> Optional[float]:
    if not best_json.exists():
        return None
    try:
        data = json.loads(best_json.read_text(encoding="utf-8"))
        return float(data["score"])
    except Exception:
        return None


def create_study(
    study_name: str,
    storage_url: str,
    direction: str = "maximize",
    pruner: Any = None,
) -> "optuna.Study":
    sampler = TPESampler(seed=42)
    if pruner is None:
        pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=0)
    return optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )


def enqueue_baseline_if_needed(study: "optuna.Study", baseline: Dict[str, Any]) -> None:
    for t in study.trials:
        if t.user_attrs.get("is_baseline"):
            return
    try:
        study.enqueue_trial(baseline, user_attrs={"is_baseline": True})
    except Exception:
        pass


def maybe_save_optuna_figures(study: "optuna.Study", figures_dir: Path) -> None:
    ensure_dir(figures_dir)
    try:
        from optuna.visualization.matplotlib import (
            plot_optimization_history,
            plot_param_importances,
        )
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        finished = [t for t in study.trials if t.state.is_finished() and t.value is not None]
        if len(finished) >= 1:
            ax = plot_optimization_history(study)
            fig = ax.figure if hasattr(ax, "figure") else plt.gcf()
            fig.savefig(figures_dir / "optimization_history.png", dpi=140, bbox_inches="tight")
            plt.close(fig)
        if len(finished) >= 2:
            try:
                ax = plot_param_importances(study)
                fig = ax.figure if hasattr(ax, "figure") else plt.gcf()
                fig.savefig(figures_dir / "param_importance.png", dpi=140, bbox_inches="tight")
                plt.close(fig)
            except Exception:
                pass
    except Exception:
        pass


CSV_FIELDS = [
    "trial",
    "score",
    "params_json",
    "run_dir",
    "pruned",
    "failed",
    "is_baseline",
    "timestamp",
]


def record_trial_row(
    csv_path: Path,
    trial: "optuna.Trial",
    score: float,
    run_dir: str = "",
    pruned: bool = False,
    failed: bool = False,
) -> None:
    append_csv_row(
        csv_path,
        CSV_FIELDS,
        {
            "trial": trial.number,
            "score": score,
            "params_json": json.dumps(trial.params, sort_keys=True),
            "run_dir": run_dir,
            "pruned": int(pruned),
            "failed": int(failed),
            "is_baseline": int(bool(trial.user_attrs.get("is_baseline"))),
            "timestamp": utc_now_iso(),
        },
    )


def update_best_if_improved(
    *,
    best_json: Path,
    best_run: Path,
    trial: "optuna.Trial",
    score: float,
    run_dir: Optional[Path],
    higher_is_better: bool = True,
    extra: Optional[Dict[str, Any]] = None,
) -> bool:
    prev = load_best_score(best_json)
    improved = prev is None or (score > prev if higher_is_better else score < prev)
    if not improved:
        return False
    payload = {
        "score": score,
        "trial": trial.number,
        "params": trial.params,
        "run_dir": str(run_dir) if run_dir else "",
        "updated_at": utc_now_iso(),
        "is_baseline": bool(trial.user_attrs.get("is_baseline")),
        "extra": extra or {},
    }
    write_best_json(best_json, payload)
    refresh_best_run(best_run, run_dir, extra_files=payload)
    print(
        f"NEW_BEST trial={trial.number} score={score:.6g} -> {best_run.as_posix()}",
        flush=True,
    )
    return True


def run_timed_study(
    *,
    root: Path,
    study_name: str,
    hours: float,
    resume: bool,
    n_trials: Optional[int],
    baseline_params: Dict[str, Any],
    objective: Callable[["optuna.Trial"], float],
    direction: str = "maximize",
    tag: str = "",
    pruner: Any = None,
) -> "optuna.Study":
    state = SearchState(root, hours=hours, resume=resume, tag=tag)
    storage = f"sqlite:///{state.db_path.as_posix()}"
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = create_study(study_name, storage, direction=direction, pruner=pruner)
    enqueue_baseline_if_needed(study, baseline_params)

    stop = {"flag": False}

    def _handle_signal(signum, _frame):  # noqa: ANN001
        stop["flag"] = True
        print(f"[SEARCH] signal {signum} — will stop after current trial", flush=True)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle_signal)
        except Exception:
            pass

    def _should_stop(study_obj: "optuna.Study", _trial: "optuna.trial.FrozenTrial") -> None:
        if stop["flag"] or state.time_up():
            study_obj.stop()

    completed_before = len([t for t in study.trials if t.state.is_finished()])
    print(
        f"[SEARCH] study={study_name} resume={resume} "
        f"hours={hours} completed_before={completed_before} "
        f"deadline_in={state.seconds_left()/3600:.2f}h",
        flush=True,
    )

    def wrapped(trial: "optuna.Trial") -> float:
        if state.time_up() or stop["flag"]:
            study.stop()
            raise optuna.TrialPruned("time budget exhausted")
        value = float(objective(trial))
        state.save()
        return value

    study.optimize(
        wrapped,
        n_trials=n_trials,
        callbacks=[_should_stop],
        catch=(Exception,),
        gc_after_trial=True,
    )

    maybe_save_optuna_figures(study, state.figures_dir)
    state.save()

    if study.best_trial is not None:
        print(
            f"[SEARCH] DONE best_score={study.best_value:.6g} "
            f"params={study.best_params} trials={len(study.trials)}",
            flush=True,
        )
    else:
        print("[SEARCH] DONE — no successful trials", flush=True)
    return study


def add_common_args(parser) -> None:
    parser.add_argument("--hours", type=float, default=8.0, help="Wall-clock budget for this method")
    parser.add_argument("--resume", action="store_true", help="Resume Optuna study from sqlite DB")
    parser.add_argument("--trials", type=int, default=None, help="Optional hard cap on new trials")
    parser.add_argument("--budget", type=int, default=None, help="RL probe timesteps (method default if omitted)")
    parser.add_argument("--verbose", action="store_true", help="Show trainer stdout")
    parser.add_argument("--threads", type=int, default=2, help="torch_num_threads for CPU coexistence")


def import_search_common():
    """Ensure lab3/ is on sys.path when scripts run from method folders."""
    import sys

    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
