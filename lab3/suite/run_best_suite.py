#!/usr/bin/env python3
"""One-command Lab3 Best suite: PPO ablations, distill, leftover DQN, BEST.md.

Usage:
    python run_best_suite.py --hours 6
    python run_best_suite.py --hours 6 --resume
    python run_best_suite.py --dry-run
    python run_best_suite.py --smoke-only
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

SUITE_DIR = Path(__file__).resolve().parent
LAB3 = SUITE_DIR.parent
PPO_DIR = LAB3 / "3.PPO" / "train"
DQN_DIR = LAB3 / "2.DQN" / "train"
DEPLOY_DIR = LAB3 / "3.PPO" / "deploy"
PRISTINE_DIR = SUITE_DIR / "_suite_pristine"
STATE_PATH = SUITE_DIR / "suite_state.json"
LOG_CSV = SUITE_DIR / "suite_experiment_log.csv"
BEST_MD = SUITE_DIR / "BEST.md"
SUITE_LOGS = SUITE_DIR / "suite_logs"

PPO_BASELINE_SCORE = 20741.16
PPO_BASELINE_RUN = PPO_DIR / "runs" / "ppo_sb3_sim2real_balance_20260720_211933"

RESERVE_S = 600
DQN_MIN_BUDGET_S = 5400
SEED_REPEAT = [7, 123, 2026]

LOG_HEADER = [
    "trial_id", "track", "hypothesis", "overrides_json", "seed",
    "run_dir", "best_score", "wall_s", "status", "notes",
]

_shutdown_requested = False
# When False (default), child trainer stdout goes only to suite_logs/*.log;
# the terminal gets progress/status lines. Use --verbose for full echo.
_echo_all_child_output = False

# Lines from PPO/DQN/distill subprocesses that are still useful on a quiet terminal.
_ECHO_LINE_RE = re.compile(
    r"\[(?:PROGRESS|RUN_DIR|DONE|BEST|EVAL|ERROR|WARN|COLLECT|SELECTED|CANCELLED|DR)\]|"
    r"\[TRAIN\] Stage|"
    r"^Traceback|"
    r"Error:|"
    r"Exception|"
    r"FAILED|"
    r"依赖缺失|"
    r"导入失败"
)


def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _import_config_editor(track: str):
    import importlib.util

    pkg = PPO_DIR if track == "ppo" else DQN_DIR
    spec = importlib.util.spec_from_file_location(
        f"{track}_config_editor",
        pkg / "config_editor.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load config_editor for {track}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ConfigEditError, mod.update_config_file


def pristine_paths() -> Tuple[Path, Path, Path, Path]:
    return (
        PRISTINE_DIR / "ppo_config.py",
        PRISTINE_DIR / "dqn_config.py",
        PPO_DIR / "config.py",
        DQN_DIR / "config.py",
    )


def ensure_pristine() -> None:
    PRISTINE_DIR.mkdir(parents=True, exist_ok=True)
    ppo_p, dqn_p, ppo_cfg, dqn_cfg = pristine_paths()
    if not ppo_p.exists():
        shutil.copy2(ppo_cfg, ppo_p)
    if not dqn_p.exists():
        shutil.copy2(dqn_cfg, dqn_p)


def restore_all_configs() -> None:
    ensure_pristine()
    ppo_p, dqn_p, ppo_cfg, dqn_cfg = pristine_paths()
    shutil.copy2(ppo_p, ppo_cfg)
    shutil.copy2(dqn_p, dqn_cfg)


@contextmanager
def config_session(track: str) -> Iterator[None]:
    restore_all_configs()
    try:
        yield
    finally:
        restore_all_configs()


def apply_overrides(track: str, overrides: Dict[str, Any]) -> None:
    restore_all_configs()
    _, _, ppo_cfg, dqn_cfg = pristine_paths()
    cfg_path = ppo_cfg if track == "ppo" else dqn_cfg
    ConfigEditError, update_config_file = _import_config_editor(track)
    try:
        update_config_file(cfg_path, overrides)
    except ConfigEditError as exc:
        raise RuntimeError(f"Config edit failed for {track}: {exc}") from exc


def _subprocess_env() -> Dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_subprocess(cwd: Path, args: List[str], log_path: Path) -> Tuple[int, str]:
    SUITE_LOGS.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    combined: List[str] = []
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            args,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=_subprocess_env(),
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            logf.write(line)
            combined.append(line)
            if _echo_all_child_output or _ECHO_LINE_RE.search(line):
                print(line, end="", flush=True)
        rc = proc.wait()
    if rc != 0 and not _echo_all_child_output:
        # On failure, surface the last log lines so debugging does not require opening the file.
        tail = "".join(combined[-40:])
        print(f"[SUITE] Subprocess failed (rc={rc}). Log: {log_path}", flush=True)
        if tail.strip():
            print("--- last log lines ---", flush=True)
            print(tail, end="" if tail.endswith("\n") else "\n", flush=True)
            print("--- end ---", flush=True)
    return rc, "".join(combined)


def parse_run_dir_from_log(text: str, track: str, experiment_name: Optional[str] = None) -> Optional[Path]:
    if track == "ppo":
        m = re.search(r"\[RUN_DIR\] path=(.+)", text)
        if m:
            return Path(m.group(1).strip())
        if experiment_name:
            candidate = PPO_DIR / "runs" / experiment_name
            if candidate.exists():
                return candidate
    else:
        m = re.search(r"run_dir\s+:\s*(.+)", text)
        if m:
            return Path(m.group(1).strip())
    return None


def newest_run_dir(track: str, after: float) -> Optional[Path]:
    runs = PPO_DIR / "runs" if track == "ppo" else DQN_DIR / "runs"
    if not runs.is_dir():
        return None
    prefix = "ppo_" if track == "ppo" else "dqn_"
    candidates = [
        p for p in runs.iterdir()
        if p.is_dir() and p.name.startswith(prefix) and p.stat().st_mtime >= after - 1.0
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def ppo_best_randomized_score(run_dir: Path) -> Tuple[float, Dict[str, Any]]:
    csv_path = run_dir / "eval_logs" / "eval_metrics.csv"
    if not csv_path.is_file():
        return float("-inf"), {}
    best_score = float("-inf")
    best_row: Dict[str, Any] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("eval_type") != "randomized":
                continue
            score = float(row["score"])
            if score > best_score:
                best_score = score
                best_row = dict(row)
    return best_score, best_row


def ppo_measure_fps(run_dir: Path) -> Optional[float]:
    progress = run_dir / "training_progress.json"
    if not progress.is_file():
        return None
    try:
        data = json.loads(progress.read_text(encoding="utf-8"))
        fps = float(data.get("fps", 0))
        return fps if fps > 0 else None
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def dqn_summary(run_dir: Path) -> Dict[str, Any]:
    summary_path = run_dir / "training_summary.json"
    if summary_path.is_file():
        try:
            return json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    out: Dict[str, Any] = {"run_dir": str(run_dir.resolve())}
    eval_csv = run_dir / "eval_history.csv"
    if eval_csv.is_file():
        best_success = -1.0
        best_row: Dict[str, Any] = {}
        with eval_csv.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    sr = float(row.get("stable_success_rate", 0))
                except (TypeError, ValueError):
                    continue
                if sr > best_success:
                    best_success = sr
                    best_row = dict(row)
        out["best_eval"] = best_row
        out["best_stable_success_rate"] = best_success
    selected = run_dir / "selected_best_model.pt"
    deploy_h = run_dir / "deploy" / "model_weights.h"
    if selected.is_file():
        out["selected_model"] = str(selected.resolve())
    if deploy_h.is_file():
        out["deploy_header"] = str(deploy_h.resolve())
    return out


@dataclass
class Trial:
    trial_id: str
    track: str
    hypothesis: str
    overrides: Dict[str, Any]
    estimate_s: float
    seed: int = 42


def build_ppo_queue(measured_fps: Optional[float] = None) -> List[Trial]:
    p0_steps = 4_000_000
    if measured_fps and measured_fps < 1200:
        p0_steps = 3_000_000

    def est(steps: int) -> float:
        if measured_fps and measured_fps > 0:
            return steps / measured_fps * 1.15
        if steps >= 4_000_000:
            return 2400.0
        return 1200.0

    # MLP PPO is typically faster on CPU than CUDA (env sim dominates; SB3 warns on GPU).
    base = {"RUN.device": "cpu", "RUN.seed": 42}
    ts = _now_ts()
    trials = [
        Trial("P0", "ppo", "longer train 4M", {**base, "PPO.total_timesteps": p0_steps}, est(p0_steps)),
        Trial("P1", "ppo", "slower DR ramp", {**base, "DOMAIN_RANDOMIZATION.dr_curriculum_fraction": 0.4}, est(2_000_000)),
        Trial("P2", "ppo", "more entropy", {**base, "PPO.ent_coef": 0.005}, est(2_000_000)),
        Trial("P3", "ppo", "tighter KL", {**base, "PPO.target_kl": 0.025}, est(2_000_000)),
        Trial("P4", "ppo", "lower LR", {**base, "PPO.learning_rate": 0.00015}, est(2_000_000)),
    ]
    for t in trials:
        t.overrides["RUN.experiment_name"] = f"suite_{t.trial_id}_{ts}"
    return trials


def build_dqn_queue() -> List[Trial]:
    ts = _now_ts()
    base = {"RUN.device": "cpu", "RUN.seed": 42}
    return [
        Trial(
            "D0", "dqn", "benchmark 5M",
            {**base},
            7200.0,
        ),
        Trial(
            "D1", "dqn", "more stage replay warmup",
            {**base, "DOMAIN_RANDOMIZATION.stage_replay_warmup_steps": 50000},
            7200.0,
        ),
    ]


def seed_repeat_trials(best: Dict[str, Any]) -> List[Trial]:
    if not best or "overrides" not in best:
        return []
    ts = _now_ts()
    base_overrides = dict(best["overrides"])
    base_overrides["RUN.device"] = "cpu"
    winning_seed = int(best.get("seed", 42))
    out: List[Trial] = []
    for seed in SEED_REPEAT:
        if seed == winning_seed:
            continue
        tid = f"P5_seed{seed}"
        ov = dict(base_overrides)
        ov["RUN.seed"] = seed
        ov["RUN.experiment_name"] = f"suite_{tid}_{ts}"
        out.append(Trial(tid, "ppo", f"seed repeat of {best.get('trial_id')}", ov, 1200.0, seed=seed))
    return out


def load_state() -> Dict[str, Any]:
    if STATE_PATH.is_file():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {
        "deadline_unix": 0.0,
        "completed_ids": [],
        "best_ppo": None,
        "best_dqn": None,
        "distill_header": None,
        "measured_ppo_fps": None,
        "hours": 6.0,
    }


def save_state(state: Dict[str, Any]) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def append_log_row(row: Dict[str, Any]) -> None:
    write_header = not LOG_CSV.is_file()
    with LOG_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_HEADER)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LOG_HEADER})


def log_has_ok(trial_id: str) -> bool:
    if not LOG_CSV.is_file():
        return False
    with LOG_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("trial_id") == trial_id and row.get("status") == "ok":
                return True
    return False


def can_start(estimate_s: float, deadline: float, reserve_s: float = RESERVE_S) -> bool:
    return time.time() + estimate_s + reserve_s <= deadline


def time_left(deadline: float) -> float:
    return max(0.0, deadline - time.time())


def run_smoke(track: str) -> Tuple[bool, str]:
    with config_session(track):
        if track == "ppo":
            rc, text = run_subprocess(
                PPO_DIR,
                [sys.executable, "run.py", "smoke"],
                SUITE_LOGS / "smoke_ppo.log",
            )
        else:
            rc, text = run_subprocess(
                DQN_DIR,
                [sys.executable, "run.py", "--worker", "smoke"],
                SUITE_LOGS / "smoke_dqn.log",
            )
    return rc == 0, text


def run_train_trial(trial: Trial) -> Tuple[str, Optional[Path], float, float, str]:
    """Returns status, run_dir, score, wall_s, notes."""
    exp_name = trial.overrides.get("RUN.experiment_name", "")
    start = time.time()
    log_path = SUITE_LOGS / f"{trial.trial_id}.log"

    with config_session(trial.track):
        apply_overrides(trial.track, trial.overrides)
        if trial.track == "ppo":
            rc, text = run_subprocess(
                PPO_DIR,
                [sys.executable, "run.py", "train"],
                log_path,
            )
        else:
            rc, text = run_subprocess(
                DQN_DIR,
                [sys.executable, "run.py", "--worker", "train"],
                log_path,
            )

    wall = time.time() - start
    run_dir = parse_run_dir_from_log(text, trial.track, str(exp_name) if exp_name else None)
    if run_dir is None:
        run_dir = newest_run_dir(trial.track, start)

    if rc != 0:
        return "fail", run_dir, float("-inf"), wall, f"subprocess exit {rc}"

    if run_dir is None or not run_dir.is_dir():
        return "fail", run_dir, float("-inf"), wall, "run_dir not found"

    if trial.track == "ppo":
        score, row = ppo_best_randomized_score(run_dir)
        notes = f"best_randomized_timesteps={row.get('timesteps', 'n/a')}"
        return "ok", run_dir, score, wall, notes

    summary = dqn_summary(run_dir)
    sr = float(summary.get("best_stable_success_rate", summary.get("best_eval", {}).get("stable_success_rate", 0) or 0))
    return "ok", run_dir, sr, wall, json.dumps(summary.get("selected_reason", summary.get("best_eval", {})))


def update_best_ppo(state: Dict[str, Any], trial: Trial, run_dir: Path, score: float) -> None:
    current = state.get("best_ppo")
    current_score = float(current["score"]) if current else float("-inf")
    if score > current_score:
        state["best_ppo"] = {
            "trial_id": trial.trial_id,
            "run_dir": str(run_dir.resolve()),
            "score": score,
            "overrides": dict(trial.overrides),
            "seed": trial.seed,
        }


def update_best_dqn(state: Dict[str, Any], trial: Trial, run_dir: Path, summary: Dict[str, Any]) -> None:
    sr = float(summary.get("best_stable_success_rate", 0))
    current = state.get("best_dqn")
    current_sr = float(current.get("best_stable_success_rate", 0)) if current else -1.0
    if sr >= current_sr:
        state["best_dqn"] = {
            "trial_id": trial.trial_id,
            "run_dir": str(run_dir.resolve()),
            "best_stable_success_rate": sr,
            "summary": summary,
        }


def run_distill(run_dir: Path) -> Optional[Path]:
    log_path = SUITE_LOGS / "distill.log"
    rc, _ = run_subprocess(
        PPO_DIR,
        [sys.executable, "distill_student64_deploy.py", "--teacher-dir", str(run_dir.resolve()), "--no-gui"],
        log_path,
    )
    if rc != 0:
        return None
    candidates = [
        run_dir / "model_weights.h",
        run_dir / "distill_student64_deploy" / "best" / "model_weights.h",
        run_dir / "distill_student64_deploy" / "best" / "ppo_model_weights.h",
        run_dir / "best_model" / "distill_student64_deploy" / "best" / "model_weights.h",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def run_headless_checks(header: Path) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for seed in (2026, 2027, 2028):
        log_path = SUITE_LOGS / f"headless_{seed}.log"
        rc, text = run_subprocess(
            DEPLOY_DIR,
            [
                sys.executable, "rip_ppo_sim_test.py", "--headless",
                "--model", str(header.resolve()),
                "--duration", "30",
                "--seed", str(seed),
                "--randomization", "0.0",
            ],
            log_path,
        )
        captures = re.search(r"captures=(\d+)", text)
        stable = re.search(r"stable_start=([\d.]+|None)", text)
        results.append({
            "seed": seed,
            "ok": rc == 0,
            "captures": int(captures.group(1)) if captures else -1,
            "stable_start": stable.group(1) if stable else "unknown",
        })
    return results


def write_best_md(state: Dict[str, Any], hours: float, incomplete: bool = False) -> None:
    lines = [
        "# Lab3 Best Artifacts",
        "",
        f"Generated: {_iso_now()}",
        f"Suite budget: {hours} h",
        "",
    ]
    if incomplete:
        lines.append("> Suite stopped early or was interrupted. Re-run with `--resume`.")
        lines.append("")

    best_ppo = state.get("best_ppo")
    lines.append("## PPO (primary)")
    if best_ppo:
        run = Path(best_ppo["run_dir"])
        lines.extend([
            f"- Trial: {best_ppo.get('trial_id', 'n/a')}",
            f"- Run dir: `{run}`",
            f"- Best randomized score: **{best_ppo.get('score', 'n/a'):.3f}** (baseline ~{PPO_BASELINE_SCORE:.2f})",
            f"- Teacher: `{run / 'best_model' / 'best_model.zip'}`",
            f"- VecNormalize: `{run / 'best_model' / 'vecnormalize.pkl'}`",
        ])
        hdr = state.get("distill_header")
        if hdr:
            lines.append(f"- Distilled header: `{hdr}`")
        else:
            lines.append("- Distilled header: *(distill not run or failed)*")
        why = "Beat baseline" if float(best_ppo.get("score", 0)) > PPO_BASELINE_SCORE else "Best among suite trials"
        lines.append(f"- Why Best: {why}")
    else:
        lines.append("- No PPO trial completed successfully.")

    lines.extend(["", "## DQN (if completed)"])
    best_dqn = state.get("best_dqn")
    if best_dqn:
        run = Path(best_dqn["run_dir"])
        summary = best_dqn.get("summary", {})
        lines.extend([
            f"- Trial: {best_dqn.get('trial_id', 'n/a')}",
            f"- Run dir: `{run}`",
            f"- Selected model: `{summary.get('selected_model', run / 'selected_best_model.pt')}`",
            f"- Deploy header: `{summary.get('deploy_header', run / 'deploy' / 'model_weights.h')}`",
            f"- Best stable success rate: {best_dqn.get('best_stable_success_rate', 'n/a')}",
        ])
    else:
        lines.append("- DQN not completed or skipped.")

    lines.extend([
        "",
        "## How this was produced",
        "```powershell",
        "cd lab3",
        "python run_best_suite.py --hours 6",
        "```",
        "See `suite_experiment_log.csv` for all trials (including failures).",
    ])
    BEST_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _handle_sigint(signum: int, frame: Any) -> None:
    global _shutdown_requested
    _shutdown_requested = True
    print("\n[SUITE] Interrupt received — finishing current step and restoring configs...", flush=True)


def execute_trial(
    trial: Trial,
    state: Dict[str, Any],
    deadline: float,
) -> bool:
    """Run one trial. Returns False if should stop suite (time/shutdown)."""
    global _shutdown_requested
    if _shutdown_requested:
        return False
    if trial.trial_id in state.get("completed_ids", []):
        print(f"[SUITE] Skip {trial.trial_id} (already completed)", flush=True)
        return True
    if log_has_ok(trial.trial_id):
        if trial.trial_id not in state["completed_ids"]:
            state["completed_ids"].append(trial.trial_id)
            save_state(state)
        print(f"[SUITE] Skip {trial.trial_id} (logged ok)", flush=True)
        return True
    if not can_start(trial.estimate_s, deadline):
        print(f"[SUITE] Not enough time for {trial.trial_id} (need ~{trial.estimate_s:.0f}s)", flush=True)
        return False

    print(f"\n{'=' * 80}\n[SUITE] Starting {trial.trial_id}: {trial.hypothesis}\n{'=' * 80}", flush=True)
    status, run_dir, score, wall_s, notes = run_train_trial(trial)

    row = {
        "trial_id": trial.trial_id,
        "track": trial.track,
        "hypothesis": trial.hypothesis,
        "overrides_json": json.dumps(trial.overrides, sort_keys=True),
        "seed": trial.seed,
        "run_dir": str(run_dir.resolve()) if run_dir else "",
        "best_score": f"{score:.6f}" if score != float("-inf") else "",
        "wall_s": f"{wall_s:.1f}",
        "status": status,
        "notes": notes,
    }
    append_log_row(row)

    if status == "ok" and run_dir:
        if trial.track == "ppo":
            fps = ppo_measure_fps(run_dir)
            if fps:
                state["measured_ppo_fps"] = fps
            update_best_ppo(state, trial, run_dir, score)
        else:
            summary = dqn_summary(run_dir)
            update_best_dqn(state, trial, run_dir, summary)

        state["completed_ids"].append(trial.trial_id)
        save_state(state)
        print(f"[SUITE] {trial.trial_id} done | score={score:.3f} | wall={wall_s:.0f}s", flush=True)
    else:
        print(f"[SUITE] {trial.trial_id} FAILED: {notes}", flush=True)

    return True


def run_suite(args: argparse.Namespace) -> int:
    global _shutdown_requested
    signal.signal(signal.SIGINT, _handle_sigint)

    ensure_pristine()
    restore_all_configs()

    if args.dry_run:
        fps = None
        state = load_state()
        if state.get("measured_ppo_fps"):
            fps = float(state["measured_ppo_fps"])
        print("[DRY-RUN] PPO queue:")
        for t in build_ppo_queue(fps):
            print(f"  {t.trial_id}: {t.hypothesis} ~{t.estimate_s:.0f}s")
        print("[DRY-RUN] DQN queue:")
        for t in build_dqn_queue():
            print(f"  {t.trial_id}: {t.hypothesis} ~{t.estimate_s:.0f}s")
        print(f"[DRY-RUN] Baseline PPO score: {PPO_BASELINE_SCORE}")
        return 0

    if args.smoke_only:
        print("[SUITE] Smoke-only mode")
        ok_ppo, _ = run_smoke("ppo")
        ok_dqn, _ = run_smoke("dqn")
        restore_all_configs()
        if ok_ppo and ok_dqn:
            print("[SUITE] Both smokes passed.")
            return 0
        print(f"[SUITE] Smoke failed: PPO={ok_ppo} DQN={ok_dqn}")
        return 1

    state = load_state()
    if args.resume and STATE_PATH.is_file():
        deadline = float(state.get("deadline_unix", 0))
        if deadline <= time.time():
            if args.hours:
                deadline = time.time() + args.hours * 3600
                state["deadline_unix"] = deadline
                print(f"[SUITE] Extended deadline by --hours {args.hours}", flush=True)
            else:
                print("[SUITE] Resume deadline expired. Pass --hours N to extend.", flush=True)
                return 2
    else:
        deadline = time.time() + args.hours * 3600
        state["deadline_unix"] = deadline
        state["hours"] = args.hours
        save_state(state)

    print(f"[SUITE] Deadline: {datetime.fromtimestamp(deadline).isoformat(timespec='seconds')}", flush=True)

    try:
        if not args.skip_smoke and "smoke" not in state.get("completed_ids", []):
            ok_ppo, _ = run_smoke("ppo")
            if not ok_ppo:
                print("[SUITE] PPO smoke failed — aborting.", flush=True)
                return 1
            ok_dqn, _ = run_smoke("dqn")
            if not ok_dqn:
                print("[SUITE] DQN smoke failed — aborting.", flush=True)
                return 1
            state["completed_ids"].append("smoke")
            save_state(state)

        measured_fps = state.get("measured_ppo_fps")
        fps_val = float(measured_fps) if measured_fps else None

        if not args.dqn_only:
            for trial in build_ppo_queue(fps_val):
                if not execute_trial(trial, state, deadline):
                    break
                if _shutdown_requested:
                    break
                fps_val = state.get("measured_ppo_fps")
                if fps_val:
                    fps_val = float(fps_val)

            if state.get("best_ppo") and time_left(deadline) > 600 and not _shutdown_requested:
                for trial in seed_repeat_trials(state["best_ppo"]):
                    if not execute_trial(trial, state, deadline):
                        break
                    if _shutdown_requested:
                        break

            if (
                state.get("best_ppo")
                and "distill" not in state.get("completed_ids", [])
                and time_left(deadline) > 600
                and not _shutdown_requested
            ):
                run_dir = Path(state["best_ppo"]["run_dir"])
                print(f"\n[SUITE] Distilling {run_dir}...", flush=True)
                header = run_distill(run_dir)
                if header:
                    state["distill_header"] = str(header.resolve())
                    print(f"[SUITE] Distill OK: {header}", flush=True)
                    if time_left(deadline) > 300:
                        checks = run_headless_checks(header)
                        append_log_row({
                            "trial_id": "distill_checks",
                            "track": "ppo",
                            "hypothesis": "headless sim validation",
                            "overrides_json": "{}",
                            "seed": "",
                            "run_dir": str(run_dir),
                            "best_score": "",
                            "wall_s": "",
                            "status": "ok",
                            "notes": json.dumps(checks),
                        })
                else:
                    print("[SUITE] Distill failed — BEST.md will list teacher only.", flush=True)
                state["completed_ids"].append("distill")
                save_state(state)

        if not args.ppo_only and time_left(deadline) >= DQN_MIN_BUDGET_S and not _shutdown_requested:
            for trial in build_dqn_queue():
                est = trial.estimate_s
                if not can_start(est, deadline, reserve_s=300):
                    print(f"[SUITE] Skipping {trial.trial_id} — insufficient time for DQN", flush=True)
                    break
                if trial.trial_id in state.get("completed_ids", []):
                    continue
                if not execute_trial(trial, state, deadline):
                    break
                if _shutdown_requested:
                    break
                if trial.trial_id == "D0" and time_left(deadline) < DQN_MIN_BUDGET_S + 300:
                    break

        incomplete = _shutdown_requested or time_left(deadline) < 60
        write_best_md(state, float(state.get("hours", args.hours)), incomplete=incomplete)

        print("\n" + "=" * 80)
        print("[SUITE] Finished")
        if state.get("best_ppo"):
            print(f"  Best PPO: {state['best_ppo']['trial_id']} score={state['best_ppo']['score']:.3f}")
            print(f"  Run: {state['best_ppo']['run_dir']}")
        if state.get("distill_header"):
            print(f"  Header: {state['distill_header']}")
        if state.get("best_dqn"):
            print(f"  Best DQN: {state['best_dqn']['trial_id']}")
        print(f"  See: {BEST_MD}")
        if incomplete:
            print("  Resume: python run_best_suite.py --hours 6 --resume")
        print("=" * 80)
        return 0

    finally:
        restore_all_configs()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lab3 one-command Best training suite")
    p.add_argument("--hours", type=float, default=6.0, help="Wall-clock budget (default 6)")
    p.add_argument("--resume", action="store_true", help="Resume from suite_state.json")
    p.add_argument("--dry-run", action="store_true", help="Print queue and exit")
    p.add_argument("--smoke-only", action="store_true", help="Run PPO+DQN smoke tests only")
    p.add_argument("--ppo-only", action="store_true", help="Skip DQN trials")
    p.add_argument("--dqn-only", action="store_true", help="Skip PPO trials")
    p.add_argument("--skip-smoke", action="store_true", help="Skip initial smoke tests")
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Echo full child trainer stdout (default: quiet; only progress/status + suite messages)",
    )
    return p.parse_args()


def main() -> int:
    global _echo_all_child_output
    args = parse_args()
    _echo_all_child_output = bool(args.verbose)
    if args.ppo_only and args.dqn_only:
        print("Cannot use --ppo-only and --dqn-only together.", file=sys.stderr)
        return 2
    if not args.verbose:
        print("[SUITE] Quiet mode: full logs in suite_logs/. Pass --verbose for full echo.", flush=True)
    return run_suite(args)


if __name__ == "__main__":
    sys.exit(main())
