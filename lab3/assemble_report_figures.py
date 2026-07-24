"""Assemble Lab Report 3 training/evaluation curves and representative figures.

Workers do not write the report placeholder names. This script builds them from
existing train logs and results/{software,hardware}/*_R##.png files into:

  submission/GroupXX_LabReport3/training/{ALG}_reward_curve.png
  submission/GroupXX_LabReport3/training/{ALG}_evaluation_curve.png
  submission/GroupXX_LabReport3/figures/{ALG}_{SIM|HW}_representative.png

Usage:
  python lab3/assemble_report_figures.py
  python lab3/assemble_report_figures.py --group 01
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LAB3 = Path(__file__).resolve().parent


def moving_mean_std(y: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        return np.array([]), np.array([]), np.array([])
    w = max(1, min(int(window), n))
    idx = np.arange(w - 1, n)
    mean = np.array([y[i - w + 1 : i + 1].mean() for i in idx])
    std = np.array([y[i - w + 1 : i + 1].std(ddof=0) for i in idx])
    return idx.astype(float), mean, std


def mark_stages(
    ax,
    steps: np.ndarray,
    stage_labels: list[str] | None,
    stage_steps: list[float] | list[tuple[float, str]] | None,
) -> None:
    boundaries: list[tuple[float, str]] = []
    if stage_steps:
        for item in stage_steps:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                boundaries.append((float(item[0]), str(item[1])))
            else:
                boundaries.append((float(item), ""))
    elif stage_labels is not None and len(stage_labels) == len(steps):
        prev = stage_labels[0]
        for step, label in zip(steps, stage_labels):
            if label != prev:
                boundaries.append((float(step), str(label)))
                prev = label
    ymin, ymax = ax.get_ylim()
    for i, (x, label) in enumerate(boundaries):
        ax.axvline(x, color="0.35", linestyle="--", linewidth=1.0, alpha=0.85)
        if label:
            ax.text(
                x,
                ymax - 0.04 * (ymax - ymin) * (1 + (i % 3) * 0.35),
                label,
                rotation=90,
                va="top",
                ha="right",
                fontsize=8,
                color="0.25",
            )


def save_reward_curve(
    out: Path,
    steps: np.ndarray,
    reward: np.ndarray,
    title: str,
    stage_labels: list[str] | None = None,
    stage_steps: list[float] | list[tuple[float, str]] | None = None,
    window: int = 30,
) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.plot(steps, reward, color="C0", alpha=0.35, linewidth=0.8, label="reward / step")
    idx, mean, std = moving_mean_std(reward, window)
    if len(idx):
        x = steps[idx.astype(int)]
        ax.plot(x, mean, color="C0", linewidth=2.0, label=f"moving mean ({window})")
        ax.fill_between(x, mean - std, mean + std, color="C0", alpha=0.18, label=f"moving std ({window})")
    ax.set_xlabel("timesteps")
    ax.set_ylabel("reward per step")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    mark_stages(ax, steps, stage_labels, stage_steps)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_eval_curve_rates(
    out: Path,
    steps: np.ndarray,
    reward: np.ndarray,
    capture: np.ndarray | None,
    stable: np.ndarray | None,
    title: str,
    reward_label: str = "mean reward / step",
    extra_series: list[tuple[np.ndarray, np.ndarray, str]] | None = None,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.2), sharex=True)
    ax_r, ax_s = axes
    ax_r.plot(steps, reward, marker="o", markersize=3, linewidth=1.4, label=reward_label)
    if extra_series:
        for xs, ys, label in extra_series:
            ax_r.plot(xs, ys, marker="o", markersize=3, linewidth=1.4, label=label)
    ax_r.set_ylabel("reward / step")
    ax_r.set_title(title)
    ax_r.grid(True, alpha=0.3)
    ax_r.legend(loc="best", fontsize=8)

    if capture is not None:
        ax_s.plot(steps, capture, marker="o", markersize=3, linewidth=1.4, label="capture rate")
    if stable is not None:
        ax_s.plot(steps, stable, marker="s", markersize=3, linewidth=1.4, label="stable-success rate")
    ax_s.set_xlabel("timesteps")
    ax_s.set_ylabel("rate")
    ax_s.set_ylim(-0.05, 1.05)
    ax_s.grid(True, alpha=0.3)
    ax_s.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_stage_steps(path: Path) -> list[tuple[float, str]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    out: list[tuple[float, str]] = []
    for row in data:
        step = float(row.get("step", 0))
        label = str(row.get("stage_name") or row.get("name") or f"stage@{int(step)}")
        out.append((step, label))
    return out


def _ppo_curriculum_markers(run: Path) -> list[tuple[float, str]]:
    cfg_path = run / "config_snapshot.json"
    if not cfg_path.exists():
        return []
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    dr = cfg.get("DOMAIN_RANDOMIZATION", {})
    ppo = cfg.get("PPO", {})
    if not bool(dr.get("enabled", False)):
        return []
    total = float(ppo.get("total_timesteps", 0) or 0)
    frac = float(dr.get("dr_curriculum_fraction", 0.0) or 0.0)
    if total <= 0 or frac <= 0:
        return []
    end_step = total * frac
    lo = float(dr.get("dr_initial_level", 0.0))
    hi = float(dr.get("dr_final_level", 0.0))
    return [(end_step, f"DR {lo:g}->{hi:g} curriculum end")]


def _find_episode_log(run: Path) -> Path | None:
    matches = sorted(run.glob("env_logs/**/episode_log.csv"))
    return matches[0] if matches else None


def _episode_reward_series(episode_log: Path) -> tuple[np.ndarray, np.ndarray]:
    ep = pd.read_csv(episode_log)
    lengths = ep["episode_length"].to_numpy(float)
    rewards = ep["episode_reward"].to_numpy(float)
    rps = rewards / np.maximum(lengths, 1.0)
    # Cumulative env steps across parallel envs is approximate; use cumsum of lengths.
    steps = np.cumsum(lengths)
    return steps, rps


def assemble_dqn(train_dir: Path, fig_dir: Path) -> dict[str, str]:
    # Prefer full overnight run over smoke best_run.
    candidates = [
        LAB3 / "2.DQN" / "train" / "runs" / "dqn_20260722_195223",
        LAB3 / "2.DQN" / "benchmark_run",
        LAB3 / "2.DQN" / "train" / "best_run" / "run",
    ]
    run = next(p for p in candidates if (p / "episode_history.csv").exists())
    ep = pd.read_csv(run / "episode_history.csv")
    ev = pd.read_csv(run / "eval_history.csv")

    save_reward_curve(
        train_dir / "DQN_reward_curve.png",
        ep["global_step"].to_numpy(float),
        ep["reward_per_step"].to_numpy(float),
        "DQN training reward per step",
        stage_labels=ep["stage_name"].astype(str).tolist(),
    )
    save_eval_curve_rates(
        train_dir / "DQN_evaluation_curve.png",
        ev["global_step"].to_numpy(float),
        ev["mean_reward_per_step"].to_numpy(float),
        ev["capture_rate"].to_numpy(float),
        ev["stable_success_rate"].to_numpy(float),
        "DQN deterministic evaluation",
    )

    sim_src = _pick_lowest_alpha_png(LAB3 / "2.DQN" / "results" / "software", "DQN_SIM_R*.csv")
    hw_src = _pick_lowest_alpha_png(LAB3 / "2.DQN" / "results" / "hardware", "DQN_HW_R*.csv")
    sim_dst = fig_dir / "DQN_SIM_representative.png"
    hw_dst = fig_dir / "DQN_HW_representative.png"
    shutil.copy2(sim_src, sim_dst)
    shutil.copy2(hw_src, hw_dst)
    return {
        "run": str(run),
        "sim": str(sim_src),
        "hw": str(hw_src),
    }


def assemble_ppo(train_dir: Path, fig_dir: Path) -> dict[str, str]:
    candidates = [
        LAB3 / "3.PPO" / "train" / "best_run_long" / "run",
        LAB3 / "3.PPO" / "train" / "best_run" / "run",
    ]
    run = next(p for p in candidates if (p / "training_metrics.csv").exists())
    ep = pd.read_csv(run / "training_metrics.csv")
    curriculum = _ppo_curriculum_markers(run)
    save_reward_curve(
        train_dir / "PPO_reward_curve.png",
        ep["timesteps"].to_numpy(float),
        ep["reward_per_step"].to_numpy(float),
        "PPO training reward per step",
        stage_steps=curriculum,
        window=30,
    )

    eval_csv = run / "eval_logs" / "eval_metrics.csv"
    if eval_csv.exists():
        ev = pd.read_csv(eval_csv)
        nom = ev[ev["eval_type"] == "nominal"]
        rnd = ev[ev["eval_type"] == "randomized"]
        fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.2), sharex=True)
        ax_r, ax_s = axes
        ax_r.plot(nom["timesteps"], nom["reward_per_step"], marker="o", markersize=3, label="nominal")
        if len(rnd):
            ax_r.plot(rnd["timesteps"], rnd["reward_per_step"], marker="o", markersize=3, label="randomized")
        ax_r.set_ylabel("reward / step")
        ax_r.set_title("PPO deterministic nominal and randomized evaluation")
        ax_r.grid(True, alpha=0.3)
        ax_r.legend(loc="best", fontsize=8)
        if "success_rate" in nom.columns:
            ax_s.plot(nom["timesteps"], nom["success_rate"], marker="s", markersize=3, label="nominal success")
        if len(rnd) and "success_rate" in rnd.columns:
            ax_s.plot(rnd["timesteps"], rnd["success_rate"], marker="s", markersize=3, label="randomized success")
        ax_s.set_xlabel("timesteps")
        ax_s.set_ylabel("success rate")
        ax_s.set_ylim(-0.05, 1.05)
        ax_s.grid(True, alpha=0.3)
        ax_s.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(train_dir / "PPO_evaluation_curve.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        built_in = run / "eval_logs" / "eval_curve.png"
        if built_in.exists():
            shutil.copy2(built_in, train_dir / "PPO_evaluation_curve.png")

    sim_src = _pick_from_summary_or_alpha(
        LAB3 / "3.PPO" / "results" / "software",
        "PPO_SIM_summary.csv",
        "PPO_SIM_R*.csv",
        prefer_captured=True,
        score_col="alpha_abs_mean",
    )
    shutil.copy2(sim_src, fig_dir / "PPO_SIM_representative.png")

    hw_dir = LAB3 / "3.PPO" / "results" / "hardware"
    hw_src = None
    if hw_dir.exists():
        try:
            hw_src = _pick_lowest_alpha_png(hw_dir, "PPO_HW_R*.csv")
            shutil.copy2(hw_src, fig_dir / "PPO_HW_representative.png")
        except FileNotFoundError:
            hw_src = None

    notes = []
    if curriculum:
        notes.append(
            f"Curriculum marker at step {int(curriculum[0][0])} ({curriculum[0][1]})."
        )
    if hw_src is None:
        notes.append(
            "PPO_HW_representative.png missing: no hardware trial PNGs under 3.PPO/results/hardware."
        )

    return {
        "run": str(run),
        "sim": str(sim_src),
        "hw": str(hw_src) if hw_src else "MISSING",
        "curriculum_markers": curriculum,
        "notes": notes,
    }


def assemble_td3(train_dir: Path, fig_dir: Path) -> dict[str, str | list]:
    # Evaluation: prefer full student curriculum run (rich eval_history + stages).
    eval_candidates = [
        LAB3 / "4.TD3" / "train" / "runs" / "td3_rip_original_2m_then_dr_20260722_200150",
        LAB3 / "4.TD3" / "train" / "best_run" / "run",
    ]
    eval_run = next(p for p in eval_candidates if (p / "eval_history.json").exists())
    rows = json.loads((eval_run / "eval_history.json").read_text(encoding="utf-8"))
    eval_steps = np.array([float(r["step"]) for r in rows])
    eval_reward = np.array(
        [float(r["mean_reward"]) / max(float(r.get("mean_length", 1.0)), 1.0) for r in rows]
    )
    capture = np.array([float(r.get("capture_rate", np.nan)) for r in rows])
    stable = np.array([float(r.get("stable_success_rate", np.nan)) for r in rows])

    save_eval_curve_rates(
        train_dir / "TD3_evaluation_curve.png",
        eval_steps,
        eval_reward,
        capture,
        stable,
        "TD3 deterministic evaluation reward, capture rate and stable-success rate",
    )

    # Reward: prefer true episode_log; fall back to longest matching run with logs.
    reward_run = eval_run
    episode_log = _find_episode_log(reward_run)
    reward_notes: list[str] = []
    if episode_log is None:
        fallback_candidates = [
            LAB3 / "4.TD3" / "train" / "runs" / "benchmark_td3_rip_original_2m_then_dr_20260721_113956",
            LAB3 / "4.TD3" / "train" / "runs" / "td3_rip_original_2m_then_dr_20260722_115841",
            LAB3 / "4.TD3" / "train" / "runs" / "td3_rip_original_2m_then_dr_20260722_121830",
        ]
        # Prefer the episode_log with the most rows among candidates.
        best: tuple[int, Path, Path] | None = None
        for cand in fallback_candidates:
            log = _find_episode_log(cand)
            if log is None:
                continue
            n = sum(1 for _ in log.open(encoding="utf-8")) - 1
            if best is None or n > best[0]:
                best = (n, cand, log)
        if best is None:
            raise FileNotFoundError("No TD3 episode_log.csv found for reward curve")
        _, reward_run, episode_log = best
        reward_notes.append(
            "Student long run has no env_logs/episode_log.csv; "
            f"TD3_reward_curve.png uses episode_log from {reward_run.name}."
        )

    steps, rps = _episode_reward_series(episode_log)
    stage_steps = load_stage_steps(reward_run / "stage_transitions.json")
    if not stage_steps:
        # Infer nominal→DR boundaries from eval_history phases when stages file is absent.
        stage_steps = _infer_stages_from_eval(reward_run / "eval_history.json")

    save_reward_curve(
        train_dir / "TD3_reward_curve.png",
        steps,
        rps,
        "TD3 episode reward per step",
        stage_steps=stage_steps,
        window=min(30, max(5, len(rps) // 10)),
    )

    soft = LAB3 / "4.TD3" / "results" / "software"
    sim_src = _pick_td3_sim_representative(soft)
    hw_src = _pick_lowest_alpha_png(LAB3 / "4.TD3" / "results" / "hardware", "TD3_HW_R*.csv", min_rows=50)
    shutil.copy2(sim_src, fig_dir / "TD3_SIM_representative.png")
    shutil.copy2(hw_src, fig_dir / "TD3_HW_representative.png")

    sim_notes = _td3_sim_stabilization_note(soft, sim_src)
    notes = reward_notes + ([sim_notes] if sim_notes else [])

    return {
        "eval_run": str(eval_run),
        "reward_run": str(reward_run),
        "episode_log": str(episode_log),
        "sim": str(sim_src),
        "hw": str(hw_src),
        "notes": notes,
    }


def _infer_stages_from_eval(eval_path: Path) -> list[tuple[float, str]]:
    if not eval_path.exists():
        return []
    rows = json.loads(eval_path.read_text(encoding="utf-8"))
    out: list[tuple[float, str]] = []
    prev_phase = None
    prev_level = None
    for row in rows:
        phase = str(row.get("phase", ""))
        level = row.get("eval_randomization_level", row.get("level"))
        if prev_phase is None:
            prev_phase, prev_level = phase, level
            continue
        if phase != prev_phase or level != prev_level:
            label = phase if level in (None, 0, 0.0) else f"{phase}_{level}"
            out.append((float(row["step"]), str(label)))
            prev_phase, prev_level = phase, level
    return out


def _td3_sim_stabilization_note(folder: Path, chosen: Path) -> str:
    summary = folder / "TD3_SIM_summary.csv"
    if not summary.exists():
        return (
            f"TD3_SIM_representative.png={chosen.name}; "
            "no summary available to confirm upright stabilization."
        )
    df = pd.read_csv(summary)
    if "stable_duration_s" in df.columns and float(df["stable_duration_s"].fillna(0).max()) <= 0:
        return (
            f"TD3_SIM_representative.png={chosen.name} is best capture among available trials, "
            "but no SIM trial reached stable upright hold (all stable_duration_s=0). "
            "Replace after a successful SIM campaign."
        )
    return f"TD3_SIM_representative.png={chosen.name}."


def _alpha_mean_from_csv(csv_path: Path) -> float:
    df = pd.read_csv(csv_path)
    col = next((c for c in df.columns if "alpha" in c.lower() and "dot" not in c.lower()), None)
    if col is None:
        return float("inf")
    return float(df[col].abs().mean())


def _pick_lowest_alpha_png(folder: Path, csv_glob: str, min_rows: int = 1) -> Path:
    best_csv = None
    best_score = float("inf")
    for csv_path in sorted(folder.glob(csv_glob)):
        df = pd.read_csv(csv_path)
        if len(df) < min_rows:
            continue
        score = _alpha_mean_from_csv(csv_path)
        if score < best_score:
            best_score = score
            best_csv = csv_path
    if best_csv is None:
        raise FileNotFoundError(f"No usable trials matching {csv_glob} in {folder}")
    png = best_csv.with_suffix(".png")
    if not png.exists():
        raise FileNotFoundError(f"Missing PNG for {best_csv}")
    return png


def _pick_from_summary_or_alpha(
    folder: Path,
    summary_name: str,
    csv_glob: str,
    *,
    prefer_captured: bool = False,
    score_col: str = "alpha_abs_mean",
    fallback_score_col: str = "max_abs_theta_rad",
) -> Path:
    summary = folder / summary_name
    if summary.exists():
        df = pd.read_csv(summary)
        pool = df
        if prefer_captured and "captured" in df.columns:
            captured = df[df["captured"] == 1]
            if len(captured):
                pool = captured
        col = score_col if score_col in pool.columns else fallback_score_col
        if col in pool.columns and "png" in pool.columns:
            row = pool.loc[pool[col].idxmin()]
            png = folder / str(row["png"])
            if png.exists():
                return png
    return _pick_lowest_alpha_png(folder, csv_glob)


def _pick_td3_sim_representative(folder: Path) -> Path:
    return _pick_from_summary_or_alpha(
        folder,
        "TD3_SIM_summary.csv",
        "TD3_SIM_R*.csv",
        prefer_captured=True,
        score_col="max_abs_theta_rad",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", default="XX")
    args = parser.parse_args()

    out = LAB3 / "submission" / f"Group{args.group}_LabReport3"
    train_dir = out / "training"
    fig_dir = out / "figures"
    train_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        "DQN": assemble_dqn(train_dir, fig_dir),
        "PPO": assemble_ppo(train_dir, fig_dir),
        "TD3": assemble_td3(train_dir, fig_dir),
    }

    notes: list[str] = []
    for alg, info in sources.items():
        for note in info.get("notes", []) if isinstance(info, dict) else []:
            notes.append(f"{alg}: {note}")

    manifest = {
        "output": str(out),
        "sources": sources,
        "notes": notes,
        "figures": sorted(p.name for p in train_dir.glob("*.png"))
        + sorted(p.name for p in fig_dir.glob("*.png")),
    }
    (out / "FIGURE_SOURCES.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Also mirror report-include paths next to docs/report3.tex.
    docs_train = LAB3 / "docs" / "training"
    docs_fig = LAB3 / "docs" / "figures"
    docs_train.mkdir(parents=True, exist_ok=True)
    docs_fig.mkdir(parents=True, exist_ok=True)
    for png in train_dir.glob("*_curve.png"):
        shutil.copy2(png, docs_train / png.name)
    for png in fig_dir.glob("*_representative.png"):
        shutil.copy2(png, docs_fig / png.name)

    expected = [
        "DQN_reward_curve.png",
        "DQN_evaluation_curve.png",
        "PPO_reward_curve.png",
        "PPO_evaluation_curve.png",
        "TD3_reward_curve.png",
        "TD3_evaluation_curve.png",
        "DQN_SIM_representative.png",
        "DQN_HW_representative.png",
        "PPO_SIM_representative.png",
        "TD3_SIM_representative.png",
        "TD3_HW_representative.png",
    ]
    missing = []
    for name in expected:
        path = train_dir / name if "curve" in name else fig_dir / name
        if not path.exists():
            missing.append(name)

    print(json.dumps(manifest, indent=2))
    for note in notes:
        print("NOTE:", note)
    if missing:
        print("Missing:", ", ".join(missing))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
