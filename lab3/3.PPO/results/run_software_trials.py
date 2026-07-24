"""Run 10 nominal PPO digital-twin trials and write course-format CSV/PNG.

Outputs (under results/software/):
  PPO_SIM_R01.csv ... PPO_SIM_R10.csv
  PPO_SIM_R01.png ... PPO_SIM_R10.png
  PPO_SIM_summary.csv
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy"
SOFTWARE = Path(__file__).resolve().parent / "software"
sys.path.insert(0, str(DEPLOY))

from rip_ppo_sim_test import (  # noqa: E402
    build_result_figure,
    result_metrics,
    run_headless,
    save_result_csv,
)

MODEL = (
    ROOT
    / "train"
    / "runs"
    / "ppo_sb3_sim2real_balance_20260723_224048"
    / "model_weights.h"
)
DURATION = 30.0
RANDOMIZATION = 0.0
PWM_LIMIT = 150.0
SWING_PWM = 120.0
# Predeclared seed list (change only seed across the batch).
SEEDS = [2026 + i for i in range(10)]


def main() -> int:
    if not MODEL.is_file():
        raise SystemExit(f"Missing model: {MODEL}")
    SOFTWARE.mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "hardware").mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for i, seed in enumerate(SEEDS, start=1):
        tag = f"PPO_SIM_R{i:02d}"
        print(f"[{tag}] seed={seed} duration={DURATION}s model={MODEL.name}", flush=True)
        result = run_headless(
            str(MODEL),
            DURATION,
            seed,
            RANDOMIZATION,
            PWM_LIMIT,
            SWING_PWM,
        )
        metrics = result_metrics(result)
        csv_path = SOFTWARE / f"{tag}.csv"
        png_path = SOFTWARE / f"{tag}.png"
        save_result_csv(result, str(csv_path))
        fig = build_result_figure(result, title_suffix=f"Hybrid PPO | {tag} seed={seed}")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        import matplotlib.pyplot as plt

        plt.close(fig)

        captured = int(metrics["capture_count"]) > 0 or (
            math.isfinite(metrics["stable_start_time"]) and metrics["stable_duration"] > 0.0
        )
        summary_rows.append(
            {
                "trial": tag,
                "seed": seed,
                "duration_s": DURATION,
                "randomization": RANDOMIZATION,
                "pwm_limit": PWM_LIMIT,
                "swing_pwm": SWING_PWM,
                "steps": len(result.time),
                "final_alpha_rad": float(result.alpha[-1]) if result.alpha.size else math.nan,
                "captured": int(captured),
                "capture_count": int(metrics["capture_count"]),
                "stable_start_time_s": metrics["stable_start_time"],
                "stable_duration_s": metrics["stable_duration"],
                "alpha_abs_mean": metrics["alpha_abs_mean"],
                "alpha_abs_std": metrics["alpha_abs_std"],
                "pwm_abs_mean": metrics["pwm_abs_mean"],
                "pwm_abs_std": metrics["pwm_abs_std"],
                "max_abs_theta_rad": metrics["max_abs_theta"],
                "csv": csv_path.name,
                "png": png_path.name,
                "model": str(MODEL),
            }
        )
        print(
            f"  steps={len(result.time)} captured={captured} "
            f"stable_start={metrics['stable_start_time']} "
            f"stable_dur={metrics['stable_duration']:.3f} "
            f"captures={int(metrics['capture_count'])}",
            flush=True,
        )

    summary_path = SOFTWARE / "PPO_SIM_summary.csv"
    fields = list(summary_rows[0].keys())
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    n_ok = sum(int(r["captured"]) for r in summary_rows)
    print(f"[DONE] {n_ok}/10 captured -> {SOFTWARE}", flush=True)
    print(f"[DONE] summary -> {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
