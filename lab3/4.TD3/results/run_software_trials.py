"""Run 10 nominal TD3 digital-twin trials and write course-format CSV/PNG.

Outputs (under results/software/):
  TD3_SIM_R01.csv ... TD3_SIM_R10.csv
  TD3_SIM_R01.png ... TD3_SIM_R10.png
  TD3_SIM_summary.csv
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

from rip_td3_sim_test import (  # noqa: E402
    build_result_figure,
    result_metrics,
    run_headless,
    save_result_csv,
)

MODEL = ROOT / "train" / "best_run" / "run" / "best_model" / "best_model.zip"
DURATION = 10.0
RANDOMIZATION = 0.0
PWM_LIMIT = 150.0
# Protocol seed list (change only seed across the batch).
SEEDS = list(range(10))


def main() -> int:
    if not MODEL.is_file():
        raise SystemExit(f"Missing model: {MODEL}")
    SOFTWARE.mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "hardware").mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for i, seed in enumerate(SEEDS, start=1):
        tag = f"TD3_SIM_R{i:02d}"
        print(f"[{tag}] seed={seed} duration={DURATION}s model={MODEL.name}", flush=True)
        result = run_headless(
            str(MODEL),
            DURATION,
            seed,
            RANDOMIZATION,
            PWM_LIMIT,
        )
        metrics = result_metrics(result)
        csv_path = SOFTWARE / f"{tag}.csv"
        png_path = SOFTWARE / f"{tag}.png"
        save_result_csv(result, str(csv_path))
        fig = build_result_figure(result, title_suffix=f"Direct TD3 | {tag} seed={seed}")
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
                "steps": len(result.time),
                "final_alpha_rad": float(result.alpha[-1]) if result.alpha.size else math.nan,
                "captured": int(captured),
                "capture_count": int(metrics["capture_count"]),
                "stable_start_time_s": metrics["stable_start_time"],
                "stable_duration_s": metrics["stable_duration"],
                "alpha_abs_mean": metrics["alpha_abs_mean"],
                "pwm_abs_mean": metrics["pwm_abs_mean"],
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

    summary_path = SOFTWARE / "TD3_SIM_summary.csv"
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
