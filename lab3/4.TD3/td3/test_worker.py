"""Thirty-second TD3 student test and trace export."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

import config
from distill_worker import load_student_checkpoint, resolve_device
from runtime import dump_json, emit_event, extract_physical_state, make_raw_env, student_input


def _prepare_model_input(obs: np.ndarray, payload: Dict[str, Any]) -> np.ndarray:
    x = np.asarray(student_input(obs, "current"), dtype=np.float32).reshape(7).copy()
    x[0:2] = np.clip(x[0:2], -1.0, 1.0)
    x[2] = np.clip(
        x[2],
        -float(payload.get("theta_dot_scale", 45.0)),
        float(payload.get("theta_dot_scale", 45.0)),
    )
    x[3:5] = np.clip(x[3:5], -1.0, 1.0)
    x[5] = np.clip(
        x[5],
        -float(payload.get("alpha_dot_scale", 40.0)),
        float(payload.get("alpha_dot_scale", 40.0)),
    )
    x[6] = np.clip(x[6], -1.0, 1.0)
    return x


def run_test(
    *, run_dir: str, variant: str = "current", smoke: bool = False
) -> Dict[str, Any]:
    if variant not in {"current", "both", "history", "no_history"}:
        raise ValueError("variant must map to current")

    run_path = Path(run_dir).resolve()
    checkpoint = run_path / "distillation" / "current" / "best" / "student64.pt"
    device = resolve_device(str(config.DISTILL["device"]))
    student, payload = load_student_checkpoint(checkpoint, device)

    duration = float(
        config.SMOKE["test_duration_seconds"]
        if smoke
        else config.TEST["duration_seconds"]
    )
    dt = float(config.ENV["physical_dt"]) * int(config.ENV["action_repeat"])
    steps = max(1, int(round(duration / dt)))
    env = make_raw_env(
        randomization_level=float(config.TEST["randomization_level"]),
        seed=int(config.TEST["seed"]),
        max_physical_steps=steps,
    )
    obs, _ = env.reset()
    rows = []
    total_reward = 0.0

    emit_event(
        "stage",
        stage="test",
        status="started",
        run_dir=str(run_path),
        variant="current",
        duration_s=duration,
        input_dim=7,
        output_dim=1,
    )

    for index in range(steps):
        x = _prepare_model_input(obs, payload)
        with torch.no_grad():
            action_norm = float(
                student(
                    torch.as_tensor(
                        x.reshape(1, 7), dtype=torch.float32, device=device
                    )
                )
                .cpu()
                .numpy()[0, 0]
            )
        action_norm = float(np.clip(action_norm, -1.0, 1.0))
        action = np.array([action_norm], dtype=np.float32)
        next_obs, reward, term, trunc, info = env.step(action)
        state = extract_physical_state(info, next_obs)
        pwm = float(
            info.get(
                "effective_pwm",
                info.get("pwm", action_norm * float(config.ENV["pwm_limit"])),
            )
        )
        rows.append(
            [
                index,
                index * dt,
                *map(float, x),
                action_norm,
                pwm,
                float(reward),
                *map(float, state),
                int(term),
                int(trunc),
            ]
        )
        total_reward += float(reward)
        obs = next_obs
        if term or trunc:
            break
    env.close()

    csv_path = run_path / "test_trace_current.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "step",
                "time_s",
                "sin_theta",
                "cos_theta",
                "theta_dot",
                "sin_alpha",
                "cos_alpha",
                "alpha_dot",
                "last_action_norm_input",
                "action_norm_output",
                "pwm",
                "reward",
                "theta",
                "theta_dot_state",
                "alpha",
                "alpha_dot_state",
                "terminated",
                "truncated",
            ]
        )
        writer.writerows(rows)

    png_path = run_path / "test_trace_current.png"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        array = np.asarray(rows, dtype=float)
        figure = plt.figure(figsize=(11, 6))
        axis = figure.add_subplot(111)
        if len(array):
            axis.plot(array[:, 1], np.rad2deg(array[:, 14]), label="alpha (deg)")
            axis.plot(array[:, 1], array[:, 10], label="PWM")
        axis.set_xlabel("Time (s)")
        axis.grid(True)
        axis.legend()
        figure.tight_layout()
        figure.savefig(png_path, dpi=140)
        plt.close(figure)
    except Exception as exc:
        png_path.write_text(f"plot unavailable: {exc}\n", encoding="utf-8")

    result = {
        "variant": "current",
        "duration_requested_s": duration,
        "steps": len(rows),
        "duration_actual_s": len(rows) * dt,
        "total_reward": total_reward,
        "student_input_dim": int(payload["input_dim"]),
        "student_output_dim": int(payload["output_dim"]),
        "continuous_action": True,
        "randomization_level": float(config.TEST["randomization_level"]),
        "result_json": str(run_path / "test_result_current.json"),
        "trace_csv": str(csv_path),
        "trace_png": str(png_path),
    }
    dump_json(run_path / "test_result_current.json", result)
    emit_event(
        "test_finished",
        run_dir=str(run_path),
        variant="current",
        result=result,
    )
    return result
