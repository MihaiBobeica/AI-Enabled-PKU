"""Thirty-second deterministic test for the DQN network."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import config
from runtime import (
    MatlabAlignedRIPEnv,
    choose_device,
    dump_json,
    emit_event,
    greedy_action,
    load_model,
    resolve_model_path,
)


def normalize_variant(variant: str) -> str:
    return "current" if str(variant).lower() in {"current", "both", "history", "no_history"} else str(variant)


def run_test(run_dir: str, variant: str = "current", smoke: bool = False) -> Dict[str, Any]:
    variant = normalize_variant(variant)
    if variant != "current":
        raise ValueError("Only current 6-D input is supported")
    root = Path(run_dir).resolve()
    model_path = resolve_model_path(root)
    device = choose_device()
    loaded = load_model(model_path, device)

    duration = float(config.SMOKE["test_duration_seconds"] if smoke else config.TEST["duration_seconds"])
    dt = float(config.ENV["physical_dt"])
    total_steps = max(1, int(round(duration / dt)))
    level = float(config.TEST["randomization_level"])
    seed = int(config.TEST["seed"])

    rows: List[Dict[str, Any]] = []
    episode = 0
    global_i = 0
    episode_successes: List[bool] = []
    episode_rewards: List[float] = []
    episode_maintain: List[int] = []

    while global_i < total_steps:
        env = MatlabAlignedRIPEnv(randomization_level=level, seed=seed + episode)
        obs, _ = env.reset(seed=seed + episode)
        episode += 1
        ep_reward = 0.0
        ep_max = 0
        for ep_step in range(int(config.ENV["max_physical_steps"])):
            if global_i >= total_steps:
                break
            with torch.no_grad():
                q = loaded.network(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))[0]
                action = int(torch.argmax(q).item())
                qmax = float(torch.max(q).detach().cpu().item())
            next_obs, reward, terminated, truncated, info = env.step(action)
            state = np.asarray(info["state"], dtype=float)
            true_state = np.asarray(info["true_state"], dtype=float)
            rows.append({
                "step": global_i,
                "time_s": global_i * dt,
                "episode": episode,
                "episode_step": ep_step,
                "theta": state[0],
                "theta_dot": state[1],
                "alpha": state[2],
                "alpha_dot": state[3],
                "true_theta": true_state[0],
                "true_theta_dot": true_state[1],
                "true_alpha": true_state[2],
                "true_alpha_dot": true_state[3],
                "action_index": action,
                "pwm": float(info["pwm"]),
                "pwm_effective": float(info["pwm_effective"]),
                "reward": float(reward),
                "qmax": qmax,
                "maintain_cur_stable": int(info["maintain_cur_stable"]),
                "maintain_max_stable": int(info["maintain_max_stable"]),
                "terminated": int(terminated),
                "truncated": int(truncated),
                "randomization_level": level,
            })
            ep_reward += float(reward)
            ep_max = max(ep_max, int(info["maintain_max_stable"]))
            global_i += 1
            obs = next_obs
            if terminated or truncated:
                break
        episode_rewards.append(ep_reward)
        episode_maintain.append(ep_max)
        episode_successes.append(ep_max >= int(config.TEST["stable_hold_steps"]))
        env.close()

    csv_path = root / "test_trace_current.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    t = np.asarray([r["time_s"] for r in rows])
    alpha = np.asarray([r["alpha"] for r in rows])
    theta = np.asarray([r["theta"] for r in rows])
    theta_dot = np.asarray([r["theta_dot"] for r in rows])
    alpha_dot = np.asarray([r["alpha_dot"] for r in rows])
    pwm = np.asarray([r["pwm"] for r in rows])
    reward = np.asarray([r["reward"] for r in rows])

    fig, axes = plt.subplots(5, 1, figsize=(12, 13), sharex=True)
    axes[0].plot(t, theta); axes[0].set_ylabel("theta / rad"); axes[0].grid(True)
    axes[1].plot(t, alpha); axes[1].axhline(math.radians(15), linestyle="--"); axes[1].axhline(-math.radians(15), linestyle="--"); axes[1].set_ylabel("alpha / rad"); axes[1].grid(True)
    axes[2].plot(t, theta_dot, label="theta_dot"); axes[2].plot(t, alpha_dot, label="alpha_dot"); axes[2].legend(); axes[2].set_ylabel("rad/s"); axes[2].grid(True)
    axes[3].step(t, pwm, where="post"); axes[3].set_ylabel("PWM"); axes[3].grid(True)
    axes[4].plot(t, reward); axes[4].set_ylabel("reward"); axes[4].set_xlabel("time / s"); axes[4].grid(True)
    fig.suptitle(f"DQN test, DR level={level:.2f}")
    fig.tight_layout()
    png_path = root / "test_trace_current.png"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    result = {
        "variant": "current",
        "model": str(model_path),
        "duration_seconds": duration,
        "steps": total_steps,
        "episodes": episode,
        "randomization_level": level,
        "mean_reward_per_step": float(np.mean(reward)),
        "total_reward": float(np.sum(reward)),
        "stable_success_rate": float(np.mean(episode_successes)),
        "max_maintain_stable_steps": int(max(episode_maintain, default=0)),
        "mean_abs_alpha": float(np.mean(np.abs(alpha))),
        "mean_abs_alpha_dot": float(np.mean(np.abs(alpha_dot))),
        "mean_abs_pwm": float(np.mean(np.abs(pwm))),
        "result_json": str(root / "test_result_current.json"),
        "trace_csv": str(csv_path),
        "trace_png": str(png_path),
    }
    dump_json(root / "test_result_current.json", result)
    emit_event("test_finished", run_dir=str(root), result=result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    raise SystemExit("Use: python run.py --worker test --run-dir <dir> --variant current")
