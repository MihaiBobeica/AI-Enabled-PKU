"""Distil the selected TD3 actor into one 7-D current/history-compatible student."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import config
from runtime import (
    TD3PaperAligned,
    dump_json,
    emit_event,
    ensure_dir,
    make_raw_env,
    student_input,
)


class StudentActor(nn.Module):
    def __init__(self, input_dim: int = 7, hidden_sizes=(64, 64)):
        super().__init__()
        dims = [int(input_dim), *map(int, hidden_sizes), 1]
        layers = []
        for a, b in zip(dims[:-2], dims[1:-1]):
            layers.extend([nn.Linear(a, b), nn.ReLU()])
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.tanh(self.net(x))


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _teacher_path(run_dir: Path) -> Path:
    for path in (
        run_dir / "selected_best_model.zip",
        run_dir / "best_model" / "best_model.zip",
        run_dir / "best_nominal_model" / "best_model.zip",
        run_dir / "best_randomized_model" / "best_model.zip",
        run_dir / "final_model.zip",
    ):
        if path.exists():
            return path
    raise FileNotFoundError("No selected TD3 teacher checkpoint found")


def teacher_action(
    teacher: TD3PaperAligned,
    obs: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    x = torch.as_tensor(obs, dtype=torch.float32, device=device)
    with torch.no_grad():
        action = teacher.actor(x)
    return action.detach().cpu().numpy().astype(np.float32)


def collect_dataset(
    teacher: TD3PaperAligned,
    student: StudentActor | None,
    env,
    steps: int,
    device: torch.device,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    obs, _ = env.reset()
    xs, ys = [], []
    student_prob = float(config.DISTILL["student_action_probability"])
    for _ in range(int(steps)):
        x = student_input(obs, "current").reshape(1, 7)
        target = teacher_action(teacher, x, device)
        xs.append(x[0])
        ys.append(target[0])
        if student is None or rng.random() >= student_prob:
            action = target[0]
        else:
            with torch.no_grad():
                action = (
                    student(torch.as_tensor(x, dtype=torch.float32, device=device))
                    .cpu()
                    .numpy()[0]
                )
        obs, _, term, trunc, _ = env.step(np.asarray(action, dtype=np.float32))
        if term or trunc:
            obs, _ = env.reset()
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32).reshape(-1, 1)


def train_student(
    student: StudentActor,
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> float:
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True)
    opt = torch.optim.Adam(
        student.parameters(),
        lr=float(config.DISTILL["learning_rate"]),
        weight_decay=float(config.DISTILL["weight_decay"]),
    )
    last = 0.0
    student.train()
    for _ in range(int(epochs)):
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = student(xb)
            loss = torch.mean((pred - yb) ** 2)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                student.parameters(), float(config.DISTILL["grad_clip_norm"])
            )
            opt.step()
            losses.append(float(loss.item()))
        last = float(np.mean(losses)) if losses else 0.0
    return last


def evaluate_student(
    student: StudentActor,
    env,
    episodes: int,
    max_steps: int,
    device: torch.device,
) -> Dict[str, Any]:
    rewards, lengths = [], []
    student.eval()
    for _ in range(int(episodes)):
        obs, _ = env.reset()
        total = 0.0
        n = 0
        for _ in range(int(max_steps)):
            x = torch.as_tensor(
                student_input(obs).reshape(1, 7), dtype=torch.float32, device=device
            )
            with torch.no_grad():
                action = student(x).cpu().numpy()[0]
            obs, reward, term, trunc, _ = env.step(action)
            total += float(reward)
            n += 1
            if term or trunc:
                break
        rewards.append(total)
        lengths.append(n)
    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_length": float(np.mean(lengths)),
        "episodes": len(rewards),
    }


def _metadata() -> Dict[str, Any]:
    return {
        "variant": "current",
        "input_dim": 7,
        "output_dim": 1,
        "continuous_action": True,
        "pwm_limit": float(config.ENV["pwm_limit"]),
        "input_pre_scaled": False,
        "theta_dot_scale": float(config.ENV["theta_dot_limit"]),
        "alpha_dot_scale": float(config.ENV["alpha_dot_limit"]),
        "state_history_len": 1,
        "action_history_len": 1,
        "input_order": [
            "sin(theta)", "cos(theta)", "theta_dot", "sin(alpha)",
            "cos(alpha)", "alpha_dot", "last_action_norm",
        ],
        "obs_mean": np.zeros(7, dtype=np.float32),
        "obs_var": np.ones(7, dtype=np.float32),
    }


def save_student_checkpoint(
    path: Path,
    student: StudentActor,
    metrics: Dict[str, Any],
) -> None:
    payload = {
        **_metadata(),
        "hidden_sizes": tuple(config.DISTILL["student_hidden_sizes"]),
        "state_dict": student.state_dict(),
        "metrics": metrics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _c_array(name: str, arr: np.ndarray) -> str:
    flat = np.asarray(arr, np.float32).reshape(-1)
    values = ", ".join(f"{float(value):.9g}f" for value in flat)
    return f"static const float {name}[{flat.size}] = {{{values}}};\n"


def export_header(path: Path, student: StudentActor) -> None:
    linears = [module for module in student.net if isinstance(module, nn.Linear)]
    text = [
        "#pragma once\n",
        "/* TD3 student actor: ReLU, ReLU, linear, tanh.\n",
        " * Input: sin(theta), cos(theta), theta_dot, sin(alpha), cos(alpha),\n",
        " *        alpha_dot, last_action_norm. Velocities are raw rad/s clipped\n",
        " *        by the environment to +/-45 and +/-40.\n",
        " * Output: normalized continuous action in [-1,1], PWM=action*150.\n",
        " */\n",
        "#define MODEL_INPUT_DIM 7\n",
        "#define MODEL_OUTPUT_DIM 1\n",
        "#define MODEL_STATE_HISTORY_LEN 1\n",
        "#define MODEL_ACTION_HISTORY_LEN 1\n",
        "#define MODEL_INPUT_PRE_SCALED 0\n",
        "#define MODEL_CONTINUOUS_ACTION 1\n",
        f"#define MODEL_PWM_LIMIT {float(config.ENV['pwm_limit']):.1f}f\n",
        f"#define MODEL_THETA_DOT_CLIP {float(config.ENV['theta_dot_limit']):.1f}f\n",
        f"#define MODEL_ALPHA_DOT_CLIP {float(config.ENV['alpha_dot_limit']):.1f}f\n",
        f"#define MODEL_HIDDEN1 {int(config.DISTILL['student_hidden_sizes'][0])}\n",
        f"#define MODEL_HIDDEN2 {int(config.DISTILL['student_hidden_sizes'][1])}\n",
    ]
    for index, layer in enumerate(linears, 1):
        text.append(_c_array(f"model_w{index}", layer.weight.detach().cpu().numpy()))
        text.append(_c_array(f"model_b{index}", layer.bias.detach().cpu().numpy()))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(text), encoding="utf-8")


def load_student_checkpoint(path: Path, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    if int(payload.get("input_dim", -1)) != 7:
        raise ValueError("TD3 student input_dim is not 7")
    if int(payload.get("output_dim", -1)) != 1:
        raise ValueError("TD3 student output_dim is not 1")
    model = StudentActor(7, payload.get("hidden_sizes", (64, 64))).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def run_distillation(
    *, run_dir: str, target: str = "current", smoke: bool = False
) -> Dict[str, Any]:
    if target not in {"current", "both", "history", "no_history"}:
        raise ValueError("target must map to current")

    run_path = Path(run_dir).resolve()
    device = resolve_device(str(config.DISTILL["device"]))
    teacher_path = _teacher_path(run_path)
    teacher = TD3PaperAligned.load(str(teacher_path), device=device)
    if tuple(teacher.observation_space.shape) != (7,):
        raise ValueError(
            f"Teacher checkpoint expected 7-D input, got {teacher.observation_space.shape}"
        )
    if tuple(teacher.action_space.shape) != (1,):
        raise ValueError("Teacher checkpoint is not one-dimensional continuous TD3")

    out_dir = ensure_dir(run_path / "distillation" / "current" / "best")
    deploy = ensure_dir(run_path / "deploy")
    student = StudentActor(7, config.DISTILL["student_hidden_sizes"]).to(device)
    rng = np.random.default_rng(int(config.DISTILL["seed"]))

    iterations = 1 if smoke else int(config.DISTILL["dagger_iterations"])
    collect_steps = int(
        config.SMOKE["distill_collect_steps"]
        if smoke
        else config.DISTILL["collect_steps_per_iter"]
    )
    epochs = int(
        config.SMOKE["distill_epochs"]
        if smoke
        else config.DISTILL["epochs_per_iter"]
    )
    xs = np.empty((0, 7), np.float32)
    ys = np.empty((0, 1), np.float32)
    best_reward = -np.inf
    best_metrics: Dict[str, Any] = {}

    emit_event(
        "stage",
        stage="distillation",
        status="started",
        run_dir=str(run_path),
        target="current",
        teacher=str(teacher_path),
        input_dim=7,
        output_dim=1,
    )

    levels = list(config.DISTILL["collect_randomization_levels"])
    for index in range(iterations):
        level = float(levels[index % len(levels)])
        env = make_raw_env(
            randomization_level=level,
            seed=int(config.DISTILL["seed"]) + index,
        )
        x, y = collect_dataset(
            teacher,
            None if index == 0 else student,
            env,
            collect_steps,
            device,
            rng,
        )
        env.close()
        xs = np.concatenate([xs, x])
        ys = np.concatenate([ys, y])
        max_size = int(config.DISTILL["max_dataset_size"])
        if len(xs) > max_size:
            selected = rng.choice(len(xs), max_size, replace=False)
            xs, ys = xs[selected], ys[selected]

        loss = train_student(
            student,
            xs,
            ys,
            epochs=epochs,
            batch_size=int(config.DISTILL["batch_size"]),
            device=device,
        )
        eval_env = make_raw_env(
            randomization_level=float(config.DISTILL["eval_randomization_level"]),
            seed=9000 + index,
            max_physical_steps=int(
                config.SMOKE["max_eval_policy_steps"]
                if smoke
                else config.DISTILL["eval_max_policy_steps"]
            ),
        )
        metrics = evaluate_student(
            student,
            eval_env,
            1 if smoke else int(config.DISTILL["eval_episodes"]),
            int(
                config.SMOKE["max_eval_policy_steps"]
                if smoke
                else config.DISTILL["eval_max_policy_steps"]
            ),
            device,
        )
        eval_env.close()
        metrics.update(
            iteration=index + 1,
            loss=loss,
            dataset_size=len(xs),
            randomization_level=level,
        )
        if metrics["mean_reward"] > best_reward:
            best_reward = metrics["mean_reward"]
            best_metrics = metrics
            save_student_checkpoint(out_dir / "student64.pt", student, metrics)
            export_header(out_dir / "model_weights.h", student)
        emit_event("distillation_iteration", **metrics)

    shutil.copy2(out_dir / "model_weights.h", deploy / "model_weights_current.h")
    shutil.copy2(out_dir / "model_weights.h", deploy / "model_weights.h")
    (deploy / "active_variant.txt").write_text("current\n", encoding="utf-8")

    summary = {
        "variant": "current",
        "teacher": str(teacher_path),
        "student_input_dim": 7,
        "student_output_dim": 1,
        "continuous_action": True,
        "checkpoint": str(out_dir / "student64.pt"),
        "header": str(out_dir / "model_weights.h"),
        "deploy_header": str(deploy / "model_weights.h"),
        "metrics": best_metrics,
    }
    dump_json(run_path / "distillation_summary.json", summary)
    emit_event(
        "distillation_finished",
        run_dir=str(run_path),
        results={"current": summary},
    )
    return summary
