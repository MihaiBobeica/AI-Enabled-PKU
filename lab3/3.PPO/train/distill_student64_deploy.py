"""
distill_student64_deploy.py

DAgger-style distillation from an SB3 PPO teacher into the compact deployment
actor used by the latest hybrid energy-swing-up + PPO-balance firmware.

Normal use:
    python3 distill_student64_deploy.py

Running without command-line model arguments opens a folder-selection window.
Select the PPO teacher run directory (the folder containing best_model/ and/or
final_model.zip).  The best 64x64 compact7 student is exported directly as:

    <selected teacher run>/model_weights.h

Deployment contract:
    raw input = [sin(theta), cos(theta), theta_dot,
                 sin(alpha), cos(alpha), alpha_dot, u_prev_norm]
    actor     = 7 -> 64 -> 64 -> 1
    hidden    = tanh
    output    = tanh, continuous normalized action in [-1, 1]
    header    = model_weights.h with internal compact7 normalization

The script also stores checkpoints, logs, and a second header copy below:
    <selected teacher run>/distill_student64_deploy/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent

# Make local vendored dependencies visible before importing config/run/SB3.
LOCAL_THIRD_PARTY = PROJECT_ROOT / "third_party"
if LOCAL_THIRD_PARTY.exists():
    sys.path.insert(0, str(LOCAL_THIRD_PARTY))

LOCAL_SB3_PARENT = PROJECT_ROOT / "stable_baselines3"
if LOCAL_SB3_PARENT.exists():
    sys.path.insert(0, str(LOCAL_SB3_PARENT))

sys.path.insert(0, str(PROJECT_ROOT))


def _early_config_name() -> str:
    argv = sys.argv[1:]
    if "--config" in argv:
        idx = argv.index("--config")
        if idx + 1 < len(argv):
            return str(argv[idx + 1]).replace(".py", "")
    return "config"


import importlib

config = importlib.import_module(_early_config_name())
sys.modules["config"] = config

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from run import make_vec_env
from rip_env.register_envs import register_all_envs

# Keep run.config aligned if run.py early-loaded a different default.
import run as _run_module

_run_module.config = config
sys.modules["config"] = config


DEFAULT_DISTILL: Dict[str, Any] = {
    "teacher_model_path": "",
    "output_subdir": "distill_student64_deploy",

    # GUI-selected teacher run is the default export location.
    "deploy_export_dir": "",
    "deploy_header_name": "model_weights.h",

    # Student deployed on STM32.  Keep this compatible with deploy_ppo.ino.
    "student_hidden_sizes": (64, 64),
    "student_activation": "Tanh",       # compatible/smooth; choices: Tanh, ReLU, Hardtanh
    "student_final_tanh": True,
    "deploy_obs_mode": "compact7",      # [sinθ, cosθ, θdot, sinα, cosα, αdot, u_prev_norm]

    # DAgger schedule.
    "dagger_iterations": 8,
    "collect_steps_per_iter": 40_000,
    "max_dataset_size": 320_000,
    "first_iter_teacher_rollout": True,
    "collect_randomization_levels": [0.0, 0.25, 0.50, 0.75],

    # Supervised fitting.
    "epochs_per_iter": 25,
    "batch_size": 4096,
    "learning_rate": 8e-4,
    "weight_decay": 1e-6,
    "grad_clip_norm": 1.0,
    "val_fraction": 0.10,

    # Closed-loop evaluation of the student.
    "eval_randomization_level": 0.75,
    "eval_episodes": 64,
    "eval_max_policy_steps": 3000,       # action_repeat=1 run uses 3000 policy steps = 15s

    "seed": 123,
    "device": "auto",
    "export_c_header": True,
}


def get_distill_cfg() -> Dict[str, Any]:
    cfg = dict(DEFAULT_DISTILL)
    user_cfg = getattr(config, "DISTILL_DEPLOY", None)
    if user_cfg is None:
        user_cfg = getattr(config, "DISTILL", {})
    if user_cfg:
        cfg.update(user_cfg)
    return cfg


class StudentActor(nn.Module):
    """Small deterministic actor for STM32 deployment."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes=(64, 64), activation="Tanh", final_tanh=True):
        super().__init__()
        act_table = {"ReLU": nn.ReLU, "Tanh": nn.Tanh, "Hardtanh": nn.Hardtanh}
        if activation not in act_table:
            raise ValueError(f"Unsupported activation {activation}; choose {list(act_table)}")
        layers: List[nn.Module] = []
        last = int(obs_dim)
        for h in hidden_sizes:
            h = int(h)
            layers.append(nn.Linear(last, h))
            layers.append(act_table[activation]())
            last = h
        layers.append(nn.Linear(last, int(action_dim)))
        if final_tanh:
            layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_sizes = tuple(int(x) for x in hidden_sizes)
        self.activation = str(activation)
        self.final_tanh = bool(final_tanh)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def ensure_dir(p: str | Path) -> None:
    os.makedirs(p, exist_ok=True)


def dump_json(path: str | Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def wrap_to_pi(x: float) -> float:
    return float((x + np.pi) % (2.0 * np.pi) - np.pi)


def find_latest_teacher_model() -> str:
    runs_root = PROJECT_ROOT / str(config.RUN.get("root_log_dir", "./runs"))
    if not runs_root.exists():
        raise FileNotFoundError(f"runs directory not found: {runs_root}")
    candidates: List[Path] = []
    for run_dir in runs_root.glob("ppo_sb3_sim2real_balance_*"):
        candidates += list(run_dir.glob("best_model/best_model.zip"))
        candidates += list(run_dir.glob("final_model.zip"))
    if not candidates:
        raise FileNotFoundError(f"No teacher model found under {runs_root}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


def infer_run_dir_from_model(model_path: str | Path) -> Path:
    p = Path(model_path).resolve()
    if p.parent.name == "best_model":
        return p.parent.parent
    return p.parent


def _is_sb3_model_zip(path: Path) -> bool:
    """Return True for an SB3 model archive, not for an arbitrary run backup ZIP."""
    if not path.is_file() or path.suffix.lower() != ".zip":
        return False
    try:
        import zipfile
        with zipfile.ZipFile(path, "r") as archive:
            names = set(archive.namelist())
        return "data" in names and any(name.endswith("policy.pth") for name in names)
    except Exception:
        return False


def resolve_teacher_model_from_directory(selection: str | Path) -> Path:
    """Resolve the intended PPO teacher model from a user-selected directory."""
    root = Path(selection).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Teacher selection is not a directory: {root}")

    priorities = [
        "best_model/best_model.zip",
        "selected_best_model.zip",
        "best_model.zip",
        "final_model.zip",
    ]
    for relative in priorities:
        candidate = root / relative
        if _is_sb3_model_zip(candidate):
            return candidate

    # Allow selecting the best_model directory itself.
    direct = root / "best_model.zip"
    if _is_sb3_model_zip(direct):
        return direct

    candidates = [p for p in root.rglob("*.zip") if _is_sb3_model_zip(p)]
    if not candidates:
        raise FileNotFoundError(
            "No SB3 PPO model archive was found below the selected directory. "
            "Expected best_model/best_model.zip or final_model.zip."
        )

    def rank(path: Path) -> tuple[int, float]:
        rel = str(path.relative_to(root)).replace("\\", "/")
        if rel == "best_model/best_model.zip":
            priority = 0
        elif path.name == "selected_best_model.zip":
            priority = 1
        elif path.name == "best_model.zip":
            priority = 2
        elif path.name == "final_model.zip":
            priority = 3
        else:
            priority = 10
        return priority, -path.stat().st_mtime

    candidates.sort(key=rank)
    return candidates[0]


def select_teacher_directory_gui(initial_dir: Optional[Path] = None) -> Optional[Path]:
    """Open a native directory chooser.  Tk is preferred; PyQt5 is a fallback."""
    title = "Select PPO Teacher Run Directory"
    initial = str((initial_dir or (PROJECT_ROOT / "runs")).expanduser())
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.update_idletasks()
        selected = filedialog.askdirectory(title=title, initialdir=initial, mustexist=True)
        root.destroy()
        return Path(selected).expanduser().resolve() if selected else None
    except Exception as tk_error:
        try:
            from PyQt5 import QtWidgets
            app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
            selected = QtWidgets.QFileDialog.getExistingDirectory(None, title, initial)
            return Path(selected).expanduser().resolve() if selected else None
        except Exception as qt_error:
            raise RuntimeError(
                "Could not open a directory-selection window. "
                f"Tk error: {tk_error}; PyQt5 error: {qt_error}. "
                "Use --teacher-dir /path/to/run as a fallback."
            ) from qt_error


def show_completion_dialog(title: str, message: str, *, error: bool = False) -> None:
    """Best-effort completion dialog; terminal output remains authoritative."""
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        root.update_idletasks()
        if error:
            messagebox.showerror(title, message)
        else:
            messagebox.showinfo(title, message)
        root.destroy()
    except Exception:
        pass


def find_vecnormalize_path(model_path: str | Path) -> Optional[Path]:
    p = Path(model_path).resolve()
    run_dir = infer_run_dir_from_model(p)
    candidates = [p.parent / "vecnormalize.pkl", run_dir / "vecnormalize.pkl"]
    # If a periodic checkpoint is used, try the matching vecnormalize file if present.
    stem = p.stem
    if "_steps" in stem:
        prefix = stem.replace(".zip", "")
        # ppo_stage1_balance_500000_steps.zip -> ppo_stage1_balance_vecnormalize_500000_steps.pkl
        parts = stem.split("_")
        if len(parts) >= 2:
            step_token = parts[-2] if parts[-1] == "steps" else None
            if step_token:
                candidates.insert(0, run_dir / "models" / f"ppo_stage1_balance_vecnormalize_{step_token}_steps.pkl")
    for c in candidates:
        if c.exists():
            return c
    return None


def make_eval_or_collect_env(randomization_level: float, vecnormalize_path: Optional[Path]):
    register_all_envs()
    reward_fn = config.build_reward_fn()
    done_fn = config.build_done_fn()
    env_cfg = config.build_env_config(randomization_level=float(randomization_level), nominal=False)
    env = make_vec_env(
        n_envs=1,
        env_cfg=env_cfg,
        reward_fn=reward_fn,
        done_fn=done_fn,
        monitor_dir=None,
        seed_offset=5000,
    )
    if vecnormalize_path is not None and vecnormalize_path.exists():
        env = VecNormalize.load(str(vecnormalize_path), env)
        env.training = False
        env.norm_reward = False
    else:
        print("[WARN] VecNormalize not found; using raw observations. This is usually wrong for an SB3 teacher.")
    return env


def set_env_randomization_level(env, level: float) -> None:
    try:
        env.env_method("set_randomization_level", float(level))
    except Exception:
        try:
            env.venv.env_method("set_randomization_level", float(level))
        except Exception:
            pass


def reset_with_random_level(env, levels: List[float]) -> np.ndarray:
    level = float(random.choice(levels)) if levels else 0.75
    set_env_randomization_level(env, level)
    return env.reset()


def get_true_state_from_vec(env) -> np.ndarray:
    """Return current true state [theta, theta_dot, alpha, alpha_dot] from a VecNormalize/VecEnv stack."""
    try:
        st = env.env_method("get_true_state")[0]
    except Exception:
        st = env.venv.env_method("get_true_state")[0]
    return np.asarray(st, dtype=np.float32).reshape(4)


def compact7_from_state(state: np.ndarray, prev_action_norm: float) -> np.ndarray:
    theta = wrap_to_pi(float(state[0]))
    theta_dot = float(state[1])
    alpha = wrap_to_pi(float(state[2]))
    alpha_dot = float(state[3])
    return np.asarray([
        math.sin(theta), math.cos(theta), theta_dot,
        math.sin(alpha), math.cos(alpha), alpha_dot,
        float(np.clip(prev_action_norm, -1.0, 1.0)),
    ], dtype=np.float32)


def predict_student_np(student: StudentActor, raw_obs: np.ndarray, obs_mean: np.ndarray, obs_std: np.ndarray, device: torch.device) -> np.ndarray:
    student.eval()
    x = (np.asarray(raw_obs, dtype=np.float32) - obs_mean) / obs_std
    with torch.no_grad():
        xt = torch.as_tensor(x.reshape(1, -1), dtype=torch.float32, device=device)
        y = student(xt).detach().cpu().numpy()
    return y.astype(np.float32)


# -----------------------------------------------------------------------------
# Dataset collection
# -----------------------------------------------------------------------------
def collect_labeled_data(
    *,
    teacher: PPO,
    student: StudentActor,
    env,
    device: torch.device,
    steps: int,
    randomization_levels: List[float],
    rollout_source: str,
    student_obs_mean: Optional[np.ndarray],
    student_obs_std: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect compact raw obs -> teacher deterministic action pairs.

    The teacher is evaluated on the original SB3 normalized observation.
    The student is trained/deployed on compact7 raw observation with internal normalization.
    """
    assert rollout_source in {"teacher", "student"}
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []

    sb3_obs = reset_with_random_level(env, randomization_levels)
    prev_action_norm = 0.0

    for _ in range(int(steps)):
        state = get_true_state_from_vec(env)
        deploy_obs = compact7_from_state(state, prev_action_norm)
        teacher_action, _ = teacher.predict(sb3_obs, deterministic=True)
        teacher_action = np.asarray(teacher_action, dtype=np.float32).reshape(1, -1)

        xs.append(deploy_obs.copy())
        ys.append(teacher_action[0].copy())

        if rollout_source == "teacher":
            action_to_env = teacher_action
        else:
            if student_obs_mean is None or student_obs_std is None:
                # Before the first normalization statistics exist, drive with teacher to avoid nonsense rollout.
                action_to_env = teacher_action
            else:
                action_to_env = predict_student_np(student, deploy_obs, student_obs_mean, student_obs_std, device=device)

        sb3_obs, _, done, _infos = env.step(action_to_env)
        prev_action_norm = float(np.asarray(action_to_env).reshape(-1)[0])

        if bool(np.asarray(done).reshape(-1)[0]):
            sb3_obs = reset_with_random_level(env, randomization_levels)
            prev_action_norm = 0.0

    return np.stack(xs, axis=0).astype(np.float32), np.stack(ys, axis=0).astype(np.float32)


def trim_dataset(x: np.ndarray, y: np.ndarray, max_size: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(x) <= max_size:
        return x, y
    idx = np.random.choice(len(x), size=int(max_size), replace=False)
    return x[idx], y[idx]


def compute_student_norm(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0).astype(np.float32)
    std = np.std(x, axis=0).astype(np.float32)
    std = np.maximum(std, 1e-4).astype(np.float32)
    return mean, std


def train_student_supervised(
    *,
    student: StudentActor,
    x_raw: np.ndarray,
    y: np.ndarray,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    cfg: Dict[str, Any],
    device: torch.device,
) -> Dict[str, float]:
    x = ((x_raw.astype(np.float32) - obs_mean) / obs_std).astype(np.float32)
    y = y.astype(np.float32)
    dataset = TensorDataset(torch.as_tensor(x), torch.as_tensor(y))
    n_val = max(1, int(len(dataset) * float(cfg["val_fraction"])))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=0)

    opt = torch.optim.AdamW(student.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    loss_fn = nn.MSELoss()
    student.to(device)
    student.train()
    last_train = math.nan
    for _epoch in range(int(cfg["epochs_per_iter"])):
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = student(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), float(cfg["grad_clip_norm"]))
            opt.step()
            losses.append(float(loss.item()))
        last_train = float(np.mean(losses)) if losses else math.nan

    student.eval()
    val_mse, val_mae, n = 0.0, 0.0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = student(xb)
            diff = pred - yb
            bs = int(xb.shape[0])
            val_mse += float(torch.mean(diff.pow(2)).item()) * bs
            val_mae += float(torch.mean(torch.abs(diff)).item()) * bs
            n += bs
    return {"train_mse": last_train, "val_mse": val_mse / max(n, 1), "val_mae": val_mae / max(n, 1)}


# -----------------------------------------------------------------------------
# Closed-loop evaluation
# -----------------------------------------------------------------------------
def evaluate_student_closed_loop(
    *,
    student: StudentActor,
    env,
    device: torch.device,
    episodes: int,
    max_policy_steps: int,
    randomization_level: float,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
) -> Dict[str, float]:
    student.eval()
    ep_rewards, ep_lengths = [], []
    alpha_abs_all, theta_abs_all, theta_dot_abs_all, alpha_dot_abs_all, pwm_abs_all = [], [], [], [], []
    terminated_flags = []

    for _ in range(int(episodes)):
        _sb3_obs = reset_with_random_level(env, [float(randomization_level)])
        prev_action_norm = 0.0
        total_reward = 0.0
        length = 0
        terminated_early = False
        theta_abs, alpha_abs, theta_dot_abs, alpha_dot_abs, pwm_abs = [], [], [], [], []

        for _step in range(int(max_policy_steps)):
            state = get_true_state_from_vec(env)
            deploy_obs = compact7_from_state(state, prev_action_norm)
            action = predict_student_np(student, deploy_obs, obs_mean, obs_std, device=device)
            _sb3_obs, reward, done, infos = env.step(action)
            prev_action_norm = float(action.reshape(-1)[0])
            total_reward += float(np.asarray(reward).reshape(-1)[0])
            length += 1

            info = infos[0] if isinstance(infos, (list, tuple)) and len(infos) > 0 else {}
            st = info.get("true_state", state)
            st = np.asarray(st, dtype=np.float64).reshape(4)
            theta_abs.append(abs(wrap_to_pi(float(st[0]))))
            theta_dot_abs.append(abs(float(st[1])))
            alpha_abs.append(abs(wrap_to_pi(float(st[2]))))
            alpha_dot_abs.append(abs(float(st[3])))
            pwm = info.get("effective_pwm", info.get("last_pwm", info.get("pwm", 0.0)))
            pwm_abs.append(abs(float(pwm)) / max(float(config.ENV["pwm_limit"]), 1e-6))

            if bool(np.asarray(done).reshape(-1)[0]):
                terminated_early = length < int(max_policy_steps)
                break

        ep_rewards.append(total_reward)
        ep_lengths.append(length)
        terminated_flags.append(float(terminated_early))
        theta_abs_all.append(np.mean(theta_abs) if theta_abs else np.nan)
        alpha_abs_all.append(np.mean(alpha_abs) if alpha_abs else np.nan)
        theta_dot_abs_all.append(np.mean(theta_dot_abs) if theta_dot_abs else np.nan)
        alpha_dot_abs_all.append(np.mean(alpha_dot_abs) if alpha_dot_abs else np.nan)
        pwm_abs_all.append(np.mean(pwm_abs) if pwm_abs else np.nan)

    mean_reward = float(np.mean(ep_rewards))
    mean_length = float(np.mean(ep_lengths))
    mean_abs_alpha = float(np.nanmean(alpha_abs_all))
    mean_abs_theta = float(np.nanmean(theta_abs_all))
    mean_abs_theta_dot = float(np.nanmean(theta_dot_abs_all))
    mean_abs_alpha_dot = float(np.nanmean(alpha_dot_abs_all))
    mean_abs_pwm_norm = float(np.nanmean(pwm_abs_all))
    terminated_rate = float(np.mean(terminated_flags))
    score = mean_reward + 0.02 * mean_length - 40.0 * mean_abs_alpha - 0.4 * mean_abs_alpha_dot - 0.1 * mean_abs_theta - 5.0 * terminated_rate
    return {
        "mean_reward": mean_reward,
        "mean_length": mean_length,
        "mean_abs_theta": mean_abs_theta,
        "mean_abs_alpha": mean_abs_alpha,
        "mean_abs_theta_dot": mean_abs_theta_dot,
        "mean_abs_alpha_dot": mean_abs_alpha_dot,
        "mean_abs_pwm_norm": mean_abs_pwm_norm,
        "terminated_rate": terminated_rate,
        "score": float(score),
    }


# -----------------------------------------------------------------------------
# Export
# -----------------------------------------------------------------------------
def c_float_literal(value: float) -> str:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite value cannot be exported to C: {value}")
    if value == 0.0:
        return "0.0f"
    token = f"{value:.9g}"
    if "." not in token and "e" not in token.lower():
        token += ".0"
    return token + "f"


def c_2d(name: str, arr: np.ndarray) -> str:
    arr = np.asarray(arr, dtype=np.float32)
    rows = []
    for r in arr:
        rows.append("    {" + ", ".join(c_float_literal(x) for x in r.reshape(-1)) + "}")
    return f"static const float {name}[{arr.shape[0]}][{arr.shape[1]}] = {{\n" + ",\n".join(rows) + "\n};\n"


def c_1d(name: str, arr: np.ndarray) -> str:
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    return f"static const float {name}[{arr.size}] = {{" + ", ".join(c_float_literal(x) for x in arr) + "};\n"


def activation_c(name: str) -> str:
    if name == "ReLU":
        return "static inline float PPO_activation(float x) { return x > 0.0f ? x : 0.0f; }\n"
    if name == "Hardtanh":
        return "static inline float PPO_activation(float x) { if (x < -1.0f) return -1.0f; if (x > 1.0f) return 1.0f; return x; }\n"
    return "static inline float PPO_activation(float x) { return tanhf(x); }\n"


def export_deploy_header(
    path: str | Path,
    student: StudentActor,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    teacher_path: str,
    metrics: Dict[str, float],
) -> None:
    linear_layers = [m for m in student.net if isinstance(m, nn.Linear)]
    if len(linear_layers) != 3:
        raise ValueError("deploy_ppo header exporter currently expects exactly 3 Linear layers.")
    W0 = linear_layers[0].weight.detach().cpu().numpy().astype(np.float32)
    B0 = linear_layers[0].bias.detach().cpu().numpy().astype(np.float32)
    W1 = linear_layers[1].weight.detach().cpu().numpy().astype(np.float32)
    B1 = linear_layers[1].bias.detach().cpu().numpy().astype(np.float32)
    W2 = linear_layers[2].weight.detach().cpu().numpy().astype(np.float32)
    B2 = linear_layers[2].bias.detach().cpu().numpy().astype(np.float32)

    obs_dim = int(student.obs_dim)
    h0 = int(W0.shape[0])
    h1 = int(W1.shape[0])
    action_dim = int(student.action_dim)
    max_layer_dim = max(h0, h1, action_dim)
    pwm_limit = float(config.ENV.get("pwm_limit", 255.0))

    lines: List[str] = []
    lines.append("#pragma once\n")
    lines.append("#ifndef PPO_MODEL_WEIGHTS_H\n#define PPO_MODEL_WEIGHTS_H\n\n")
    lines.append("#include <stdint.h>\n#include <math.h>\n\n")
    lines.append("/*\n")
    lines.append(" * Auto-generated by distill_student64_deploy.py\n")
    lines.append(" * Deployment actor distilled from SB3 PPO teacher.\n")
    lines.append(" * Teacher: " + str(teacher_path) + "\n")
    lines.append(" * Input convention: compact7 raw observation\n")
    lines.append(" *   [sin(theta), cos(theta), theta_dot, sin(alpha), cos(alpha), alpha_dot, u_prev_norm]\n")
    lines.append(" * The header normalizes compact7 internally before the MLP forward pass.\n")
    lines.append(" * Closed-loop student eval metrics at export time:\n")
    for k in ["mean_length", "terminated_rate", "mean_abs_alpha", "mean_abs_theta", "mean_abs_pwm_norm", "score"]:
        if k in metrics:
            lines.append(f" *   {k}: {metrics[k]:.9g}\n")
    lines.append(" */\n\n")
    lines.append(f"#define PPO_OBS_DIM {obs_dim}\n#define PPO_STATE_DIM {obs_dim}\n#define PPO_ACT_DIM {action_dim}\n")
    lines.append("#define PPO_LINEAR_LAYER_COUNT 3\n")
    lines.append(f"#define PPO_CONTINUOUS_PWM_LIMIT {c_float_literal(pwm_limit)}\n")
    lines.append("#define PPO_ACTION_TYPE_DISCRETE 0\n#define PPO_ACTION_TYPE_CONTINUOUS 1\n")
    lines.append(f"#define PPO_L0_IN {obs_dim}\n#define PPO_L0_OUT {h0}\n")
    lines.append(f"#define PPO_L1_IN {h0}\n#define PPO_L1_OUT {h1}\n")
    lines.append(f"#define PPO_L2_IN {h1}\n#define PPO_L2_OUT {action_dim}\n")
    lines.append(f"#define PPO_MAX_LAYER_DIM {max_layer_dim}\n")
    lines.append("#define PPO_INTERNAL_OBS_NORMALIZATION 1\n\n")

    lines.append(c_1d("PPO_OBS_MEAN", obs_mean))
    lines.append(c_1d("PPO_OBS_STD", obs_std))
    lines.append("\n")
    lines.append(c_2d("PPO_W0", W0))
    lines.append(c_1d("PPO_B0", B0))
    lines.append("\n")
    lines.append(c_2d("PPO_W1", W1))
    lines.append(c_1d("PPO_B1", B1))
    lines.append("\n")
    lines.append(c_2d("PPO_W2", W2))
    lines.append(c_1d("PPO_B2", B2))
    lines.append("\n")
    lines.append("static inline float PPO_clip_float(float x, float lo, float hi) { if (x < lo) return lo; if (x > hi) return hi; return x; }\n")
    lines.append(activation_c(student.activation))
    lines.append("""
static inline void PPO_linear_forward(
    const float *input,
    float *output,
    int in_dim,
    int out_dim,
    const float *W,
    const float *B
) {
    for (int i = 0; i < out_dim; ++i) {
        float acc = B[i];
        for (int j = 0; j < in_dim; ++j) {
            acc += W[i * in_dim + j] * input[j];
        }
        output[i] = acc;
    }
}

static inline void PPO_normalize_obs(const float obs_raw[PPO_OBS_DIM], float obs_norm[PPO_OBS_DIM]) {
    for (int i = 0; i < PPO_OBS_DIM; ++i) {
        obs_norm[i] = (obs_raw[i] - PPO_OBS_MEAN[i]) / PPO_OBS_STD[i];
        obs_norm[i] = PPO_clip_float(obs_norm[i], -10.0f, 10.0f);
    }
}

static inline void PPO_actor_forward(const float obs[PPO_OBS_DIM], float out[PPO_ACT_DIM]) {
    float obs_n[PPO_OBS_DIM];
    float buf_a[PPO_MAX_LAYER_DIM];
    float buf_b[PPO_MAX_LAYER_DIM];

    PPO_normalize_obs(obs, obs_n);

    PPO_linear_forward(obs_n, buf_a, PPO_L0_IN, PPO_L0_OUT, &PPO_W0[0][0], PPO_B0);
    for (int i = 0; i < PPO_L0_OUT; ++i) buf_a[i] = PPO_activation(buf_a[i]);

    PPO_linear_forward(buf_a, buf_b, PPO_L1_IN, PPO_L1_OUT, &PPO_W1[0][0], PPO_B1);
    for (int i = 0; i < PPO_L1_OUT; ++i) buf_b[i] = PPO_activation(buf_b[i]);

    PPO_linear_forward(buf_b, buf_a, PPO_L2_IN, PPO_L2_OUT, &PPO_W2[0][0], PPO_B2);
""")
    if student.final_tanh:
        lines.append("    for (int i = 0; i < PPO_ACT_DIM; ++i) out[i] = tanhf(buf_a[i]);\n")
    else:
        lines.append("    for (int i = 0; i < PPO_ACT_DIM; ++i) out[i] = buf_a[i];\n")
    lines.append("""
}

static inline float PPO_predict_action_norm(const float obs[PPO_OBS_DIM]) {
    float out[PPO_ACT_DIM];
    PPO_actor_forward(obs, out);
    return PPO_clip_float(out[0], -1.0f, 1.0f);
}

static inline float PPO_predict_pwm_float(const float obs[PPO_OBS_DIM]) {
    return PPO_predict_action_norm(obs) * PPO_CONTINUOUS_PWM_LIMIT;
}

static inline int16_t PPO_predict_pwm_int16(const float obs[PPO_OBS_DIM]) {
    float pwm_f = PPO_predict_pwm_float(obs);
    if (pwm_f >= 0.0f) return (int16_t)(pwm_f + 0.5f);
    return (int16_t)(pwm_f - 0.5f);
}

#endif  // PPO_MODEL_WEIGHTS_H
""")

    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(lines))


def export_npz(path: str | Path, student: StudentActor, obs_mean: np.ndarray, obs_std: np.ndarray) -> None:
    arrays: Dict[str, np.ndarray] = {"obs_mean": obs_mean.astype(np.float32), "obs_std": obs_std.astype(np.float32)}
    layer_idx = 0
    for module in student.net:
        if isinstance(module, nn.Linear):
            arrays[f"W{layer_idx}"] = module.weight.detach().cpu().numpy().astype(np.float32)
            arrays[f"b{layer_idx}"] = module.bias.detach().cpu().numpy().astype(np.float32)
            layer_idx += 1
    np.savez(path, **arrays)


def save_checkpoint(output_dir: Path, student: StudentActor, obs_mean: np.ndarray, obs_std: np.ndarray, cfg: Dict[str, Any], metrics: Dict[str, float]) -> None:
    ensure_dir(output_dir)
    torch.save({
        "state_dict": student.state_dict(),
        "obs_dim": student.obs_dim,
        "action_dim": student.action_dim,
        "hidden_sizes": student.hidden_sizes,
        "activation": student.activation,
        "final_tanh": student.final_tanh,
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "distill_cfg": cfg,
        "metrics": metrics,
    }, output_dir / "student64_deploy.pt")
    export_npz(output_dir / "student64_deploy_weights.npz", student, obs_mean, obs_std)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distill an SB3 PPO teacher into the latest deploy-compatible compact7 64x64 model_weights.h"
    )
    parser.add_argument("--teacher", type=str, default="", help="Direct path to an SB3 PPO model ZIP")
    parser.add_argument("--teacher-dir", type=str, default="", help="Teacher run directory; skips the folder chooser")
    parser.add_argument("--out", type=str, default="", help="Distillation logs/checkpoints directory")
    parser.add_argument("--header-out", type=str, default="", help="Exact output path for model_weights.h")
    parser.add_argument("--no-gui", action="store_true", help="Do not show chooser/completion dialogs")
    parser.add_argument("--smoke", action="store_true", help="Very short integration test, not a useful final distillation")
    parser.add_argument(
        "--config",
        type=str,
        default="config",
        help="Training config module (use config_bonus2 for Bonus-2 teachers)",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="",
        choices=["", "Tanh", "ReLU", "Hardtanh"],
        help="Student hidden activation. Bonus-2 deploy expects ReLU; hybrid firmware expects Tanh.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="Alias for --teacher-dir (Bonus-2 docs use this name)",
    )
    args = parser.parse_args()
    if args.run_dir and not args.teacher_dir:
        args.teacher_dir = args.run_dir
    # Re-bind if argparse config differs from early peek (should match).
    cfg_name = str(args.config).replace(".py", "")
    if cfg_name != getattr(config, "__name__", cfg_name):
        import importlib as _il
        new_cfg = _il.import_module(cfg_name)
        globals()["config"] = new_cfg
        sys.modules["config"] = new_cfg
        _run_module.config = new_cfg

    gui_used = False
    selected_dir: Optional[Path] = None
    try:
        if args.teacher:
            teacher_model = Path(args.teacher).expanduser().resolve()
            if not _is_sb3_model_zip(teacher_model):
                raise ValueError(f"Not a valid SB3 model ZIP: {teacher_model}")
            selected_dir = infer_run_dir_from_model(teacher_model)
        else:
            if args.teacher_dir:
                selected_dir = Path(args.teacher_dir).expanduser().resolve()
            else:
                if args.no_gui:
                    raise ValueError("Use --teacher or --teacher-dir when --no-gui is specified.")
                selected_dir = select_teacher_directory_gui()
                gui_used = True
                if selected_dir is None:
                    print("[CANCELLED] No teacher directory selected.")
                    return
            teacher_model = resolve_teacher_model_from_directory(selected_dir)

        run_dir = infer_run_dir_from_model(teacher_model)
        # If a containing run directory was explicitly selected, keep outputs there.
        export_root = selected_dir if selected_dir and selected_dir.is_dir() else run_dir

        cfg = get_distill_cfg()
        # Fixed deployment contract required by the latest firmware/panels.
        # Hybrid PPO balance uses Tanh; Bonus-2 always-on panels match teacher ReLU.
        activation = str(args.activation or "Tanh")
        cfg["student_hidden_sizes"] = (64, 64)
        cfg["student_activation"] = activation
        cfg["student_final_tanh"] = True
        cfg["deploy_obs_mode"] = "compact7"
        cfg["deploy_header_name"] = "model_weights.h"
        if args.smoke:
            cfg.update({
                "dagger_iterations": 1,
                "collect_steps_per_iter": 256,
                "max_dataset_size": 512,
                "epochs_per_iter": 1,
                "batch_size": 128,
                "eval_episodes": 1,
                "eval_max_policy_steps": 128,
                "collect_randomization_levels": [0.0],
                "eval_randomization_level": 0.0,
            })

        set_seed(int(cfg["seed"]))
        device = choose_device(str(cfg["device"]))
        output_dir = Path(args.out).expanduser().resolve() if args.out else export_root / str(cfg["output_subdir"])
        deploy_header_path = Path(args.header_out).expanduser().resolve() if args.header_out else export_root / "model_weights.h"
        ensure_dir(output_dir)
        ensure_dir(deploy_header_path.parent)

        teacher_model_path = str(teacher_model)
        vecnormalize_path = find_vecnormalize_path(teacher_model)
        print("=" * 100)
        print(f"[SELECTED DIR] {export_root}")
        print(f"[TEACHER]      {teacher_model}")
        print(f"[VECNORM]      {vecnormalize_path if vecnormalize_path else 'NOT FOUND'}")
        print(f"[OUTPUT]       {output_dir}")
        print(f"[HEADER]       {deploy_header_path}")
        print(f"[DEVICE]       {device}")
        print(f"[NETWORK]      compact7 7 -> 64 -> 64 -> 1, {activation}/{activation}/Tanh")
        print("=" * 100)

        collect_env = make_eval_or_collect_env(float(cfg["eval_randomization_level"]), vecnormalize_path)
        try:
            teacher = PPO.load(teacher_model_path, env=collect_env, device=device)

            obs_dim = 7
            action_dim = int(collect_env.action_space.shape[0])
            if action_dim != 1:
                raise ValueError(f"Latest deploy exporter expects action_dim=1, got {action_dim}")

            student = StudentActor(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_sizes=(64, 64),
                activation=activation,
                final_tanh=True,
            ).to(device)

            dump_json(output_dir / "distill_config_snapshot.json", cfg)
            log_path = output_dir / "distill_log.csv"
            with open(log_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "iter", "rollout_source", "dataset_size", "train_mse", "val_mse", "val_mae",
                    "eval_score", "eval_length", "eval_terminated_rate", "eval_abs_alpha", "eval_abs_theta", "eval_pwm_norm",
                ])

            all_x: Optional[np.ndarray] = None
            all_y: Optional[np.ndarray] = None
            obs_mean: Optional[np.ndarray] = None
            obs_std: Optional[np.ndarray] = None
            best_score = -1e18
            best_metrics: Dict[str, float] = {}
            levels = [float(x) for x in cfg["collect_randomization_levels"]]

            for it in range(int(cfg["dagger_iterations"])):
                source = "teacher" if (it == 0 and bool(cfg["first_iter_teacher_rollout"])) else "student"
                print(f"\n[COLLECT] iter={it} source={source} steps={cfg['collect_steps_per_iter']} levels={levels}")
                x_new, y_new = collect_labeled_data(
                    teacher=teacher,
                    student=student,
                    env=collect_env,
                    device=device,
                    steps=int(cfg["collect_steps_per_iter"]),
                    randomization_levels=levels,
                    rollout_source=source,
                    student_obs_mean=obs_mean,
                    student_obs_std=obs_std,
                )
                if all_x is None:
                    all_x, all_y = x_new, y_new
                else:
                    all_x = np.concatenate([all_x, x_new], axis=0)
                    all_y = np.concatenate([all_y, y_new], axis=0)
                all_x, all_y = trim_dataset(all_x, all_y, int(cfg["max_dataset_size"]))
                obs_mean, obs_std = compute_student_norm(all_x)

                print(f"[TRAIN] iter={it} dataset={len(all_x)} epochs={cfg['epochs_per_iter']}")
                losses = train_student_supervised(
                    student=student,
                    x_raw=all_x,
                    y=all_y,
                    obs_mean=obs_mean,
                    obs_std=obs_std,
                    cfg=cfg,
                    device=device,
                )
                metrics = evaluate_student_closed_loop(
                    student=student,
                    env=collect_env,
                    device=device,
                    episodes=int(cfg["eval_episodes"]),
                    max_policy_steps=int(cfg["eval_max_policy_steps"]),
                    randomization_level=float(cfg["eval_randomization_level"]),
                    obs_mean=obs_mean,
                    obs_std=obs_std,
                )
                print(
                    f"[EVAL] iter={it} score={metrics['score']:.3f} len={metrics['mean_length']:.1f} "
                    f"term={metrics['terminated_rate']:.3f} abs_alpha={metrics['mean_abs_alpha']:.5f} "
                    f"abs_theta={metrics['mean_abs_theta']:.5f} pwm={metrics['mean_abs_pwm_norm']:.3f} "
                    f"val_mse={losses['val_mse']:.6g}"
                )
                with open(log_path, "a", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        it, source, len(all_x), losses["train_mse"], losses["val_mse"], losses["val_mae"],
                        metrics["score"], metrics["mean_length"], metrics["terminated_rate"],
                        metrics["mean_abs_alpha"], metrics["mean_abs_theta"], metrics["mean_abs_pwm_norm"],
                    ])
                save_checkpoint(output_dir, student, obs_mean, obs_std, cfg, metrics)

                if metrics["score"] > best_score:
                    best_score = float(metrics["score"])
                    best_metrics = dict(metrics)
                    best_dir = output_dir / "best"
                    save_checkpoint(best_dir, student, obs_mean, obs_std, cfg, metrics)
                    export_deploy_header(best_dir / "model_weights.h", student, obs_mean, obs_std, teacher_model_path, metrics)
                    export_deploy_header(deploy_header_path, student, obs_mean, obs_std, teacher_model_path, metrics)
                    print(f"[BEST] iter={it} score={best_score:.3f}")
                    print(f"       best header:   {best_dir / 'model_weights.h'}")
                    print(f"       deploy header: {deploy_header_path}")

            if not deploy_header_path.is_file():
                raise RuntimeError("Distillation completed without producing model_weights.h")

            print("\n" + "=" * 100)
            print("[DONE] Distillation finished")
            print(f"Output directory: {output_dir}")
            print(f"Deployment header: {deploy_header_path}")
            print(f"Best score: {best_score:.3f}")
            for k, v in best_metrics.items():
                print(f"  {k}: {v:.6f}")
            print("=" * 100)
        finally:
            collect_env.close()

        if gui_used and not args.no_gui:
            show_completion_dialog(
                "PPO Distillation Complete",
                "The deploy-compatible student model was exported successfully:\n\n"
                f"{deploy_header_path}",
            )
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        if gui_used and not args.no_gui:
            show_completion_dialog("PPO Distillation Failed", str(exc), error=True)
        raise


if __name__ == "__main__":
    main()
