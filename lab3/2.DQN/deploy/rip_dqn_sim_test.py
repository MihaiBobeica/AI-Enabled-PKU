#!/usr/bin/env python3
"""MATLAB-aligned DQN test panel for the rotary inverted pendulum.

The script is deliberately self-contained.  Students can select either a
``.pt`` checkpoint, an exported ``model_weights.h`` file, or a training/run
folder.  The policy is evaluated from the hanging-down state through swing-up
and balance in the same 200 Hz nonlinear Furuta model and six-element
observation convention used by the MATLAB-reproduction training package.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

CONTROL_DT = 0.005
DEFAULT_ACTIONS = np.asarray([-150, -120, -90, -60, -30, 30, 60, 90, 120, 150], dtype=np.int16)
THETA_LIMIT = 12.0 * math.pi
THETA_DOT_LIMIT = 45.0
ALPHA_DOT_LIMIT = 40.0
ARM_LENGTH = 0.18
PEND_LENGTH = 0.24
MOTOR_RADIUS = 0.035
MOTOR_HEIGHT = 0.08
ARM_Z = MOTOR_HEIGHT
MODE_NAMES = {0: "DISABLED", 1: "DQN_SWING", 2: "DQN_CAPTURE", 3: "DQN_BALANCE"}

PHYSICAL_NOMINAL = {
    "g": 9.8, "c_theta": 0.025, "c_alpha": 0.001,
    "k_t": 0.2310, "k_b": 0.1875, "k_u": 0.04706, "R": 4.2857,
    "m1": 0.20625, "m2": 0.15845, "l1cg": 0.080305,
    "l1": 0.151894, "l2cg": 0.066733,
    "I1z": 0.00049228, "I2x": 0.00036892,
    "I2y": 2.3641e-05, "I2z": 0.00036139,
}
PARAM_SCALE_RANGES = {
    "g": (0.95, 1.05), "m1": (0.75, 1.25), "m2": (0.85, 1.15),
    "l1": (0.90, 1.10), "l1cg": (0.85, 1.15), "l2cg": (0.75, 1.25),
    "I1z": (0.55, 1.70), "I2x": (0.80, 1.20), "I2y": (0.50, 1.80),
    "I2z": (0.50, 1.80), "c_theta": (0.25, 3.50), "c_alpha": (0.25, 4.50),
    "k_t": (0.75, 1.25), "k_b": (0.75, 1.25), "k_u": (0.70, 1.30),
    "R": (0.80, 1.25),
}


def wrap_to_pi(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _float_tokens(text: str) -> np.ndarray:
    values = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", text.replace("f", ""))
    return np.asarray([float(v) for v in values], dtype=np.float32)


def _header_array(text: str, name: str, dtype: str = "float") -> np.ndarray:
    pattern = rf"static\s+const\s+{dtype}\s+{re.escape(name)}\s*[^=]*=\s*\{{(.*?)\}}\s*;"
    match = re.search(pattern, text, flags=re.S)
    if not match:
        raise ValueError(f"Array {name} was not found in the header")
    return _float_tokens(match.group(1))


def _state_value(state: Dict[str, object], suffix: str) -> np.ndarray:
    candidates = []
    for key, value in state.items():
        clean = str(key)
        if clean == suffix or clean.endswith("." + suffix):
            candidates.append(value)
    if len(candidates) != 1:
        raise KeyError(f"Could not uniquely locate {suffix}; matches={len(candidates)}")
    value = candidates[0]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


@dataclass
class DQNModel:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    w3: np.ndarray
    b3: np.ndarray
    actions: np.ndarray
    velocity_lpf: float
    source: Path

    def __post_init__(self) -> None:
        self.w1 = np.ascontiguousarray(self.w1, dtype=np.float32).reshape(64, 6)
        self.b1 = np.ascontiguousarray(self.b1, dtype=np.float32).reshape(64)
        self.w2 = np.ascontiguousarray(self.w2, dtype=np.float32).reshape(64, 64)
        self.b2 = np.ascontiguousarray(self.b2, dtype=np.float32).reshape(64)
        self.w3 = np.ascontiguousarray(self.w3, dtype=np.float32).reshape(10, 64)
        self.b3 = np.ascontiguousarray(self.b3, dtype=np.float32).reshape(10)
        self.actions = np.ascontiguousarray(self.actions, dtype=np.int16).reshape(10)
        if not all(np.all(np.isfinite(a)) for a in (self.w1, self.b1, self.w2, self.b2, self.w3, self.b3)):
            raise ValueError("Model contains non-finite parameters")
        if not 0.0 < float(self.velocity_lpf) <= 1.0:
            raise ValueError("velocity_lpf must be in (0, 1]")

    def q_values(self, observation: np.ndarray) -> np.ndarray:
        x = np.asarray(observation, dtype=np.float32).reshape(6)
        h1 = np.maximum(self.w1 @ x + self.b1, 0.0)
        h2 = np.maximum(self.w2 @ h1 + self.b2, 0.0)
        return self.w3 @ h2 + self.b3

    def predict(self, observation: np.ndarray) -> Tuple[int, int, float]:
        q = self.q_values(observation)
        index = int(np.argmax(q))
        return index, int(self.actions[index]), float(q[index])

    @property
    def digest(self) -> str:
        h = hashlib.sha256()
        for array in (self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.actions):
            h.update(np.ascontiguousarray(array).tobytes())
        return h.hexdigest()[:16]


def load_header_model(path: Path) -> DQNModel:
    text = path.read_text(encoding="utf-8", errors="ignore")
    w1 = _header_array(text, "DQN_W1").reshape(64, 6)
    b1 = _header_array(text, "DQN_b1").reshape(64)
    w2 = _header_array(text, "DQN_W2").reshape(64, 64)
    b2 = _header_array(text, "DQN_b2").reshape(64)
    w3 = _header_array(text, "DQN_W3").reshape(10, 64)
    b3 = _header_array(text, "DQN_b3").reshape(10)
    try:
        actions = _header_array(text, "DQN_ACTIONS", "int16_t").astype(np.int16)
    except ValueError:
        actions = DEFAULT_ACTIONS.copy()
    match = re.search(r"#define\s+MODEL_VELOCITY_LPF\s+([-+0-9.eE]+)f?", text)
    velocity_lpf = float(match.group(1)) if match else 0.25
    return DQNModel(w1, b1, w2, b2, w3, b3, actions, velocity_lpf, path)


def load_pt_model(path: Path) -> DQNModel:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to open .pt checkpoints") from exc
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if hasattr(checkpoint, "state_dict") and not isinstance(checkpoint, dict):
        state = checkpoint.state_dict()
        metadata: Dict[str, object] = {}
    elif isinstance(checkpoint, dict):
        metadata = checkpoint
        state = checkpoint.get("online_state_dict", checkpoint.get("state_dict"))
        if state is None and all(hasattr(v, "shape") for v in checkpoint.values()):
            state = checkpoint
    else:
        raise TypeError("Unsupported .pt checkpoint format")
    if not isinstance(state, dict):
        raise KeyError("Checkpoint has no online_state_dict/state_dict")
    actions = metadata.get("actions_pwm", metadata.get("action_values_pwm", DEFAULT_ACTIONS))
    velocity_lpf = float(metadata.get("velocity_lpf", 0.25))
    return DQNModel(
        _state_value(state, "fc1.weight"), _state_value(state, "fc1.bias"),
        _state_value(state, "fc2.weight"), _state_value(state, "fc2.bias"),
        _state_value(state, "fc3.weight"), _state_value(state, "fc3.bias"),
        np.asarray(actions, dtype=np.int16), velocity_lpf, path,
    )


def resolve_model_path(selection: str | Path) -> Path:
    path = Path(selection).expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() not in {".pt", ".h"}:
            raise ValueError("Select a .pt checkpoint or model_weights.h")
        return path
    if not path.is_dir():
        raise FileNotFoundError(path)
    priorities = [
        "selected_best_model.pt",
        "best_nominal_model/best_model.pt",
        "best_model.pt",
        "recovery_model/nominal_2m_last.pt",
        "final_model.pt",
        "model_weights.h",
        "best_nominal_model/model_weights.h",
    ]
    for relative in priorities:
        candidate = path / relative
        if candidate.is_file():
            return candidate.resolve()
    candidates = sorted(path.rglob("*.pt")) + sorted(path.rglob("model_weights*.h"))
    if len(candidates) == 1:
        return candidates[0].resolve()
    if not candidates:
        raise FileNotFoundError(f"No supported model found below {path}")
    raise RuntimeError("Several model files were found. Select the intended .pt or .h file directly.\n" + "\n".join(map(str, candidates[:20])))


def load_model(selection: str | Path) -> DQNModel:
    path = resolve_model_path(selection)
    return load_pt_model(path) if path.suffix.lower() == ".pt" else load_header_model(path)


def furuta_derivative(state: np.ndarray, pwm: float, p: Dict[str, float]) -> np.ndarray:
    _, thd, al, ald = map(float, state)
    s_a, c_a = math.sin(al), math.cos(al)
    k1 = p["k_t"] * p["k_u"] / p["R"]
    k2 = p["k_t"] * p["k_b"] / p["R"]
    a = (p["m1"] * p["l1cg"] ** 2 + p["I1z"] + p["m2"] * p["l1"] ** 2
         + (p["m2"] * p["l2cg"] ** 2 + p["I2z"]) * s_a ** 2 + p["I2y"] * c_a ** 2)
    b = p["m2"] * p["l1"] * p["l2cg"] * c_a
    c = -p["m2"] * p["l1"] * p["l2cg"] * s_a
    d = 2.0 * (p["I2z"] + p["m2"] * p["l2cg"] ** 2 - p["I2y"]) * s_a * c_a
    e = k1 * float(pwm) - k2 * thd - p["c_theta"] * thd
    f = -(p["m2"] * p["l2cg"] ** 2 + p["I2x"])
    g_term = -(p["m2"] * p["l1"] * p["l2cg"] * c_a)
    h = (p["m2"] * p["l2cg"] ** 2 - p["I2y"] + p["I2z"]) * s_a * c_a
    grav = p["m2"] * p["g"] * p["l2cg"] * s_a
    pend_friction = p["c_alpha"] * ald
    den1, den2 = a * f - g_term * b, g_term * b - a * f
    if abs(den1) < 1e-12 or abs(den2) < 1e-12:
        raise FloatingPointError("Singular Furuta dynamics")
    thdd = ((-f * c) * ald ** 2 + (-f * d) * ald * thd + (b * h) * thd ** 2 + b * grav + f * e - b * pend_friction) / den1
    aldd = ((-g_term * c) * ald ** 2 + (-g_term * d) * ald * thd + (a * h) * thd ** 2 + a * grav + g_term * e - a * pend_friction) / den2
    return np.asarray([thd, thdd, ald, aldd], dtype=np.float64)


def rk4_step(state: np.ndarray, pwm: float, p: Dict[str, float]) -> np.ndarray:
    k1 = furuta_derivative(state, pwm, p)
    k2 = furuta_derivative(state + 0.5 * CONTROL_DT * k1, pwm, p)
    k3 = furuta_derivative(state + 0.5 * CONTROL_DT * k2, pwm, p)
    k4 = furuta_derivative(state + CONTROL_DT * k3, pwm, p)
    nxt = state + CONTROL_DT * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    nxt[2] = wrap_to_pi(nxt[2])
    return nxt


@dataclass
class SimulationConfig:
    model_selection: str
    duration: float = 30.0
    playback_speed: float = 1.0
    seed: int = 2026
    randomization_level: float = 0.0
    initial_condition: str = "MATLAB random downward"
    save_csv: bool = True
    output_dir: str = os.path.expanduser("~/rip_twin_logs")


@dataclass
class SimulationResult:
    time: np.ndarray
    theta: np.ndarray
    theta_dot: np.ndarray
    alpha: np.ndarray
    alpha_dot: np.ndarray
    pwm: np.ndarray
    mode: np.ndarray
    blend: np.ndarray


def stable_phase_start_index(result: SimulationResult, threshold_deg: float = 15.0) -> Optional[int]:
    if result.alpha.size == 0:
        return None
    inside = np.abs(result.alpha) <= math.radians(threshold_deg)
    outside = np.flatnonzero(~inside)
    start = int(outside[-1] + 1) if outside.size else 0
    return start if start < inside.size else None


def result_metrics(result: SimulationResult) -> Dict[str, float]:
    index = stable_phase_start_index(result)
    metrics = {"stable_start_time": math.nan, "stable_duration": 0.0,
               "alpha_abs_mean": math.nan, "alpha_abs_std": math.nan,
               "pwm_abs_mean": math.nan, "pwm_abs_std": math.nan,
               "max_abs_theta": float(np.max(np.abs(result.theta))) if result.theta.size else math.nan}
    if index is not None:
        metrics.update(stable_start_time=float(result.time[index]),
                       stable_duration=float(result.time[-1] - result.time[index]),
                       alpha_abs_mean=float(np.mean(np.abs(result.alpha[index:]))),
                       alpha_abs_std=float(np.std(np.abs(result.alpha[index:]))),
                       pwm_abs_mean=float(np.mean(np.abs(result.pwm[index:]))),
                       pwm_abs_std=float(np.std(np.abs(result.pwm[index:]))))
    return metrics


def build_result_figure(result: SimulationResult, title_suffix: str = "DQN Digital Twin"):
    from matplotlib.figure import Figure
    metrics = result_metrics(result)
    fig = Figure(figsize=(10.5, 7.6), tight_layout=True)
    axs = fig.subplots(2, 2)
    ax_alpha, ax_theta, ax_pwm, ax_hist = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]
    t, alpha, theta, pwm = result.time, result.alpha, result.theta, result.pwm
    fig.suptitle(f"Rotary Inverted Pendulum {title_suffix} Response", fontsize=15, fontweight="bold")
    ax_alpha.plot(t, alpha, linewidth=1.8, label=r"$\alpha$")
    ax_alpha.axhline(0.0, linewidth=1.0, linestyle="--")
    for sign in (-1, 1): ax_alpha.axhline(sign * math.radians(15), linewidth=0.8, linestyle=":")
    ax_alpha.set(title=r"Pendulum Angle $\alpha(t)$", xlabel="Time / s", ylabel=r"$\alpha$ / rad")
    ax_alpha.grid(True, linestyle="--", linewidth=0.6, alpha=0.55); ax_alpha.legend(loc="lower right")
    index = stable_phase_start_index(result)
    if index is None:
        text = "Stable phase: not reached"
    else:
        text = (rf"$\mathrm{{mean}}(|\alpha|)$ = {metrics['alpha_abs_mean']:.6f} rad" "\n"
                rf"$\mathrm{{std}}(|\alpha|)$ = {metrics['alpha_abs_std']:.6f} rad" "\n"
                f"stable from t = {metrics['stable_start_time']:.3f} s")
        ax_alpha.axvspan(metrics["stable_start_time"], t[-1], alpha=0.10)
    ax_alpha.text(0.98, 0.96, text, transform=ax_alpha.transAxes, ha="right", va="top", fontsize=9.5,
                  bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="0.35"))
    ax_theta.plot(t, theta, linewidth=1.8, label=r"$\theta$"); ax_theta.axhline(0, linewidth=1, linestyle="--")
    ax_theta.set(title=r"Rotary Arm Angle $\theta(t)$", xlabel="Time / s", ylabel=r"$\theta$ / rad")
    ax_theta.grid(True, linestyle="--", linewidth=0.6, alpha=0.55); ax_theta.legend(loc="upper right")
    ax_pwm.plot(t, pwm, linewidth=1.5, label="PWM"); ax_pwm.axhline(0, linewidth=1, linestyle="--")
    ax_pwm.set(title="Control Input PWM(t)", xlabel="Time / s", ylabel="PWM")
    ax_pwm.set_ylim(-max(160.0, float(np.max(np.abs(pwm))) * 1.1), max(160.0, float(np.max(np.abs(pwm))) * 1.1))
    ax_pwm.grid(True, linestyle="--", linewidth=0.6, alpha=0.55); ax_pwm.legend(loc="lower right")
    if index is not None:
        ax_pwm.axvspan(metrics["stable_start_time"], t[-1], alpha=0.10)
        ptext = rf"$\mathrm{{mean}}(|PWM|)$ = {metrics['pwm_abs_mean']:.3f}" "\n" rf"$\mathrm{{std}}(|PWM|)$ = {metrics['pwm_abs_std']:.3f}"
    else: ptext = "Stable phase: not reached"
    ax_pwm.text(0.98, 0.96, ptext, transform=ax_pwm.transAxes, ha="right", va="top", fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="0.35"))
    if index is None:
        ax_hist.text(0.5, 0.5, "No final stable phase", transform=ax_hist.transAxes, ha="center", va="center")
    else:
        ax_hist.hist(pwm[index:], bins=np.arange(-255, 271, 15), edgecolor="black", linewidth=0.45)
    ax_hist.axvline(0, linewidth=1, linestyle="--"); ax_hist.set_xlim(-255, 255)
    ax_hist.set(title="Stable-stage PWM Distribution", xlabel="PWM", ylabel="Count")
    ax_hist.grid(True, linestyle="--", linewidth=0.6, alpha=0.55)
    xmax = max(float(t[-1]), 0.1)
    for ax in (ax_alpha, ax_theta, ax_pwm): ax.set_xlim(0, xmax)
    for ax in (ax_alpha, ax_theta, ax_pwm, ax_hist):
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.tick_params(direction="in")
    return fig


def save_result_csv(result: SimulationResult, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_s", "theta_rad", "theta_dot_rad_s", "alpha_rad", "alpha_dot_rad_s", "pwm", "mode", "blend"])
        writer.writerows(zip(result.time, result.theta, result.theta_dot, result.alpha, result.alpha_dot, result.pwm, result.mode, result.blend))


def default_output_paths(output_dir: str, duration: float) -> Tuple[str, str]:
    directory = Path(output_dir).expanduser().resolve(); directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{duration:.3f}".rstrip("0").rstrip(".").replace(".", "p")
    return str(directory / f"rip_dqn_sim_{tag}s_{stamp}.png"), str(directory / f"rip_dqn_sim_{tag}s_{stamp}.csv")


def sample_physical(rng: np.random.Generator, level: float) -> Dict[str, float]:
    level = float(np.clip(level, 0.0, 1.0)); values = dict(PHYSICAL_NOMINAL)
    for name, (lo, hi) in PARAM_SCALE_RANGES.items():
        slo, shi = 1 + level * (lo - 1), 1 + level * (hi - 1)
        values[name] *= float(rng.uniform(slo, shi))
    return values


class DQNSimulation:
    def __init__(self, config: SimulationConfig, model: DQNModel):
        self.config, self.model = config, model
        self.rng = np.random.default_rng(config.seed)
        self.physical = sample_physical(self.rng, config.randomization_level)
        self.state = np.zeros(4, dtype=np.float64)
        self.theta_prev = self.alpha_prev = self.theta_unwrapped = self.theta_unwrapped_prev = 0.0
        self.theta_dot_est = self.alpha_dot_est = 0.0
        self.step_count = 0
        self.reset()

    def reset(self) -> None:
        name = self.config.initial_condition.lower()
        if "matlab" in name:
            theta = self.rng.normal(0.0, math.radians(5.0))
            alpha = wrap_to_pi(math.pi + self.rng.normal(0.0, math.radians(8.0)))
        elif "exact" in name:
            theta, alpha = 0.0, math.pi
        elif "+8" in name:
            theta, alpha = 0.0, math.radians(8.0)
        else:
            theta, alpha = 0.0, math.radians(-8.0)
        self.state[:] = [theta, 0.0, alpha, 0.0]
        self.theta_prev = self.theta_unwrapped = self.theta_unwrapped_prev = theta
        self.alpha_prev = alpha
        self.theta_dot_est = self.alpha_dot_est = 0.0
        self.step_count = 0

    def observation(self) -> np.ndarray:
        return np.asarray([math.sin(self.theta_unwrapped), math.cos(self.theta_unwrapped), self.theta_dot_est,
                           math.sin(self.state[2]), math.cos(self.state[2]), self.alpha_dot_est], dtype=np.float32)

    def step(self) -> Tuple[np.ndarray, int, int, float, int, float, bool]:
        action_index, pwm, qmax = self.model.predict(self.observation())
        self.state = rk4_step(self.state, pwm, self.physical)
        dtheta = wrap_to_pi(self.state[0] - self.theta_prev)
        self.theta_unwrapped += dtheta
        theta_dot_raw = (self.theta_unwrapped - self.theta_unwrapped_prev) / CONTROL_DT
        alpha_dot_raw = wrap_to_pi(self.state[2] - self.alpha_prev) / CONTROL_DT
        beta = self.model.velocity_lpf
        self.theta_dot_est = float(np.clip((1-beta)*self.theta_dot_est + beta*theta_dot_raw, -THETA_DOT_LIMIT, THETA_DOT_LIMIT))
        self.alpha_dot_est = float(np.clip((1-beta)*self.alpha_dot_est + beta*alpha_dot_raw, -ALPHA_DOT_LIMIT, ALPHA_DOT_LIMIT))
        self.theta_prev = self.state[0]; self.alpha_prev = self.state[2]; self.theta_unwrapped_prev = self.theta_unwrapped
        self.step_count += 1
        abs_alpha = abs(wrap_to_pi(self.state[2]))
        mode = 3 if abs_alpha <= math.radians(15) else (2 if abs_alpha <= math.radians(35) else 1)
        blend = float(np.clip((math.radians(35)-abs_alpha)/math.radians(20), 0.0, 1.0))
        terminated = (not np.all(np.isfinite(self.state)) or abs(self.theta_unwrapped) > THETA_LIMIT
                      or abs(self.state[1]) > THETA_DOT_LIMIT or abs(self.state[3]) > ALPHA_DOT_LIMIT)
        shown = np.asarray([self.state[0], self.state[1], self.state[2], self.state[3]], dtype=float)
        return shown, action_index, pwm, qmax, mode, blend, terminated


def rip_points(theta: float, alpha: float):
    center = np.array([0.0, 0.0, ARM_Z]); radial = np.array([math.cos(theta), math.sin(theta), 0.0])
    tangent = np.array([-math.sin(theta), math.cos(theta), 0.0]); vertical = np.array([0.0, 0.0, 1.0])
    joint = center + ARM_LENGTH * radial; direction = math.sin(alpha) * tangent + math.cos(alpha) * vertical
    return center, joint, joint + PEND_LENGTH * direction, joint + PEND_LENGTH * vertical, tangent


def set_line3d(line, p0, p1):
    line.set_data([p0[0], p1[0]], [p0[1], p1[1]]); line.set_3d_properties([p0[2], p1[2]])


def set_point3d(point, p):
    point.set_data([p[0]], [p[1]]); point.set_3d_properties([p[2]])


def run_headless(model_path: str, duration: float, seed: int, randomization: float = 0.0) -> SimulationResult:
    model = load_model(model_path)
    cfg = SimulationConfig(model_selection=str(model_path), duration=duration, seed=seed, randomization_level=randomization)
    sim = DQNSimulation(cfg, model); rows = []
    for k in range(max(1, int(round(duration / CONTROL_DT)))):
        state, _, pwm, _, mode, blend, terminated = sim.step()
        rows.append([k * CONTROL_DT, *state, pwm, mode, blend])
        if terminated: break
    data = np.asarray(rows, dtype=float)
    return SimulationResult(data[:,0], data[:,1], data[:,2], data[:,3], data[:,4], data[:,5], data[:,6].astype(int), data[:,7])


def launch_gui() -> int:
    try:
        from PyQt5 import QtCore, QtWidgets
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
    except ImportError as exc:
        print(f"GUI dependency missing: {exc}", file=sys.stderr); return 2

    class Window(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__(); self.setWindowTitle("RIP DQN Digital Twin | MATLAB-aligned Model Test"); self.resize(1210, 790)
            self.saved: Optional[SimulationConfig] = None; self.model: Optional[DQNModel] = None; self.sim: Optional[DQNSimulation] = None
            self.running = False; self.rows: List[List[float]] = []; self.result: Optional[SimulationResult] = None
            self.last_png_path = self.last_csv_path = None; self.sim_time = 0.0; self.wall_prev = time.perf_counter(); self.accum = 0.0
            self.build_ui(); self.build_3d(); self.timer = QtCore.QTimer(self); self.timer.setTimerType(QtCore.Qt.PreciseTimer)
            self.timer.timeout.connect(self.tick); self.timer.start(16); self.update_buttons()

        def dspin(self, value, lo, hi, decimals, step):
            w=QtWidgets.QDoubleSpinBox(); w.setRange(lo,hi); w.setDecimals(decimals); w.setSingleStep(step); w.setValue(value); w.setKeyboardTracking(False); return w

        def build_ui(self):
            central=QtWidgets.QWidget(); self.setCentralWidget(central); root=QtWidgets.QHBoxLayout(central)
            scroll=QtWidgets.QScrollArea(); scroll.setWidgetResizable(True); scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff); scroll.setMinimumWidth(410); scroll.setMaximumWidth(475)
            panel=QtWidgets.QWidget(); left=QtWidgets.QVBoxLayout(panel); scroll.setWidget(panel); root.addWidget(scroll)
            group=QtWidgets.QGroupBox("DQN Model"); layout=QtWidgets.QGridLayout(group)
            self.model_path=QtWidgets.QLineEdit(); self.model_path.setPlaceholderText("Select .pt, model_weights.h, or a training folder")
            file_btn=QtWidgets.QPushButton("Choose Model File"); dir_btn=QtWidgets.QPushButton("Choose Run Folder")
            file_btn.clicked.connect(self.choose_file); dir_btn.clicked.connect(self.choose_dir); self.model_path.textChanged.connect(self.dirty)
            layout.addWidget(self.model_path,0,0,1,2); layout.addWidget(file_btn,1,0); layout.addWidget(dir_btn,1,1)
            self.model_info=QtWidgets.QLabel("No model loaded"); self.model_info.setWordWrap(True); layout.addWidget(self.model_info,2,0,1,2); left.addWidget(group)
            exp=QtWidgets.QGroupBox("Simulation Experiment"); form=QtWidgets.QFormLayout(exp)
            self.initial=QtWidgets.QComboBox(); self.initial.addItems(["MATLAB random downward","Exact downward","Near upright +8°","Near upright -8°"])
            self.duration=self.dspin(30,0.1,600,3,1); self.speed=self.dspin(1,0.05,50,2,0.25); self.randomization=self.dspin(0,0,1,2,0.1)
            self.seed=QtWidgets.QSpinBox(); self.seed.setRange(0,2_000_000_000); self.seed.setValue(2026)
            form.addRow("Initial condition:",self.initial); form.addRow("Duration / s:",self.duration); form.addRow("Playback speed:",self.speed)
            form.addRow("Domain randomization:",self.randomization); form.addRow("Seed:",self.seed)
            note=QtWidgets.QLabel("The policy controls the complete motion. Its input is [sin θ, cos θ, θ̇LPF, sin α, cos α, α̇LPF] at 200 Hz.")
            note.setWordWrap(True); note.setStyleSheet("QLabel {background:#f2f2f2;padding:6px;}"); form.addRow(note); left.addWidget(exp)
            run=QtWidgets.QGroupBox("Run"); v=QtWidgets.QVBoxLayout(run); self.save_btn=QtWidgets.QPushButton("SAVE / Load Model & Apply Settings")
            self.go_btn=QtWidgets.QPushButton("GO"); self.stop_btn=QtWidgets.QPushButton("STOP"); self.reset_btn=QtWidgets.QPushButton("RESET")
            self.save_btn.clicked.connect(self.save_settings); self.go_btn.clicked.connect(self.start); self.stop_btn.clicked.connect(self.stop); self.reset_btn.clicked.connect(self.reset)
            v.addWidget(self.save_btn); row=QtWidgets.QHBoxLayout(); [row.addWidget(x) for x in (self.go_btn,self.stop_btn,self.reset_btn)]; v.addLayout(row)
            self.status=QtWidgets.QLabel("Choose a model, then SAVE."); self.status.setWordWrap(True); self.state_label=QtWidgets.QLabel(); v.addWidget(self.status); v.addWidget(self.state_label); left.addWidget(run)
            out=QtWidgets.QGroupBox("Result & Logging"); ov=QtWidgets.QVBoxLayout(out); self.csv_check=QtWidgets.QCheckBox("Generate CSV log"); self.csv_check.setChecked(True)
            self.output_dir=QtWidgets.QLineEdit(os.path.expanduser("~/rip_twin_logs")); browse=QtWidgets.QPushButton("Browse"); browse.clicked.connect(self.choose_output)
            show=QtWidgets.QPushButton("Show Last Result Curves"); show.clicked.connect(self.show_result); ov.addWidget(self.csv_check)
            rr=QtWidgets.QHBoxLayout(); rr.addWidget(self.output_dir,1); rr.addWidget(browse); ov.addLayout(rr); ov.addWidget(show); left.addWidget(out); left.addStretch(1)
            self.right=QtWidgets.QWidget(); root.addWidget(self.right,1)
            for w in (self.initial,self.duration,self.speed,self.randomization,self.seed,self.csv_check,self.output_dir):
                signal = getattr(w,"valueChanged",None) or getattr(w,"currentIndexChanged",None) or getattr(w,"stateChanged",None) or getattr(w,"textChanged",None)
                if signal is not None: signal.connect(self.dirty)

        def build_3d(self):
            layout=QtWidgets.QVBoxLayout(self.right); self.figure=Figure(figsize=(9,7),tight_layout=True); self.canvas=FigureCanvas(self.figure); layout.addWidget(self.canvas)
            self.axis=self.figure.add_subplot(111,projection="3d"); self.axis.set_title("MATLAB-aligned DQN Digital Twin",pad=2)
            self.axis.set_xlabel("X / m"); self.axis.set_ylabel("Y / m"); self.axis.set_zlabel("Z / m"); lim=ARM_LENGTH+PEND_LENGTH+0.05
            self.axis.set_xlim(-lim,lim); self.axis.set_ylim(-lim,lim); self.axis.set_zlim(-0.28,0.38); self.axis.view_init(elev=24,azim=-55)
            angle=np.linspace(0,2*math.pi,80); self.axis.plot(MOTOR_RADIUS*np.cos(angle),MOTOR_RADIUS*np.sin(angle),MOTOR_HEIGHT*np.ones_like(angle),linewidth=2)
            self.axis.plot(ARM_LENGTH*np.cos(angle),ARM_LENGTH*np.sin(angle),ARM_Z*np.ones_like(angle),linestyle="--",linewidth=1)
            self.arm_line,=self.axis.plot([],[],[],linewidth=6); self.pend_line,=self.axis.plot([],[],[],linewidth=5)
            self.joint_dot,=self.axis.plot([],[],[],marker="o",markersize=8); self.tip_dot,=self.axis.plot([],[],[],marker="o",markersize=10)
            self.tangent_line,=self.axis.plot([],[],[],linestyle=":",linewidth=2); self.reference_line,=self.axis.plot([],[],[],linestyle="--",linewidth=2.5)
            self.text=self.axis.text2D(0.03,0.88,"",transform=self.axis.transAxes,fontsize=11)

        def dirty(self,*_):
            self.saved=None; self.save_btn.setText("SAVE / Load Model & Apply Settings *"); self.update_buttons()

        def choose_file(self):
            path,_=QtWidgets.QFileDialog.getOpenFileName(self,"Select DQN model",str(Path(self.model_path.text() or Path.home()).expanduser()),"DQN model (*.pt *.h)")
            if path:self.model_path.setText(path)

        def choose_dir(self):
            path=QtWidgets.QFileDialog.getExistingDirectory(self,"Select DQN run folder",str(Path(self.model_path.text() or Path.home()).expanduser()))
            if path:self.model_path.setText(path)

        def choose_output(self):
            path=QtWidgets.QFileDialog.getExistingDirectory(self,"Output directory",self.output_dir.text())
            if path:self.output_dir.setText(path)

        def capture(self):
            return SimulationConfig(self.model_path.text().strip(),float(self.duration.value()),float(self.speed.value()),int(self.seed.value()),float(self.randomization.value()),self.initial.currentText(),self.csv_check.isChecked(),self.output_dir.text().strip())

        def save_settings(self):
            try:
                cfg=self.capture()
                if not cfg.model_selection: raise ValueError("Select a model file or run folder.")
                model=load_model(cfg.model_selection)
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self,"Model/settings error",str(exc)); return
            self.saved=cfg; self.model=model; self.model_path.setText(str(model.source)); self.save_btn.setText("SAVE / Load Model & Apply Settings")
            self.model_info.setText(f"6→64→64→10 ReLU | actions {model.actions.tolist()} | LPF={model.velocity_lpf:g} | ID {model.digest}")
            self.status.setText(f"Model loaded: {model.source}"); self.reset(force=True); self.update_buttons()

        def reset(self,*_,force=False):
            self.running=False
            if self.saved and self.model:
                self.sim=DQNSimulation(self.saved,self.model); self.sim_time=0; self.rows=[]; self.result=None; self.wall_prev=time.perf_counter(); self.accum=0
                self.draw_state(self.sim.state,0,0,0); self.status.setText("Ready. Click GO for a complete DQN swing-up and balance test.")
            elif force: self.status.setText("Choose a model, then SAVE.")
            self.update_buttons()

        def start(self):
            if not self.saved or not self.model: return
            self.reset(force=True); self.running=True; self.wall_prev=time.perf_counter(); self.status.setText("DQN simulation running at 200 Hz."); self.update_buttons()

        def stop(self):
            if self.running: self.running=False; self.finalize(False); self.status.setText("Simulation stopped."); self.update_buttons()

        def tick(self):
            if not self.running or not self.sim or not self.saved:return
            now=time.perf_counter(); wall=min(now-self.wall_prev,0.1); self.wall_prev=now; self.accum += wall*self.saved.playback_speed
            steps=0
            while self.accum>=CONTROL_DT and steps<500 and self.running:
                state,_,pwm,qmax,mode,blend,terminated=self.sim.step(); self.sim_time += CONTROL_DT
                self.rows.append([self.sim_time,*state,pwm,mode,blend]); self.accum-=CONTROL_DT; steps+=1
                if terminated or self.sim_time+1e-12>=self.saved.duration:
                    self.running=False; self.finalize(True); self.status.setText("Configured DQN experiment duration completed." if not terminated else "Simulation terminated by a safety limit."); self.update_buttons(); break
            if self.rows:
                row=self.rows[-1]; self.draw_state(np.asarray(row[1:5]),row[5],int(row[6]),row[7],qmax if 'qmax' in locals() else 0)

        def draw_state(self,state,pwm,mode,blend,qmax=0):
            center,joint,tip,ref,tangent=rip_points(float(state[0]),float(state[2])); set_line3d(self.arm_line,center,joint); set_line3d(self.pend_line,joint,tip)
            set_point3d(self.joint_dot,joint); set_point3d(self.tip_dot,tip); set_line3d(self.reference_line,joint,ref); set_line3d(self.tangent_line,joint,joint+0.11*tangent)
            self.text.set_text(f"t = {self.sim_time:7.3f} s\nθ = {state[0]: .4f} rad\nα = {state[2]: .4f} rad\nPWM = {pwm: .0f}\nmode = {MODE_NAMES.get(mode,mode)}\nQmax = {qmax: .3f}")
            self.state_label.setText(f"θ={state[0]:+.4f}, θ̇={state[1]:+.4f}, α={state[2]:+.4f}, α̇={state[3]:+.4f}, PWM={pwm:+.0f}, capture={blend:.2f}")
            self.canvas.draw_idle()

        def rows_result(self):
            if not self.rows:return None
            d=np.asarray(self.rows,float); return SimulationResult(d[:,0],d[:,1],d[:,2],d[:,3],d[:,4],d[:,5],d[:,6].astype(int),d[:,7])

        def finalize(self,show):
            self.result=self.rows_result()
            if self.result is None:return
            png,csv_path=default_output_paths(self.saved.output_dir,self.saved.duration); fig=build_result_figure(self.result); fig.savefig(png,dpi=300,bbox_inches="tight"); self.last_png_path=png
            if self.saved.save_csv: save_result_csv(self.result,csv_path); self.last_csv_path=csv_path
            if show:self.show_result()

        def show_result(self):
            if self.result is None: QtWidgets.QMessageBox.information(self,"No result","No completed result is available."); return
            dialog=QtWidgets.QDialog(self); dialog.setWindowTitle(f"RIP DQN Digital Twin Result | {self.result.time[-1]:.3f} s Response"); dialog.resize(1080,800)
            layout=QtWidgets.QVBoxLayout(dialog); fig=build_result_figure(self.result); canvas=FigureCanvas(fig); layout.addWidget(canvas); close=QtWidgets.QPushButton("Close"); close.clicked.connect(dialog.close); layout.addWidget(close); dialog.exec_()

        def update_buttons(self):
            ready=self.saved is not None and self.model is not None
            self.go_btn.setEnabled(ready and not self.running); self.stop_btn.setEnabled(self.running); self.reset_btn.setEnabled(ready); self.save_btn.setEnabled(not self.running)

    app=QtWidgets.QApplication(sys.argv); win=Window(); win.show(); return app.exec_()


def main() -> int:
    parser=argparse.ArgumentParser(description="MATLAB-aligned RIP DQN model test")
    parser.add_argument("--headless",action="store_true"); parser.add_argument("--model",default="")
    parser.add_argument("--duration",type=float,default=10.0); parser.add_argument("--seed",type=int,default=2026); parser.add_argument("--randomization",type=float,default=0.0)
    args=parser.parse_args()
    if not args.headless:return launch_gui()
    if not args.model: parser.error("--model is required with --headless")
    result=run_headless(args.model,args.duration,args.seed,args.randomization); metrics=result_metrics(result)
    print(f"steps={len(result.time)} final_alpha={result.alpha[-1]:.6f} mean_abs_alpha={np.mean(np.abs(result.alpha)):.6f} stable_start={metrics['stable_start_time']}")
    return 0

if __name__=="__main__": raise SystemExit(main())
