#!/usr/bin/env python3
"""Hybrid energy-swing-up + distilled PPO-balance digital-twin panel.

Far from upright, the controller uses the MPC-style phase/energy-pumping law.
Near upright, it switches through hysteresis and first-order blending to the
student PPO actor exported in ``ppo_model_weights.h``.

Supported model selections:
  * the generated C header ``ppo_model_weights.h``;
  * a cached ``.npz`` file produced from that header;
  * a run/deploy directory containing one supported header.

Expected actor: compact7 raw input, 7 -> 64 -> 64 -> 1, Tanh.  The seven inputs
are [sin(theta), cos(theta), theta_dot, sin(alpha), cos(alpha), alpha_dot,
previous_applied_pwm / model_pwm_scale].  Observation normalization stored in
the header is applied before inference.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import math
import os
import pickle
import re
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

CONTROL_DT = 0.005
THETA_LIMIT = 12.0 * math.pi
THETA_DOT_LIMIT = 45.0
ALPHA_DOT_LIMIT = 40.0
ARM_LENGTH = 0.18
PEND_LENGTH = 0.24
MOTOR_RADIUS = 0.035
MOTOR_HEIGHT = 0.08
ARM_Z = MOTOR_HEIGHT
MODE_NAMES = {0: "DISABLED", 1: "ENERGY_SWING", 2: "BLEND", 3: "PPO_BALANCE"}

PHYSICAL_NOMINAL = {
    "g": 9.8, "c_theta": 0.025, "c_alpha": 0.001,
    "k_t": 0.2310, "k_b": 0.1875, "k_u": 0.04706, "R": 4.2857,
    "m1": 0.20625, "m2": 0.15845, "l1cg": 0.080305,
    "l1": 0.151894, "l2cg": 0.066733,
    "I1z": 0.00049228, "I2x": 0.00036892,
    "I2y": 2.3641e-05, "I2z": 0.00036139,
}
PARAM_SCALE_RANGES = {
    "g": (0.98, 1.02), "m1": (0.75, 1.25), "m2": (0.65, 1.35),
    "l1": (0.90, 1.10), "l1cg": (0.85, 1.15), "l2cg": (0.75, 1.25),
    "I1z": (0.55, 1.70), "I2x": (0.50, 1.80), "I2y": (0.50, 1.80),
    "I2z": (0.50, 1.80), "c_theta": (0.25, 3.50), "c_alpha": (0.25, 4.50),
    "k_t": (0.75, 1.25), "k_b": (0.75, 1.25), "k_u": (0.70, 1.30),
    "R": (0.80, 1.25),
}

OBS_A_LC = np.asarray([
    [0.82469568, 0.00438721, -0.31915638, -0.00085553],
    [-1.75095359, 0.94056709, -0.04181246, 0.00193564],
    [-0.05215683, 0.00005656, 0.76549687, 0.00439733],
    [-1.38405407, 0.07685667, -15.83946960, 0.95027468],
], dtype=np.float64)
OBS_B_U = np.asarray([0.00001119, 0.00396389, -0.00001398, -0.00583846], dtype=np.float64)
OBS_L = np.asarray([
    [0.17530432, 0.31833008],
    [1.75095359, -0.19974721],
    [0.05215683, 0.23646539],
    [1.38405407, 16.66931771],
], dtype=np.float64)
OBS_RESET_ERR_LIMIT = math.radians(35.0)


def wrap_to_pi(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _float_tokens(text: str) -> np.ndarray:
    values = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", text.replace("f", ""))
    return np.asarray([float(v) for v in values], dtype=np.float32)


def _header_array(text: str, name: str) -> np.ndarray:
    pattern = rf"static\s+const\s+float\s+{re.escape(name)}\s*[^=]*=\s*\{{(.*?)\}}\s*;"
    match = re.search(pattern, text, flags=re.S)
    if not match:
        raise ValueError(f"Array {name} was not found in the PPO header")
    return _float_tokens(match.group(1))


def _header_macro_float(text: str, name: str, default: float) -> float:
    match = re.search(rf"#define\s+{re.escape(name)}\s+([-+0-9.eE]+)f?", text)
    return float(match.group(1)) if match else float(default)


@dataclass
class PPOModel:
    w0: np.ndarray
    b0: np.ndarray
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    obs_mean: np.ndarray
    obs_std: np.ndarray
    model_pwm_scale: float
    source: Path
    clip_obs: float = 10.0

    def __post_init__(self) -> None:
        self.w0 = np.ascontiguousarray(self.w0, dtype=np.float32).reshape(64, 7)
        self.b0 = np.ascontiguousarray(self.b0, dtype=np.float32).reshape(64)
        self.w1 = np.ascontiguousarray(self.w1, dtype=np.float32).reshape(64, 64)
        self.b1 = np.ascontiguousarray(self.b1, dtype=np.float32).reshape(64)
        self.w2 = np.ascontiguousarray(self.w2, dtype=np.float32).reshape(1, 64)
        self.b2 = np.ascontiguousarray(self.b2, dtype=np.float32).reshape(1)
        self.obs_mean = np.ascontiguousarray(self.obs_mean, dtype=np.float32).reshape(7)
        self.obs_std = np.ascontiguousarray(self.obs_std, dtype=np.float32).reshape(7)
        arrays = (self.w0, self.b0, self.w1, self.b1, self.w2, self.b2, self.obs_mean, self.obs_std)
        if not all(np.all(np.isfinite(a)) for a in arrays):
            raise ValueError("PPO model contains non-finite values")
        if np.any(self.obs_std <= 0.0):
            raise ValueError("PPO_OBS_STD must be strictly positive")
        if not (0.0 < self.clip_obs <= 1.0e6 and 1.0 <= self.model_pwm_scale <= 255.0):
            raise ValueError("Invalid observation clip or PWM scale")

    @property
    def architecture(self) -> str:
        return "compact7 7→64→64→1 Tanh"

    @property
    def digest(self) -> str:
        h = hashlib.sha256()
        for array in (self.w0, self.b0, self.w1, self.b1, self.w2, self.b2,
                      self.obs_mean, self.obs_std):
            h.update(np.ascontiguousarray(array).tobytes())
        h.update(np.asarray([self.clip_obs, self.model_pwm_scale], dtype="<f4").tobytes())
        return h.hexdigest()[:16]

    def normalize(self, observation: np.ndarray) -> np.ndarray:
        x = np.asarray(observation, dtype=np.float32).reshape(7)
        z = (x - self.obs_mean) / self.obs_std
        return np.clip(z, -self.clip_obs, self.clip_obs).astype(np.float32)

    def policy_raw(self, observation: np.ndarray) -> float:
        x = self.normalize(observation)
        h0 = np.tanh(self.w0 @ x + self.b0)
        h1 = np.tanh(self.w1 @ h0 + self.b1)
        return float((self.w2 @ h1 + self.b2)[0])

    def predict(self, observation: np.ndarray) -> Tuple[float, float, float]:
        raw = self.policy_raw(observation)
        action = float(np.tanh(raw))
        return action, action * float(self.model_pwm_scale), raw

    def float_blob(self) -> bytes:
        inv_std = (1.0 / self.obs_std).astype(np.float32)
        parts = [
            np.ascontiguousarray(self.w0, dtype="<f4").tobytes(),
            np.ascontiguousarray(self.b0, dtype="<f4").tobytes(),
            np.ascontiguousarray(self.w1, dtype="<f4").tobytes(),
            np.ascontiguousarray(self.b1, dtype="<f4").tobytes(),
            np.ascontiguousarray(self.w2, dtype="<f4").tobytes(),
            np.ascontiguousarray(self.b2, dtype="<f4").tobytes(),
            np.ascontiguousarray(self.obs_mean, dtype="<f4").tobytes(),
            np.ascontiguousarray(inv_std, dtype="<f4").tobytes(),
            np.asarray([self.clip_obs], dtype="<f4").tobytes(),
            np.asarray([self.model_pwm_scale], dtype="<f4").tobytes(),
        ]
        return b"".join(parts)

    def save_npz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, w0=self.w0, b0=self.b0, w1=self.w1, b1=self.b1,
                            w2=self.w2, b2=self.b2, obs_mean=self.obs_mean,
                            obs_std=self.obs_std, clip_obs=np.float32(self.clip_obs),
                            model_pwm_scale=np.float32(self.model_pwm_scale))


def load_header_model(path: Path) -> PPOModel:
    text = path.read_text(encoding="utf-8", errors="ignore")
    dims = {
        "obs": int(_header_macro_float(text, "PPO_OBS_DIM", 0)),
        "l0": int(_header_macro_float(text, "PPO_L0_OUT", 0)),
        "l1": int(_header_macro_float(text, "PPO_L1_OUT", 0)),
        "l2": int(_header_macro_float(text, "PPO_L2_OUT", 0)),
    }
    if dims != {"obs": 7, "l0": 64, "l1": 64, "l2": 1}:
        raise ValueError(f"Expected PPO header 7->64->64->1, got {dims}")
    model = PPOModel(
        _header_array(text, "PPO_W0").reshape(64, 7),
        _header_array(text, "PPO_B0"),
        _header_array(text, "PPO_W1").reshape(64, 64),
        _header_array(text, "PPO_B1"),
        _header_array(text, "PPO_W2").reshape(1, 64),
        _header_array(text, "PPO_B2"),
        _header_array(text, "PPO_OBS_MEAN"),
        _header_array(text, "PPO_OBS_STD"),
        _header_macro_float(text, "PPO_CONTINUOUS_PWM_LIMIT", 255.0),
        path,
        10.0,
    )
    cache = path.with_suffix(".ppo64_deploy.npz")
    try:
        model.save_npz(cache)
    except Exception:
        pass
    return model


def _load_npz(path: Path) -> PPOModel:
    d = np.load(path, allow_pickle=False)
    return PPOModel(d["w0"], d["b0"], d["w1"], d["b1"], d["w2"], d["b2"],
                    d["obs_mean"], d["obs_std"], float(d["model_pwm_scale"]), path,
                    float(d.get("clip_obs", 10.0)))


def resolve_model_path(selection: str | Path) -> Path:
    path = Path(selection).expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() not in {".h", ".npz"}:
            raise ValueError("Select ppo_model_weights.h, a compatible .npz, or a run/deploy folder")
        return path
    if not path.is_dir():
        raise FileNotFoundError(path)
    priorities = [
        "ppo_model_weights.h", "deploy/ppo_model_weights.h", "deploy/model_weights.h",
        "model_weights.h", "selected_best_model.ppo64_deploy.npz",
    ]
    for rel in priorities:
        candidate = path / rel
        if candidate.is_file():
            return candidate.resolve()
    headers = sorted(path.rglob("ppo_model_weights*.h")) or sorted(path.rglob("model_weights*.h"))
    npzs = sorted(path.rglob("*.ppo64_deploy.npz"))
    candidates = headers + npzs
    if len(candidates) == 1:
        return candidates[0].resolve()
    if not candidates:
        raise FileNotFoundError(f"No compatible 7->64->64->1 PPO header found below {path}")
    raise RuntimeError("Several compatible model files were found. Select the intended header directly.\n" +
                       "\n".join(map(str, candidates[:20])))


def load_model(selection: str | Path) -> PPOModel:
    path = resolve_model_path(selection)
    return _load_npz(path) if path.suffix.lower() == ".npz" else load_header_model(path)


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


def sample_physical(rng: np.random.Generator, level: float) -> Dict[str, float]:
    level = float(np.clip(level, 0.0, 1.0)); values = dict(PHYSICAL_NOMINAL)
    for name, (lo, hi) in PARAM_SCALE_RANGES.items():
        slo, shi = 1.0 + level * (lo - 1.0), 1.0 + level * (hi - 1.0)
        values[name] *= float(rng.uniform(slo, shi))
    return values


@dataclass
class SimulationConfig:
    model_selection: str
    duration: float = 30.0
    playback_speed: float = 1.0
    seed: int = 2026
    randomization_level: float = 0.0
    initial_condition: str = "Random hanging-down"
    safety_pwm_limit: float = 150.0
    swing_pwm: float = 120.0
    kick_time: float = 0.10
    enter_deg: float = 15.0
    exit_deg: float = 25.0
    blend_alpha: float = 0.18
    velocity_lpf: float = 0.25
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
    action_norm: np.ndarray
    policy_raw: np.ndarray


def stable_phase_start_index(result: SimulationResult, threshold_deg: float = 15.0) -> Optional[int]:
    if result.alpha.size == 0:
        return None
    inside = np.abs(result.alpha) <= math.radians(threshold_deg)
    outside = np.flatnonzero(~inside)
    start = int(outside[-1] + 1) if outside.size else 0
    return start if start < inside.size else None


def result_metrics(result: SimulationResult) -> Dict[str, float]:
    idx = stable_phase_start_index(result)
    out = {
        "stable_start_time": math.nan, "stable_duration": 0.0,
        "alpha_abs_mean": math.nan, "alpha_abs_std": math.nan,
        "pwm_abs_mean": math.nan, "pwm_abs_std": math.nan,
        "max_abs_theta": float(np.max(np.abs(result.theta))) if result.theta.size else math.nan,
        "capture_count": float(np.count_nonzero(np.diff((np.abs(result.alpha) <= math.radians(15)).astype(np.int8)) == 1)),
    }
    if idx is not None:
        out.update(stable_start_time=float(result.time[idx]), stable_duration=float(result.time[-1] - result.time[idx]),
                   alpha_abs_mean=float(np.mean(np.abs(result.alpha[idx:]))), alpha_abs_std=float(np.std(np.abs(result.alpha[idx:]))),
                   pwm_abs_mean=float(np.mean(np.abs(result.pwm[idx:]))), pwm_abs_std=float(np.std(np.abs(result.pwm[idx:]))))
    return out


def build_result_figure(result: SimulationResult, title_suffix: str = "Hybrid PPO Digital Twin"):
    from matplotlib.figure import Figure
    metrics = result_metrics(result)
    fig = Figure(figsize=(10.5, 7.6), tight_layout=True)
    axs = fig.subplots(2, 2)
    ax_alpha, ax_theta, ax_pwm, ax_hist = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]
    t, alpha, theta, pwm = result.time, result.alpha, result.theta, result.pwm
    fig.suptitle(f"Rotary Inverted Pendulum {title_suffix} Response", fontsize=15, fontweight="bold")
    ax_alpha.plot(t, alpha, linewidth=1.8, label=r"$\alpha$")
    ax_alpha.axhline(0.0, linewidth=1.0, linestyle="--")
    for sign in (-1, 1):
        ax_alpha.axhline(sign * math.radians(15), linewidth=0.8, linestyle=":")
    ax_alpha.set(title=r"Pendulum Angle $\alpha(t)$", xlabel="Time / s", ylabel=r"$\alpha$ / rad")
    ax_alpha.grid(True, linestyle="--", linewidth=0.6, alpha=0.55)
    ax_alpha.legend(loc="lower right")
    index = stable_phase_start_index(result)
    if index is None:
        text = "Stable phase: not reached"
    else:
        text = (
            rf"$\mathrm{{mean}}(|\alpha|)$ = {metrics['alpha_abs_mean']:.6f} rad" "\n"
            rf"$\mathrm{{std}}(|\alpha|)$ = {metrics['alpha_abs_std']:.6f} rad" "\n"
            f"stable from t = {metrics['stable_start_time']:.3f} s"
        )
        ax_alpha.axvspan(metrics["stable_start_time"], t[-1], alpha=0.10)
    ax_alpha.text(
        0.98, 0.96, text, transform=ax_alpha.transAxes, ha="right", va="top", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="0.35"),
    )
    ax_theta.plot(t, theta, linewidth=1.8, label=r"$\theta$")
    ax_theta.axhline(0, linewidth=1, linestyle="--")
    ax_theta.set(title=r"Rotary Arm Angle $\theta(t)$", xlabel="Time / s", ylabel=r"$\theta$ / rad")
    ax_theta.grid(True, linestyle="--", linewidth=0.6, alpha=0.55)
    ax_theta.legend(loc="upper right")
    ax_pwm.plot(t, pwm, linewidth=1.5, label="PWM")
    ax_pwm.axhline(0, linewidth=1, linestyle="--")
    ax_pwm.set(title="Control Input PWM(t)", xlabel="Time / s", ylabel="PWM")
    lim = max(160.0, float(np.max(np.abs(pwm))) * 1.1) if pwm.size else 160.0
    ax_pwm.set_ylim(-lim, lim)
    ax_pwm.grid(True, linestyle="--", linewidth=0.6, alpha=0.55)
    ax_pwm.legend(loc="lower right")
    if index is not None:
        ax_pwm.axvspan(metrics["stable_start_time"], t[-1], alpha=0.10)
        ptext = (
            rf"$\mathrm{{mean}}(|PWM|)$ = {metrics['pwm_abs_mean']:.3f}" "\n"
            rf"$\mathrm{{std}}(|PWM|)$ = {metrics['pwm_abs_std']:.3f}"
        )
    else:
        ptext = "Stable phase: not reached"
    ax_pwm.text(
        0.98, 0.96, ptext, transform=ax_pwm.transAxes, ha="right", va="top", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="0.35"),
    )
    if index is None:
        ax_hist.text(0.5, 0.5, "No final stable phase", transform=ax_hist.transAxes, ha="center", va="center")
    else:
        ax_hist.hist(pwm[index:], bins=np.arange(-255, 271, 15), edgecolor="black", linewidth=0.45)
    ax_hist.axvline(0, linewidth=1, linestyle="--")
    ax_hist.set_xlim(-255, 255)
    ax_hist.set(title="Stable-stage PWM Distribution", xlabel="PWM", ylabel="Count")
    ax_hist.grid(True, linestyle="--", linewidth=0.6, alpha=0.55)
    xmax = max(float(t[-1]), 0.1)
    for ax in (ax_alpha, ax_theta, ax_pwm):
        ax.set_xlim(0, xmax)
    for ax in (ax_alpha, ax_theta, ax_pwm, ax_hist):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="in")
    return fig


def save_result_csv(result: SimulationResult, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        import csv
        writer = csv.writer(handle)
        writer.writerow(["time_s", "theta_rad", "theta_dot_rad_s", "alpha_rad", "alpha_dot_rad_s", "pwm", "mode", "blend", "action_norm", "policy_raw"])
        writer.writerows(zip(result.time, result.theta, result.theta_dot, result.alpha, result.alpha_dot,
                             result.pwm, result.mode, result.blend, result.action_norm, result.policy_raw))


def default_output_paths(output_dir: str, duration: float) -> Tuple[str, str]:
    directory = Path(output_dir).expanduser().resolve(); directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S"); tag = f"{duration:.3f}".rstrip("0").rstrip(".").replace(".", "p")
    return str(directory / f"rip_ppo_hybrid_sim_{tag}s_{stamp}.png"), str(directory / f"rip_ppo_hybrid_sim_{tag}s_{stamp}.csv")


class HybridPPOSimulation:
    def __init__(self, config: SimulationConfig, model: PPOModel):
        self.config, self.model = config, model
        self.rng = np.random.default_rng(config.seed)
        self.physical = sample_physical(self.rng, config.randomization_level)
        self.state = np.zeros(4, dtype=np.float64)
        self.theta_unwrapped = 0.0
        self.prev_theta = 0.0
        self.prev_alpha = 0.0
        self.theta_dot_lpf = 0.0
        self.alpha_dot_lpf = 0.0
        self.xhat = np.zeros(4, dtype=np.float64)
        self.observer_ready = False
        self.ppo_active = False
        self.blend = 0.0
        self.previous_controller_pwm = 0.0
        self.step_count = 0
        self.reset()

    def reset(self) -> None:
        name = self.config.initial_condition.lower()
        if "random" in name:
            theta = self.rng.normal(0.0, math.radians(5.0))
            alpha = wrap_to_pi(math.pi + self.rng.normal(0.0, math.radians(8.0)))
        elif "exact" in name:
            theta, alpha = 0.0, math.pi
        elif "+8" in name:
            theta, alpha = 0.0, math.radians(8.0)
        else:
            theta, alpha = 0.0, math.radians(-8.0)
        self.state[:] = [theta, 0.0, alpha, 0.0]
        self.theta_unwrapped = theta
        self.prev_theta = theta
        self.prev_alpha = alpha
        self.theta_dot_lpf = self.alpha_dot_lpf = 0.0
        self.xhat[:] = self.state
        self.observer_ready = False
        self.ppo_active = False
        self.blend = 0.0
        self.previous_controller_pwm = 0.0
        self.step_count = 0

    def _observer_initialize(self, theta: float, alpha: float) -> None:
        self.xhat[:] = [theta, self.theta_dot_lpf, wrap_to_pi(alpha), self.alpha_dot_lpf]
        self.observer_ready = True

    def _observer_update(self, previous_pwm: float, theta: float, alpha: float) -> None:
        if not self.observer_ready:
            self._observer_initialize(theta, alpha)
            return
        nxt = OBS_A_LC @ self.xhat + OBS_B_U * previous_pwm + OBS_L @ np.asarray([theta, alpha])
        nxt[2] = wrap_to_pi(nxt[2])
        if not np.all(np.isfinite(nxt)) or abs(wrap_to_pi(nxt[2] - alpha)) > OBS_RESET_ERR_LIMIT:
            self._observer_initialize(theta, alpha)
            return
        nxt[1] = float(np.clip(nxt[1], -500, 500))
        nxt[3] = float(np.clip(nxt[3], -500, 500))
        self.xhat[:] = nxt

    def _compact_obs(self, control_state: np.ndarray) -> np.ndarray:
        theta, theta_dot, alpha, alpha_dot = map(float, control_state)
        previous_action = float(np.clip(self.previous_controller_pwm / self.model.model_pwm_scale, -1.0, 1.0))
        return np.asarray([
            math.sin(theta), math.cos(theta), np.clip(theta_dot, -THETA_DOT_LIMIT, THETA_DOT_LIMIT),
            math.sin(alpha), math.cos(alpha), np.clip(alpha_dot, -ALPHA_DOT_LIMIT, ALPHA_DOT_LIMIT),
            previous_action,
        ], dtype=np.float32)

    def _measure(self) -> np.ndarray:
        theta, _, alpha, _ = map(float, self.state)
        dtheta = wrap_to_pi(theta - self.prev_theta)
        dalpha = wrap_to_pi(alpha - self.prev_alpha)
        self.theta_unwrapped += dtheta
        raw_th = dtheta / CONTROL_DT
        raw_al = dalpha / CONTROL_DT
        beta = float(np.clip(self.config.velocity_lpf, 0.0, 1.0))
        self.theta_dot_lpf += beta * (raw_th - self.theta_dot_lpf)
        self.alpha_dot_lpf += beta * (raw_al - self.alpha_dot_lpf)
        self.prev_theta = theta
        self.prev_alpha = alpha
        return np.asarray([self.theta_unwrapped, self.theta_dot_lpf, alpha, self.alpha_dot_lpf], dtype=np.float64)

    def step(self) -> Tuple[np.ndarray, float, int, float, float, float, bool]:
        measured = self._measure()
        theta, _, alpha, _ = measured
        enter = math.radians(self.config.enter_deg)
        exit_angle = math.radians(self.config.exit_deg)
        abs_alpha = abs(wrap_to_pi(alpha))
        if not self.ppo_active and abs_alpha <= enter:
            self.ppo_active = True
            self._observer_initialize(theta, alpha)
        elif self.ppo_active and abs_alpha >= exit_angle:
            self.ppo_active = False
            self.observer_ready = False
        if self.ppo_active:
            self._observer_update(self.previous_controller_pwm, theta, alpha)
            control_state = self.xhat.copy()
        else:
            control_state = measured.copy()
        control_state[2] = wrap_to_pi(control_state[2])
        action_norm, policy_pwm, raw = self.model.predict(self._compact_obs(control_state))
        run_time = self.step_count * CONTROL_DT
        if run_time < self.config.kick_time:
            swing = self.config.swing_pwm
        else:
            phase = self.alpha_dot_lpf * math.cos(alpha)
            swing = -self.config.swing_pwm * (1.0 if phase >= 0.0 else -1.0)
        target = 1.0 if self.ppo_active else 0.0
        self.blend = float(np.clip(self.blend + self.config.blend_alpha * (target - self.blend), 0.0, 1.0))
        command = (1.0 - self.blend) * swing + self.blend * policy_pwm
        command = float(np.clip(command, -self.config.safety_pwm_limit, self.config.safety_pwm_limit))
        self.previous_controller_pwm = command
        self.state = rk4_step(self.state, command, self.physical)
        self.step_count += 1
        mode = 1 if self.blend <= 0.01 else (3 if self.blend >= 0.99 else 2)
        terminated = (not np.all(np.isfinite(self.state)) or abs(self.theta_unwrapped) > THETA_LIMIT
                      or abs(self.state[1]) > THETA_DOT_LIMIT or abs(self.state[3]) > ALPHA_DOT_LIMIT)
        shown = np.asarray([self.state[0], control_state[1], self.state[2], control_state[3]], dtype=float)
        return shown, command, mode, self.blend, action_norm, raw, terminated


def rip_points(theta: float, alpha: float):
    center = np.array([0.0, 0.0, ARM_Z]); radial = np.array([math.cos(theta), math.sin(theta), 0.0])
    tangent = np.array([-math.sin(theta), math.cos(theta), 0.0]); vertical = np.array([0.0, 0.0, 1.0])
    joint = center + ARM_LENGTH * radial; direction = math.sin(alpha) * tangent + math.cos(alpha) * vertical
    return center, joint, joint + PEND_LENGTH * direction, joint + PEND_LENGTH * vertical, tangent


def set_line3d(line, p0, p1):
    line.set_data([p0[0], p1[0]], [p0[1], p1[1]]); line.set_3d_properties([p0[2], p1[2]])


def set_point3d(point, p):
    point.set_data([p[0]], [p[1]]); point.set_3d_properties([p[2]])


def run_headless(model_path: str, duration: float, seed: int, randomization: float = 0.0,
                 safety_pwm_limit: float = 150.0, swing_pwm: float = 120.0) -> SimulationResult:
    model = load_model(model_path)
    cfg = SimulationConfig(model_selection=str(model_path), duration=duration, seed=seed,
                           randomization_level=randomization, safety_pwm_limit=safety_pwm_limit, swing_pwm=swing_pwm)
    sim = HybridPPOSimulation(cfg, model); rows = []
    for k in range(max(1, int(round(duration / CONTROL_DT)))):
        state, pwm, mode, blend, action, raw, terminated = sim.step()
        rows.append([k * CONTROL_DT, *state, pwm, mode, blend, action, raw])
        if terminated: break
    d = np.asarray(rows, dtype=float)
    return SimulationResult(d[:, 0], d[:, 1], d[:, 2], d[:, 3], d[:, 4], d[:, 5], d[:, 6].astype(int), d[:, 7], d[:, 8], d[:, 9])


def launch_gui() -> int:
    try:
        from PyQt5 import QtCore, QtWidgets
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
    except ImportError as exc:
        print(f"GUI dependency missing: {exc}", file=sys.stderr); return 2

    class Window(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__(); self.setWindowTitle("RIP PPO Hybrid Digital Twin | Energy Swing-up + PPO Balance"); self.resize(1240, 820)
            self.saved: Optional[SimulationConfig] = None; self.model: Optional[PPOModel] = None; self.sim: Optional[HybridPPOSimulation] = None
            self.running = False; self.rows: List[List[float]] = []; self.result: Optional[SimulationResult] = None
            self.sim_time = 0.0; self.wall_prev = time.perf_counter(); self.accum = 0.0
            self.build_ui(); self.build_3d(); self.timer = QtCore.QTimer(self); self.timer.setTimerType(QtCore.Qt.PreciseTimer)
            self.timer.timeout.connect(self.tick); self.timer.start(16); self.update_buttons()

        def dspin(self, value, lo, hi, decimals, step):
            w = QtWidgets.QDoubleSpinBox(); w.setRange(lo, hi); w.setDecimals(decimals); w.setSingleStep(step); w.setValue(value); w.setKeyboardTracking(False); return w

        def build_ui(self):
            central = QtWidgets.QWidget(); self.setCentralWidget(central); root = QtWidgets.QHBoxLayout(central)
            scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True); scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff); scroll.setMinimumWidth(430); scroll.setMaximumWidth(510)
            panel = QtWidgets.QWidget(); left = QtWidgets.QVBoxLayout(panel); scroll.setWidget(panel); root.addWidget(scroll)
            group = QtWidgets.QGroupBox("PPO Balance Model"); layout = QtWidgets.QGridLayout(group)
            self.model_path = QtWidgets.QLineEdit(); self.model_path.setPlaceholderText("Select ppo_model_weights.h, cached .npz, or a run/deploy folder")
            file_btn = QtWidgets.QPushButton("Choose Model File"); dir_btn = QtWidgets.QPushButton("Choose Run Folder")
            file_btn.clicked.connect(self.choose_file); dir_btn.clicked.connect(self.choose_dir); self.model_path.textChanged.connect(self.dirty)
            layout.addWidget(self.model_path, 0, 0, 1, 2); layout.addWidget(file_btn, 1, 0); layout.addWidget(dir_btn, 1, 1)
            self.model_info = QtWidgets.QLabel("No model loaded"); self.model_info.setWordWrap(True); layout.addWidget(self.model_info, 2, 0, 1, 2); left.addWidget(group)
            exp = QtWidgets.QGroupBox("Hybrid Simulation Experiment"); form = QtWidgets.QFormLayout(exp)
            self.initial = QtWidgets.QComboBox(); self.initial.addItems(["Random hanging-down", "Exact hanging-down", "Near upright +8°", "Near upright -8°"])
            self.duration = self.dspin(30, 0.1, 600, 3, 1); self.speed = self.dspin(1, 0.05, 50, 2, 0.25); self.randomization = self.dspin(0, 0, 1, 2, 0.1)
            self.seed = QtWidgets.QSpinBox(); self.seed.setRange(0, 2_000_000_000); self.seed.setValue(2026)
            self.pwm_limit = self.dspin(150, 1, 255, 2, 1); self.swing_pwm = self.dspin(120, 0, 255, 2, 1); self.kick_time = self.dspin(0.10, 0, 2, 3, 0.01)
            self.enter_deg = self.dspin(15, 1, 60, 2, 1); self.exit_deg = self.dspin(25, 2, 90, 2, 1); self.blend_alpha = self.dspin(0.18, 0.001, 1, 4, 0.01); self.velocity_lpf = self.dspin(0.25, 0.001, 1, 4, 0.01)
            form.addRow("Initial condition:", self.initial); form.addRow("Duration / s:", self.duration); form.addRow("Playback speed:", self.speed); form.addRow("Domain randomization:", self.randomization); form.addRow("Seed:", self.seed)
            form.addRow("Safety PWM max:", self.pwm_limit); form.addRow("Energy swing PWM:", self.swing_pwm); form.addRow("Initial kick / s:", self.kick_time)
            form.addRow("PPO enter / deg:", self.enter_deg); form.addRow("PPO exit / deg:", self.exit_deg); form.addRow("Blend λ:", self.blend_alpha); form.addRow("Large-angle velocity LPF β:", self.velocity_lpf)
            note = QtWidgets.QLabel("Far from upright: MPC-firmware phase/energy pumping. Near upright: fixed 5 ms Luenberger observer + distilled compact7 7→64→64→1 PPO actor. Hysteresis and blending prevent abrupt switching.")
            note.setWordWrap(True); note.setStyleSheet("QLabel {background:#f2f2f2;padding:6px;}"); form.addRow(note); left.addWidget(exp)
            run = QtWidgets.QGroupBox("Run"); v = QtWidgets.QVBoxLayout(run); self.save_btn = QtWidgets.QPushButton("SAVE / Load Model & Apply Settings")
            self.go_btn = QtWidgets.QPushButton("GO"); self.stop_btn = QtWidgets.QPushButton("STOP"); self.reset_btn = QtWidgets.QPushButton("RESET")
            self.save_btn.clicked.connect(self.save_settings); self.go_btn.clicked.connect(self.start); self.stop_btn.clicked.connect(self.stop); self.reset_btn.clicked.connect(self.reset)
            v.addWidget(self.save_btn); row = QtWidgets.QHBoxLayout(); [row.addWidget(x) for x in (self.go_btn, self.stop_btn, self.reset_btn)]; v.addLayout(row)
            self.status = QtWidgets.QLabel("Choose a PPO model or run folder, then SAVE."); self.status.setWordWrap(True); self.state_label = QtWidgets.QLabel(); v.addWidget(self.status); v.addWidget(self.state_label); left.addWidget(run)
            out = QtWidgets.QGroupBox("Result & Logging"); ov = QtWidgets.QVBoxLayout(out); self.csv_check = QtWidgets.QCheckBox("Generate CSV log"); self.csv_check.setChecked(True)
            self.output_dir = QtWidgets.QLineEdit(os.path.expanduser("~/rip_twin_logs")); browse = QtWidgets.QPushButton("Browse"); browse.clicked.connect(self.choose_output)
            show = QtWidgets.QPushButton("Show Last Result Curves"); show.clicked.connect(self.show_result); ov.addWidget(self.csv_check); rr = QtWidgets.QHBoxLayout(); rr.addWidget(self.output_dir, 1); rr.addWidget(browse); ov.addLayout(rr); ov.addWidget(show); left.addWidget(out); left.addStretch(1)
            self.right = QtWidgets.QWidget(); root.addWidget(self.right, 1)
            for w in (self.initial, self.duration, self.speed, self.randomization, self.seed, self.pwm_limit, self.swing_pwm, self.kick_time, self.enter_deg, self.exit_deg, self.blend_alpha, self.velocity_lpf, self.csv_check, self.output_dir):
                signal = getattr(w, "valueChanged", None) or getattr(w, "currentIndexChanged", None) or getattr(w, "stateChanged", None) or getattr(w, "textChanged", None)
                if signal is not None: signal.connect(self.dirty)

        def build_3d(self):
            layout = QtWidgets.QVBoxLayout(self.right); self.figure = Figure(figsize=(9, 7), tight_layout=True); self.canvas = FigureCanvas(self.figure); layout.addWidget(self.canvas)
            self.axis = self.figure.add_subplot(111, projection="3d"); self.axis.set_title("Energy Swing-up + PPO Balance Digital Twin", pad=2)
            self.axis.set_xlabel("X / m"); self.axis.set_ylabel("Y / m"); self.axis.set_zlabel("Z / m"); lim = ARM_LENGTH + PEND_LENGTH + 0.05
            self.axis.set_xlim(-lim, lim); self.axis.set_ylim(-lim, lim); self.axis.set_zlim(-0.28, 0.38); self.axis.view_init(elev=24, azim=-55)
            angle = np.linspace(0, 2 * math.pi, 80); self.axis.plot(MOTOR_RADIUS * np.cos(angle), MOTOR_RADIUS * np.sin(angle), MOTOR_HEIGHT * np.ones_like(angle), linewidth=2)
            self.axis.plot(ARM_LENGTH * np.cos(angle), ARM_LENGTH * np.sin(angle), ARM_Z * np.ones_like(angle), linestyle="--", linewidth=1)
            self.arm_line, = self.axis.plot([], [], [], linewidth=6); self.pend_line, = self.axis.plot([], [], [], linewidth=5); self.joint_dot, = self.axis.plot([], [], [], marker="o", markersize=8); self.tip_dot, = self.axis.plot([], [], [], marker="o", markersize=10)
            self.tangent_line, = self.axis.plot([], [], [], linestyle=":", linewidth=2); self.reference_line, = self.axis.plot([], [], [], linestyle="--", linewidth=2.5); self.text = self.axis.text2D(0.03, 0.86, "", transform=self.axis.transAxes, fontsize=11)

        def dirty(self, *_): self.saved = None; self.save_btn.setText("SAVE / Load Model & Apply Settings *"); self.update_buttons()
        def choose_file(self):
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select PPO model", str(Path(self.model_path.text() or Path.home()).expanduser()), "PPO model (*.h *.npz)")
            if path: self.model_path.setText(path)
        def choose_dir(self):
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select PPO run folder", str(Path(self.model_path.text() or Path.home()).expanduser()))
            if path: self.model_path.setText(path)
        def choose_output(self):
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "Output directory", self.output_dir.text())
            if path: self.output_dir.setText(path)
        def capture(self):
            return SimulationConfig(self.model_path.text().strip(), float(self.duration.value()), float(self.speed.value()), int(self.seed.value()), float(self.randomization.value()), self.initial.currentText(), float(self.pwm_limit.value()), float(self.swing_pwm.value()), float(self.kick_time.value()), float(self.enter_deg.value()), float(self.exit_deg.value()), float(self.blend_alpha.value()), float(self.velocity_lpf.value()), self.csv_check.isChecked(), self.output_dir.text().strip())
        def save_settings(self):
            try:
                cfg = self.capture()
                if not cfg.model_selection: raise ValueError("Select a PPO model file or run folder.")
                if cfg.enter_deg >= cfg.exit_deg: raise ValueError("PPO enter angle must be smaller than PPO exit angle.")
                model = load_model(cfg.model_selection)
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "Model/settings error", str(exc)); return
            self.saved = cfg; self.model = model; self.model_path.setText(str(model.source)); self.save_btn.setText("SAVE / Load Model & Apply Settings")
            self.model_info.setText(f"{model.architecture} | internal normalization | model PWM scale={model.model_pwm_scale:g}\nID {model.digest} | float32 STM32 upload")
            self.status.setText(f"Model loaded: {model.source}"); self.reset(force=True); self.update_buttons()
        def reset(self, *_, force=False):
            self.running = False
            if self.saved and self.model:
                self.sim = HybridPPOSimulation(self.saved, self.model); self.sim_time = 0.0; self.rows = []; self.result = None; self.wall_prev = time.perf_counter(); self.accum = 0.0
                self.draw_state(self.sim.state, 0, 0, 0, 0, 0); self.status.setText("Ready. GO runs energy swing-up and switches to PPO balance near upright.")
            elif force: self.status.setText("Choose a PPO model, then SAVE.")
            self.update_buttons()
        def start(self):
            if not self.saved or not self.model: return
            self.reset(force=True); self.running = True; self.wall_prev = time.perf_counter(); self.status.setText("Hybrid PPO simulation running at 200 Hz."); self.update_buttons()
        def stop(self):
            if self.running: self.running = False; self.finalize(False); self.status.setText("Simulation stopped."); self.update_buttons()
        def tick(self):
            if not self.running or not self.sim or not self.saved: return
            now = time.perf_counter(); wall = min(now - self.wall_prev, 0.1); self.wall_prev = now; self.accum += wall * self.saved.playback_speed; steps = 0
            while self.accum >= CONTROL_DT and steps < 500 and self.running:
                state, pwm, mode, blend, action, raw, terminated = self.sim.step(); self.sim_time += CONTROL_DT
                self.rows.append([self.sim_time, *state, pwm, mode, blend, action, raw]); self.accum -= CONTROL_DT; steps += 1
                if terminated or self.sim_time + 1e-12 >= self.saved.duration:
                    self.running = False; self.finalize(True); self.status.setText("Configured hybrid experiment completed." if not terminated else "Simulation terminated by a safety limit."); self.update_buttons(); break
            if self.rows:
                r = self.rows[-1]; self.draw_state(np.asarray(r[1:5]), r[5], int(r[6]), r[7], r[8], r[9])
        def draw_state(self, state, pwm, mode, blend, action, raw):
            center, joint, tip, ref, tangent = rip_points(float(state[0]), float(state[2])); set_line3d(self.arm_line, center, joint); set_line3d(self.pend_line, joint, tip); set_point3d(self.joint_dot, joint); set_point3d(self.tip_dot, tip); set_line3d(self.reference_line, joint, ref); set_line3d(self.tangent_line, joint, joint + 0.11 * tangent)
            self.text.set_text(f"t = {self.sim_time:7.3f} s\nθ = {state[0]: .4f} rad\nα = {state[2]: .4f} rad\nPWM = {pwm: .0f}\nmode = {MODE_NAMES.get(mode, mode)}\nPPO blend = {blend:.3f}\naction={action:+.3f}, raw={raw:+.3f}")
            self.state_label.setText(f"θ={state[0]:+.4f}, θ̇={state[1]:+.4f}, α={state[2]:+.4f}, α̇={state[3]:+.4f}, PWM={pwm:+.0f}, {MODE_NAMES.get(mode, mode)}"); self.canvas.draw_idle()
        def rows_result(self):
            if not self.rows: return None
            d = np.asarray(self.rows, float); return SimulationResult(d[:,0], d[:,1], d[:,2], d[:,3], d[:,4], d[:,5], d[:,6].astype(int), d[:,7], d[:,8], d[:,9])
        def finalize(self, show):
            self.result = self.rows_result()
            if self.result is None: return
            png, csv_path = default_output_paths(self.saved.output_dir, self.saved.duration); fig = build_result_figure(self.result); fig.savefig(png, dpi=300, bbox_inches="tight")
            if self.saved.save_csv: save_result_csv(self.result, csv_path)
            if show: self.show_result()
        def show_result(self):
            if self.result is None: QtWidgets.QMessageBox.information(self, "No result", "No completed result is available."); return
            dialog = QtWidgets.QDialog(self); dialog.setWindowTitle(f"Hybrid PPO Digital Twin Result | {self.result.time[-1]:.3f} s"); dialog.resize(1080, 800)
            layout = QtWidgets.QVBoxLayout(dialog); canvas = FigureCanvas(build_result_figure(self.result)); layout.addWidget(canvas); close = QtWidgets.QPushButton("Close"); close.clicked.connect(dialog.close); layout.addWidget(close); dialog.exec_()
        def update_buttons(self):
            ready = self.saved is not None and self.model is not None; self.go_btn.setEnabled(ready and not self.running); self.stop_btn.setEnabled(self.running); self.reset_btn.setEnabled(ready); self.save_btn.setEnabled(not self.running)

    app = QtWidgets.QApplication(sys.argv); win = Window(); win.show(); return app.exec_()


def main() -> int:
    parser = argparse.ArgumentParser(description="Hybrid energy-swing-up + PPO-balance RIP test")
    parser.add_argument("--headless", action="store_true"); parser.add_argument("--model", default="")
    parser.add_argument("--duration", type=float, default=10.0); parser.add_argument("--seed", type=int, default=2026); parser.add_argument("--randomization", type=float, default=0.0)
    parser.add_argument("--pwm-limit", type=float, default=150.0); parser.add_argument("--swing-pwm", type=float, default=120.0)
    args = parser.parse_args()
    if not args.headless: return launch_gui()
    if not args.model: parser.error("--model is required with --headless")
    result = run_headless(args.model, args.duration, args.seed, args.randomization, args.pwm_limit, args.swing_pwm); metrics = result_metrics(result)
    print(f"steps={len(result.time)} final_alpha={result.alpha[-1]:.6f} stable_start={metrics['stable_start_time']} captures={int(metrics['capture_count'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
