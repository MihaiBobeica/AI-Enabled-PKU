"""Runtime for the Furuta DQN package.

The nominal environment in this file is a direct Python translation of the
physics, observation construction, reward and termination path in RN_DQN(3).m.
No rip_env environment is used during the first 2M steps, avoiding hidden
wrapper, actuator or observation differences. Domain randomization is applied
inside the same equations only when ``randomization_level > 0``.
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import shutil
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import sys
PROJECT_ROOT = Path(__file__).resolve().parent
for _local in (PROJECT_ROOT / "third_party", PROJECT_ROOT / "stable_baselines3", PROJECT_ROOT):
    if _local.exists() and str(_local) not in sys.path:
        sys.path.insert(0, str(_local))

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def dump_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def emit_event(event: str, **payload: Any) -> None:
    print("[PANEL_JSON] " + json.dumps({"event": event, **payload}, ensure_ascii=False, default=str), flush=True)


def config_snapshot() -> Dict[str, Any]:
    return {name: copy.deepcopy(getattr(config, name)) for name in config.config_sections()}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_config_string(value: Any) -> str:
    """Remove accidental whole-string quote layers introduced by GUI editing.

    Examples: ``auto``, ``'auto'`` and ``"'auto'"`` all normalize to
    ``auto``. Only quote layers that wrap the *entire* value are removed.
    """
    text = str(value).strip()
    for _ in range(4):
        if len(text) < 2 or text[0] not in {"'", '"'} or text[-1] != text[0]:
            break
        try:
            parsed = __import__("ast").literal_eval(text)
        except Exception:
            break
        if not isinstance(parsed, str) or parsed == text:
            break
        text = parsed.strip()
    return text


def choose_device(name: str | None = None) -> torch.device:
    requested = normalize_config_string(name if name is not None else config.RUN.get("device", "auto")).lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("RUN.device='cuda' but CUDA is unavailable")
    if requested == "mps" and not (
        getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    ):
        raise RuntimeError("RUN.device='mps' but Apple MPS is unavailable")
    return torch.device(requested)


def wrap_to_pi(value: float) -> float:
    """Match the original DQN while-loop semantics."""
    x = float(value)
    while x > math.pi:
        x -= 2.0 * math.pi
    while x < -math.pi:
        x += 2.0 * math.pi
    return x


def clip(value: float, limit: float) -> float:
    return max(-float(limit), min(float(limit), float(value)))


def _interp_scale_range(bounds: Sequence[float], level: float) -> tuple[float, float]:
    lo, hi = map(float, bounds)
    return 1.0 + level * (lo - 1.0), 1.0 + level * (hi - 1.0)


def _interp_zero_range(bounds: Sequence[float], level: float) -> tuple[float, float]:
    lo, hi = map(float, bounds)
    return level * lo, level * hi


def _sample_uniform(rng: np.random.Generator, bounds: Sequence[float]) -> float:
    lo, hi = map(float, bounds)
    return float(rng.uniform(lo, hi)) if hi > lo else lo


def _sample_nominal_to_range(
    rng: np.random.Generator, nominal: float, bounds: Sequence[float], level: float
) -> float:
    target = _sample_uniform(rng, bounds)
    return float(nominal + level * (target - nominal))


@dataclass
class EpisodeDomain:
    level: float
    physical: Dict[str, float]
    init_theta_std_deg: float
    init_alpha_std_deg: float
    init_theta_dot_std: float
    init_alpha_dot_std: float
    pwm_limit: float
    pwm_gain: float
    pwm_bias: float
    pwm_deadzone: float
    pwm_noise_sigma: float
    actuator_tau: float
    action_delay_steps: int
    theta_bias: float
    alpha_bias: float
    theta_sigma: float
    alpha_sigma: float
    theta_dot_sigma: float
    alpha_dot_sigma: float
    encoder_quantization_rad: float
    velocity_lpf: float
    process_theta_dot_sigma: float
    process_alpha_dot_sigma: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "physical": dict(self.physical),
            "init_theta_std_deg": self.init_theta_std_deg,
            "init_alpha_std_deg": self.init_alpha_std_deg,
            "init_theta_dot_std": self.init_theta_dot_std,
            "init_alpha_dot_std": self.init_alpha_dot_std,
            "pwm_limit": self.pwm_limit,
            "pwm_gain": self.pwm_gain,
            "pwm_bias": self.pwm_bias,
            "pwm_deadzone": self.pwm_deadzone,
            "pwm_noise_sigma": self.pwm_noise_sigma,
            "actuator_tau": self.actuator_tau,
            "action_delay_steps": self.action_delay_steps,
            "theta_bias": self.theta_bias,
            "alpha_bias": self.alpha_bias,
            "theta_sigma": self.theta_sigma,
            "alpha_sigma": self.alpha_sigma,
            "theta_dot_sigma": self.theta_dot_sigma,
            "alpha_dot_sigma": self.alpha_dot_sigma,
            "encoder_quantization_rad": self.encoder_quantization_rad,
            "velocity_lpf": self.velocity_lpf,
            "process_theta_dot_sigma": self.process_theta_dot_sigma,
            "process_alpha_dot_sigma": self.process_alpha_dot_sigma,
        }


def sample_episode_domain(rng: np.random.Generator, level: float) -> EpisodeDomain:
    dr = config.DOMAIN_RANDOMIZATION
    level = float(np.clip(level if dr.get("enabled", True) else 0.0, 0.0, 1.0))
    physical = dict(config.PHYSICAL_PARAMS)

    if level > 0.0:
        for scale_name, bounds in dr["param_scale_ranges"].items():
            param_name = str(scale_name).removesuffix("_scale")
            lo, hi = _interp_scale_range(bounds, level)
            physical[param_name] *= float(rng.uniform(lo, hi))

    nominal_theta_std = float(config.ENV["init_theta_std_deg"])
    nominal_alpha_std = float(config.ENV["init_alpha_std_deg"])
    nominal_theta_dot_std = float(config.ENV["init_theta_dot_std"])
    nominal_alpha_dot_std = float(config.ENV["init_alpha_dot_std"])

    if level <= 0.0:
        return EpisodeDomain(
            level=0.0,
            physical=physical,
            init_theta_std_deg=nominal_theta_std,
            init_alpha_std_deg=nominal_alpha_std,
            init_theta_dot_std=nominal_theta_dot_std,
            init_alpha_dot_std=nominal_alpha_dot_std,
            pwm_limit=float(config.ENV["pwm_limit"]),
            pwm_gain=1.0,
            pwm_bias=0.0,
            pwm_deadzone=0.0,
            pwm_noise_sigma=0.0,
            actuator_tau=0.0,
            action_delay_steps=0,
            theta_bias=0.0,
            alpha_bias=0.0,
            theta_sigma=0.0,
            alpha_sigma=0.0,
            theta_dot_sigma=0.0,
            alpha_dot_sigma=0.0,
            encoder_quantization_rad=0.0,
            velocity_lpf=float(config.ENV["velocity_lpf"]),
            process_theta_dot_sigma=0.0,
            process_alpha_dot_sigma=0.0,
        )

    pwm_limit_scale = _sample_uniform(rng, _interp_scale_range(dr["pwm_limit_scale_range"], level))
    pwm_gain = _sample_uniform(rng, _interp_scale_range(dr["pwm_gain_range"], level))
    pwm_bias = _sample_uniform(rng, _interp_zero_range(dr["pwm_bias_range"], level))
    pwm_deadzone = _sample_uniform(rng, _interp_zero_range(dr["pwm_deadzone_range"], level))
    pwm_noise = _sample_uniform(rng, _interp_zero_range(dr["pwm_noise_sigma_range"], level))
    tau = _sample_uniform(rng, _interp_zero_range(dr["actuator_tau_range"], level))

    delay_lo, delay_hi = map(int, dr["action_delay_steps_range"])
    if delay_hi <= 0:
        delay_steps = 0
    else:
        # At level=1 this reproduces uniform integer sampling over the configured range.
        possible = np.arange(delay_lo, delay_hi + 1, dtype=int)
        sampled_full = int(rng.choice(possible))
        delay_steps = int(math.floor(level * sampled_full + 0.5))

    use_random_lpf = rng.random() < level * float(dr["use_lpf_velocity_probability"])
    velocity_lpf = (
        _sample_nominal_to_range(rng, float(config.ENV["velocity_lpf"]), dr["velocity_lpf_range"], level)
        if use_random_lpf
        else float(config.ENV["velocity_lpf"])
    )

    return EpisodeDomain(
        level=level,
        physical=physical,
        init_theta_std_deg=_sample_nominal_to_range(rng, nominal_theta_std, dr["init_theta_std_deg_range"], level),
        init_alpha_std_deg=_sample_nominal_to_range(rng, nominal_alpha_std, dr["init_alpha_std_deg_range"], level),
        init_theta_dot_std=_sample_nominal_to_range(rng, nominal_theta_dot_std, dr["init_theta_dot_std_range"], level),
        init_alpha_dot_std=_sample_nominal_to_range(rng, nominal_alpha_dot_std, dr["init_alpha_dot_std_range"], level),
        pwm_limit=float(config.ENV["pwm_limit"]) * pwm_limit_scale,
        pwm_gain=pwm_gain,
        pwm_bias=pwm_bias,
        pwm_deadzone=pwm_deadzone,
        pwm_noise_sigma=pwm_noise,
        actuator_tau=tau,
        action_delay_steps=delay_steps,
        theta_bias=_sample_uniform(rng, _interp_zero_range(dr["theta_bias_range"], level)),
        alpha_bias=_sample_uniform(rng, _interp_zero_range(dr["alpha_bias_range"], level)),
        theta_sigma=_sample_uniform(rng, _interp_zero_range(dr["theta_sigma_range"], level)),
        alpha_sigma=_sample_uniform(rng, _interp_zero_range(dr["alpha_sigma_range"], level)),
        theta_dot_sigma=_sample_uniform(rng, _interp_zero_range(dr["theta_dot_sigma_range"], level)),
        alpha_dot_sigma=_sample_uniform(rng, _interp_zero_range(dr["alpha_dot_sigma_range"], level)),
        encoder_quantization_rad=_sample_uniform(rng, _interp_zero_range(dr["encoder_quantization_rad_range"], level)),
        velocity_lpf=float(np.clip(velocity_lpf, 0.0, 1.0)),
        process_theta_dot_sigma=_sample_uniform(rng, _interp_zero_range(dr["process_theta_dot_sigma_range"], level)),
        process_alpha_dot_sigma=_sample_uniform(rng, _interp_zero_range(dr["process_alpha_dot_sigma_range"], level)),
    )


def furuta_derivative(state: np.ndarray, pwm: float, p: Dict[str, float]) -> np.ndarray:
    """Direct translation of RN_DQN(3).m ``furuta_f``."""
    th, thd, al, ald = map(float, state)
    del th  # theta does not enter the equations explicitly.

    g = p["g"]
    ct = p["c_theta"]
    ca = p["c_alpha"]
    kt = p["k_t"]
    kb = p["k_b"]
    ku = p["k_u"]
    resistance = p["R"]
    k1_motor = kt * ku / resistance
    k2_motor = kt * kb / resistance

    m1 = p["m1"]
    m2 = p["m2"]
    l1 = p["l1"]
    l1c = p["l1cg"]
    l2c = p["l2cg"]
    i1z = p["I1z"]
    i2x = p["I2x"]
    i2y = p["I2y"]
    i2z = p["I2z"]

    s_a = math.sin(al)
    c_a = math.cos(al)
    sac = s_a * c_a

    a = m1 * l1c**2 + i1z + m2 * l1**2 + (m2 * l2c**2 + i2z) * s_a**2 + i2y * c_a**2
    b = m2 * l1 * l2c * c_a
    c = -m2 * l1 * l2c * s_a
    d = 2.0 * (i2z + m2 * l2c**2 - i2y) * sac
    e = k1_motor * float(pwm) - k2_motor * thd - ct * thd

    f = -(m2 * l2c**2 + i2x)
    g_term = -(m2 * l1 * l2c * c_a)
    h = (m2 * l2c**2 - i2y + i2z) * sac
    k = m2 * g * l2c * s_a
    l = ca * ald

    den1 = a * f - g_term * b
    den2 = g_term * b - a * f
    if abs(den1) < 1e-12 or abs(den2) < 1e-12:
        raise FloatingPointError(f"Singular Furuta dynamics denominator: den1={den1}, den2={den2}")

    thdd = ((-f * c) * ald**2 + (-f * d) * ald * thd + (b * h) * thd**2 + (b * k + f * e - b * l)) / den1
    aldd = ((-g_term * c) * ald**2 + (-g_term * d) * ald * thd + (a * h) * thd**2 + (a * k + g_term * e - a * l)) / den2

    return np.asarray([thd, thdd, ald, aldd], dtype=np.float64)


def furuta_rk4_step(state: np.ndarray, pwm: float, p: Dict[str, float], dt: float) -> np.ndarray:
    s = np.asarray(state, dtype=np.float64)
    k1 = furuta_derivative(s, pwm, p)
    k2 = furuta_derivative(s + 0.5 * dt * k1, pwm, p)
    k3 = furuta_derivative(s + 0.5 * dt * k2, pwm, p)
    k4 = furuta_derivative(s + dt * k3, pwm, p)
    return s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


class MatlabAlignedRIPEnv(gym.Env):
    """Nominal dynamics plus optional episode-wise randomization."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        randomization_level: float = 0.0,
        seed: int = 42,
        max_steps: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.randomization_level = float(randomization_level)
        self.base_seed = int(seed)
        self.rng = np.random.default_rng(self.base_seed)
        self.dt = float(config.ENV["physical_dt"])
        self.max_steps = int(max_steps or config.ENV["max_physical_steps"])
        self.actions = np.asarray(config.ENV["discrete_actions"], dtype=np.float64)
        self.action_space = gym.spaces.Discrete(len(self.actions))
        self.observation_space = gym.spaces.Box(
            low=np.asarray([-1.0, -1.0, -config.ENV["theta_dot_limit"], -1.0, -1.0, -config.ENV["alpha_dot_limit"]], dtype=np.float32),
            high=np.asarray([1.0, 1.0, config.ENV["theta_dot_limit"], 1.0, 1.0, config.ENV["alpha_dot_limit"]], dtype=np.float32),
            dtype=np.float32,
        )
        self.domain = sample_episode_domain(self.rng, self.randomization_level)
        self.x_true = np.zeros(4, dtype=np.float64)
        self.theta_meas_prev = 0.0
        self.alpha_meas_prev = 0.0
        self.theta_unwrap = 0.0
        self.theta_unwrap_prev = 0.0
        self.theta_dot_est = 0.0
        self.alpha_dot_est = 0.0
        self.control_state = np.zeros(4, dtype=np.float64)
        self.applied_pwm = 0.0
        self.delay_queue: deque[float] = deque()
        self.step_count = 0
        self.maintain_cur_stable = 0
        self.maintain_max_stable = 0

    def set_randomization_level(self, level: float) -> None:
        self.randomization_level = float(level)

    def _measurement(self) -> tuple[float, float]:
        theta = float(self.x_true[0]) + self.domain.theta_bias
        alpha = float(self.x_true[2]) + self.domain.alpha_bias
        if self.domain.theta_sigma > 0.0:
            theta += float(self.rng.normal(0.0, self.domain.theta_sigma))
        if self.domain.alpha_sigma > 0.0:
            alpha += float(self.rng.normal(0.0, self.domain.alpha_sigma))
        q = self.domain.encoder_quantization_rad
        if q > 0.0:
            theta = round(theta / q) * q
            alpha = round(alpha / q) * q
        return theta, alpha

    def _observation(self) -> np.ndarray:
        theta, theta_dot, alpha, alpha_dot = map(float, self.control_state)
        return np.asarray(
            [math.sin(theta), math.cos(theta), theta_dot, math.sin(alpha), math.cos(alpha), alpha_dot],
            dtype=np.float32,
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        del options
        self.domain = sample_episode_domain(self.rng, self.randomization_level)

        theta = math.radians(float(config.ENV["init_theta_mean_deg"])) + self.rng.normal(0.0, math.radians(self.domain.init_theta_std_deg))
        theta_dot = float(config.ENV["init_theta_dot_mean"]) + self.rng.normal(0.0, self.domain.init_theta_dot_std)
        alpha = wrap_to_pi(math.radians(float(config.ENV["init_alpha_mean_deg"])) + self.rng.normal(0.0, math.radians(self.domain.init_alpha_std_deg)))
        alpha_dot = float(config.ENV["init_alpha_dot_mean"]) + self.rng.normal(0.0, self.domain.init_alpha_dot_std)
        self.x_true = np.asarray([theta, theta_dot, alpha, alpha_dot], dtype=np.float64)

        theta_meas, alpha_meas = self._measurement()
        self.theta_meas_prev = theta_meas
        self.alpha_meas_prev = alpha_meas
        self.theta_unwrap = theta_meas
        self.theta_unwrap_prev = theta_meas
        self.theta_dot_est = 0.0
        self.alpha_dot_est = 0.0
        self.control_state = np.asarray([self.theta_unwrap, 0.0, wrap_to_pi(alpha_meas), 0.0], dtype=np.float64)
        self.applied_pwm = 0.0
        self.delay_queue = deque([0.0] * self.domain.action_delay_steps)
        self.step_count = 0
        self.maintain_cur_stable = 0
        self.maintain_max_stable = 0
        return self._observation(), self._info(0.0, 0.0)

    def _apply_actuator(self, pwm_command: float) -> float:
        command = float(pwm_command) * self.domain.pwm_gain + self.domain.pwm_bias
        if abs(command) < self.domain.pwm_deadzone:
            command = 0.0
        if self.domain.pwm_noise_sigma > 0.0:
            command += float(self.rng.normal(0.0, self.domain.pwm_noise_sigma))
        command = float(np.clip(command, -self.domain.pwm_limit, self.domain.pwm_limit))

        if self.domain.action_delay_steps > 0:
            self.delay_queue.append(command)
            target = float(self.delay_queue.popleft())
        else:
            target = command

        tau = self.domain.actuator_tau
        if tau > 0.0:
            factor = self.dt / (tau + self.dt)
            self.applied_pwm += factor * (target - self.applied_pwm)
        else:
            self.applied_pwm = target
        return float(self.applied_pwm)

    def _reward(self, alpha: float, theta_dot: float, alpha_dot: float, pwm_command: float) -> float:
        rw = config.REWARD
        reward = (
            float(rw["k_cos_alpha"]) * math.cos(alpha)
            - float(rw["k_alpha_dot"]) * alpha_dot**2
            - float(rw["k_theta_dot"]) * theta_dot**2
            - float(rw["k_theta"]) * float(self.control_state[0]) ** 2
            - float(rw["alpha_penalty_value"]) * float(abs(alpha) > math.radians(float(rw["alpha_penalty_deg"])))
        )
        if float(rw.get("action_l2", 0.0)) != 0.0:
            reward -= float(rw["action_l2"]) * pwm_command**2
        return float(reward)

    def _info(self, pwm_command: float, pwm_effective: float) -> Dict[str, Any]:
        return {
            "state": self.control_state.astype(np.float64).copy(),
            "true_state": self.x_true.astype(np.float64).copy(),
            "pwm": float(pwm_command),
            "pwm_effective": float(pwm_effective),
            "randomization_level": float(self.randomization_level),
            "maintain_cur_stable": int(self.maintain_cur_stable),
            "maintain_max_stable": int(self.maintain_max_stable),
            "domain": self.domain.as_dict(),
        }

    def step(self, action: int):
        action_index = int(np.asarray(action).reshape(-1)[0])
        if not self.action_space.contains(action_index):
            raise ValueError(f"Invalid action index {action_index}")
        pwm_command = float(self.actions[action_index])
        pwm_effective = self._apply_actuator(pwm_command)

        self.x_true = furuta_rk4_step(self.x_true, pwm_effective, self.domain.physical, self.dt)
        if self.domain.process_theta_dot_sigma > 0.0:
            self.x_true[1] += float(self.rng.normal(0.0, self.domain.process_theta_dot_sigma))
        if self.domain.process_alpha_dot_sigma > 0.0:
            self.x_true[3] += float(self.rng.normal(0.0, self.domain.process_alpha_dot_sigma))

        theta_meas, alpha_meas = self._measurement()
        dtheta_wrapped = wrap_to_pi(theta_meas - self.theta_meas_prev)
        self.theta_unwrap += dtheta_wrapped
        theta_dot_raw = (self.theta_unwrap - self.theta_unwrap_prev) / self.dt
        alpha_dot_raw = wrap_to_pi(alpha_meas - self.alpha_meas_prev) / self.dt

        lpf = self.domain.velocity_lpf
        self.theta_dot_est = (1.0 - lpf) * self.theta_dot_est + lpf * theta_dot_raw
        self.alpha_dot_est = (1.0 - lpf) * self.alpha_dot_est + lpf * alpha_dot_raw

        theta_dot_ctrl = self.theta_dot_est
        alpha_dot_ctrl = self.alpha_dot_est
        if self.domain.theta_dot_sigma > 0.0:
            theta_dot_ctrl += float(self.rng.normal(0.0, self.domain.theta_dot_sigma))
        if self.domain.alpha_dot_sigma > 0.0:
            alpha_dot_ctrl += float(self.rng.normal(0.0, self.domain.alpha_dot_sigma))

        theta_ctrl = self.theta_unwrap
        alpha_ctrl = wrap_to_pi(alpha_meas)
        theta_dot_clipped = clip(theta_dot_ctrl, float(config.ENV["theta_dot_limit"]))
        alpha_dot_clipped = clip(alpha_dot_ctrl, float(config.ENV["alpha_dot_limit"]))
        self.control_state = np.asarray([theta_ctrl, theta_dot_clipped, alpha_ctrl, alpha_dot_clipped], dtype=np.float64)

        finite = bool(np.all(np.isfinite(self.control_state)) and np.all(np.isfinite(self.x_true)))
        terminated = (
            abs(theta_ctrl) > float(config.ENV["theta_limit"])
            or abs(theta_dot_ctrl) > float(config.ENV["theta_dot_limit"])
            or abs(alpha_dot_ctrl) > float(config.ENV["alpha_dot_limit"])
            or not finite
        )

        reward = self._reward(alpha_ctrl, theta_dot_clipped, alpha_dot_clipped, pwm_command)

        if abs(alpha_ctrl) < math.radians(15.0) and abs(alpha_dot_clipped) < 4.0:
            self.maintain_cur_stable += 1
        else:
            self.maintain_cur_stable = 0
        self.maintain_max_stable = max(self.maintain_max_stable, self.maintain_cur_stable)

        self.step_count += 1
        truncated = self.step_count >= self.max_steps

        self.theta_meas_prev = theta_meas
        self.alpha_meas_prev = alpha_meas
        self.theta_unwrap_prev = self.theta_unwrap

        return self._observation(), reward, bool(terminated), bool(truncated), self._info(pwm_command, pwm_effective)


class QNetwork(nn.Module):
    def __init__(self, input_dim: int = 6, hidden_sizes: Sequence[int] = (64, 64), output_dim: int = 10):
        super().__init__()
        if len(tuple(hidden_sizes)) != 2:
            raise ValueError("DQN requires exactly two hidden layers")
        h1, h2 = map(int, hidden_sizes)
        self.fc1 = nn.Linear(int(input_dim), h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, int(output_dim))
        self.reset_matlab_parameters(float(config.DQN["weight_init_std"]))

    def reset_matlab_parameters(self, std: float = 0.01) -> None:
        for layer in (self.fc1, self.fc2, self.fc3):
            nn.init.normal_(layer.weight, mean=0.0, std=float(std))
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int = 6):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        i = self.ptr
        self.states[i] = np.asarray(state, dtype=np.float32)
        self.actions[i] = int(action)
        self.rewards[i] = float(reward)
        self.next_states[i] = np.asarray(next_state, dtype=np.float32)
        self.dones[i] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def reset(self) -> None:
        self.ptr = 0
        self.size = 0

    def sample(self, batch_size: int, rng: np.random.Generator, device: torch.device):
        if self.size <= 0:
            raise RuntimeError("Cannot sample empty replay buffer")
        indices = rng.integers(0, self.size, size=int(batch_size))
        return (
            torch.as_tensor(self.states[indices], device=device),
            torch.as_tensor(self.actions[indices], device=device),
            torch.as_tensor(self.rewards[indices], device=device),
            torch.as_tensor(self.next_states[indices], device=device),
            torch.as_tensor(self.dones[indices], device=device),
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "state_dim": self.state_dim,
            "states": self.states[: self.size].copy(),
            "actions": self.actions[: self.size].copy(),
            "rewards": self.rewards[: self.size].copy(),
            "next_states": self.next_states[: self.size].copy(),
            "dones": self.dones[: self.size].copy(),
            "ptr": self.ptr,
            "size": self.size,
        }


@dataclass
class LoadedModel:
    network: QNetwork
    checkpoint: Dict[str, Any]
    path: Path


def build_network(device: torch.device) -> QNetwork:
    network = QNetwork(
        input_dim=int(config.ENV["observation_dim"]),
        hidden_sizes=tuple(config.DQN["net_arch"]),
        output_dim=len(config.ENV["discrete_actions"]),
    )
    return network.to(device)


def make_optimizer(network: QNetwork) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        network.parameters(),
        lr=float(config.DQN["learning_rate"]),
        betas=(float(config.DQN["adam_beta1"]), float(config.DQN["adam_beta2"])),
        eps=float(config.DQN["adam_eps"]),
    )


def epsilon_at(step: int) -> float:
    start = float(config.DQN["exploration_initial_eps"])
    final = float(config.DQN["exploration_final_eps"])
    decay = float(config.DQN["exploration_decay"])
    return final + (start - final) * math.exp(-float(step) / max(decay, 1e-12))


def greedy_action(network: QNetwork, observation: np.ndarray, device: torch.device) -> int:
    with torch.no_grad():
        x = torch.as_tensor(np.asarray(observation, dtype=np.float32), device=device).unsqueeze(0)
        return int(torch.argmax(network(x), dim=1).item())


def dqn_update(
    online: QNetwork,
    target: QNetwork,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    rng: np.random.Generator,
    device: torch.device,
    batch_size: int,
) -> Dict[str, float]:
    states, actions, rewards, next_states, dones = replay.sample(batch_size, rng, device)
    q_all = online(states)
    q_selected = q_all.gather(1, actions.long().unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        # Vanilla DQN: the target network both selects and evaluates the max action.
        next_q = target(next_states).max(dim=1).values
        targets = rewards + float(config.DQN["gamma"]) * next_q * (1.0 - dones)
    delta = float(config.DQN["huber_delta"])
    try:
        loss = F.huber_loss(q_selected, targets, reduction="mean", delta=delta)
    except TypeError:  # older torch
        loss = F.smooth_l1_loss(q_selected, targets, reduction="mean", beta=delta)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    max_norm = float(config.DQN.get("max_grad_norm", 0.0))
    if max_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(online.parameters(), max_norm)
    optimizer.step()
    td = q_selected.detach() - targets.detach()
    return {
        "loss": float(loss.detach().cpu().item()),
        "mean_q": float(q_selected.detach().mean().cpu().item()),
        "mean_target": float(targets.detach().mean().cpu().item()),
        "mean_abs_td": float(td.abs().mean().cpu().item()),
    }


def checkpoint_payload(
    online: QNetwork,
    target: QNetwork,
    behavior: QNetwork,
    optimizer: torch.optim.Optimizer,
    *,
    global_step: int,
    episode: int,
    stage_index: int,
    stage_name: str,
    stage_level: float,
    stats: Dict[str, Any],
    replay: Optional[ReplayBuffer] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "format": "dqn_v1",
        "algorithm": config.ALGORITHM_NAME,
        "global_step": int(global_step),
        "episode": int(episode),
        "stage_index": int(stage_index),
        "stage_name": str(stage_name),
        "stage_level": float(stage_level),
        "input_dim": int(config.ENV["observation_dim"]),
        "hidden_sizes": tuple(config.DQN["net_arch"]),
        "output_dim": len(config.ENV["discrete_actions"]),
        "actions_pwm": tuple(config.ENV["discrete_actions"]),
        "observation_order": tuple(config.ENV["observation_order"]),
        "input_pre_scaled": False,
        "velocity_lpf": float(config.ENV["velocity_lpf"]),
        "online_state_dict": {k: v.detach().cpu() for k, v in online.state_dict().items()},
        "target_state_dict": {k: v.detach().cpu() for k, v in target.state_dict().items()},
        "behavior_state_dict": {k: v.detach().cpu() for k, v in behavior.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "stats": stats,
        "config": config_snapshot(),
    }
    if replay is not None and bool(config.DQN.get("save_replay_buffer", False)):
        payload["replay_buffer"] = replay.state_dict()
    return payload


def save_checkpoint(path: str | Path, payload: Dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, p)
    return p


def load_model(path: str | Path, device: Optional[torch.device] = None) -> LoadedModel:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    device = device or choose_device()
    checkpoint = torch.load(p, map_location=device, weights_only=False)
    input_dim = int(checkpoint.get("input_dim", 6))
    hidden = tuple(checkpoint.get("hidden_sizes", (64, 64)))
    output_dim = int(checkpoint.get("output_dim", len(config.ENV["discrete_actions"])))
    network = QNetwork(input_dim, hidden, output_dim).to(device)
    state = checkpoint.get("online_state_dict", checkpoint.get("state_dict"))
    if state is None:
        raise KeyError(f"Checkpoint {p} has no online_state_dict/state_dict")
    network.load_state_dict(state)
    network.eval()
    return LoadedModel(network=network, checkpoint=checkpoint, path=p)


def resolve_model_path(run_dir: str | Path) -> Path:
    root = Path(run_dir)
    candidates = [
        root / "selected_best_model.pt",
        root / "best_nominal_model" / "best_model.pt",
        root / "recovery_model" / "nominal_2m_last.pt",
        root / "final_model.pt",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError("No model checkpoint found. Tried: " + ", ".join(map(str, candidates)))


def _c_float(value: float) -> str:
    text = f"{float(value):.9g}"
    if "." not in text and "e" not in text.lower():
        text += ".0"
    return text + "f"


def _format_float_array(values: np.ndarray, indent: str = "  ") -> str:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    chunks = []
    for start in range(0, len(flat), 8):
        chunks.append(indent + ", ".join(_c_float(v) for v in flat[start : start + 8]))
    return ",\n".join(chunks)


def _write_2d(name: str, array: np.ndarray) -> str:
    a = np.asarray(array, dtype=np.float32)
    rows = []
    for row in a:
        rows.append("  {" + ", ".join(_c_float(v) for v in row) + "}")
    return f"static const float {name}[{a.shape[0]}][{a.shape[1]}] = {{\n" + ",\n".join(rows) + "\n};\n"


def _write_1d(name: str, array: np.ndarray) -> str:
    a = np.asarray(array, dtype=np.float32).reshape(-1)
    return f"static const float {name}[{len(a)}] = {{\n{_format_float_array(a)}\n}};\n"


def export_c_header(network: QNetwork, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().numpy() for k, v in network.state_dict().items()}
    actions = tuple(int(v) for v in config.ENV["discrete_actions"])
    lines = [
        "#pragma once",
        "// Auto-generated from Python DQN.",
        "// Input order: sin(theta), cos(theta), theta_dot_LPF, sin(alpha), cos(alpha), alpha_dot_LPF.",
        "// Velocities are raw rad/s estimates (LPF coefficient configured in training); no extra normalization.",
        "#include <stdint.h>",
        "",
        "#define MODEL_INPUT_DIM 6",
        "#define MODEL_H1 64",
        "#define MODEL_H2 64",
        f"#define MODEL_OUTPUT_DIM {len(actions)}",
        "#define MODEL_STATE_HISTORY_LEN 1",
        "#define MODEL_ACTION_HISTORY_LEN 0",
        "#define MODEL_INPUT_PRE_SCALED 0",
        "#define MODEL_VELOCITY_LPF 0.25f",
        "",
        _write_2d("DQN_W1", state["fc1.weight"]),
        _write_1d("DQN_b1", state["fc1.bias"]),
        _write_2d("DQN_W2", state["fc2.weight"]),
        _write_1d("DQN_b2", state["fc2.bias"]),
        _write_2d("DQN_W3", state["fc3.weight"]),
        _write_1d("DQN_b3", state["fc3.bias"]),
        "static const int16_t DQN_ACTIONS[MODEL_OUTPUT_DIM] = {" + ", ".join(map(str, actions)) + "};",
        "",
    ]
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def episode_summary(rewards: list[float], infos: list[Dict[str, Any]], length: int) -> Dict[str, Any]:
    maintain_max = int(max((int(i.get("maintain_max_stable", 0)) for i in infos), default=0))
    states = np.asarray([i.get("state", np.zeros(4)) for i in infos], dtype=np.float64)
    return {
        "reward_total": float(np.sum(rewards)),
        "reward_per_step": float(np.sum(rewards) / max(1, length)),
        "length": int(length),
        "maintain_max_stable": maintain_max,
        "stable_success": bool(maintain_max >= int(config.EVAL["stable_hold_steps"])),
        "capture": bool(states.size and np.any(np.abs(states[:, 2]) < math.radians(float(config.EVAL["capture_angle_deg"])))),
        "mean_abs_theta": float(np.mean(np.abs(states[:, 0]))) if states.size else math.nan,
        "mean_abs_theta_dot": float(np.mean(np.abs(states[:, 1]))) if states.size else math.nan,
        "mean_abs_alpha": float(np.mean(np.abs(states[:, 2]))) if states.size else math.nan,
        "mean_abs_alpha_dot": float(np.mean(np.abs(states[:, 3]))) if states.size else math.nan,
        "mean_abs_pwm": float(np.mean([abs(float(i.get("pwm", 0.0))) for i in infos])) if infos else 0.0,
    }


def evaluate_network(
    network: QNetwork,
    *,
    device: torch.device,
    randomization_level: float,
    episodes: int,
    max_steps: int,
    seed: int,
) -> Dict[str, Any]:
    rows = []
    network.eval()
    for ep in range(int(episodes)):
        env = MatlabAlignedRIPEnv(randomization_level=randomization_level, seed=seed + ep, max_steps=max_steps)
        obs, _ = env.reset(seed=seed + ep)
        rewards: list[float] = []
        infos: list[Dict[str, Any]] = []
        for _ in range(int(max_steps)):
            action = greedy_action(network, obs, device)
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(float(reward))
            infos.append(info)
            if terminated or truncated:
                break
        rows.append(episode_summary(rewards, infos, len(rewards)))
        env.close()
    network.train()
    return {
        "episodes": len(rows),
        "randomization_level": float(randomization_level),
        "mean_episode_reward": float(np.mean([r["reward_total"] for r in rows])),
        "mean_reward_per_step": float(np.mean([r["reward_per_step"] for r in rows])),
        "mean_length": float(np.mean([r["length"] for r in rows])),
        "mean_maintain_max_stable": float(np.mean([r["maintain_max_stable"] for r in rows])),
        "stable_success_rate": float(np.mean([r["stable_success"] for r in rows])),
        "capture_rate": float(np.mean([r["capture"] for r in rows])),
        "mean_abs_theta": float(np.mean([r["mean_abs_theta"] for r in rows])),
        "mean_abs_theta_dot": float(np.mean([r["mean_abs_theta_dot"] for r in rows])),
        "mean_abs_alpha": float(np.mean([r["mean_abs_alpha"] for r in rows])),
        "mean_abs_alpha_dot": float(np.mean([r["mean_abs_alpha_dot"] for r in rows])),
        "mean_abs_pwm": float(np.mean([r["mean_abs_pwm"] for r in rows])),
    }


def copy_model(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination
