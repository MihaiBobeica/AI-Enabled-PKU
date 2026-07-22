# rip_env/envs/cartpole_rip_sim2real.py
# Domain-randomized sim-to-real benchmark wrapper around the existing CartPoleRIPEenv.

from __future__ import annotations

import copy
import math
from collections import deque
from typing import Optional, Dict, Any

import numpy as np
import gymnasium as gym

from .cartpole_rip import CartPoleRIPEenv
from .types import RIPPhysicalParams
from .dynamics import wrap_to_pi
from .sim2real_types import DomainRandomizationConfig, EpisodeRandomizationSnapshot


class CartPoleRIPSim2RealEnv(CartPoleRIPEenv):
    """Rotary inverted pendulum environment with episode-level domain randomization.

    This class intentionally keeps the original environment intact.  It only
    adds the sim-to-real ingredients needed for Stage-1 PPO balance:

    1. physical-parameter randomization at reset;
    2. actuator saturation/gain/bias/dead-zone/lag/delay/noise;
    3. sensor bias/noise/quantization and optional LPF velocity estimation;
    4. an externally settable randomization level for curriculum training.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.nominal_params: RIPPhysicalParams = copy.deepcopy(self.cfg.physical_params)
        self.nominal_pwm_limit = float(self.cfg.continuous_pwm_limit)
        self.randomization: DomainRandomizationConfig = getattr(
            self.cfg, "randomization", DomainRandomizationConfig(enabled=False, level=0.0)
        )
        self.randomization_level = float(np.clip(getattr(self.randomization, "level", 0.0), 0.0, 1.0))
        self.episode_randomization = EpisodeRandomizationSnapshot()

        self._cmd_delay_queue: deque[float] = deque([0.0], maxlen=1)
        self._actuator_prev_pwm = 0.0
        self._last_commanded_pwm = 0.0
        self._last_delayed_pwm = 0.0
        self._last_effective_pwm = 0.0

    # ------------------------------------------------------------
    # Randomization helpers
    # ------------------------------------------------------------
    def set_randomization_level(self, level: float) -> None:
        self.randomization_level = float(np.clip(level, 0.0, 1.0))
        if hasattr(self.cfg, "randomization"):
            self.cfg.randomization.level = self.randomization_level

    def _scaled_range_around_one(self, low: float, high: float) -> tuple[float, float]:
        level = float(np.clip(self.randomization_level, 0.0, 1.0))
        return 1.0 + level * (float(low) - 1.0), 1.0 + level * (float(high) - 1.0)

    def _scaled_abs_range_from_zero(self, low: float, high: float) -> tuple[float, float]:
        level = float(np.clip(self.randomization_level, 0.0, 1.0))
        return level * float(low), level * float(high)

    def _sample_uniform(self, low: float, high: float) -> float:
        if abs(high - low) < 1e-12:
            return float(low)
        return float(self.np_random.uniform(float(low), float(high)))

    def _sample_scale(self, name: str, default_range: tuple[float, float]) -> float:
        if not getattr(self.randomization, "enabled", False) or self.randomization_level <= 0.0:
            return 1.0
        lo, hi = self.randomization.param_scale_ranges.get(name, default_range)
        lo2, hi2 = self._scaled_range_around_one(lo, hi)
        # log-uniform avoids biasing wide multiplicative ranges toward large values.
        lo2 = max(lo2, 1e-6)
        hi2 = max(hi2, lo2 + 1e-9)
        return float(math.exp(self.np_random.uniform(math.log(lo2), math.log(hi2))))

    def _apply_episode_physical_randomization(self) -> None:
        p = copy.deepcopy(self.nominal_params)
        scales: Dict[str, float] = {}

        for field_name in [
            "g", "m1", "m2", "l1", "l1cg", "l2cg", "I1z", "I2x", "I2y", "I2z",
            "c_theta", "c_alpha", "k_t", "k_b", "k_u", "R",
        ]:
            key = f"{field_name}_scale"
            s = self._sample_scale(key, (1.0, 1.0))
            setattr(p, field_name, float(getattr(p, field_name)) * s)
            scales[key] = s

        self.params = p
        self.cfg.physical_params = p
        self.episode_randomization.physical_param_scales = scales

    def _sample_episode_actuator_and_sensor(self) -> None:
        dr = self.randomization
        level = float(np.clip(self.randomization_level, 0.0, 1.0))
        snap = EpisodeRandomizationSnapshot(level=level)
        snap.physical_param_scales = dict(self.episode_randomization.physical_param_scales)

        if not getattr(dr, "enabled", False) or level <= 0.0:
            snap.pwm_limit = self.nominal_pwm_limit
            self.episode_randomization = snap
            return

        # Actuator parameters.
        lo, hi = self._scaled_range_around_one(*dr.pwm_limit_scale_range)
        snap.pwm_limit = float(self.nominal_pwm_limit * self._sample_uniform(lo, hi))
        lo, hi = self._scaled_range_around_one(*dr.pwm_gain_range)
        snap.pwm_gain = self._sample_uniform(lo, hi)
        lo, hi = self._scaled_abs_range_from_zero(*dr.pwm_bias_range)
        snap.pwm_bias = self._sample_uniform(lo, hi)
        lo, hi = self._scaled_abs_range_from_zero(*dr.pwm_deadzone_range)
        snap.pwm_deadzone = max(0.0, self._sample_uniform(lo, hi))
        lo, hi = self._scaled_abs_range_from_zero(*dr.pwm_noise_sigma_range)
        snap.pwm_noise_sigma = max(0.0, self._sample_uniform(lo, hi))
        lo, hi = self._scaled_abs_range_from_zero(*dr.actuator_tau_range)
        snap.actuator_tau = max(0.0, self._sample_uniform(lo, hi))
        d_lo, d_hi = dr.action_delay_steps_range
        d_max = int(round(level * int(d_hi)))
        snap.action_delay_steps = int(self.np_random.integers(int(d_lo), max(d_lo, d_max) + 1))

        # Sensor parameters.
        for name in [
            "theta_bias", "alpha_bias", "theta_sigma", "alpha_sigma",
            "theta_dot_sigma", "alpha_dot_sigma", "encoder_quantization_rad",
        ]:
            range_name = f"{name}_range"
            lo, hi = self._scaled_abs_range_from_zero(*getattr(dr, range_name))
            val = self._sample_uniform(lo, hi)
            if "sigma" in name or "quantization" in name:
                val = max(0.0, val)
            setattr(snap, name, float(val))

        # Velocity-estimator randomization.
        prob = float(np.clip(dr.use_lpf_velocity_probability * level, 0.0, 1.0))
        snap.use_lpf_velocity = bool(self.np_random.random() < prob)
        lo, hi = dr.velocity_lpf_range
        snap.velocity_lpf = self._sample_uniform(lo, hi)

        self.episode_randomization = snap

    def _randomize_initial_distribution(self) -> None:
        """Curriculum over initial perturbations for upright balance."""
        dr = self.randomization
        if not getattr(dr, "enabled", False):
            return
        level = float(np.clip(self.randomization_level, 0.0, 1.0))
        self.init_cfg.theta_std_deg = self._sample_uniform(
            dr.init_theta_std_deg_range[0],
            dr.init_theta_std_deg_range[0] + level * (dr.init_theta_std_deg_range[1] - dr.init_theta_std_deg_range[0]),
        )
        self.init_cfg.alpha_std_deg = self._sample_uniform(
            dr.init_alpha_std_deg_range[0],
            dr.init_alpha_std_deg_range[0] + level * (dr.init_alpha_std_deg_range[1] - dr.init_alpha_std_deg_range[0]),
        )
        self.init_cfg.theta_dot_std = self._sample_uniform(
            dr.init_theta_dot_std_range[0],
            dr.init_theta_dot_std_range[0] + level * (dr.init_theta_dot_std_range[1] - dr.init_theta_dot_std_range[0]),
        )
        self.init_cfg.alpha_dot_std = self._sample_uniform(
            dr.init_alpha_dot_std_range[0],
            dr.init_alpha_dot_std_range[0] + level * (dr.init_alpha_dot_std_range[1] - dr.init_alpha_dot_std_range[0]),
        )

    def _reset_actuator_state(self) -> None:
        delay = max(0, int(self.episode_randomization.action_delay_steps))
        self._cmd_delay_queue = deque([0.0 for _ in range(delay + 1)], maxlen=delay + 1)
        self._actuator_prev_pwm = 0.0
        self._last_commanded_pwm = 0.0
        self._last_delayed_pwm = 0.0
        self._last_effective_pwm = 0.0
        self.continuous_pwm_limit = float(self.episode_randomization.pwm_limit)
        self.cfg.continuous_pwm_limit = float(self.episode_randomization.pwm_limit)

    # ------------------------------------------------------------
    # Overrides used by base env
    # ------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the freshly seeded RNG before sampling randomized parameters.
        # Calling gym.Env.reset only seeds np_random; it does not run the base
        # CartPoleRIPEenv reset logic.  The parent reset is called once below.
        if seed is not None:
            gym.Env.reset(self, seed=seed)

        self.randomization = getattr(self.cfg, "randomization", self.randomization)
        self.randomization_level = float(np.clip(getattr(self.randomization, "level", self.randomization_level), 0.0, 1.0))
        self._apply_episode_physical_randomization()
        self._sample_episode_actuator_and_sensor()
        self._randomize_initial_distribution()

        # Randomly enable the LPF velocity estimator per episode if configured.
        self.use_lpf_velocity = bool(self.episode_randomization.use_lpf_velocity)
        self.cfg.use_lpf_velocity = bool(self.episode_randomization.use_lpf_velocity)
        self.velocity_lpf = float(self.episode_randomization.velocity_lpf)
        self.cfg.velocity_lpf = float(self.episode_randomization.velocity_lpf)

        self._reset_actuator_state()
        obs, info = super().reset(seed=None, options=options)
        info["domain_randomization"] = self.episode_randomization.to_dict()
        return obs, info

    def _action_to_pwm(self, action) -> float:
        commanded = super()._action_to_pwm(action)
        snap = self.episode_randomization
        self._last_commanded_pwm = float(commanded)

        # Delay: use the oldest queued command after appending current command.
        self._cmd_delay_queue.append(float(commanded))
        delayed = float(self._cmd_delay_queue[0])
        self._last_delayed_pwm = delayed

        u = snap.pwm_gain * delayed + snap.pwm_bias

        if snap.pwm_noise_sigma > 0.0:
            u += float(self.np_random.normal(0.0, snap.pwm_noise_sigma))

        dz = max(0.0, float(snap.pwm_deadzone))
        if abs(u) <= dz:
            u = 0.0
        else:
            u = math.copysign(abs(u) - dz, u)

        tau = max(0.0, float(snap.actuator_tau))
        if tau > 1e-9:
            alpha = float(self.dt / (tau + self.dt))
            u = self._actuator_prev_pwm + alpha * (u - self._actuator_prev_pwm)

        u = float(np.clip(u, -snap.pwm_limit, snap.pwm_limit))
        self._actuator_prev_pwm = u
        self._last_effective_pwm = u
        return u

    def _apply_observation_noise(self, state: np.ndarray) -> np.ndarray:
        noisy = np.asarray(state, dtype=np.float64).reshape(4).copy()
        snap = self.episode_randomization

        # Existing fixed noise from EnvConfig remains available for old scripts.
        if getattr(self.noise_cfg, "enabled", False):
            noisy = super()._apply_observation_noise(noisy)

        noisy[0] += snap.theta_bias
        noisy[2] = wrap_to_pi(float(noisy[2] + snap.alpha_bias))

        if snap.theta_sigma > 0.0:
            noisy[0] += float(self.np_random.normal(0.0, snap.theta_sigma))
        if snap.theta_dot_sigma > 0.0:
            noisy[1] += float(self.np_random.normal(0.0, snap.theta_dot_sigma))
        if snap.alpha_sigma > 0.0:
            noisy[2] = wrap_to_pi(float(noisy[2] + self.np_random.normal(0.0, snap.alpha_sigma)))
        if snap.alpha_dot_sigma > 0.0:
            noisy[3] += float(self.np_random.normal(0.0, snap.alpha_dot_sigma))

        q = max(0.0, float(snap.encoder_quantization_rad))
        if q > 1e-12:
            noisy[0] = round(float(noisy[0]) / q) * q
            noisy[2] = wrap_to_pi(round(float(noisy[2]) / q) * q)

        return noisy

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["commanded_pwm"] = float(self._last_commanded_pwm)
        info["delayed_pwm"] = float(self._last_delayed_pwm)
        info["effective_pwm"] = float(self._last_effective_pwm)
        info["domain_randomization"] = self.episode_randomization.to_dict()
        return obs, reward, terminated, truncated, info
