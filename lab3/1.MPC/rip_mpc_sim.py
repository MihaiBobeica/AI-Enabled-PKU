
"""
TA Ju Zhixiang
Class 9: MPC control of a rotary inverted pendulum
===================================================
STUDENT VERSION.

This file is a minimal Class-9 modification of the Class-8 nonlinear digital
Twin.  The 200 Hz timing, Furuta dynamics, swing-up logic, noise model, 3D
visualisation, SAVE/GO workflow and result plots are intentionally retained.
Only the upright stabiliser is changed from dual-loop PID to constrained linear MPC.

State order:
    x = [theta, theta_dot, alpha, alpha_dot]^T
The panel configures N, diagonal Q, R, solver iterations and state estimation.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

# Keep imports backend-neutral until command-line arguments are parsed.
import matplotlib
from matplotlib.figure import Figure


# ============================================================
# ROSRIP-aligned constants
# ============================================================

CONTROL_DT = 0.005          # 200 Hz, same as firmware
DEFAULT_DURATION = 10.0
DEFAULT_PWM_LIMIT = 150.0

ARM_LENGTH = 0.18
PEND_LENGTH = 0.24
MOTOR_RADIUS = 0.035
MOTOR_HEIGHT = 0.08
ARM_Z = MOTOR_HEIGHT
PEND_TANGENTIAL_SIGN = 1.0

MODE_DISABLED = 0
MODE_SWING_MPC = 1
MODE_BLEND = 2
MODE_MPC = 3
MODE_NAMES = {
    MODE_DISABLED: "DISABLED",
    MODE_SWING_MPC: "SWING_PUMP",
    MODE_BLEND: "BLEND",
    MODE_MPC: "MPC",
}


# ============================================================
# Physical model copied into this single file
# ============================================================

@dataclass
class RIPPhysicalParams:
    """Furuta / rotary inverted pendulum physical parameters."""

    g: float = 9.8

    c_theta: float = 0.025
    c_alpha: float = 0.001

    k_t: float = 0.2310
    k_b: float = 0.1875
    k_u: float = 0.04706
    R: float = 4.2857

    m1: float = 0.20625
    m2: float = 0.15845
    l1cg: float = 0.080305
    l1: float = 0.151894
    l2cg: float = 0.066733
    I1z: float = 0.00049228
    I2x: float = 0.00036892
    I2y: float = 2.3641e-05
    I2z: float = 0.00036139

    @property
    def K1(self) -> float:
        return self.k_t * self.k_u / self.R

    @property
    def K2(self) -> float:
        return self.k_t * self.k_b / self.R


def wrap_to_pi(x: float) -> float:
    return (float(x) + math.pi) % (2.0 * math.pi) - math.pi


def furuta_derivatives(state: np.ndarray, pwm: float, p: RIPPhysicalParams) -> np.ndarray:
    """
    Nonlinear Furuta-pendulum dynamics.

    State:
        [theta, theta_dot, alpha, alpha_dot]

    Convention:
        alpha = 0      upright
        alpha = +/-pi  downward
    """
    theta, theta_dot, alpha, alpha_dot = [float(v) for v in state]

    s_a = math.sin(alpha)
    c_a = math.cos(alpha)
    s_a_c_a = s_a * c_a

    a11 = (
        p.m1 * p.l1cg * p.l1cg
        + p.I1z
        + p.m2 * p.l1 * p.l1
        + (p.m2 * p.l2cg * p.l2cg + p.I2z) * (s_a * s_a)
        + p.I2y * (c_a * c_a)
    )
    b12 = p.m2 * p.l1 * p.l2cg * c_a
    c_term = -p.m2 * p.l1 * p.l2cg * s_a
    d_term = 2.0 * (p.I2z + p.m2 * p.l2cg * p.l2cg - p.I2y) * s_a_c_a

    motor_torque = p.K1 * float(pwm) - p.K2 * theta_dot - p.c_theta * theta_dot

    f22 = -(p.m2 * p.l2cg * p.l2cg + p.I2x)
    g21 = -(p.m2 * p.l1 * p.l2cg * c_a)
    h_term = (p.m2 * p.l2cg * p.l2cg - p.I2y + p.I2z) * s_a_c_a
    gravity_term = p.m2 * p.g * p.l2cg * s_a
    pend_friction = p.c_alpha * alpha_dot

    den1 = a11 * f22 - g21 * b12
    den2 = g21 * b12 - a11 * f22
    eps = 1e-12
    if abs(den1) < eps:
        den1 = eps if den1 >= 0.0 else -eps
    if abs(den2) < eps:
        den2 = eps if den2 >= 0.0 else -eps

    theta_ddot = (
        (-f22 * c_term) * (alpha_dot ** 2)
        + (-f22 * d_term) * alpha_dot * theta_dot
        + (b12 * h_term) * (theta_dot ** 2)
        + (b12 * gravity_term + f22 * motor_torque - b12 * pend_friction)
    ) / den1

    alpha_ddot = (
        (-g21 * c_term) * (alpha_dot ** 2)
        + (-g21 * d_term) * alpha_dot * theta_dot
        + (a11 * h_term) * (theta_dot ** 2)
        + (a11 * gravity_term + g21 * motor_torque - a11 * pend_friction)
    ) / den2

    return np.array([theta_dot, theta_ddot, alpha_dot, alpha_ddot], dtype=np.float64)


def rk4_step(state: np.ndarray, pwm: float, dt: float, p: RIPPhysicalParams) -> np.ndarray:
    k1 = furuta_derivatives(state, pwm, p)
    k2 = furuta_derivatives(state + 0.5 * dt * k1, pwm, p)
    k3 = furuta_derivatives(state + 0.5 * dt * k2, pwm, p)
    k4 = furuta_derivatives(state + dt * k3, pwm, p)

    nxt = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    # Match the supplied environment: alpha is wrapped; theta remains unwrapped.
    nxt[2] = wrap_to_pi(float(nxt[2]))
    return nxt


# ============================================================
# Hybrid swing-up + constrained linear MPC controller
# ============================================================

# Discrete linear model obtained by zero-order-hold discretisation of the
# nonlinear Furuta model above at Ts = 0.005 s.  These are the matrices used
# by the validated legacy MPC firmware.
MPC_AD = np.array(
    [
        [1.0, 0.0048612691704992, -0.0006112340140108, 0.0000048745269118],
        [0.0, 0.9450502885145126, -0.2419612332597028, 0.0017237656430326],
        [0.0, 0.0002070777337968, 1.0021161748142602, 0.0049831142945805],
        [0.0, 0.0819731603634566, 0.8422019829266282, 0.9939886689544905],
    ],
    dtype=np.float64,
)
MPC_BD = np.array(
    [[0.0000100237507910], [0.0039702942449759], [-0.0000149620355143], [-0.0059228257625436]],
    dtype=np.float64,
)

# Luenberger observer copied from the previously supplied MPC firmware.
OBS_A_LC = np.array(
    [
        [0.82469568, 0.00438721, -0.31915638, -0.00085553],
        [-1.75095359, 0.94056709, -0.04181246, 0.00193564],
        [-0.05215683, 0.00005656, 0.76549687, 0.00439733],
        [-1.38405407, 0.07685667, -15.83946960, 0.95027468],
    ],
    dtype=np.float64,
)
OBS_BD_U = np.array(
    [0.00001119, 0.00396389, -0.00001398, -0.00583846],
    dtype=np.float64,
)
OBS_L = np.array(
    [
        [0.17530432, 0.31833008],
        [1.75095359, -0.19974721],
        [0.05215683, 0.23646539],
        [1.38405407, 16.66931771],
    ],
    dtype=np.float64,
)

ESTIMATOR_DIFFERENTIAL = "differential"
ESTIMATOR_LUENBERGER = "luenberger"


@dataclass
class MPCConfig:
    horizon: int = 8
    q_theta: float = 1.0
    q_theta_dot: float = 0.05
    q_alpha: float = 80.0
    q_alpha_dot: float = 2.0
    r_input: float = 0.001
    pgd_iterations: int = 16
    estimator: str = ESTIMATOR_DIFFERENTIAL
    velocity_lpf: float = 0.25

    def validate(self) -> None:
        if not 4 <= int(self.horizon) <= 20:
            raise ValueError("Prediction horizon N must be between 4 and 20.")
        weights = [
            self.q_theta,
            self.q_theta_dot,
            self.q_alpha,
            self.q_alpha_dot,
        ]
        if any((not math.isfinite(float(v))) or float(v) < 0.0 for v in weights):
            raise ValueError("All four state weights must be finite and non-negative.")
        if sum(float(v) for v in weights) <= 0.0:
            raise ValueError("At least one state weight must be positive.")
        if not math.isfinite(float(self.r_input)) or float(self.r_input) <= 0.0:
            raise ValueError("Input weight R must be finite and greater than zero.")
        if not 1 <= int(self.pgd_iterations) <= 100:
            raise ValueError("PGD iterations must be between 1 and 100.")
        if self.estimator not in {ESTIMATOR_DIFFERENTIAL, ESTIMATOR_LUENBERGER}:
            raise ValueError("Unknown state estimator selection.")
        if not 0.001 <= float(self.velocity_lpf) <= 1.0:
            raise ValueError("Velocity LPF beta must be in [0.001, 1].")

    def q_matrix(self) -> np.ndarray:
        return np.diag(
            [self.q_theta, self.q_theta_dot, self.q_alpha, self.q_alpha_dot]
        ).astype(np.float64)


TEACHER_DEMO_MPC = MPCConfig(
    horizon=8,
    q_theta=1.0,
    q_theta_dot=0.05,
    q_alpha=80.0,
    q_alpha_dot=2.0,
    r_input=0.001,
    pgd_iterations=16,
    estimator=ESTIMATOR_LUENBERGER,
    velocity_lpf=0.25,
)


@dataclass
class ControllerConfig:
    pwm_limit: float = DEFAULT_PWM_LIMIT
    swing_pwm: float = 120.0
    kick_time: float = 0.10
    enter_deg: float = 15.0
    exit_deg: float = 25.0
    blend_alpha: float = 0.18


@dataclass
class NoiseConfig:
    """Independent white Gaussian noise applied once per 200 Hz control step.

    Only theta and alpha are measured.  The selected estimator reconstructs
    angular velocity from those noisy angle samples.
    """

    enabled: bool = True
    theta_obs_mean: float = 0.030
    theta_obs_std: float = 0.020
    alpha_obs_mean: float = 0.015
    alpha_obs_std: float = 0.010
    pwm_mean: float = 5.0
    pwm_std: float = 2.5


@dataclass
class SimulationSettings:
    """Settings committed by the GUI SAVE button."""

    initial_name: str
    duration: float
    playback_speed: float
    mpc_cfg: MPCConfig
    controller_cfg: ControllerConfig
    noise_cfg: NoiseConfig
    save_csv: bool
    output_dir: str


def solve_discrete_are_iterative(
    a: np.ndarray,
    b: np.ndarray,
    q: np.ndarray,
    r: float,
    max_iterations: int = 20000,
    tolerance: float = 1.0e-11,
) -> np.ndarray:
    """Solve the scalar-input DARE without requiring SciPy."""
    p = np.asarray(q, dtype=np.float64).copy()
    for _ in range(max_iterations):
        denominator = float(r + (b.T @ p @ b)[0, 0])
        if not math.isfinite(denominator) or denominator <= 1.0e-15:
            raise ValueError("Riccati iteration encountered a non-positive denominator.")
        gain = (b.T @ p @ a) / denominator
        p_next = q + a.T @ p @ a - (a.T @ p @ b) @ gain
        p_next = 0.5 * (p_next + p_next.T)
        if not np.all(np.isfinite(p_next)):
            raise ValueError("Riccati iteration became non-finite.")
        if float(np.max(np.abs(p_next - p))) < tolerance:
            return p_next
        p = p_next
    raise ValueError("Riccati iteration did not converge for the selected Q/R.")


def build_condensed_mpc_matrices(config: MPCConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return H, G, terminal P and a conservative PGD step.

    Objective convention:
        0.5 U.T H U + (G x0).T U
    subject to the box input constraint applied by projected gradient descent.
    """
    config.validate()
    n = int(config.horizon)
    q = config.q_matrix()
    p = solve_discrete_are_iterative(MPC_AD, MPC_BD, q, float(config.r_input))

    nx = 4
    sx = np.zeros((n * nx, nx), dtype=np.float64)
    su = np.zeros((n * nx, n), dtype=np.float64)
    for k in range(n):
        sx[k * nx : (k + 1) * nx, :] = np.linalg.matrix_power(MPC_AD, k + 1)
        for j in range(k + 1):
            su[k * nx : (k + 1) * nx, j] = (
                np.linalg.matrix_power(MPC_AD, k - j) @ MPC_BD
            )[:, 0]

    qbar = np.kron(np.eye(n, dtype=np.float64), q)
    qbar[-nx:, -nx:] = p
    h = 2.0 * (
        su.T @ qbar @ su + float(config.r_input) * np.eye(n, dtype=np.float64)
    )
    g = 2.0 * su.T @ qbar @ sx
    h = 0.5 * (h + h.T)
    row_sum_bound = float(np.max(np.sum(np.abs(h), axis=1)))
    if not math.isfinite(row_sum_bound) or row_sum_bound <= 1.0e-15:
        raise ValueError("Invalid MPC Hessian generated from the selected parameters.")
    pgd_step = 0.95 / row_sum_bound
    return h, g, p, pgd_step


class ProjectedGradientMPC:
    def __init__(self, config: MPCConfig, pwm_limit: float) -> None:
        config.validate()
        self.config = config
        self.pwm_limit = float(pwm_limit)
        self.h, self.g, self.terminal_p, self.step = build_condensed_mpc_matrices(config)
        self.sequence = np.zeros(int(config.horizon), dtype=np.float64)

    def reset(self) -> None:
        self.sequence.fill(0.0)

    def compute(self, state: np.ndarray) -> float:
        x = np.asarray(state, dtype=np.float64).reshape(4).copy()
        x[2] = wrap_to_pi(float(x[2]))
        u = np.empty_like(self.sequence)
        u[:-1] = self.sequence[1:]
        u[-1] = self.sequence[-1]
        np.clip(u, -self.pwm_limit, self.pwm_limit, out=u)
        linear = self.g @ x
        for _ in range(int(self.config.pgd_iterations)):
            u -= self.step * (self.h @ u + linear)
            np.clip(u, -self.pwm_limit, self.pwm_limit, out=u)
        if not np.all(np.isfinite(u)):
            u.fill(0.0)
        self.sequence[:] = u
        return float(np.clip(u[0], -self.pwm_limit, self.pwm_limit))


class DifferentialVelocityEstimator:
    def __init__(self, beta: float) -> None:
        self.beta = float(beta)
        self.reset()

    def reset(self) -> None:
        self.ready = False
        self.previous_theta = 0.0
        self.previous_alpha = 0.0
        self.theta_dot = 0.0
        self.alpha_dot = 0.0

    def update(self, theta: float, alpha: float, dt: float) -> np.ndarray:
        theta = float(theta)
        alpha = wrap_to_pi(float(alpha))
        if not self.ready:
            self.previous_theta = theta
            self.previous_alpha = alpha
            self.ready = True
        else:
            raw_theta_dot = (theta - self.previous_theta) / float(dt)
            raw_alpha_dot = wrap_to_pi(alpha - self.previous_alpha) / float(dt)
            self.theta_dot += self.beta * (raw_theta_dot - self.theta_dot)
            self.alpha_dot += self.beta * (raw_alpha_dot - self.alpha_dot)
            self.previous_theta = theta
            self.previous_alpha = alpha
        return np.array([theta, self.theta_dot, alpha, self.alpha_dot], dtype=np.float64)


class LuenbergerObserver:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.ready = False
        self.state = np.zeros(4, dtype=np.float64)

    def initialize(self, theta: float, alpha: float) -> None:
        self.state[:] = [float(theta), 0.0, wrap_to_pi(float(alpha)), 0.0]
        self.ready = True

    def update(self, previous_input: float, theta: float, alpha: float) -> np.ndarray:
        theta = float(theta)
        alpha = wrap_to_pi(float(alpha))
        if not self.ready:
            self.initialize(theta, alpha)
            return self.state.copy()
        measurement = np.array([theta, alpha], dtype=np.float64)
        next_state = OBS_A_LC @ self.state + OBS_BD_U * float(previous_input) + OBS_L @ measurement
        if (
            not np.all(np.isfinite(next_state))
            or abs(float(next_state[2] - alpha)) > math.radians(35.0)
        ):
            self.initialize(theta, alpha)
            return self.state.copy()
        next_state[1] = float(np.clip(next_state[1], -500.0, 500.0))
        next_state[3] = float(np.clip(next_state[3], -500.0, 500.0))
        next_state[2] = wrap_to_pi(float(next_state[2]))
        self.state[:] = next_state
        return self.state.copy()


class HybridMPCController:
    """Energy swing-up plus constrained local MPC stabilisation."""

    def __init__(self, mpc_cfg: MPCConfig, cfg: ControllerConfig) -> None:
        mpc_cfg.validate()
        self.mpc_cfg = mpc_cfg
        self.cfg = cfg
        self.mpc = ProjectedGradientMPC(mpc_cfg, cfg.pwm_limit)
        self.differential = DifferentialVelocityEstimator(mpc_cfg.velocity_lpf)
        self.observer = LuenbergerObserver()
        self.reset()

    def reset(self) -> None:
        self.upright_mode = False
        self.blend = 0.0
        self.mode = MODE_DISABLED
        self.capture_time: Optional[float] = None
        self.previous_input = 0.0
        self.mpc.reset()
        self.differential.reset()
        self.observer.reset()

    def accept_applied_pwm(self, pwm: float) -> None:
        self.previous_input = float(pwm)

    def _swing_pwm(self, t: float, alpha: float, alpha_dot: float) -> float:
        if t < self.cfg.kick_time:
            return float(self.cfg.swing_pwm)
        phase = alpha_dot * math.cos(alpha)
        direction = 1.0 if phase >= 0.0 else -1.0
        return -float(self.cfg.swing_pwm) * direction

    def compute(self, theta_measured: float, alpha_measured: float, t: float, dt: float) -> Tuple[float, Dict[str, float]]:
        alpha_measured = wrap_to_pi(float(alpha_measured))
        differential_state = self.differential.update(theta_measured, alpha_measured, dt)
        abs_alpha = abs(alpha_measured)
        enter = math.radians(self.cfg.enter_deg)
        exit_ = math.radians(self.cfg.exit_deg)

        previous_upright = self.upright_mode
        if (not self.upright_mode) and abs_alpha <= enter:
            self.upright_mode = True
            self.mpc.reset()
            if self.mpc_cfg.estimator == ESTIMATOR_LUENBERGER:
                self.observer.initialize(theta_measured, alpha_measured)
            if self.capture_time is None:
                self.capture_time = float(t)
        elif self.upright_mode and abs_alpha >= exit_:
            self.upright_mode = False
            self.mpc.reset()
            self.observer.reset()

        if self.mpc_cfg.estimator == ESTIMATOR_LUENBERGER and self.upright_mode:
            estimated_state = self.observer.update(
                self.previous_input, theta_measured, alpha_measured
            )
        else:
            estimated_state = differential_state

        u_swing = self._swing_pwm(t, alpha_measured, float(differential_state[3]))
        u_mpc = self.mpc.compute(estimated_state) if self.upright_mode or self.blend > 1.0e-6 else 0.0

        target = 1.0 if self.upright_mode else 0.0
        self.blend += self.cfg.blend_alpha * (target - self.blend)
        self.blend = float(np.clip(self.blend, 0.0, 1.0))
        command = (1.0 - self.blend) * u_swing + self.blend * u_mpc
        command = float(np.clip(command, -self.cfg.pwm_limit, self.cfg.pwm_limit))

        if self.blend <= 0.01:
            self.mode = MODE_SWING_MPC
        elif self.blend >= 0.99:
            self.mode = MODE_MPC
        else:
            self.mode = MODE_BLEND

        details = {
            "u_swing": float(u_swing),
            "u_mpc": float(u_mpc),
            "blend": float(self.blend),
            "upright": float(self.upright_mode),
            "mode": float(self.mode),
            "entered_now": float((not previous_upright) and self.upright_mode),
            "theta_hat": float(estimated_state[0]),
            "theta_dot_hat": float(estimated_state[1]),
            "alpha_hat": float(estimated_state[2]),
            "alpha_dot_hat": float(estimated_state[3]),
        }
        return command, details


# ============================================================
# Simulation and tuning helpers# ============================================================
# Simulation and tuning helpers
# ============================================================

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
    capture_time: Optional[float]


def initial_state_from_name(name: str) -> np.ndarray:
    key = name.strip().lower()
    if key in {"downward", "down", "downward swing-up"}:
        return np.array([0.0, 0.0, math.pi, 0.0], dtype=np.float64)
    if key in {"upright+8", "+8", "near upright +8°"}:
        return np.array([0.0, 0.0, math.radians(8.0), 0.0], dtype=np.float64)
    if key in {"upright-8", "-8", "near upright -8°"}:
        return np.array([0.0, 0.0, math.radians(-8.0), 0.0], dtype=np.float64)
    raise ValueError(f"Unknown initial condition: {name}")


def noisy_angle_measurements(
    true_state: np.ndarray,
    noise_cfg: NoiseConfig,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """Return the only two quantities available to a real controller."""
    theta = float(true_state[0])
    alpha = float(true_state[2])
    if noise_cfg.enabled:
        theta += float(rng.normal(noise_cfg.theta_obs_mean, noise_cfg.theta_obs_std))
        alpha += float(rng.normal(noise_cfg.alpha_obs_mean, noise_cfg.alpha_obs_std))
    return theta, wrap_to_pi(alpha)


def noisy_applied_pwm(
    pwm_command: float,
    noise_cfg: NoiseConfig,
    pwm_limit: float,
    rng: np.random.Generator,
) -> float:
    """Apply Gaussian actuator error after controller computation."""
    pwm = float(pwm_command)
    if noise_cfg.enabled:
        pwm += float(rng.normal(noise_cfg.pwm_mean, noise_cfg.pwm_std))
    return float(np.clip(pwm, -float(pwm_limit), float(pwm_limit)))


def run_simulation(
    mpc_cfg: MPCConfig,
    controller_cfg: ControllerConfig,
    duration: float = DEFAULT_DURATION,
    dt: float = CONTROL_DT,
    initial_state: Optional[np.ndarray] = None,
    physical_params: Optional[RIPPhysicalParams] = None,
    noise_config: Optional[NoiseConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> SimulationResult:
    p = physical_params if physical_params is not None else RIPPhysicalParams()
    noise_cfg = noise_config if noise_config is not None else NoiseConfig()
    noise_rng = rng if rng is not None else np.random.default_rng()
    controller = HybridMPCController(mpc_cfg, controller_cfg)
    state = (
        np.asarray(initial_state, dtype=np.float64).reshape(4).copy()
        if initial_state is not None
        else initial_state_from_name("downward")
    )

    steps = int(round(float(duration) / float(dt)))
    data = np.zeros((steps + 1, 8), dtype=np.float64)
    data[0, 0] = 0.0
    data[0, 1:5] = state
    data[0, 5] = 0.0
    data[0, 6] = MODE_DISABLED
    data[0, 7] = 0.0

    for k in range(steps):
        t = k * dt
        theta_measured, alpha_measured = noisy_angle_measurements(state, noise_cfg, noise_rng)
        pwm_command, details = controller.compute(theta_measured, alpha_measured, t, dt)
        pwm_applied = noisy_applied_pwm(
            pwm_command,
            noise_cfg,
            controller_cfg.pwm_limit,
            noise_rng,
        )
        controller.accept_applied_pwm(pwm_applied)
        state = rk4_step(state, pwm_applied, dt, p)
        if not np.all(np.isfinite(state)):
            raise FloatingPointError(f"Simulation became non-finite at t={t:.3f} s")

        data[k + 1, 0] = (k + 1) * dt
        data[k + 1, 1:5] = state
        data[k + 1, 5] = pwm_applied
        data[k + 1, 6] = details["mode"]
        data[k + 1, 7] = details["blend"]

    return SimulationResult(
        time=data[:, 0],
        theta=data[:, 1],
        theta_dot=data[:, 2],
        alpha=data[:, 3],
        alpha_dot=data[:, 4],
        pwm=data[:, 5],
        mode=data[:, 6].astype(np.int32),
        blend=data[:, 7],
        capture_time=controller.capture_time,
    )


# ============================================================
# ROSRIP-style result output# ============================================================
# ROSRIP-style result output
# ============================================================


def stable_phase_start_index(
    result: SimulationResult,
    threshold_deg: float = 15.0,
) -> Optional[int]:
    """Return the start index of the final continuous stable phase.

    The stable phase is defined as the earliest sample in the final suffix for
    which |alpha| <= threshold_deg at every remaining sample until the end of
    the run. If the final sample is outside the threshold, no stable phase is
    reported.
    """
    if result.alpha.size == 0:
        return None

    inside = np.abs(result.alpha) <= math.radians(float(threshold_deg))
    outside_indices = np.flatnonzero(~inside)
    start_index = int(outside_indices[-1] + 1) if outside_indices.size else 0

    if start_index >= inside.size:
        return None
    return start_index


def result_metrics(result: SimulationResult) -> Dict[str, float]:
    stable_index = stable_phase_start_index(result, threshold_deg=15.0)

    if stable_index is None:
        stable_start_time = math.nan
        stable_duration = 0.0
        alpha_abs_mean = math.nan
        alpha_abs_std = math.nan
        pwm_abs_mean = math.nan
        pwm_abs_std = math.nan
    else:
        stable_alpha_abs = np.abs(result.alpha[stable_index:])
        stable_pwm_abs = np.abs(result.pwm[stable_index:])
        stable_start_time = float(result.time[stable_index])
        stable_duration = float(result.time[-1] - result.time[stable_index])
        alpha_abs_mean = float(np.mean(stable_alpha_abs))
        alpha_abs_std = float(np.std(stable_alpha_abs))
        pwm_abs_mean = float(np.mean(stable_pwm_abs))
        pwm_abs_std = float(np.std(stable_pwm_abs))

    return {
        "stable_start_time": stable_start_time,
        "stable_duration": stable_duration,
        "alpha_abs_mean": alpha_abs_mean,
        "alpha_abs_std": alpha_abs_std,
        "pwm_abs_mean": pwm_abs_mean,
        "pwm_abs_std": pwm_abs_std,
        "max_abs_theta": float(np.max(np.abs(result.theta))),
        "capture_time": float(result.capture_time) if result.capture_time is not None else math.nan,
    }


def build_result_figure(result: SimulationResult, title_suffix: str = "MPC Digital Twin") -> Figure:
    metrics = result_metrics(result)
    fig = Figure(figsize=(10.5, 7.6), tight_layout=True)
    fig.patch.set_facecolor("white")
    axs = fig.subplots(2, 2)
    ax_alpha, ax_theta = axs[0, 0], axs[0, 1]
    ax_pwm, ax_hist = axs[1, 0], axs[1, 1]

    t = result.time
    alpha = result.alpha
    theta = result.theta
    pwm = result.pwm

    fig.suptitle(
        f"Rotary Inverted Pendulum {title_suffix} Response",
        fontsize=15,
        fontweight="bold",
    )

    ax_alpha.plot(t, alpha, linewidth=1.8, label=r"$\alpha$")
    ax_alpha.axhline(0.0, linewidth=1.0, linestyle="--")
    ax_alpha.axhline(math.radians(15.0), linewidth=0.8, linestyle=":")
    ax_alpha.axhline(-math.radians(15.0), linewidth=0.8, linestyle=":")
    ax_alpha.set_title(r"Pendulum Angle $\alpha(t)$", fontsize=12, fontweight="bold")
    ax_alpha.set_xlabel("Time / s")
    ax_alpha.set_ylabel(r"$\alpha$ / rad")
    ax_alpha.grid(True, linestyle="--", linewidth=0.6, alpha=0.55)
    ax_alpha.legend(loc="lower right", frameon=True)
    stable_index = stable_phase_start_index(result, threshold_deg=15.0)
    if stable_index is None:
        alpha_text = (
            "Stable phase: not reached\n"
            r"criterion: $|\alpha|\leq15^\circ$ until the end"
        )
    else:
        alpha_text = (
            r"$\mathrm{stable\ mean}(|\alpha|)$"
            + f" = {metrics['alpha_abs_mean']:.5f} rad\n"
            + f"stable from t = {metrics['stable_start_time']:.3f} s"
        )
        ax_alpha.axvspan(metrics["stable_start_time"], t[-1], alpha=0.10)
    ax_alpha.text(
        0.98, 0.96, alpha_text, transform=ax_alpha.transAxes,
        ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="0.35"),
    )

    ax_theta.plot(t, theta, linewidth=1.8, label=r"$\theta$")
    ax_theta.axhline(0.0, linewidth=1.0, linestyle="--")
    ax_theta.set_title(r"Rotary Arm Angle $\theta(t)$", fontsize=12, fontweight="bold")
    ax_theta.set_xlabel("Time / s")
    ax_theta.set_ylabel(r"$\theta$ / rad")
    ax_theta.grid(True, linestyle="--", linewidth=0.6, alpha=0.55)
    ax_theta.legend(loc="upper right", frameon=True)

    ax_pwm.plot(t, pwm, linewidth=1.5, label="PWM")
    ax_pwm.axhline(0.0, linewidth=1.0, linestyle="--")
    ax_pwm.set_title("Control Input PWM(t)", fontsize=12, fontweight="bold")
    ax_pwm.set_xlabel("Time / s")
    ax_pwm.set_ylabel("PWM")
    lim = max(160.0, float(np.max(np.abs(pwm))) * 1.10)
    ax_pwm.set_ylim(-lim, lim)
    ax_pwm.grid(True, linestyle="--", linewidth=0.6, alpha=0.55)
    ax_pwm.legend(loc="lower right", frameon=True)
    if stable_index is None:
        pwm_text = "Stable phase: not reached"
    else:
        pwm_text = (
            r"$\mathrm{stable\ mean}(|PWM|)$"
            + f" = {metrics['pwm_abs_mean']:.2f}\n"
            + r"$\mathrm{stable\ std}(|PWM|)$"
            + f" = {metrics['pwm_abs_std']:.2f}"
        )
        ax_pwm.axvspan(metrics["stable_start_time"], t[-1], alpha=0.10)
    ax_pwm.text(
        0.98, 0.96, pwm_text, transform=ax_pwm.transAxes,
        ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="0.35"),
    )

    bins = np.arange(-255, 256 + 1, 15)
    if stable_index is None:
        ax_hist.text(
            0.5, 0.5, "No final stable phase",
            transform=ax_hist.transAxes, ha="center", va="center", fontsize=11,
        )
    else:
        ax_hist.hist(
            pwm[stable_index:], bins=bins, edgecolor="black", linewidth=0.45
        )
    ax_hist.axvline(0.0, linewidth=1.0, linestyle="--")
    ax_hist.set_xlim(-255, 255)
    ax_hist.set_title("Stable-stage PWM Distribution", fontsize=12, fontweight="bold")
    ax_hist.set_xlabel("PWM")
    ax_hist.set_ylabel("Count")
    ax_hist.grid(True, linestyle="--", linewidth=0.6, alpha=0.55)

    for ax in (ax_alpha, ax_theta, ax_pwm):
        ax.set_xlim(0.0, max(float(t[-1]), 0.1))
    for ax in (ax_alpha, ax_theta, ax_pwm, ax_hist):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="in", length=4, width=0.8)

    return fig


def save_result_csv(result: SimulationResult, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time_s", "theta_rad", "theta_dot_rad_s", "alpha_rad",
            "alpha_dot_rad_s", "pwm", "mode", "blend",
        ])
        for row in zip(
            result.time, result.theta, result.theta_dot, result.alpha,
            result.alpha_dot, result.pwm, result.mode, result.blend,
        ):
            writer.writerow(row)


def default_output_paths(output_dir: str, duration_s: float) -> Tuple[str, str]:
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    duration_tag = f"{float(duration_s):.3f}".rstrip("0").rstrip(".").replace(".", "p")
    return (
        os.path.join(output_dir, f"rip_mpc_twin_{duration_tag}s_{stamp}.png"),
        os.path.join(output_dir, f"rip_mpc_twin_{duration_tag}s_{stamp}.csv"),
    )


# ============================================================
# GUI
# ============================================================


def rip_points(theta: float, alpha: float):
    center = np.array([0.0, 0.0, ARM_Z])
    e_r = np.array([math.cos(theta), math.sin(theta), 0.0])
    e_t = np.array([-math.sin(theta), math.cos(theta), 0.0])
    e_z = np.array([0.0, 0.0, 1.0])
    joint = center + ARM_LENGTH * e_r
    pend_dir = PEND_TANGENTIAL_SIGN * math.sin(alpha) * e_t + math.cos(alpha) * e_z
    tip = joint + PEND_LENGTH * pend_dir
    ref_tip = joint + PEND_LENGTH * e_z
    return center, joint, tip, ref_tip, e_t


def set_line3d(line, p0, p1) -> None:
    line.set_data([p0[0], p1[0]], [p0[1], p1[1]])
    line.set_3d_properties([p0[2], p1[2]])


def set_point3d(point, p) -> None:
    point.set_data([p[0]], [p[1]])
    point.set_3d_properties([p[2]])



def launch_gui() -> int:
    try:
        from PyQt5 import QtCore, QtGui, QtWidgets
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    except ImportError as exc:
        print(f"GUI dependency missing: {exc}", file=sys.stderr)
        print("Install PyQt5 or run with --headless.", file=sys.stderr)
        return 2

    class RIPDigitalTwinWindow(QtWidgets.QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("RIP MPC Digital Twin | Rotary Inverted Pendulum")
            self.resize(1180, 760)

            self.params = RIPPhysicalParams()
            self.state = initial_state_from_name("downward")
            self.controller: Optional[HybridMPCController] = None
            self.running = False
            self.sim_time = 0.0
            self.duration = DEFAULT_DURATION
            self.result: Optional[SimulationResult] = None
            self.rows: List[List[float]] = []
            self.last_png_path: Optional[str] = None
            self.last_csv_path: Optional[str] = None

            self.saved_settings: Optional[SimulationSettings] = None
            self.run_settings: Optional[SimulationSettings] = None
            self.settings_dirty = False
            self._suppress_dirty = False
            self._last_wall_time = time.perf_counter()
            self._sim_accumulator = 0.0
            self.noise_rng = np.random.default_rng()
            self.help_dialog = None

            self.build_ui()
            self.build_3d_figure()

            # Commit the initial widget values before the first reset.
            self.saved_settings = self.capture_settings_from_ui()
            self.update_saved_summary()
            self.connect_dirty_signals()
            self.reset_simulation()

            self.timer = QtCore.QTimer(self)
            self.timer.setTimerType(QtCore.Qt.PreciseTimer)
            self.timer.timeout.connect(self.on_simulation_tick)
            self.timer.start(16)

        def _dspin(
            self,
            value: float,
            lo: float,
            hi: float,
            decimals: int = 3,
            step: float = 1.0,
        ):
            """Editable numeric input.

            The arrow step is only a convenience. Users may type any value,
            including values such as 111, 113, 124 or 442.3.
            """
            box = QtWidgets.QDoubleSpinBox()
            box.setRange(lo, hi)
            box.setDecimals(decimals)
            box.setSingleStep(step)
            box.setValue(value)
            box.setKeyboardTracking(False)
            box.setAccelerated(False)
            box.setAlignment(QtCore.Qt.AlignRight)
            box.setCorrectionMode(QtWidgets.QAbstractSpinBox.CorrectToPreviousValue)
            box.lineEdit().setClearButtonEnabled(True)
            return box

        def build_ui(self) -> None:
            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            root = QtWidgets.QHBoxLayout(central)

            left_scroll = QtWidgets.QScrollArea()
            left_scroll.setWidgetResizable(True)
            left_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            left_scroll.setMinimumWidth(370)
            left_scroll.setMaximumWidth(430)

            left = QtWidgets.QWidget()
            left_layout = QtWidgets.QVBoxLayout(left)
            left_scroll.setWidget(left)
            root.addWidget(left_scroll)

            group_ctrl = QtWidgets.QGroupBox("Controller")
            form = QtWidgets.QFormLayout(group_ctrl)

            self.combo_initial = QtWidgets.QComboBox()
            self.combo_initial.addItems([
                "Downward swing-up",
                "Near upright +8°",
                "Near upright -8°",
            ])
            self.spin_duration = self._dspin(10.0, 0.1, 600.0, 3, 1.0)
            self.spin_speed = self._dspin(1.0, 0.1, 3.0, 2, 0.1)
            self.spin_speed.setSuffix(" ×")
            self.spin_pwm_limit = self._dspin(150.0, 1.0, 255.0, 3, 1.0)
            self.spin_swing_pwm = self._dspin(120.0, 0.0, 255.0, 3, 1.0)
            self.spin_enter = self._dspin(15.0, 0.1, 89.0, 3, 0.1)
            self.spin_exit = self._dspin(25.0, 0.2, 179.0, 3, 0.1)
            self.spin_blend = self._dspin(0.18, 0.001, 1.0, 4, 0.01)

            form.addRow("Controller profile:", QtWidgets.QLabel("Swing pump → constrained linear MPC"))
            form.addRow("Initial condition:", self.combo_initial)
            form.addRow("Duration / s:", self.spin_duration)
            form.addRow("Playback speed:", self.spin_speed)
            form.addRow("PWM max:", self.spin_pwm_limit)
            form.addRow("Swing pump PWM:", self.spin_swing_pwm)
            form.addRow("MPC enter / deg:", self.spin_enter)
            form.addRow("MPC exit / deg:", self.spin_exit)
            form.addRow("Soft blend λ:", self.spin_blend)

            self.label_profile = QtWidgets.QLabel(
                "200 Hz nonlinear twin. Configure MPC and state estimation, then click SAVE. "
                "Unsaved changes are deliberately not used by GO."
            )
            self.label_profile.setWordWrap(True)
            self.label_profile.setStyleSheet(
                "QLabel { color: #333; background: #f2f2f2; padding: 6px; }"
            )
            form.addRow(self.label_profile)

            self.btn_help = QtWidgets.QPushButton("Linear MPC & State Estimation Help")
            self.btn_help.clicked.connect(self.show_formula_help)
            form.addRow(self.btn_help)

            self.label_saved = QtWidgets.QLabel()
            self.label_saved.setWordWrap(True)
            self.label_saved.setStyleSheet(
                "QLabel { color: #174a7e; background: #edf5ff; padding: 6px; }"
            )
            form.addRow(self.label_saved)
            left_layout.addWidget(group_ctrl)

            group_mpc = QtWidgets.QGroupBox("Linear MPC Design & State Estimation")
            mpc_form = QtWidgets.QFormLayout(group_mpc)
            self.spin_horizon = QtWidgets.QSpinBox()
            self.spin_horizon.setRange(4, 20)
            self.spin_horizon.setValue(8)
            self.spin_horizon.setKeyboardTracking(False)
            self.spin_q_theta = self._dspin(1.0, 0.0, 100000.0, 6, 0.1)
            self.spin_q_theta_dot = self._dspin(0.05, 0.0, 100000.0, 6, 0.01)
            self.spin_q_alpha = self._dspin(80.0, 0.0, 100000.0, 6, 1.0)
            self.spin_q_alpha_dot = self._dspin(2.0, 0.0, 100000.0, 6, 0.1)
            self.spin_r_input = self._dspin(0.001, 0.000001, 1000.0, 6, 0.001)
            self.spin_pgd_iterations = QtWidgets.QSpinBox()
            self.spin_pgd_iterations.setRange(1, 100)
            self.spin_pgd_iterations.setValue(16)
            self.spin_pgd_iterations.setKeyboardTracking(False)
            self.combo_estimator = QtWidgets.QComboBox()
            self.combo_estimator.addItem("Differential + low-pass filter", ESTIMATOR_DIFFERENTIAL)
            self.combo_estimator.addItem("Luenberger observer", ESTIMATOR_LUENBERGER)
            self.spin_velocity_lpf = self._dspin(0.25, 0.001, 1.0, 4, 0.01)

            mpc_form.addRow("Prediction horizon N:", self.spin_horizon)
            mpc_form.addRow("Q theta:", self.spin_q_theta)
            mpc_form.addRow("Q theta-dot:", self.spin_q_theta_dot)
            mpc_form.addRow("Q alpha:", self.spin_q_alpha)
            mpc_form.addRow("Q alpha-dot:", self.spin_q_alpha_dot)
            mpc_form.addRow("Input weight R:", self.spin_r_input)
            mpc_form.addRow("PGD iterations:", self.spin_pgd_iterations)
            mpc_form.addRow("State estimator:", self.combo_estimator)
            mpc_form.addRow("Velocity LPF beta:", self.spin_velocity_lpf)

            mpc_note = QtWidgets.QLabel(
                "At SAVE, the terminal Riccati matrix and condensed QP are rebuilt. "
                "Differential mode uses the LPF value; Luenberger mode uses the fixed "
                "validated observer from the earlier firmware."
            )
            mpc_note.setWordWrap(True)
            mpc_note.setStyleSheet(
                "QLabel { color: #333; background: #f2f2f2; padding: 6px; }"
            )
            mpc_form.addRow(mpc_note)

            self.btn_save = QtWidgets.QPushButton("SAVE / Apply Settings")
            self.btn_save.setMinimumHeight(34)
            self.btn_save.clicked.connect(self.save_settings)
            mpc_form.addRow(self.btn_save)
            left_layout.addWidget(group_mpc)

            group_noise = QtWidgets.QGroupBox("Random Noise (Gaussian)")
            group_noise = QtWidgets.QGroupBox("Random Noise (Gaussian)")
            noise_form = QtWidgets.QFormLayout(group_noise)
            self.check_noise = QtWidgets.QCheckBox("Enable random noise")
            self.check_noise.setChecked(True)

            self.theta_noise_mean = self._dspin(0.030, -3.1416, 3.1416, 6, 0.001)
            self.theta_noise_std = self._dspin(0.020, 0.0, 3.1416, 6, 0.001)
            self.alpha_noise_mean = self._dspin(0.015, -3.1416, 3.1416, 6, 0.001)
            self.alpha_noise_std = self._dspin(0.010, 0.0, 3.1416, 6, 0.001)
            self.pwm_noise_mean = self._dspin(5.0, -255.0, 255.0, 4, 0.1)
            self.pwm_noise_std = self._dspin(2.5, 0.0, 255.0, 4, 0.1)

            noise_form.addRow(self.check_noise)
            noise_form.addRow("Theta observation μ / rad:", self.theta_noise_mean)
            noise_form.addRow("Theta observation σ / rad:", self.theta_noise_std)
            noise_form.addRow("Alpha observation μ / rad:", self.alpha_noise_mean)
            noise_form.addRow("Alpha observation σ / rad:", self.alpha_noise_std)
            noise_form.addRow("PWM output μ / PWM:", self.pwm_noise_mean)
            noise_form.addRow("PWM output σ / PWM:", self.pwm_noise_std)

            self.label_noise = QtWidgets.QLabel(
                "Independent white Gaussian samples are drawn at every 5 ms "
                "control step. μ is bias; σ is standard deviation."
            )
            self.label_noise.setWordWrap(True)
            self.label_noise.setStyleSheet(
                "QLabel { color: #333; background: #f2f2f2; padding: 6px; }"
            )
            noise_form.addRow(self.label_noise)
            left_layout.addWidget(group_noise)

            group_run = QtWidgets.QGroupBox("Simulation")
            run_layout = QtWidgets.QVBoxLayout(group_run)
            row = QtWidgets.QHBoxLayout()
            self.btn_go = QtWidgets.QPushButton("GO")
            self.btn_stop = QtWidgets.QPushButton("STOP")
            self.btn_reset = QtWidgets.QPushButton("RESET")
            row.addWidget(self.btn_go)
            row.addWidget(self.btn_stop)
            row.addWidget(self.btn_reset)
            run_layout.addLayout(row)
            self.btn_go.clicked.connect(self.start_simulation)
            self.btn_stop.clicked.connect(self.stop_simulation)
            self.btn_reset.clicked.connect(self.reset_simulation)

            self.label_status = QtWidgets.QLabel(
                "Link: digital twin | phase: stopped\n"
                "controller: MPC | runtime: DISABLED\n"
                "time: 0.000 / 10.000 s | playback: 1.00×"
            )
            self.label_status.setWordWrap(True)
            self.label_status.setStyleSheet(
                "QLabel { color: #0b5d1e; background: #e8f5e9; padding: 6px; }"
            )
            run_layout.addWidget(self.label_status)

            self.label_state = QtWidgets.QLabel()
            run_layout.addWidget(self.label_state)
            left_layout.addWidget(group_run)

            group_output = QtWidgets.QGroupBox("Result & Logging")
            output_layout = QtWidgets.QVBoxLayout(group_output)
            self.check_save_csv = QtWidgets.QCheckBox("Generate CSV log")
            self.check_save_csv.setChecked(True)
            output_layout.addWidget(self.check_save_csv)

            path_row = QtWidgets.QHBoxLayout()
            self.output_dir = QtWidgets.QLineEdit(os.path.expanduser("~/rip_twin_logs"))
            self.btn_browse_dir = QtWidgets.QPushButton("Browse")
            self.btn_browse_dir.clicked.connect(self.choose_output_directory)
            path_row.addWidget(self.output_dir, stretch=1)
            path_row.addWidget(self.btn_browse_dir)
            output_layout.addLayout(path_row)

            btn_open_result = QtWidgets.QPushButton("Show Last Result Curves")
            btn_open_result.clicked.connect(self.show_result_dialog)
            output_layout.addWidget(btn_open_result)
            left_layout.addWidget(group_output)
            left_layout.addStretch(1)

            self.right_page = QtWidgets.QWidget()
            root.addWidget(self.right_page, stretch=1)

        def build_3d_figure(self) -> None:
            layout = QtWidgets.QVBoxLayout(self.right_page)
            self.fig3d = Figure(figsize=(9, 7), tight_layout=True)
            self.canvas3d = FigureCanvas(self.fig3d)
            layout.addWidget(self.canvas3d)
            self.ax3d = self.fig3d.add_subplot(111, projection="3d")
            self.ax3d.set_title("3D View", pad=2)
            self.ax3d.set_xlabel("X / m")
            self.ax3d.set_ylabel("Y / m")
            self.ax3d.set_zlabel("Z / m")
            limit = ARM_LENGTH + PEND_LENGTH + 0.05
            self.ax3d.set_xlim(-limit, limit)
            self.ax3d.set_ylim(-limit, limit)
            self.ax3d.set_zlim(-0.28, 0.38)
            try:
                self.ax3d.set_box_aspect([1, 1, 1])
            except Exception:
                pass
            self.ax3d.view_init(elev=24, azim=-55)

            ang = np.linspace(0.0, 2.0 * math.pi, 80)
            self.ax3d.plot(
                MOTOR_RADIUS * np.cos(ang), MOTOR_RADIUS * np.sin(ang),
                MOTOR_HEIGHT * np.ones_like(ang), linewidth=2.0,
            )
            self.ax3d.plot(
                MOTOR_RADIUS * np.cos(ang), MOTOR_RADIUS * np.sin(ang),
                np.zeros_like(ang), linewidth=1.4,
            )
            for a in np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False):
                x = MOTOR_RADIUS * math.cos(a)
                y = MOTOR_RADIUS * math.sin(a)
                self.ax3d.plot([x, x], [y, y], [0.0, MOTOR_HEIGHT], linewidth=1.0)
            self.ax3d.plot(
                ARM_LENGTH * np.cos(ang), ARM_LENGTH * np.sin(ang),
                ARM_Z * np.ones_like(ang), linestyle="--", linewidth=1.0,
            )

            self.arm_line, = self.ax3d.plot([], [], [], linewidth=6)
            self.pend_line, = self.ax3d.plot([], [], [], linewidth=5)
            self.joint_dot, = self.ax3d.plot([], [], [], marker="o", markersize=8)
            self.tip_dot, = self.ax3d.plot([], [], [], marker="o", markersize=10)
            self.tangent_line, = self.ax3d.plot([], [], [], linestyle=":", linewidth=2)
            self.ref_line, = self.ax3d.plot([], [], [], linestyle="--", linewidth=2.5)
            self.state_text = self.ax3d.text2D(
                0.03, 0.89, "", transform=self.ax3d.transAxes, fontsize=11
            )

        def numeric_widgets(self):
            return [
                self.spin_duration,
                self.spin_speed,
                self.spin_pwm_limit,
                self.spin_swing_pwm,
                self.spin_enter,
                self.spin_exit,
                self.spin_blend,
                self.spin_horizon,
                self.spin_q_theta,
                self.spin_q_theta_dot,
                self.spin_q_alpha,
                self.spin_q_alpha_dot,
                self.spin_r_input,
                self.spin_pgd_iterations,
                self.spin_velocity_lpf,
                self.theta_noise_mean,
                self.theta_noise_std,
                self.alpha_noise_mean,
                self.alpha_noise_std,
                self.pwm_noise_mean,
                self.pwm_noise_std,
            ]

        def commit_editor_text(self) -> None:
            # Ensure the number currently typed by the user is committed before SAVE.
            for widget in self.numeric_widgets():
                widget.interpretText()

        def read_mpc_from_ui(self) -> MPCConfig:
            config = MPCConfig(
                horizon=int(self.spin_horizon.value()),
                q_theta=float(self.spin_q_theta.value()),
                q_theta_dot=float(self.spin_q_theta_dot.value()),
                q_alpha=float(self.spin_q_alpha.value()),
                q_alpha_dot=float(self.spin_q_alpha_dot.value()),
                r_input=float(self.spin_r_input.value()),
                pgd_iterations=int(self.spin_pgd_iterations.value()),
                estimator=str(self.combo_estimator.currentData()),
                velocity_lpf=float(self.spin_velocity_lpf.value()),
            )
            config.validate()
            build_condensed_mpc_matrices(config)
            return config

        def update_estimator_controls(self, *_args) -> None:
            self.spin_velocity_lpf.setEnabled(
                self.combo_estimator.currentData() == ESTIMATOR_DIFFERENTIAL
            )

        def read_cfg_from_ui(self) -> ControllerConfig:
            enter = float(self.spin_enter.value())
            exit_ = float(self.spin_exit.value())
            if exit_ <= enter:
                raise ValueError("MPC exit angle must be greater than MPC enter angle.")
            pwm_limit = float(self.spin_pwm_limit.value())
            swing_pwm = float(self.spin_swing_pwm.value())
            if swing_pwm > pwm_limit:
                raise ValueError("Swing pump PWM cannot exceed PWM max.")
            return ControllerConfig(
                pwm_limit=pwm_limit,
                swing_pwm=swing_pwm,
                enter_deg=enter,
                exit_deg=exit_,
                blend_alpha=float(self.spin_blend.value()),
            )

        def read_noise_from_ui(self) -> NoiseConfig:
            return NoiseConfig(
                enabled=bool(self.check_noise.isChecked()),
                theta_obs_mean=float(self.theta_noise_mean.value()),
                theta_obs_std=float(self.theta_noise_std.value()),
                alpha_obs_mean=float(self.alpha_noise_mean.value()),
                alpha_obs_std=float(self.alpha_noise_std.value()),
                pwm_mean=float(self.pwm_noise_mean.value()),
                pwm_std=float(self.pwm_noise_std.value()),
            )

        def capture_settings_from_ui(self) -> SimulationSettings:
            self.commit_editor_text()
            output_dir = self.output_dir.text().strip() or "~/rip_twin_logs"
            return SimulationSettings(
                initial_name=self.combo_initial.currentText(),
                duration=float(self.spin_duration.value()),
                playback_speed=float(self.spin_speed.value()),
                mpc_cfg=self.read_mpc_from_ui(),
                controller_cfg=self.read_cfg_from_ui(),
                noise_cfg=self.read_noise_from_ui(),
                save_csv=bool(self.check_save_csv.isChecked()),
                output_dir=os.path.abspath(os.path.expanduser(output_dir)),
            )

        def connect_dirty_signals(self) -> None:
            self.combo_initial.currentIndexChanged.connect(self.mark_settings_dirty)
            self.combo_estimator.currentIndexChanged.connect(self.mark_settings_dirty)
            self.combo_estimator.currentIndexChanged.connect(self.update_estimator_controls)
            self.update_estimator_controls()
            for widget in self.numeric_widgets():
                widget.valueChanged.connect(self.mark_settings_dirty)
            self.check_noise.toggled.connect(self.mark_settings_dirty)
            self.check_save_csv.toggled.connect(self.mark_settings_dirty)
            self.output_dir.textChanged.connect(self.mark_settings_dirty)

        def mark_settings_dirty(self, *_args) -> None:
            if self._suppress_dirty:
                return
            self.settings_dirty = True
            self.btn_save.setText("SAVE / Apply Settings *")
            self.btn_save.setStyleSheet(
                "QPushButton { background: #fff3cd; font-weight: bold; }"
            )
            self.statusBar().showMessage(
                "Unsaved settings: click SAVE before GO or RESET."
            )

        def update_saved_summary(self) -> None:
            if self.saved_settings is None:
                self.label_saved.setText("No saved settings.")
                return
            s = self.saved_settings
            m = s.mpc_cfg
            n = s.noise_cfg
            noise_text = (
                f"on: θα μ/σ=({n.theta_obs_mean:g}/{n.theta_obs_std:g}, "
                f"{n.alpha_obs_mean:g}/{n.alpha_obs_std:g}), "
                f"PWM μ/σ={n.pwm_mean:g}/{n.pwm_std:g}"
                if n.enabled else "off"
            )
            self.label_saved.setText(
                "Saved and active for the next run:\n"
                f"initial={s.initial_name}, duration={s.duration:g}s, speed={s.playback_speed:g}x\n"
                f"N={m.horizon}, Q=[{m.q_theta:g}, {m.q_theta_dot:g}, {m.q_alpha:g}, {m.q_alpha_dot:g}], "
                f"R={m.r_input:g}, PGD={m.pgd_iterations}\n"
                f"estimator={m.estimator}, Gaussian noise={noise_text}"
            )

        def save_settings(self, *_args, show_message: bool = True) -> None:
            if self.running:
                QtWidgets.QMessageBox.information(
                    self,
                    "Simulation running",
                    "STOP the current run before saving new settings.",
                )
                return
            try:
                self.saved_settings = self.capture_settings_from_ui()
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(self, "Invalid setting", str(exc))
                return

            self.settings_dirty = False
            self.btn_save.setText("SAVE / Apply Settings")
            self.btn_save.setStyleSheet("")
            self.update_saved_summary()
            self.reset_simulation(force=True)
            if show_message:
                self.statusBar().showMessage(
                    "Settings saved. The saved MPC and estimator settings "
                    "will be used by subsequent GO/RESET operations."
                )

        def choose_output_directory(self) -> None:
            start = self.output_dir.text().strip() or os.path.expanduser("~/rip_twin_logs")
            path = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Choose result directory", os.path.expanduser(start)
            )
            if path:
                self.output_dir.setText(path)

        def reset_simulation(self, *_args, force: bool = False) -> None:
            if self.settings_dirty and not force:
                QtWidgets.QMessageBox.information(
                    self,
                    "Unsaved settings",
                    "The controls contain unsaved changes. Click SAVE first; "
                    "RESET always uses the last saved settings.",
                )
                return
            if self.saved_settings is None:
                return

            self.running = False
            self.run_settings = None
            self.sim_time = 0.0
            self.duration = self.saved_settings.duration
            self.state = initial_state_from_name(self.saved_settings.initial_name)
            self.controller = HybridMPCController(
                self.saved_settings.mpc_cfg,
                self.saved_settings.controller_cfg,
            )
            self.rows = [[0.0, *self.state, 0.0, MODE_DISABLED, 0.0]]
            self.result = None
            self._sim_accumulator = 0.0
            self.noise_rng = np.random.default_rng()
            self._last_wall_time = time.perf_counter()
            self.update_3d_view()
            self.update_labels(0.0, MODE_DISABLED)
            self.statusBar().showMessage("Digital twin reset using saved settings.")

        def start_simulation(self) -> None:
            if self.settings_dirty:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Save required",
                    "Initial condition, MPC parameters or another setting was changed.\n"
                    "Click SAVE / Apply Settings before GO.",
                )
                return
            if self.saved_settings is None:
                return

            self.run_settings = self.saved_settings
            self.duration = self.run_settings.duration
            self.state = initial_state_from_name(self.run_settings.initial_name)
            self.controller = HybridMPCController(
                self.run_settings.mpc_cfg,
                self.run_settings.controller_cfg,
            )
            self.sim_time = 0.0
            self.rows = [[0.0, *self.state, 0.0, MODE_DISABLED, 0.0]]
            self.result = None
            self.running = True
            self._sim_accumulator = 0.0
            self.noise_rng = np.random.default_rng()
            self._last_wall_time = time.perf_counter()
            self.label_status.setStyleSheet(
                "QLabel { color: #0b5d1e; background: #e8f5e9; padding: 6px; }"
            )
            self.statusBar().showMessage(
                f"Run started at {self.run_settings.playback_speed:g}× playback."
            )

        def stop_simulation(self) -> None:
            if self.running:
                self.running = False
                self.finalize_result(show_dialog=True)
            else:
                self.statusBar().showMessage("Simulation is already stopped.")

        def on_simulation_tick(self) -> None:
            now = time.perf_counter()
            wall_dt = max(0.0, min(now - self._last_wall_time, 0.25))
            self._last_wall_time = now

            if not self.running or self.controller is None or self.run_settings is None:
                return

            # Wall-clock synchronized playback. At 1x, one second of wall time
            # advances one simulated second. The selected factor ranges 0.1x–3x.
            self._sim_accumulator += wall_dt * self.run_settings.playback_speed
            step_count = 0
            last_pwm = 0.0
            last_details = {"mode": MODE_DISABLED, "blend": 0.0}

            while self._sim_accumulator + 1e-12 >= CONTROL_DT and step_count < 600:
                if self.sim_time >= self.duration - 1e-12:
                    self.running = False
                    self.finalize_result(show_dialog=True)
                    return

                theta_measured, alpha_measured = noisy_angle_measurements(
                    self.state,
                    self.run_settings.noise_cfg,
                    self.noise_rng,
                )
                pwm_command, last_details = self.controller.compute(
                    theta_measured, alpha_measured, self.sim_time, CONTROL_DT
                )
                last_pwm = noisy_applied_pwm(
                    pwm_command,
                    self.run_settings.noise_cfg,
                    self.run_settings.controller_cfg.pwm_limit,
                    self.noise_rng,
                )
                self.controller.accept_applied_pwm(last_pwm)
                self.state = rk4_step(
                    self.state, last_pwm, CONTROL_DT, self.params
                )
                self.sim_time += CONTROL_DT
                self.rows.append([
                    self.sim_time,
                    *self.state,
                    last_pwm,
                    int(last_details["mode"]),
                    last_details["blend"],
                ])
                self._sim_accumulator -= CONTROL_DT
                step_count += 1

            if step_count > 0:
                self.update_3d_view()
                self.update_labels(last_pwm, int(last_details["mode"]))

        def update_labels(self, pwm: float, mode: int) -> None:
            theta, theta_dot, alpha, alpha_dot = self.state
            phase = "running" if self.running else "stopped"
            speed = (
                self.run_settings.playback_speed
                if self.running and self.run_settings is not None
                else self.saved_settings.playback_speed
                if self.saved_settings is not None
                else 1.0
            )
            self.label_status.setText(
                f"Link: digital twin | phase: {phase}\n"
                f"controller: MPC | runtime: {MODE_NAMES.get(mode, 'UNKNOWN')}\n"
                f"time: {self.sim_time:.3f} / {self.duration:.3f} s | "
                f"playback: {speed:.2f}×"
            )
            self.label_state.setText(
                f"theta: {theta:.5f} rad\n"
                f"alpha: {alpha:.5f} rad\n"
                f"theta_dot: {theta_dot:.5f} rad/s\n"
                f"alpha_dot: {alpha_dot:.5f} rad/s\n"
                f"PWM: {pwm:.1f}"
            )

        def update_3d_view(self) -> None:
            theta, _theta_dot, alpha, _alpha_dot = self.state
            center, joint, tip, ref_tip, e_t = rip_points(theta, alpha)
            set_line3d(self.arm_line, center, joint)
            set_line3d(self.pend_line, joint, tip)
            set_point3d(self.joint_dot, joint)
            set_point3d(self.tip_dot, tip)
            tangent_end = joint + 0.12 * e_t
            set_line3d(self.tangent_line, joint - 0.12 * e_t, tangent_end)
            set_line3d(self.ref_line, joint, ref_tip)

            # Requested compact right-side text: no mode and no blend.
            self.state_text.set_text(
                f"t = {self.sim_time:6.3f} s\n"
                f"theta = {theta:+.4f} rad\n"
                f"alpha = {alpha:+.4f} rad"
            )
            self.canvas3d.draw_idle()

        def rows_to_result(self) -> SimulationResult:
            arr = np.asarray(self.rows, dtype=np.float64)
            capture_time = self.controller.capture_time if self.controller is not None else None
            return SimulationResult(
                time=arr[:, 0],
                theta=arr[:, 1],
                theta_dot=arr[:, 2],
                alpha=arr[:, 3],
                alpha_dot=arr[:, 4],
                pwm=arr[:, 5],
                mode=arr[:, 6].astype(np.int32),
                blend=arr[:, 7],
                capture_time=capture_time,
            )

        def finalize_result(self, show_dialog: bool) -> None:
            if len(self.rows) < 2:
                return
            self.result = self.rows_to_result()
            settings = self.run_settings or self.saved_settings
            if settings is None:
                return

            png_path, csv_path = default_output_paths(
                settings.output_dir, self.result.time[-1]
            )
            fig = build_result_figure(self.result)
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
            self.last_png_path = png_path
            self.last_csv_path = None

            if settings.save_csv:
                save_result_csv(self.result, csv_path)
                self.last_csv_path = csv_path

            metrics = result_metrics(self.result)
            stable_start = metrics["stable_start_time"]
            stable_text = (
                "final stable phase not reached" if math.isnan(stable_start)
                else f"stable from {stable_start:.3f} s"
            )
            self.label_status.setText(
                "Link: digital twin | phase: stopped\n"
                f"controller: MPC | runtime: "
                f"{MODE_NAMES.get(int(self.result.mode[-1]), 'UNKNOWN')}\n"
                f"result: {stable_text}"
            )

            saved_items = [png_path]
            if self.last_csv_path:
                saved_items.append(self.last_csv_path)
            self.statusBar().showMessage("Saved: " + " | ".join(saved_items))
            if show_dialog:
                self.show_result_dialog()

        def _screen_aware_dialog_size(
            self,
            width_ratio: float,
            height_ratio: float,
            max_width: int,
            max_height: int,
        ) -> Tuple[int, int]:
            screen = self.screen() or QtWidgets.QApplication.primaryScreen()
            if screen is None:
                return max_width, max_height
            geo = screen.availableGeometry()
            return (
                min(max_width, max(560, int(geo.width() * width_ratio))),
                min(max_height, max(420, int(geo.height() * height_ratio))),
            )

        def show_formula_help(self) -> None:
            if self.help_dialog is not None and self.help_dialog.isVisible():
                self.help_dialog.raise_()
                self.help_dialog.activateWindow()
                return
            dialog = QtWidgets.QDialog(self)
            self.help_dialog = dialog
            dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
            dialog.destroyed.connect(lambda: setattr(self, "help_dialog", None))
            dialog.setWindowTitle("Linear MPC, State Estimation, Swing-up and Soft Blend")
            dialog.setSizeGripEnabled(True)
            width, height = self._screen_aware_dialog_size(0.72, 0.82, 1050, 900)
            dialog.resize(width, height)
            layout = QtWidgets.QVBoxLayout(dialog)
            browser = QtWidgets.QTextBrowser()
            browser.setOpenExternalLinks(True)
            browser.setHtml("""
            <h2>Class 9: Swing-up and Linear MPC Stabilisation</h2>
            <p>The nonlinear Furuta model still runs at 200 Hz.  Far from upright,
            the same phase-pumping swing controller is used.  Inside the capture
            region, a constrained linear MPC predicts the next N states.</p>
            <h3>1. MPC objective</h3>
            <pre>J = sum(x[k]^T Q x[k] + R u[k]^2) + x[N]^T P x[N]</pre>
            <p>Q is diagonal in the state order
            [theta, theta_dot, alpha, alpha_dot].  P is rebuilt by discrete Riccati
            iteration whenever SAVE is pressed.</p>
            <h3>2. Condensed constrained problem</h3>
            <pre>min 0.5 U^T H U + (G x)^T U
subject to -PWMmax &lt;= U[i] &lt;= PWMmax</pre>
            <p>A warm-started fixed-iteration projected-gradient solver is used in
            both Python and STM32 firmware.  The safe step size is generated
            automatically from H.</p>
            <h3>3. State estimation</h3>
            <p><b>Differential + LPF:</b> velocities are obtained from successive
            angle samples and filtered by beta.  <b>Luenberger:</b> the fixed
            observer from the earlier MPC firmware supplies the four-state estimate
            in the upright region.  Differential velocity remains active for swing-up.</p>
            <h3>4. Hybrid switching</h3>
            <pre>PWM = clip((1-b) uSwing + b uMPC, -PWMmax, PWMmax)</pre>
            <p>Enter/exit hysteresis and the first-order blend are unchanged from
            the LQR panel, so the comparison isolates the upright controller.</p>
            """)
            layout.addWidget(browser)
            close = QtWidgets.QPushButton("Close")
            close.clicked.connect(dialog.close)
            layout.addWidget(close, alignment=QtCore.Qt.AlignRight)
            dialog.show()

        def show_result_dialog(self) -> None:
            if self.result is None:
                self.statusBar().showMessage("No completed simulation result yet.")
                return

            dialog = QtWidgets.QDialog(self)
            dialog.setWindowTitle(
                f"RIP MPC Digital Twin Result | {self.result.time[-1]:.3f} s Response"
            )
            dialog.setSizeGripEnabled(True)
            dialog.setWindowFlags(
                dialog.windowFlags()
                | QtCore.Qt.WindowMinMaxButtonsHint
            )
            layout = QtWidgets.QVBoxLayout(dialog)
            fig = build_result_figure(self.result)
            canvas = FigureCanvas(fig)
            layout.addWidget(canvas)

            row = QtWidgets.QHBoxLayout()
            path_edit = QtWidgets.QLineEdit(self.last_png_path or "")
            btn_choose = QtWidgets.QPushButton("Choose PNG Path")
            btn_save = QtWidgets.QPushButton("Save PNG")
            btn_close = QtWidgets.QPushButton("Close")
            row.addWidget(path_edit, stretch=1)
            row.addWidget(btn_choose)
            row.addWidget(btn_save)
            row.addWidget(btn_close)
            layout.addLayout(row)

            def choose_path() -> None:
                path, _ = QtWidgets.QFileDialog.getSaveFileName(
                    dialog,
                    "Save MPC twin result",
                    path_edit.text(),
                    "PNG Image (*.png)",
                )
                if path:
                    if not path.endswith(".png"):
                        path += ".png"
                    path_edit.setText(path)

            def save_png() -> None:
                raw = path_edit.text().strip()
                if not raw:
                    return
                path = os.path.abspath(os.path.expanduser(raw))
                if not path.endswith(".png"):
                    path += ".png"
                    path_edit.setText(path)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                fig.savefig(path, dpi=300, bbox_inches="tight")
                self.statusBar().showMessage(f"Saved LQR result figure: {path}")

            btn_choose.clicked.connect(choose_path)
            btn_save.clicked.connect(save_png)
            btn_close.clicked.connect(dialog.close)

            w, h = self._screen_aware_dialog_size(0.80, 0.82, 1180, 900)
            dialog.resize(w, h)
            dialog.exec_()

    app = QtWidgets.QApplication(sys.argv)
    win = RIPDigitalTwinWindow()
    win.show()

    # Open the explanation as a separate, resizable, screen-aware window once.
    QtCore.QTimer.singleShot(300, win.show_formula_help)
    return int(app.exec_())


# ============================================================
# CLI
# ============================================================


def run_headless(args) -> int:
    matplotlib.use("Agg", force=True)
    mpc_cfg = MPCConfig(
        horizon=args.horizon,
        q_theta=args.q_theta,
        q_theta_dot=args.q_theta_dot,
        q_alpha=args.q_alpha,
        q_alpha_dot=args.q_alpha_dot,
        r_input=args.r_input,
        pgd_iterations=args.pgd_iterations,
        estimator=args.estimator,
        velocity_lpf=args.velocity_lpf,
    )
    controller_cfg = ControllerConfig(
        pwm_limit=args.pwm_limit,
        swing_pwm=args.swing_pwm,
        kick_time=args.kick_time,
        enter_deg=args.enter_deg,
        exit_deg=args.exit_deg,
        blend_alpha=args.blend_alpha,
    )
    noise_cfg = NoiseConfig(
        enabled=not args.no_noise,
        theta_obs_mean=args.theta_noise_mean,
        theta_obs_std=args.theta_noise_std,
        alpha_obs_mean=args.alpha_noise_mean,
        alpha_obs_std=args.alpha_noise_std,
        pwm_mean=args.pwm_noise_mean,
        pwm_std=args.pwm_noise_std,
    )
    result = run_simulation(
        mpc_cfg,
        controller_cfg,
        duration=args.duration,
        initial_state=initial_state_from_name(args.initial),
        noise_config=noise_cfg,
        rng=np.random.default_rng(args.seed),
    )
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    png_path, csv_path = default_output_paths(output_dir, args.duration)
    figure = build_result_figure(result)
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    if args.csv:
        save_result_csv(result, csv_path)
    metrics = result_metrics(result)
    print("=== RIP MPC DIGITAL TWIN RESULT ===")
    print(f"N / Q / R                    : {mpc_cfg.horizon} / "
          f"[{mpc_cfg.q_theta:g}, {mpc_cfg.q_theta_dot:g}, {mpc_cfg.q_alpha:g}, {mpc_cfg.q_alpha_dot:g}] / {mpc_cfg.r_input:g}")
    print(f"estimator                     : {mpc_cfg.estimator}")
    print(f"duration_s                    : {result.time[-1]:.3f}")
    print(f"capture_time_s                : {metrics['capture_time']:.6f}")
    print(f"stable_start_time_s           : {metrics['stable_start_time']:.6f}")
    print(f"alpha_abs_mean_rad            : {metrics['alpha_abs_mean']:.9f}")
    print(f"pwm_abs_mean                  : {metrics['pwm_abs_mean']:.6f}")
    print(f"figure                        : {png_path}")
    if args.csv:
        print(f"csv                           : {csv_path}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single-file ROSRIP-aligned constrained MPC digital twin"
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--initial", default="Downward swing-up")
    parser.add_argument("--output-dir", default="~/rip_mpc_logs")
    parser.add_argument("--csv", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-noise", action="store_true")
    parser.add_argument("--theta-noise-mean", type=float, default=0.030)
    parser.add_argument("--theta-noise-std", type=float, default=0.020)
    parser.add_argument("--alpha-noise-mean", type=float, default=0.015)
    parser.add_argument("--alpha-noise-std", type=float, default=0.010)
    parser.add_argument("--pwm-noise-mean", type=float, default=5.0)
    parser.add_argument("--pwm-noise-std", type=float, default=2.5)
    parser.add_argument("--pwm-limit", type=float, default=150.0)
    parser.add_argument("--swing-pwm", type=float, default=120.0)
    parser.add_argument("--kick-time", type=float, default=0.10)
    parser.add_argument("--enter-deg", type=float, default=15.0)
    parser.add_argument("--exit-deg", type=float, default=25.0)
    parser.add_argument("--blend-alpha", type=float, default=0.18)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--q-theta", type=float, default=1.0)
    parser.add_argument("--q-theta-dot", type=float, default=0.05)
    parser.add_argument("--q-alpha", type=float, default=80.0)
    parser.add_argument("--q-alpha-dot", type=float, default=2.0)
    parser.add_argument("--r-input", type=float, default=0.001)
    parser.add_argument("--pgd-iterations", type=int, default=16)
    parser.add_argument(
        "--estimator",
        choices=[ESTIMATOR_DIFFERENTIAL, ESTIMATOR_LUENBERGER],
        default=ESTIMATOR_DIFFERENTIAL,
    )
    parser.add_argument("--velocity-lpf", type=float, default=0.25)
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.headless:
        return run_headless(args)
    return launch_gui()


if __name__ == "__main__":
    raise SystemExit(main())
