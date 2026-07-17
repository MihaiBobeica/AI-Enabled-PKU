
"""
TA Ju zhixiang
2026.7.14
=================================
A self-contained rotary inverted pendulum (Furuta pendulum) simulator whose
state convention, 200 Hz timing, PWM input, 3D geometry and result plots are
aligned with real RIP  project.
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
MODE_SWING_PID = 1
MODE_BLEND = 2
MODE_PID = 3
MODE_NAMES = {
    MODE_DISABLED: "DISABLED",
    MODE_SWING_PID: "SWING_PUMP",
    MODE_BLEND: "BLEND",
    MODE_PID: "PID",
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
# Hybrid swing-up + dual PID controller
# ============================================================

@dataclass
class PIDGains:
    # Pendulum angle loop
    kp_alpha: float = 680.0
    ki_alpha: float = 0.0
    kd_alpha: float = 67.0

    # Rotary-arm centering loop
    kp_theta: float = 27.0
    ki_theta: float = 0.0
    kd_theta: float = 39.0


@dataclass
class ControllerConfig:
    pwm_limit: float = DEFAULT_PWM_LIMIT
    swing_pwm: float = 120.0
    kick_time: float = 0.10
    enter_deg: float = 15.0
    exit_deg: float = 25.0
    blend_alpha: float = 0.18
    alpha_i_limit: float = 0.50
    theta_i_limit: float = 1.00


@dataclass
class NoiseConfig:
    """Independent white Gaussian noise applied once per 200 Hz control step.

    Angle observation noise:
        theta_measured = theta_true + N(theta_mean, theta_std^2)
        alpha_measured = wrap(alpha_true + N(alpha_mean, alpha_std^2))

    PWM actuation noise:
        pwm_applied = clip(pwm_command + N(pwm_mean, pwm_std^2), +/- pwm_limit)

    The means model constant bias; the standard deviations model random spread.
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
    gains: PIDGains
    controller_cfg: ControllerConfig
    noise_cfg: NoiseConfig
    save_csv: bool
    output_dir: str


class HybridPIDController:
    """
    Complete controller used by the twin.

    Swing-up:
        phase-pumping relay. It is required because the Furuta pendulum is
        underactuated and the upright PID is only a local controller.

    Balance:
        u = PID_alpha(alpha) + PID_theta(theta)

    The transition uses the same 15/25 degree hysteresis and first-order blend
    concept as ROSRIP.
    """

    def __init__(self, gains: PIDGains, cfg: ControllerConfig) -> None:
        self.gains = gains
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.int_alpha = 0.0
        self.int_theta = 0.0
        self.upright_mode = False
        self.blend = 0.0
        self.mode = MODE_DISABLED
        self.capture_time: Optional[float] = None

    def _swing_pwm(self, t: float, alpha: float, alpha_dot: float) -> float:
        # A short deterministic kick removes the exact downward equilibrium.
        if t < self.cfg.kick_time:
            return float(self.cfg.swing_pwm)

        phase = alpha_dot * math.cos(alpha)
        direction = 1.0 if phase >= 0.0 else -1.0

        # The sign is identified for the state/input convention used by the
        # supplied Furuta dynamics. This pumps energy toward alpha = 0.
        return -float(self.cfg.swing_pwm) * direction

    def compute(self, state: np.ndarray, t: float, dt: float) -> Tuple[float, Dict[str, float]]:
        theta, theta_dot, alpha, alpha_dot = [float(v) for v in state]
        abs_alpha = abs(alpha)
        enter = math.radians(self.cfg.enter_deg)
        exit_ = math.radians(self.cfg.exit_deg)

        previous = self.upright_mode
        if (not self.upright_mode) and abs_alpha <= enter:
            self.upright_mode = True
            self.int_alpha = 0.0
            self.int_theta = 0.0
            if self.capture_time is None:
                self.capture_time = float(t)
        elif self.upright_mode and abs_alpha >= exit_:
            self.upright_mode = False
            self.int_alpha = 0.0
            self.int_theta = 0.0

        if self.upright_mode:
            self.int_alpha = float(np.clip(
                self.int_alpha + alpha * dt,
                -self.cfg.alpha_i_limit,
                self.cfg.alpha_i_limit,
            ))
            self.int_theta = float(np.clip(
                self.int_theta + theta * dt,
                -self.cfg.theta_i_limit,
                self.cfg.theta_i_limit,
            ))

        u_swing = self._swing_pwm(t, alpha, alpha_dot)
        g = self.gains
        u_pid = (
            g.kp_alpha * alpha
            + g.ki_alpha * self.int_alpha
            + g.kd_alpha * alpha_dot
            + g.kp_theta * theta
            + g.ki_theta * self.int_theta
            + g.kd_theta * theta_dot
        )

        target = 1.0 if self.upright_mode else 0.0
        self.blend += self.cfg.blend_alpha * (target - self.blend)
        self.blend = float(np.clip(self.blend, 0.0, 1.0))

        u = (1.0 - self.blend) * u_swing + self.blend * u_pid
        u = float(np.clip(u, -self.cfg.pwm_limit, self.cfg.pwm_limit))

        if self.blend <= 0.01:
            self.mode = MODE_SWING_PID
        elif self.blend >= 0.99:
            self.mode = MODE_PID
        else:
            self.mode = MODE_BLEND

        details = {
            "u_swing": float(u_swing),
            "u_pid": float(u_pid),
            "blend": float(self.blend),
            "upright": float(self.upright_mode),
            "mode": float(self.mode),
            "entered_now": float((not previous) and self.upright_mode),
        }
        return u, details


# ============================================================
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


def noisy_controller_observation(
    true_state: np.ndarray,
    noise_cfg: NoiseConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return the state seen by the controller.

    Only theta and alpha angle observations are corrupted. The simulated true
    angular velocities are retained, matching the requested angle-observation
    noise model.
    """
    observed = np.asarray(true_state, dtype=np.float64).reshape(4).copy()
    if not noise_cfg.enabled:
        return observed

    observed[0] += float(rng.normal(
        noise_cfg.theta_obs_mean,
        noise_cfg.theta_obs_std,
    ))
    observed[2] = wrap_to_pi(float(
        observed[2]
        + rng.normal(noise_cfg.alpha_obs_mean, noise_cfg.alpha_obs_std)
    ))
    return observed


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
    gains: PIDGains,
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
    controller = HybridPIDController(gains, controller_cfg)
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
        observed_state = noisy_controller_observation(state, noise_cfg, noise_rng)
        pwm_command, details = controller.compute(observed_state, t, dt)
        pwm_applied = noisy_applied_pwm(
            pwm_command,
            noise_cfg,
            controller_cfg.pwm_limit,
            noise_rng,
        )
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


def build_result_figure(result: SimulationResult, title_suffix: str = "PID Digital Twin") -> Figure:
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
        os.path.join(output_dir, f"rip_pid_twin_{duration_tag}s_{stamp}.png"),
        os.path.join(output_dir, f"rip_pid_twin_{duration_tag}s_{stamp}.csv"),
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
            self.setWindowTitle("RIP PID Digital Twin | Rotary Inverted Pendulum")
            self.resize(1180, 760)

            self.params = RIPPhysicalParams()
            self.state = initial_state_from_name("downward")
            self.controller: Optional[HybridPIDController] = None
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

            form.addRow("Controller profile:", QtWidgets.QLabel("Swing pump → dual PID"))
            form.addRow("Initial condition:", self.combo_initial)
            form.addRow("Duration / s:", self.spin_duration)
            form.addRow("Playback speed:", self.spin_speed)
            form.addRow("PWM max:", self.spin_pwm_limit)
            form.addRow("Swing pump PWM:", self.spin_swing_pwm)
            form.addRow("PID enter / deg:", self.spin_enter)
            form.addRow("PID exit / deg:", self.spin_exit)
            form.addRow("Soft blend λ:", self.spin_blend)

            self.label_profile = QtWidgets.QLabel(
                "200 Hz nonlinear twin. Change values freely, then click SAVE. "
                "Unsaved changes are deliberately not used by GO."
            )
            self.label_profile.setWordWrap(True)
            self.label_profile.setStyleSheet(
                "QLabel { color: #333; background: #f2f2f2; padding: 6px; }"
            )
            form.addRow(self.label_profile)

            self.btn_help = QtWidgets.QPushButton("PID Formula & Parameter Help")
            self.btn_help.clicked.connect(self.show_formula_help)
            form.addRow(self.btn_help)

            self.label_saved = QtWidgets.QLabel()
            self.label_saved.setWordWrap(True)
            self.label_saved.setStyleSheet(
                "QLabel { color: #174a7e; background: #edf5ff; padding: 6px; }"
            )
            form.addRow(self.label_saved)
            left_layout.addWidget(group_ctrl)

            group_pid = QtWidgets.QGroupBox("PID Gains")
            pid_form = QtWidgets.QFormLayout(group_pid)
            self.kp_alpha = self._dspin(10.0, -5000.0, 5000.0, 4, 1.0)
            self.ki_alpha = self._dspin(10.0, -1000.0, 1000.0, 4, 1.0)
            self.kd_alpha = self._dspin(10.0, -2000.0, 2000.0, 4, 1.0)
            self.kp_theta = self._dspin(10.0, -2000.0, 2000.0, 4, 1.0)
            self.ki_theta = self._dspin(10.0, -1000.0, 1000.0, 4, 1.0)
            self.kd_theta = self._dspin(10.0, -2000.0, 2000.0, 4, 1.0)

            pid_form.addRow("Pendulum Kp:", self.kp_alpha)
            pid_form.addRow("Pendulum Ki:", self.ki_alpha)
            pid_form.addRow("Pendulum Kd:", self.kd_alpha)
            pid_form.addRow("Arm Kp:", self.kp_theta)
            pid_form.addRow("Arm Ki:", self.ki_theta)
            pid_form.addRow("Arm Kd:", self.kd_theta)

            self.btn_save = QtWidgets.QPushButton("SAVE / Apply Settings")
            self.btn_save.setMinimumHeight(34)
            self.btn_save.clicked.connect(self.save_settings)
            pid_form.addRow(self.btn_save)
            left_layout.addWidget(group_pid)

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
                "controller: PID | runtime: DISABLED\n"
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
                self.kp_alpha,
                self.ki_alpha,
                self.kd_alpha,
                self.kp_theta,
                self.ki_theta,
                self.kd_theta,
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

        def read_gains_from_ui(self) -> PIDGains:
            return PIDGains(
                kp_alpha=self.kp_alpha.value(),
                ki_alpha=self.ki_alpha.value(),
                kd_alpha=self.kd_alpha.value(),
                kp_theta=self.kp_theta.value(),
                ki_theta=self.ki_theta.value(),
                kd_theta=self.kd_theta.value(),
            )

        def read_cfg_from_ui(self) -> ControllerConfig:
            enter = float(self.spin_enter.value())
            exit_ = float(self.spin_exit.value())
            if exit_ <= enter:
                raise ValueError("PID exit angle must be greater than PID enter angle.")
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
                gains=self.read_gains_from_ui(),
                controller_cfg=self.read_cfg_from_ui(),
                noise_cfg=self.read_noise_from_ui(),
                save_csv=bool(self.check_save_csv.isChecked()),
                output_dir=os.path.abspath(os.path.expanduser(output_dir)),
            )

        def connect_dirty_signals(self) -> None:
            self.combo_initial.currentIndexChanged.connect(self.mark_settings_dirty)
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
            g = s.gains
            n = s.noise_cfg
            noise_text = (
                f"on: θα μ/σ=({n.theta_obs_mean:g}/{n.theta_obs_std:g}, "
                f"{n.alpha_obs_mean:g}/{n.alpha_obs_std:g}), "
                f"PWM μ/σ={n.pwm_mean:g}/{n.pwm_std:g}"
                if n.enabled else "off"
            )
            self.label_saved.setText(
                "Saved and active for the next run:\n"
                f"initial={s.initial_name}, duration={s.duration:g}s, "
                f"speed={s.playback_speed:g}×\n"
                f"Kα=({g.kp_alpha:g}, {g.ki_alpha:g}, {g.kd_alpha:g}), "
                f"Kθ=({g.kp_theta:g}, {g.ki_theta:g}, {g.kd_theta:g})\n"
                f"Gaussian noise={noise_text}"
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
                    "Settings saved. The saved initial condition and PID gains "
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
            self.controller = HybridPIDController(
                self.saved_settings.gains,
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
                    "Initial condition, PID gains or another setting was changed.\n"
                    "Click SAVE / Apply Settings before GO.",
                )
                return
            if self.saved_settings is None:
                return

            self.run_settings = self.saved_settings
            self.duration = self.run_settings.duration
            self.state = initial_state_from_name(self.run_settings.initial_name)
            self.controller = HybridPIDController(
                self.run_settings.gains,
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

                observed_state = noisy_controller_observation(
                    self.state,
                    self.run_settings.noise_cfg,
                    self.noise_rng,
                )
                pwm_command, last_details = self.controller.compute(
                    observed_state, self.sim_time, CONTROL_DT
                )
                last_pwm = noisy_applied_pwm(
                    pwm_command,
                    self.run_settings.noise_cfg,
                    self.run_settings.controller_cfg.pwm_limit,
                    self.noise_rng,
                )
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
                f"controller: PID | runtime: {MODE_NAMES.get(mode, 'UNKNOWN')}\n"
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
                f"controller: PID | runtime: "
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
            dialog.setWindowTitle("Dual PID Formula, Swing-up and Soft Blend")
            dialog.setSizeGripEnabled(True)
            dialog.setWindowFlags(
                dialog.windowFlags()
                | QtCore.Qt.WindowMinMaxButtonsHint
            )

            layout = QtWidgets.QVBoxLayout(dialog)
            scroll = QtWidgets.QScrollArea(dialog)
            scroll.setWidgetResizable(True)
            content = QtWidgets.QLabel()
            content.setTextFormat(QtCore.Qt.RichText)
            content.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            content.setWordWrap(True)
            content.setMargin(14)

            dt_ms = CONTROL_DT * 1000.0
            lam = (
                self.saved_settings.controller_cfg.blend_alpha
                if self.saved_settings is not None
                else 0.18
            )
            if 0.0 < lam < 1.0:
                blend_99_s = math.log(0.01) / math.log(1.0 - lam) * CONTROL_DT
            else:
                blend_99_s = 0.0

            content.setText(f"""
            <h2>Rotary Inverted Pendulum: Swing-up and Dual-Channel PID</h2>

            <h3>1. State convention</h3>
            <p>
            θ is the rotary-arm angle and α is the pendulum angle.
            α=0 means upright; α=±π means downward.
            The controller runs at Δt={CONTROL_DT:.3f}s
            ({1.0 / CONTROL_DT:.0f}Hz).
            </p>

            <h3>2. Dual-channel PID near the upright equilibrium</h3>
            <pre>
Iα[k+1] = clip(Iα[k] + α[k] Δt, -Iα,max, Iα,max)
Iθ[k+1] = clip(Iθ[k] + θ[k] Δt, -Iθ,max, Iθ,max)

uα = Kpα α + Kiα Iα + Kdα α̇
uθ = Kpθ θ + Kiθ Iθ + Kdθ θ̇

uPID = uα + uθ
            </pre>
            <p>
            There is only one motor. These are therefore two feedback
            contributions, not two independent actuators. The pendulum
            contribution uα keeps α close to zero, while the arm contribution
            uθ limits rotary-arm drift. Their sum generates one PWM command.
            </p>
            <p>
            The integrators update only while the controller is in the upright
            region. They are reset during swing-up or after loss of balance to
            avoid wind-up. If Kiα and Kiθ are set to zero, the controller
            reduces to a dual-channel PD controller.
            </p>

            <h3>3. Swing-up from the downward equilibrium</h3>
            <p>
            A local PID cannot lift the pendulum from α=±π. The program first
            applies a short deterministic kick and then uses fixed-amplitude
            phase pumping:
            </p>
            <pre>
uSwing = -Us sign(α̇ cos α)
            </pre>
            <p>
            Us is the Swing pump PWM. The sign changes with pendulum phase so
            that energy is injected over successive swings.
            </p>

            <h3>4. PID enter and PID exit</h3>
            <p>
            If |αmeasured| becomes smaller than PID enter, the blend target is
            set to 1. It stays at 1 until |αmeasured| becomes larger than PID
            exit, at which point the target returns to 0. The two thresholds
            create hysteresis and prevent rapid switching near one boundary.
            </p>

            <h3>5. Soft blend λ: exact meaning and time evolution</h3>
            <p>
            <b>Soft blend λ is not an angle and is not a direct function of
            angle magnitude.</b> The angle is used only to select the binary
            target through the enter/exit thresholds. Once the target has been
            selected, the blend state b changes once per 5 ms control step:
            </p>
            <pre>
b[k+1] = clip(b[k] + λ(target - b[k]), 0, 1)

target = 1  while the upright PID region is active
target = 0  while swing-up is active

PWM = clip((1-b) uSwing + b uPID, -PWMmax, PWMmax)
            </pre>
            <p>
            Therefore b evolves with <b>simulation time</b>. For a constant
            target=1 and b[0]=0, the closed-form sequence is:
            </p>
            <pre>
b[k] = 1 - (1-λ)^k
t = k Δt
            </pre>
            <p>
            To reach 99%, solve:
            </p>
            <pre>
1 - (1-λ)^k = 0.99
k = ln(0.01) / ln(1-λ)
            </pre>
            <p>
            For the current saved value λ={lam:g} and Δt={CONTROL_DT:.3f}s,
            k≈{math.log(0.01) / math.log(1.0 - lam):.3f} steps, giving the
            interpolated nominal time
            t≈{blend_99_s:.3f}s. Because the implementation is discrete, the
            first actual sample at or above 99% occurs after
            {math.ceil(math.log(0.01) / math.log(1.0 - lam))} steps, i.e.
            {math.ceil(math.log(0.01) / math.log(1.0 - lam)) * CONTROL_DT:.3f}s.
            </p>
            <p>
            For target=0, the decay is b[k]=b[0](1-λ)^k. If the pendulum crosses
            the exit threshold during a transition, the target immediately
            changes to 0 and the same exponential law reverses the blend toward
            swing-up. Playback speed changes only wall-clock display speed; it
            does not change this simulation-time law.
            </p>

            <h3>6. Gaussian random-noise model</h3>
            <p>
            When Enable random noise is selected, independent white Gaussian
            samples are drawn at every 5 ms controller update:
            </p>
            <pre>
nθ ~ N(μθ, σθ²)
nα ~ N(μα, σα²)
nu ~ N(μu, σu²)

θmeasured = θtrue + nθ
αmeasured = wrap(αtrue + nα)
PWMactual = clip(PWMcommand + nu, -PWMmax, PWMmax)
            </pre>
            <p>
            μ is the mean and represents a sensor or actuator bias. σ is the
            standard deviation and controls random spread. For example,
            μα=0.01 rad creates a +0.01 rad average alpha-observation bias.
            The controller uses noisy θ and α, while the 3D view and angle plots
            show the true simulated state. The PWM curve records the actual
            noisy PWM applied to the dynamics.
            </p>

            <h3>7. Other controls</h3>
            <ul>
              <li><b>PWM max:</b> final saturation limit.</li>
              <li><b>Playback speed:</b> wall-clock playback from 0.1× to 3×.</li>
              <li><b>Duration:</b> simulation duration; independent of playback speed.</li>
              <li><b>SAVE:</b> commits initial condition, PID gains, blend,
              noise, playback, and logging settings for subsequent GO/RESET runs.</li>
            </ul>
            """)
            scroll.setWidget(content)
            layout.addWidget(scroll)

            close_button = QtWidgets.QPushButton("Close")
            close_button.clicked.connect(dialog.close)
            layout.addWidget(close_button, alignment=QtCore.Qt.AlignRight)

            w, h = self._screen_aware_dialog_size(0.68, 0.78, 940, 820)
            dialog.resize(w, h)
            dialog.show()

        def show_result_dialog(self) -> None:
            if self.result is None:
                self.statusBar().showMessage("No completed simulation result yet.")
                return

            dialog = QtWidgets.QDialog(self)
            dialog.setWindowTitle(
                f"RIP PID Digital Twin Result | {self.result.time[-1]:.3f} s Response"
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
                    "Save PID twin result",
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
                self.statusBar().showMessage(f"Saved PID result figure: {path}")

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
    gains = PIDGains(
        kp_alpha=args.kp_alpha,
        ki_alpha=args.ki_alpha,
        kd_alpha=args.kd_alpha,
        kp_theta=args.kp_theta,
        ki_theta=args.ki_theta,
        kd_theta=args.kd_theta,
    )
    cfg = ControllerConfig(
        pwm_limit=args.pwm_limit,
        swing_pwm=min(args.swing_pwm, args.pwm_limit),
        enter_deg=args.enter_deg,
        exit_deg=max(args.exit_deg, args.enter_deg + 0.1),
        blend_alpha=args.blend_alpha,
    )
    noise_cfg = NoiseConfig(
        enabled=bool(args.noise),
        theta_obs_mean=args.theta_noise_mean,
        theta_obs_std=max(0.0, args.theta_noise_std),
        alpha_obs_mean=args.alpha_noise_mean,
        alpha_obs_std=max(0.0, args.alpha_noise_std),
        pwm_mean=args.pwm_noise_mean,
        pwm_std=max(0.0, args.pwm_noise_std),
    )
    result = run_simulation(
        gains=gains,
        controller_cfg=cfg,
        duration=args.duration,
        dt=CONTROL_DT,
        initial_state=initial_state_from_name(args.initial),
        noise_config=noise_cfg,
    )
    png_path, csv_path = default_output_paths(args.output_dir, result.time[-1])
    fig = build_result_figure(result)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    if not args.no_csv:
        save_result_csv(result, csv_path)
    metrics = result_metrics(result)

    print("=== RIP PID DIGITAL TWIN RESULT ===")
    print(f"duration_s                    : {result.time[-1]:.3f}")
    print(f"capture_time_s                : {metrics['capture_time']:.6f}")
    print(f"stable_start_time_s           : {metrics['stable_start_time']:.6f}")
    print(f"stable_duration_s             : {metrics['stable_duration']:.6f}")
    print(f"stable_mean_abs_alpha_rad     : {metrics['alpha_abs_mean']:.8f}")
    print(f"stable_std_abs_alpha_rad      : {metrics['alpha_abs_std']:.8f}")
    print(f"max_abs_theta_rad             : {metrics['max_abs_theta']:.6f}")
    print(f"stable_mean_abs_pwm           : {metrics['pwm_abs_mean']:.4f}")
    print(f"stable_std_abs_pwm            : {metrics['pwm_abs_std']:.4f}")
    print(f"result_png              : {png_path}")
    print(f"result_csv              : {csv_path if not args.no_csv else 'disabled'}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-file ROSRIP-aligned PID digital twin with Gaussian noise")
    parser.add_argument("--headless", action="store_true", help="Run without the Qt GUI")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--initial", default="downward", choices=["downward", "upright+8", "upright-8"])
    parser.add_argument("--output-dir", default="~/rip_twin_logs")
    parser.add_argument("--no-csv", action="store_true", help="Do not generate a CSV log")
    parser.add_argument("--noise", dest="noise", action="store_true", default=True, help="Enable Gaussian observation and PWM noise")
    parser.add_argument("--no-noise", dest="noise", action="store_false", help="Disable Gaussian observation and PWM noise")
    parser.add_argument("--theta-noise-mean", type=float, default=0.030)
    parser.add_argument("--theta-noise-std", type=float, default=0.020)
    parser.add_argument("--alpha-noise-mean", type=float, default=0.015)
    parser.add_argument("--alpha-noise-std", type=float, default=0.010)
    parser.add_argument("--pwm-noise-mean", type=float, default=5.0)
    parser.add_argument("--pwm-noise-std", type=float, default=2.5)
    parser.add_argument("--pwm-limit", type=float, default=150.0)
    parser.add_argument("--swing-pwm", type=float, default=120.0)
    parser.add_argument("--enter-deg", type=float, default=15.0)
    parser.add_argument("--exit-deg", type=float, default=25.0)
    parser.add_argument("--blend-alpha", type=float, default=0.18)
    parser.add_argument("--kp-alpha", type=float, default=10.0)
    parser.add_argument("--ki-alpha", type=float, default=10.0)
    parser.add_argument("--kd-alpha", type=float, default=10.0)
    parser.add_argument("--kp-theta", type=float, default=10.0)
    parser.add_argument("--ki-theta", type=float, default=10.0)
    parser.add_argument("--kd-theta", type=float, default=10.0)
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.headless:
        return run_headless(args)
    return launch_gui()


if __name__ == "__main__":
    raise SystemExit(main())
