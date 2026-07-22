# rl_env/envs/dynamics.py

from __future__ import annotations

import math
import numpy as np

from .types import RIPPhysicalParams


def deg2rad(x: float) -> float:
    return np.deg2rad(x)


def wrap_to_pi(x: float) -> float:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def clip_scalar(v: float, lim: float) -> float:
    return max(-lim, min(lim, v))


def furuta_derivatives(state: np.ndarray, pwm: float, p: RIPPhysicalParams) -> np.ndarray:
    """
    state = [theta, theta_dot, alpha, alpha_dot]
    alpha = 0 means upright
    """
    theta, theta_dot, alpha, alpha_dot = state

    sA = math.sin(alpha)
    cA = math.cos(alpha)
    sAcA = sA * cA

    A = (
        p.m1 * p.l1cg * p.l1cg
        + p.I1z
        + p.m2 * p.l1 * p.l1
        + (p.m2 * p.l2cg * p.l2cg + p.I2z) * (sA * sA)
        + p.I2y * (cA * cA)
    )
    B = p.m2 * p.l1 * p.l2cg * cA
    Cc = -p.m2 * p.l1 * p.l2cg * sA
    D = 2.0 * (p.I2z + p.m2 * p.l2cg * p.l2cg - p.I2y) * sAcA

    E = p.K1 * pwm - p.K2 * theta_dot - p.c_theta * theta_dot

    F = -(p.m2 * p.l2cg * p.l2cg + p.I2x)
    G = -(p.m2 * p.l1 * p.l2cg * cA)
    H = (p.m2 * p.l2cg * p.l2cg - p.I2y + p.I2z) * sAcA
    K = p.m2 * p.g * p.l2cg * sA
    L = p.c_alpha * alpha_dot

    den1 = A * F - G * B
    den2 = G * B - A * F

    # Numerical safeguard
    eps = 1e-12
    if abs(den1) < eps:
        den1 = eps if den1 >= 0 else -eps
    if abs(den2) < eps:
        den2 = eps if den2 >= 0 else -eps

    theta_ddot = (
        (-F * Cc) * (alpha_dot ** 2)
        + (-F * D) * alpha_dot * theta_dot
        + (B * H) * (theta_dot ** 2)
        + (B * K + F * E - B * L)
    ) / den1

    alpha_ddot = (
        (-G * Cc) * (alpha_dot ** 2)
        + (-G * D) * alpha_dot * theta_dot
        + (A * H) * (theta_dot ** 2)
        + (A * K + G * E - A * L)
    ) / den2

    return np.array([theta_dot, theta_ddot, alpha_dot, alpha_ddot], dtype=np.float64)


def euler_step(state: np.ndarray, pwm: float, dt: float, p: RIPPhysicalParams) -> np.ndarray:
    nxt = state + dt * furuta_derivatives(state, pwm, p)
    nxt[2] = wrap_to_pi(float(nxt[2]))
    return nxt


def rk4_step(state: np.ndarray, pwm: float, dt: float, p: RIPPhysicalParams) -> np.ndarray:
    k1 = furuta_derivatives(state, pwm, p)
    k2 = furuta_derivatives(state + 0.5 * dt * k1, pwm, p)
    k3 = furuta_derivatives(state + 0.5 * dt * k2, pwm, p)
    k4 = furuta_derivatives(state + dt * k3, pwm, p)

    nxt = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    nxt[2] = wrap_to_pi(float(nxt[2]))
    return nxt


def state_to_trig_obs(
    state: np.ndarray,
    theta_dot_limit: float,
    alpha_dot_limit: float,
    clip_velocity: bool = True,
) -> np.ndarray:
    theta, theta_dot, alpha, alpha_dot = state

    if clip_velocity:
        theta_dot = clip_scalar(float(theta_dot), float(theta_dot_limit))
        alpha_dot = clip_scalar(float(alpha_dot), float(alpha_dot_limit))

    obs = np.array(
        [
            np.sin(theta),
            np.cos(theta),
            theta_dot,
            np.sin(alpha),
            np.cos(alpha),
            alpha_dot,
        ],
        dtype=np.float32,
    )
    return obs


def state_to_raw_obs(
    state: np.ndarray,
    theta_dot_limit: float,
    alpha_dot_limit: float,
    clip_velocity: bool = True,
) -> np.ndarray:
    theta, theta_dot, alpha, alpha_dot = state

    if clip_velocity:
        theta_dot = clip_scalar(float(theta_dot), float(theta_dot_limit))
        alpha_dot = clip_scalar(float(alpha_dot), float(alpha_dot_limit))

    obs = np.array([theta, theta_dot, alpha, alpha_dot], dtype=np.float32)
    return obs