# rl_env/envs/done_fns.py

from __future__ import annotations

from typing import Callable, Tuple, Dict, Any
import numpy as np


DoneFn = Callable[[np.ndarray, object, np.ndarray, object], Tuple[bool, bool, Dict[str, Any]]]


def default_done_fn(state, action, next_state, env):
    """
    默认终止条件：
    - theta 超限
    - theta_dot 超限
    - alpha_dot 超限
    - 数值崩掉
    - max_steps 截断
    """
    theta, theta_dot, alpha, alpha_dot = next_state

    terminated = (
        abs(theta) > env.theta_limit
        or abs(theta_dot) > env.theta_dot_limit
        or abs(alpha_dot) > env.alpha_dot_limit
        or (not np.all(np.isfinite(next_state)))
    )

    if env.terminate_on_alpha_abs_deg:
        alpha_abs_limit_rad = np.deg2rad(env.alpha_abs_limit_deg)
        terminated = terminated or (abs(alpha) > alpha_abs_limit_rad)

    truncated = (env.step_count + 1) >= env.max_steps

    if terminated:
        reason = "state_limit"
        if not np.all(np.isfinite(next_state)):
            reason = "non_finite"
    elif truncated:
        reason = "time_limit"
    else:
        reason = None

    info = {"done_reason": reason}
    return bool(terminated), bool(truncated), info


def strict_balance_done_fn(state, action, next_state, env):
    """
    更严格的平衡任务终止条件：
    对 alpha 本身也设限。
    """
    theta, theta_dot, alpha, alpha_dot = next_state

    terminated = (
        abs(theta) > env.theta_limit
        or abs(theta_dot) > env.theta_dot_limit
        or abs(alpha_dot) > env.alpha_dot_limit
        or abs(alpha) > np.deg2rad(60.0)
        or (not np.all(np.isfinite(next_state)))
    )

    truncated = (env.step_count + 1) >= env.max_steps
    info = {"done_reason": "strict_balance_fail" if terminated else ("time_limit" if truncated else None)}
    return bool(terminated), bool(truncated), info


def build_done_fn(cfg: Dict[str, Any]) -> DoneFn:
    """
    外部通过配置构建终止函数。
    cfg 示例：
    {
        "theta_limit": 12*np.pi,
        "theta_dot_limit": 45.0,
        "alpha_dot_limit": 40.0,
        "terminate_on_alpha_abs_deg": False,
        "alpha_abs_limit_deg": 90.0,
    }
    """
    theta_limit = float(cfg.get("theta_limit", 12.0 * np.pi))
    theta_dot_limit = float(cfg.get("theta_dot_limit", 45.0))
    alpha_dot_limit = float(cfg.get("alpha_dot_limit", 40.0))
    terminate_on_alpha_abs_deg = bool(cfg.get("terminate_on_alpha_abs_deg", False))
    alpha_abs_limit_deg = float(cfg.get("alpha_abs_limit_deg", 90.0))

    def _done_fn(state, action, next_state, env):
        theta, theta_dot, alpha, alpha_dot = next_state

        terminated = (
            abs(theta) > theta_limit
            or abs(theta_dot) > theta_dot_limit
            or abs(alpha_dot) > alpha_dot_limit
            or (not np.all(np.isfinite(next_state)))
        )

        if terminate_on_alpha_abs_deg:
            terminated = terminated or (abs(alpha) > np.deg2rad(alpha_abs_limit_deg))

        truncated = env.step_count >= env.max_steps

        if terminated:
            reason = "state_limit"
        elif truncated:
            reason = "time_limit"
        else:
            reason = None

        info = {"done_reason": reason}
        return bool(terminated), bool(truncated), info

    return _done_fn