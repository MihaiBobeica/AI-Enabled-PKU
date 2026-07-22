# rl_env/envs/reward_fns.py

from __future__ import annotations

from typing import Callable, Dict, Any
import numpy as np


RewardFn = Callable[[np.ndarray, object, np.ndarray, object], float]


def default_balance_reward_fn(state, action, next_state, env) -> float:
    """
    默认平衡奖励：
    - 鼓励 alpha 接近 0（竖直向上）
    - 轻微惩罚角速度
    - 不显式惩罚 theta 本体，避免过早限制旋转臂活动
    """
    theta, theta_dot, alpha, alpha_dot = next_state

    reward = (
        10.0 * np.cos(alpha)
        - 0.001 * (alpha_dot ** 2)
        - 0.0001 * (theta_dot ** 2)
        - 5.0 * float(abs(alpha) > np.deg2rad(15.0))
    )
    return float(reward)


def swingup_reward_fn(state, action, next_state, env) -> float:
    """
    更适合 swing-up 的奖励：
    - cos(alpha) 是主项
    - 增加对 theta 偏移的轻微惩罚
    """
    theta, theta_dot, alpha, alpha_dot = next_state

    reward = (
        12.0 * np.cos(alpha)
        - 0.002 * (alpha_dot ** 2)
        - 0.0005 * (theta_dot ** 2)
        - 0.001 * (theta ** 2)
    )
    return float(reward)


def sparse_upright_reward_fn(state, action, next_state, env) -> float:
    """
    稀疏奖励：只有在接近直立时给高奖励。
    """
    theta, theta_dot, alpha, alpha_dot = next_state

    good_alpha = abs(alpha) < np.deg2rad(12.0)
    good_alpha_dot = abs(alpha_dot) < 4.0

    if good_alpha and good_alpha_dot:
        return 10.0
    return -1.0


def build_weighted_reward_fn(cfg: Dict[str, Any]) -> RewardFn:
    """
    外部可以通过配置快速生成奖励函数。
    cfg 示例：
    {
        "k_cos_alpha": 10.0,
        "k_alpha_dot": 0.001,
        "k_theta_dot": 0.0001,
        "k_theta": 0.0,
        "alpha_penalty_deg": 15.0,
        "alpha_penalty_value": 5.0,
        "action_l2": 0.0,
    }
    """
    k_cos_alpha = float(cfg.get("k_cos_alpha", 10.0))
    k_alpha_dot = float(cfg.get("k_alpha_dot", 0.001))
    k_theta_dot = float(cfg.get("k_theta_dot", 0.0001))
    k_theta = float(cfg.get("k_theta", 0.0))
    alpha_penalty_deg = float(cfg.get("alpha_penalty_deg", 15.0))
    alpha_penalty_value = float(cfg.get("alpha_penalty_value", 5.0))
    action_l2 = float(cfg.get("action_l2", 0.0))

    alpha_penalty_rad = np.deg2rad(alpha_penalty_deg)

    def _reward_fn(state, action, next_state, env) -> float:
        theta, theta_dot, alpha, alpha_dot = next_state
        theta_dot = np.clip(theta_dot, -env.theta_dot_limit, env.theta_dot_limit)

        alpha_dot = np.clip(alpha_dot, -env.alpha_dot_limit, env.alpha_dot_limit)
        # Do not call env._action_to_pwm() here: in the sim-to-real environment
        # that method advances delay/lag/noise state. The environment has already
        # computed the effective PWM before invoking the reward function.
        if hasattr(env, "_last_effective_pwm"):
            pwm = float(env._last_effective_pwm)
        elif env.action_type == "discrete":
            idx = int(np.asarray(action).reshape(-1)[0])
            pwm = float(env.discrete_actions[idx])
        else:
            scalar = float(np.asarray(action).reshape(-1)[0])
            pwm = float(np.clip(scalar, -1.0, 1.0) * env.continuous_pwm_limit)

        reward = (
            k_cos_alpha * np.cos(alpha)
            - k_alpha_dot * (alpha_dot ** 2)
            - k_theta_dot * (theta_dot ** 2)
            - k_theta * (theta ** 2)
            - alpha_penalty_value * float(abs(alpha) > alpha_penalty_rad)
            - action_l2 * ((pwm / max(env.continuous_pwm_limit, 1e-6)) ** 2)
        )
        return float(reward)

    return _reward_fn