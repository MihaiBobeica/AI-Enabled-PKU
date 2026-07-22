# rl_env/wrappers/action_wrappers.py

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ContinuousToDiscreteActionWrapper(gym.ActionWrapper):
    """
    把一个 continuous action env 包装成 discrete action env。
    常用于：
    - 你的底层环境统一写成 continuous
    - 想给 DQN 用离散动作时，用这个 wrapper 离散化

    注意：
    这个 wrapper 只适用于底层 action_space 是 Box(shape=(1,))
    """

    def __init__(self, env: gym.Env, discrete_actions):
        super().__init__(env)
        self.discrete_actions = np.asarray(discrete_actions, dtype=np.float32).reshape(-1)
        assert self.discrete_actions.size > 0, "discrete_actions must be non-empty."

        self.action_space = spaces.Discrete(int(self.discrete_actions.size))

    def action(self, act):
        idx = int(act)
        if idx < 0 or idx >= self.discrete_actions.size:
            raise ValueError(f"Action index out of range: {idx}")

        # map discrete pwm to normalized continuous action in [-1, 1]
        if not hasattr(self.env, "continuous_pwm_limit"):
            raise AttributeError("Underlying env must have attribute 'continuous_pwm_limit'.")

        pwm_limit = float(self.env.continuous_pwm_limit)
        a = float(self.discrete_actions[idx]) / max(pwm_limit, 1e-8)
        a = np.clip(a, -1.0, 1.0)
        return np.array([a], dtype=np.float32)


class ActionRepeatWrapper(gym.Wrapper):
    """
    一个简单的 action repeat wrapper。
    可选，不一定要用。
    """

    def __init__(self, env: gym.Env, repeat: int = 1):
        super().__init__(env)
        if repeat < 1:
            raise ValueError("repeat must be >= 1")
        self.repeat = int(repeat)

    def step(self, action):
        total_reward = 0.0
        final_obs = None
        final_info = {}
        terminated = False
        truncated = False

        for _ in range(self.repeat):
            final_obs, reward, terminated, truncated, final_info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        return final_obs, total_reward, terminated, truncated, final_info