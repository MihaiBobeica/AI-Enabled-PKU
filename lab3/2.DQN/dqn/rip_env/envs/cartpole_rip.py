# rip_env/envs/cartpole_rip.py
# by Zhixiang Ju
# 2026.4.22

from __future__ import annotations

from typing import Optional, Callable, Dict, Any, Tuple
from collections import deque
import copy
import re
import json
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .types import EnvConfig, RIPPhysicalParams
from .dynamics import rk4_step, furuta_derivatives, wrap_to_pi, state_to_trig_obs, state_to_raw_obs
from .reward_fns import default_balance_reward_fn
from .done_fns import default_done_fn
from .logger import EnvLogger


RewardFnType = Callable[[np.ndarray, object, np.ndarray, object], float]
DoneFnType = Callable[[np.ndarray, object, np.ndarray, object], Tuple[bool, bool, Dict[str, Any]]]


class _ResidualMLPForEnv:
    """
    Lightweight residual-network wrapper used only when rn_enabled=True.

    Supported deployed network types:
        mode1/state residual:
            output_dim = 4
            [dtheta, dtheta_dot, dalpha, dalpha_dot]

        mode2/acceleration residual:
            output_dim = 2
            [dtheta_ddot, dalpha_ddot]

        mode3/action residual:
            output_dim = 1
            [delta_u]

    It is intentionally self-contained so the original environment still works
    without torch installed when rn_enabled=False.
    """

    def __init__(self, model_path: str, norm_path: str = "", device: str = "cpu") -> None:
        import torch
        from torch import nn

        self.torch = torch
        self.device = torch.device(device)

        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"RN model_path not found: {model_path}")

        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)

        self.input_dim = int(ckpt.get("input_dim", 25))
        self.output_dim = int(ckpt.get("output_dim", 4))
        self.seq_len = int(ckpt.get("seq_len", max(self.input_dim // 5, 1)))
        self.target_mode = str(ckpt.get("target_mode", "unknown")).strip().lower()
        self.target_names = list(ckpt.get("target_names", []))

        h1 = int(ckpt.get("h1", 64))
        h2 = int(ckpt.get("h2", 64))
        dropout = float(ckpt.get("dropout", 0.0))

        class ResidualMLP(nn.Module):
            def __init__(self, in_dim, out_dim, h1=64, h2=64, dropout=0.0):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, h1),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(h1, h2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(h2, out_dim),
                )

            def forward(self, x):
                return self.net(x)

        self.model = ResidualMLP(
            in_dim=self.input_dim,
            out_dim=self.output_dim,
            h1=h1,
            h2=h2,
            dropout=dropout,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        # New training scripts save scalers directly in the checkpoint.
        # Older scripts save them in normalization_stats.json.  Support both.
        if all(k in ckpt for k in ("x_mean", "x_std", "y_mean", "y_std")):
            self.x_mean = np.asarray(ckpt["x_mean"], dtype=np.float64)
            self.x_std = np.asarray(ckpt["x_std"], dtype=np.float64)
            self.y_mean = np.asarray(ckpt["y_mean"], dtype=np.float64)
            self.y_std = np.asarray(ckpt["y_std"], dtype=np.float64)
        else:
            if not norm_path or not os.path.exists(norm_path):
                raise FileNotFoundError(
                    "RN normalization stats not found. Either save x_mean/x_std/y_mean/y_std "
                    f"inside checkpoint or provide rn_norm_path. rn_norm_path={norm_path}"
                )
            with open(norm_path, "r", encoding="utf-8") as f:
                norm = json.load(f)
            self.x_mean = np.asarray(norm["x_scaler"]["mean"], dtype=np.float64)
            self.x_std = np.asarray(norm["x_scaler"]["std"], dtype=np.float64)
            self.y_mean = np.asarray(norm["y_scaler"]["mean"], dtype=np.float64)
            self.y_std = np.asarray(norm["y_scaler"]["std"], dtype=np.float64)

        self.x_std = np.where(np.abs(self.x_std) < 1e-12, 1.0, self.x_std)
        self.y_std = np.where(np.abs(self.y_std) < 1e-12, 1.0, self.y_std)

        if self.x_mean.shape[0] != self.input_dim:
            raise ValueError(
                f"RN input_dim mismatch: ckpt input_dim={self.input_dim}, "
                f"scaler x_dim={self.x_mean.shape[0]}"
            )
        if self.y_mean.shape[0] != self.output_dim:
            raise ValueError(
                f"RN output_dim mismatch: ckpt output_dim={self.output_dim}, "
                f"scaler y_dim={self.y_mean.shape[0]}"
            )

    def predict(self, x_raw_flat: np.ndarray) -> np.ndarray:
        x = np.asarray(x_raw_flat, dtype=np.float64).reshape(1, -1)
        if x.shape[1] != self.input_dim:
            raise ValueError(f"RN input shape mismatch: got {x.shape[1]}, expected {self.input_dim}")
        x_norm = (x - self.x_mean.reshape(1, -1)) / self.x_std.reshape(1, -1)
        with self.torch.no_grad():
            xt = self.torch.as_tensor(x_norm, dtype=self.torch.float32, device=self.device)
            y_norm = self.model(xt).detach().cpu().numpy()
        y = y_norm * self.y_std.reshape(1, -1) + self.y_mean.reshape(1, -1)
        return y.reshape(-1).astype(np.float64)


class CartPoleRIPEenv(gym.Env):
    """
    Rotary Inverted Pendulum environment.

    True state:
        [theta, theta_dot, alpha, alpha_dot]

    Convention:
        alpha = 0        -> upright
        alpha = +/- pi   -> downward

    Supported observation types:
        "trig":
            [sin(theta), cos(theta), theta_dot, sin(alpha), cos(alpha), alpha_dot]

        "raw":
            [theta, theta_dot, alpha, alpha_dot]

        "trig_hist4_act4":
            最近4步 trig observation 叠加 + 最近4步 action history
            shape = 4 * 6 + 4 = 28

        "raw_hist4_act4":
            最近4步 raw observation 叠加 + 最近4步 action history
            shape = 4 * 4 + 4 = 20

    更一般地，也支持：
        "trig_histN"
        "trig_histN_actM"
        "raw_histN"
        "raw_histN_actM"

    Important note:
        DO NOT put key "episode" into info.
        SB3's Monitor wrapper reserves info["episode"] for episode summary dict.

    新增但不破坏旧接口的初始化模式：
        - balance_random_small
          使用现有字段表达“有界均匀随机小扰动”
          其中：
              theta      ~ U(theta_mean_deg - theta_std_deg, theta_mean_deg + theta_std_deg)
              theta_dot  ~ U(theta_dot_mean - theta_dot_std, theta_dot_mean + theta_dot_std)
              alpha      ~ U(alpha_mean_deg - alpha_std_deg, alpha_mean_deg + alpha_std_deg)
              alpha_dot  ~ U(alpha_dot_mean - alpha_dot_std, alpha_dot_mean + alpha_dot_std)
    """

    metadata = {"render_modes": ["human", None], "render_fps": 60}

    def __init__(
        self,
        env_config: Optional[EnvConfig] = None,
        reward_fn: Optional[RewardFnType] = None,
        done_fn: Optional[DoneFnType] = None,
        render_mode: Optional[str] = None,
        **override_kwargs,
    ) -> None:
        super().__init__()

        if env_config is None:
            env_config = EnvConfig()

        self.cfg: EnvConfig = copy.deepcopy(env_config)

        for k, v in override_kwargs.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)

        self.render_mode = render_mode

        # ==================== basic configs ====================
        self.dt = float(self.cfg.dt)
        self.max_steps = int(self.cfg.max_steps)

        self.action_type = str(self.cfg.action_type)
        self.discrete_actions = np.asarray(self.cfg.discrete_actions, dtype=np.float32).reshape(-1)
        self.continuous_pwm_limit = float(self.cfg.continuous_pwm_limit)

        self.observation_type = str(self.cfg.observation_type)
        self.clip_velocity_in_obs = bool(self.cfg.clip_velocity_in_obs)

        self.params: RIPPhysicalParams = self.cfg.physical_params

        self.theta_limit = float(self.cfg.limits.theta_limit)
        self.theta_dot_limit = float(self.cfg.limits.theta_dot_limit)
        self.alpha_dot_limit = float(self.cfg.limits.alpha_dot_limit)

        self.terminate_on_alpha_abs_deg = bool(self.cfg.limits.terminate_on_alpha_abs_deg)
        self.alpha_abs_limit_deg = float(self.cfg.limits.alpha_abs_limit_deg)

        self.noise_cfg = self.cfg.noise
        self.init_cfg = self.cfg.init_state
        self.log_cfg = self.cfg.logging

        self.reward_fn = reward_fn if reward_fn is not None else default_balance_reward_fn
        self.done_fn = done_fn if done_fn is not None else default_done_fn

        # ==================== logger ====================
        self.logger = EnvLogger(
            log_dir=self.log_cfg.log_dir,
            enabled=self.log_cfg.enabled,
            episode_csv_name=self.log_cfg.episode_csv_name,
            step_csv_name=self.log_cfg.step_csv_name,
            episode_jsonl_name=self.log_cfg.episode_jsonl_name,
            save_step_log=self.log_cfg.save_step_log,
            flush_every_step=self.log_cfg.flush_every_step,
        )

        if self.log_cfg.enabled and self.log_cfg.write_config_json:
            self.logger.write_env_config(self.cfg.to_dict())

        # ==================== parse observation mode ====================
        self.base_observation_type, self.obs_history_len, self.act_history_len = self._parse_observation_type(
            self.observation_type
        )

        # ==================== action space ====================
        if self.action_type == "discrete":
            if self.discrete_actions.size == 0:
                raise ValueError("discrete_actions must be non-empty when action_type='discrete'.")
            self.action_space = spaces.Discrete(int(self.discrete_actions.size))

        elif self.action_type == "continuous":
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32,
            )
        else:
            raise ValueError(f"Unknown action_type: {self.action_type}")

        # ==================== observation space ====================
        self.base_obs_dim = self._get_base_obs_dim(self.base_observation_type)
        total_obs_dim = self.base_obs_dim * self.obs_history_len + self.act_history_len

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_obs_dim,),
            dtype=np.float32,
        )

        # ==================== runtime states ====================
        self.state: Optional[np.ndarray] = None
        self.step_count: int = 0
        self.episode_idx: int = 0
        self.episode_reward: float = 0.0
        self.max_abs_theta: float = 0.0
        self.max_abs_alpha: float = 0.0

        self.last_action = None
        self.last_pwm = 0.0

        # ==================== optional LPF velocity estimation ====================
        # Default is False, so TD3/PPO and old experiments keep using true simulation velocities.
        # When enabled, observation/reward/done use velocities estimated from angle differences
        # with a low-pass filter, while dynamics integration still uses the true simulation state.
        self.use_lpf_velocity = bool(getattr(self.cfg, "use_lpf_velocity", False))
        self.velocity_lpf = float(getattr(self.cfg, "velocity_lpf", 0.25))

        self.theta_meas_prev = 0.0
        self.alpha_meas_prev = 0.0
        self.theta_unwrap_meas = 0.0
        self.theta_dot_est = 0.0
        self.alpha_dot_est = 0.0
        self.measured_state: Optional[np.ndarray] = None

        # 历史缓存
        self.obs_history: deque[np.ndarray] = deque(maxlen=self.obs_history_len)
        self.act_history: deque[np.ndarray] = deque(maxlen=self.act_history_len)

        # ==================== residual network correction configs ====================
        # 默认全部关闭，因此不影响原有干净环境功能。
        # rn_mode 新旧兼容：
        #   none             : 不启用
        #   direct_state     : 直接修正四维状态，旧 direct_dynamics 的新名字
        #   acceleration     : 修正加速度，默认推荐
        #   action           : 修正输入 u
        #   replay           : 只修正 replay buffer 中的 next_obs，保留旧接口
        self.rn_enabled = bool(getattr(self.cfg, "rn_enabled", False))
        self.rn_mode = str(getattr(self.cfg, "rn_mode", "none")).strip().lower()
        alias = {
            "direct_dynamics": "direct_state",
            "state": "direct_state",
            "direct_state": "direct_state",
            "acc": "acceleration",
            "accel": "acceleration",
            "acceleration": "acceleration",
            "input": "action",
            "u": "action",
            "action": "action",
            "replay": "replay",
            "none": "none",
        }
        self.rn_mode = alias.get(self.rn_mode, self.rn_mode)
        if not self.rn_enabled:
            self.rn_mode = "none"
        valid_rn_modes = {"none", "direct_state", "acceleration", "replay", "action"}
        if self.rn_mode not in valid_rn_modes:
            raise ValueError(f"Unknown rn_mode={self.rn_mode}. Valid modes: {sorted(valid_rn_modes)}")

        self.rn_seq_len = int(getattr(self.cfg, "rn_seq_len", 5))
        self.rn_lambda = float(getattr(self.cfg, "rn_lambda", 0.3))
        self.rn_start_step = int(getattr(self.cfg, "rn_start_step", 0))
        self.rn_random_scale = bool(getattr(self.cfg, "rn_random_scale", False))
        self.rn_random_scale_low = float(getattr(self.cfg, "rn_random_scale_low", 0.0))
        self.rn_random_scale_high = float(getattr(self.cfg, "rn_random_scale_high", 1.0))
        self.rn_episode_lambda = float(self.rn_lambda)

        self.rn_du_max = float(getattr(self.cfg, "rn_du_max", 20.0))
        self.rn_acc_clip = float(getattr(self.cfg, "rn_acc_clip", 300.0))
        self.rn_state_clip = np.asarray(getattr(self.cfg, "rn_state_clip", [0.2, 5.0, 0.2, 5.0]), dtype=np.float64).reshape(4)
        self.rn_rho = float(getattr(self.cfg, "rn_rho", 1e-2))
        self.rn_jac_eps = float(getattr(self.cfg, "rn_jac_eps", 1e-3))
        self.rn_W = np.asarray(getattr(self.cfg, "rn_W", [0.0, 0.2, 5.0, 1.0]), dtype=np.float64).reshape(4)
        self.rn_history_use_corrected = bool(getattr(self.cfg, "rn_history_use_corrected", False))
        self.rn_device = str(getattr(self.cfg, "rn_device", "cpu"))

        self.rn_model = None
        if self.rn_enabled and self.rn_mode != "none":
            self.rn_model = _ResidualMLPForEnv(
                model_path=str(getattr(self.cfg, "rn_model_path", "")),
                norm_path=str(getattr(self.cfg, "rn_norm_path", "")),
                device=self.rn_device,
            )
            # 以模型 checkpoint 为准，避免配置和模型不一致。
            self.rn_seq_len = int(self.rn_model.seq_len)

        # RN 使用真实 raw state + raw pwm 的历史，独立于 agent observation history。
        self.rn_history: deque[Tuple[np.ndarray, float]] = deque(maxlen=max(self.rn_seq_len, 1))

    # ============================================================
    # parse observation type
    # ============================================================

    def _parse_observation_type(self, obs_type: str) -> Tuple[str, int, int]:
        """
        支持：
            trig
            raw
            trig_hist4
            trig_hist4_act4
            raw_hist4
            raw_hist4_act4
        """
        obs_type = obs_type.strip()

        if obs_type in ("trig", "raw"):
            return obs_type, 1, 0

        pattern = r"^(trig|raw)_hist(\d+)(?:_act(\d+))?$"
        m = re.match(pattern, obs_type)
        if m is None:
            raise ValueError(
                f"Unknown observation_type: {obs_type}. "
                f"Supported examples: 'trig', 'raw', 'trig_hist4_act4', 'raw_hist4_act4'"
            )

        base_type = m.group(1)
        obs_hist_len = int(m.group(2))
        act_hist_len = int(m.group(3)) if m.group(3) is not None else 0

        if obs_hist_len <= 0:
            raise ValueError(f"obs_history_len must be >= 1, got {obs_hist_len}")
        if act_hist_len < 0:
            raise ValueError(f"act_history_len must be >= 0, got {act_hist_len}")

        return base_type, obs_hist_len, act_hist_len

    def _get_base_obs_dim(self, base_type: str) -> int:
        if base_type == "trig":
            return 6
        if base_type == "raw":
            return 4
        raise ValueError(f"Unknown base observation type: {base_type}")

    # ============================================================
    # helpers
    # ============================================================

    def _uniform_center_halfwidth(self, center: float, halfwidth: float) -> float:
        halfwidth = abs(float(halfwidth))
        if halfwidth <= 0.0:
            return float(center)
        low = float(center) - halfwidth
        high = float(center) + halfwidth
        return float(self.np_random.uniform(low, high))

    def _sample_initial_state(self) -> np.ndarray:
        """
        Sample initial true state.

        保留旧逻辑：
            - custom
            - downward
            - upright
            - 其他模式：按 alpha_mean_deg / alpha_std_deg 高斯采样

        新增：
            - balance_random_small
              使用“有界均匀随机”初始化，便于只训练 balance 阶段
        """
        mode = str(self.init_cfg.mode).lower()

        if mode == "custom":
            if self.init_cfg.custom_state is None:
                raise ValueError("init_state.mode='custom' but custom_state is None.")
            state = np.asarray(self.init_cfg.custom_state, dtype=np.float64).reshape(4)
            state[2] = wrap_to_pi(float(state[2]))
            return state

        # 新增模式：只做附加，不影响旧模式
        if mode == "balance_random_small":
            theta = self._uniform_center_halfwidth(
                np.deg2rad(self.init_cfg.theta_mean_deg),
                np.deg2rad(self.init_cfg.theta_std_deg),
            )
            theta_dot = self._uniform_center_halfwidth(
                self.init_cfg.theta_dot_mean,
                self.init_cfg.theta_dot_std,
            )
            alpha = self._uniform_center_halfwidth(
                np.deg2rad(self.init_cfg.alpha_mean_deg),
                np.deg2rad(self.init_cfg.alpha_std_deg),
            )
            alpha = wrap_to_pi(float(alpha))
            alpha_dot = self._uniform_center_halfwidth(
                self.init_cfg.alpha_dot_mean,
                self.init_cfg.alpha_dot_std,
            )
            return np.array([theta, theta_dot, alpha, alpha_dot], dtype=np.float64)

        theta = self.np_random.normal(
            np.deg2rad(self.init_cfg.theta_mean_deg),
            np.deg2rad(self.init_cfg.theta_std_deg),
        )
        theta_dot = self.np_random.normal(
            self.init_cfg.theta_dot_mean,
            self.init_cfg.theta_dot_std,
        )

        if mode == "downward":
            alpha_center = np.pi
        elif mode == "upright":
            alpha_center = 0.0
        else:
            alpha_center = np.deg2rad(self.init_cfg.alpha_mean_deg)

        alpha = self.np_random.normal(
            alpha_center,
            np.deg2rad(self.init_cfg.alpha_std_deg),
        )
        alpha = wrap_to_pi(float(alpha))

        alpha_dot = self.np_random.normal(
            self.init_cfg.alpha_dot_mean,
            self.init_cfg.alpha_dot_std,
        )

        return np.array([theta, theta_dot, alpha, alpha_dot], dtype=np.float64)

    def _state_to_base_obs(self, state: np.ndarray) -> np.ndarray:
        if self.base_observation_type == "trig":
            return state_to_trig_obs(
                state=state,
                theta_dot_limit=self.theta_dot_limit,
                alpha_dot_limit=self.alpha_dot_limit,
                clip_velocity=self.clip_velocity_in_obs,
            )
        elif self.base_observation_type == "raw":
            return state_to_raw_obs(
                state=state,
                theta_dot_limit=self.theta_dot_limit,
                alpha_dot_limit=self.alpha_dot_limit,
                clip_velocity=self.clip_velocity_in_obs,
            )
        else:
            raise RuntimeError(f"Invalid base_observation_type: {self.base_observation_type}")

    def _apply_observation_noise(self, state: np.ndarray) -> np.ndarray:
        if not self.noise_cfg.enabled:
            return state

        noisy = state.copy()
        noisy[0] += self.np_random.normal(0.0, self.noise_cfg.theta_sigma)
        noisy[1] += self.np_random.normal(0.0, self.noise_cfg.theta_dot_sigma)
        noisy[2] = wrap_to_pi(
            float(noisy[2] + self.np_random.normal(0.0, self.noise_cfg.alpha_sigma))
        )
        noisy[3] += self.np_random.normal(0.0, self.noise_cfg.alpha_dot_sigma)
        return noisy

    def _reset_velocity_estimator(self, state: np.ndarray) -> np.ndarray:
        """
        Reset measured state used when use_lpf_velocity=True.

        measured_state = [theta_unwrap_meas, theta_dot_est, alpha_wrap, alpha_dot_est]
        """
        s = np.asarray(state, dtype=np.float64).reshape(4)
        theta = float(s[0])
        theta_dot = float(s[1])
        alpha = wrap_to_pi(float(s[2]))
        alpha_dot = float(s[3])

        self.theta_meas_prev = theta
        self.alpha_meas_prev = alpha
        self.theta_unwrap_meas = theta
        self.theta_dot_est = theta_dot
        self.alpha_dot_est = alpha_dot

        self.measured_state = np.array(
            [self.theta_unwrap_meas, self.theta_dot_est, alpha, self.alpha_dot_est],
            dtype=np.float64,
        )
        return self.measured_state.copy()

    def _update_velocity_estimator(self, next_state: np.ndarray) -> np.ndarray:
        """
        Update LPF velocity estimates from angle differences.

        MATLAB-aligned logic:
            thetaDot_raw = wrapToPi(theta_k - theta_{k-1}) / dt
            alphaDot_raw = wrapToPi(alpha_k - alpha_{k-1}) / dt
            thetaDot_est = (1-lpf)*thetaDot_est + lpf*thetaDot_raw
            alphaDot_est = (1-lpf)*alphaDot_est + lpf*alphaDot_raw
        """
        s = np.asarray(next_state, dtype=np.float64).reshape(4)
        theta_meas = float(s[0])
        alpha_meas = wrap_to_pi(float(s[2]))

        dtheta_wrapped = wrap_to_pi(theta_meas - self.theta_meas_prev)
        dalpha_wrapped = wrap_to_pi(alpha_meas - self.alpha_meas_prev)

        self.theta_unwrap_meas += dtheta_wrapped

        theta_dot_raw = dtheta_wrapped / self.dt
        alpha_dot_raw = dalpha_wrapped / self.dt

        lpf = float(np.clip(self.velocity_lpf, 0.0, 1.0))
        self.theta_dot_est = (1.0 - lpf) * self.theta_dot_est + lpf * theta_dot_raw
        self.alpha_dot_est = (1.0 - lpf) * self.alpha_dot_est + lpf * alpha_dot_raw

        self.theta_meas_prev = theta_meas
        self.alpha_meas_prev = alpha_meas

        self.measured_state = np.array(
            [self.theta_unwrap_meas, self.theta_dot_est, alpha_meas, self.alpha_dot_est],
            dtype=np.float64,
        )
        return self.measured_state.copy()

    def _get_agent_state(self, true_state: np.ndarray) -> np.ndarray:
        """Return state used by observation/reward/done.

        Dynamics always uses true_state. When LPF mode is off, this keeps the original behavior.
        """
        if self.use_lpf_velocity:
            if self.measured_state is None:
                return self._reset_velocity_estimator(true_state)
            return self.measured_state.copy()
        return np.asarray(true_state, dtype=np.float64).reshape(4).copy()

    def _action_to_pwm(self, action) -> float:
        if self.action_type == "discrete":
            idx = int(action)
            if idx < 0 or idx >= len(self.discrete_actions):
                raise ValueError(f"Discrete action index out of range: {idx}")
            return float(self.discrete_actions[idx])

        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        a = float(action_arr[0])
        a = np.clip(a, -1.0, 1.0)
        return float(a * self.continuous_pwm_limit)

    def _pwm_to_action_feature(self, pwm: float) -> np.ndarray:
        """
        历史 action 统一用归一化 PWM 表示，范围约在 [-1, 1]
        """
        if self.continuous_pwm_limit <= 0:
            raise ValueError(f"continuous_pwm_limit must be > 0, got {self.continuous_pwm_limit}")
        a = float(pwm) / float(self.continuous_pwm_limit)
        a = float(np.clip(a, -1.0, 1.0))
        return np.array([a], dtype=np.float32)

    def _build_full_observation(self, base_obs: np.ndarray) -> np.ndarray:
        """
        - 原模式：直接返回 base_obs
        - 历史模式：拼接 obs_history 和 act_history
        """
        if self.obs_history_len == 1 and self.act_history_len == 0:
            return base_obs.astype(np.float32)

        obs_parts = []
        for x in self.obs_history:
            obs_parts.append(np.asarray(x, dtype=np.float32).reshape(-1))

        act_parts = []
        for a in self.act_history:
            act_parts.append(np.asarray(a, dtype=np.float32).reshape(-1))

        if len(obs_parts) == 0:
            raise RuntimeError("obs_history is empty when building full observation.")

        full_parts = obs_parts + act_parts
        return np.concatenate(full_parts, axis=0).astype(np.float32)

    def _init_histories(self, initial_base_obs: np.ndarray) -> None:
        self.obs_history = deque(maxlen=self.obs_history_len)
        self.act_history = deque(maxlen=self.act_history_len)

        for _ in range(self.obs_history_len):
            self.obs_history.append(initial_base_obs.copy())

        for _ in range(self.act_history_len):
            self.act_history.append(np.zeros((1,), dtype=np.float32))

    def _build_full_observation_from_histories(
        self,
        obs_history_values,
        act_history_values,
        fallback_base_obs: np.ndarray,
    ) -> np.ndarray:
        """Build an observation from explicit history lists without mutating env state."""
        if self.obs_history_len == 1 and self.act_history_len == 0:
            return np.asarray(fallback_base_obs, dtype=np.float32).reshape(-1)

        obs_parts = [np.asarray(x, dtype=np.float32).reshape(-1) for x in obs_history_values]
        act_parts = [np.asarray(a, dtype=np.float32).reshape(-1) for a in act_history_values]
        full_parts = obs_parts + act_parts
        return np.concatenate(full_parts, axis=0).astype(np.float32)

    def _init_rn_history(self, initial_state: np.ndarray) -> None:
        self.rn_history = deque(maxlen=max(self.rn_seq_len, 1))
        s0 = np.asarray(initial_state, dtype=np.float64).reshape(4).copy()
        for _ in range(max(self.rn_seq_len, 1)):
            self.rn_history.append((s0.copy(), 0.0))

    def _build_rn_input(self, current_state: np.ndarray, current_pwm: float) -> np.ndarray:
        """
        Build flattened RN input as [x_{k-N+1},u_{k-N+1},...,x_k,u_k].

        The stored RN history contains states and the action that led to each state.
        For current prediction, the last entry's action is replaced by the current
        candidate pwm, matching the training target convention.
        """
        if self.rn_model is None:
            raise RuntimeError("RN model is not loaded.")

        entries = list(self.rn_history)
        if len(entries) == 0:
            entries = [(np.asarray(current_state, dtype=np.float64).reshape(4).copy(), 0.0)]

        while len(entries) < self.rn_seq_len:
            entries.insert(0, entries[0])
        entries = entries[-self.rn_seq_len:]

        entries[-1] = (np.asarray(current_state, dtype=np.float64).reshape(4).copy(), float(current_pwm))

        flat = []
        for st, u in entries:
            st = np.asarray(st, dtype=np.float64).reshape(4)
            flat.extend([float(st[0]), float(st[1]), float(st[2]), float(st[3]), float(u)])
        return np.asarray(flat, dtype=np.float64)

    def _rn_effective_lambda(self) -> float:
        """Residual strength used at this step.

        rn_start_step keeps the first part of training identical to the original
        simulator.  rn_random_scale samples one episode-level multiplier in reset().
        """
        if (not self.rn_enabled) or self.rn_mode == "none":
            return 0.0
        if self.step_count < self.rn_start_step:
            return 0.0
        return float(self.rn_episode_lambda)

    def _predict_rn_residual(self, current_state: np.ndarray, current_pwm: float) -> np.ndarray:
        if (not self.rn_enabled) or self.rn_mode == "none" or self.rn_model is None:
            return np.zeros(1, dtype=np.float64)
        rn_input = self._build_rn_input(current_state, current_pwm)
        res = self.rn_model.predict(rn_input).astype(np.float64).reshape(-1)
        return res

    def _wrap_state_angles(self, state: np.ndarray) -> np.ndarray:
        s = np.asarray(state, dtype=np.float64).reshape(4).copy()
        s[0] = wrap_to_pi(float(s[0]))
        s[2] = wrap_to_pi(float(s[2]))
        return s

    def _extract_state_residual(self, residual: np.ndarray) -> np.ndarray:
        r = np.asarray(residual, dtype=np.float64).reshape(-1)
        if r.size < 4:
            raise ValueError(f"State residual mode needs output_dim >= 4, got {r.size}")
        r4 = r[:4].copy()
        r4 = np.clip(r4, -self.rn_state_clip, self.rn_state_clip)
        return r4

    def _extract_acc_residual(self, residual: np.ndarray) -> np.ndarray:
        r = np.asarray(residual, dtype=np.float64).reshape(-1)
        if r.size == 2:
            acc = r[:2]
        elif r.size >= 6:
            # Backward compatibility with old 6D checkpoints:
            # [4D state residual, 2D acceleration residual]
            acc = r[4:6]
        else:
            raise ValueError(f"Acceleration residual mode needs output_dim=2 or >=6, got {r.size}")
        return np.clip(acc.astype(np.float64), -self.rn_acc_clip, self.rn_acc_clip)

    def _apply_state_residual(self, next_state_sim: np.ndarray, residual: np.ndarray) -> np.ndarray:
        lam = self._rn_effective_lambda()
        r4 = self._extract_state_residual(residual)
        s = np.asarray(next_state_sim, dtype=np.float64).reshape(4) + lam * r4
        return self._wrap_state_angles(s)

    def _apply_acceleration_residual(self, current_state: np.ndarray, pwm: float, residual: np.ndarray) -> np.ndarray:
        """Apply RN acceleration residual while keeping kinematic consistency.

        theta_dot_next = theta_dot + dt * (theta_ddot_sim + lambda*dtheta_ddot_res)
        alpha_dot_next = alpha_dot + dt * (alpha_ddot_sim + lambda*dalpha_ddot_res)
        theta_next     = theta + dt * theta_dot_next
        alpha_next     = alpha + dt * alpha_dot_next
        """
        lam = self._rn_effective_lambda()
        s = np.asarray(current_state, dtype=np.float64).reshape(4)
        acc = self._extract_acc_residual(residual)
        f = furuta_derivatives(s, pwm, self.params)

        theta_ddot = float(f[1] + lam * acc[0])
        alpha_ddot = float(f[3] + lam * acc[1])

        theta_dot_next = float(s[1] + self.dt * theta_ddot)
        alpha_dot_next = float(s[3] + self.dt * alpha_ddot)
        theta_next = wrap_to_pi(float(s[0] + self.dt * theta_dot_next))
        alpha_next = wrap_to_pi(float(s[2] + self.dt * alpha_dot_next))

        return np.array([theta_next, theta_dot_next, alpha_next, alpha_dot_next], dtype=np.float64)

    def _compute_rn_action_correction(self, current_state: np.ndarray, pwm: float, residual: np.ndarray) -> Tuple[float, float]:
        """Compute effective input.

        If the deployed RN has output_dim=1, use it directly as delta_u.
        If it has output_dim>=4, preserve the old behavior and project 4D state
        residual into the simulator input channel.
        """
        lam = self._rn_effective_lambda()
        r = np.asarray(residual, dtype=np.float64).reshape(-1)
        pwm = float(pwm)

        if r.size == 1:
            delta_u = float(lam * r[0])
            delta_u = float(np.clip(delta_u, -self.rn_du_max, self.rn_du_max))
            u_limit = float(self.continuous_pwm_limit)
            pwm_eff = float(np.clip(pwm + delta_u, -u_limit, u_limit))
            return pwm_eff, delta_u

        if r.size < 4:
            raise ValueError(f"Action correction needs output_dim=1 or >=4, got {r.size}")

        eps = max(abs(float(self.rn_jac_eps)), 1e-6)

        s_plus = rk4_step(state=current_state, pwm=pwm + eps, dt=self.dt, p=self.params)
        s_minus = rk4_step(state=current_state, pwm=pwm - eps, dt=self.dt, p=self.params)
        B = (np.asarray(s_plus, dtype=np.float64) - np.asarray(s_minus, dtype=np.float64)) / (2.0 * eps)
        B[0] = wrap_to_pi(float(s_plus[0] - s_minus[0])) / (2.0 * eps)
        B[2] = wrap_to_pi(float(s_plus[2] - s_minus[2])) / (2.0 * eps)

        W = np.diag(self.rn_W.astype(np.float64))
        r4 = self._extract_state_residual(r)
        denom = float(B.T @ W @ B + self.rn_rho)
        if abs(denom) < 1e-12:
            delta_u = 0.0
        else:
            delta_u = float((B.T @ W @ r4) / denom)

        delta_u = float(np.clip(lam * delta_u, -self.rn_du_max, self.rn_du_max))
        u_limit = float(self.continuous_pwm_limit)
        pwm_eff = float(np.clip(pwm + delta_u, -u_limit, u_limit))
        return pwm_eff, delta_u

    def get_true_state(self) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Environment has not been reset yet.")
        return self.state.copy()

    def set_max_steps(self, max_steps: int) -> None:
        self.max_steps = int(max_steps)
        self.cfg.max_steps = int(max_steps)

    # ============================================================
    # gymnasium api
    # ============================================================

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.episode_idx += 1
        self.step_count = 0
        self.episode_reward = 0.0
        self.max_abs_theta = 0.0
        self.max_abs_alpha = 0.0

        self.last_action = None
        self.last_pwm = 0.0

        self.state = self._sample_initial_state()
        self._init_rn_history(self.state)

        if self.use_lpf_velocity:
            agent_state = self._reset_velocity_estimator(self.state)
        else:
            self.measured_state = None
            agent_state = self.state.copy()

        obs_state = self._apply_observation_noise(agent_state)
        base_obs = self._state_to_base_obs(obs_state)

        self._init_histories(base_obs)
        obs = self._build_full_observation(base_obs)

        info = {
            "state": self.state.copy(),
            "true_state": self.state.copy(),
            "measured_state": agent_state.copy(),
            "use_lpf_velocity": bool(self.use_lpf_velocity),
            "velocity_lpf": float(self.velocity_lpf),
            "episode_index": self.episode_idx,
            "step_count": self.step_count,
            "reset_options": options,
            "base_observation_dim": self.base_obs_dim,
            "obs_history_len": self.obs_history_len,
            "act_history_len": self.act_history_len,
        }

        return obs, info

    def step(self, action):
        if self.state is None:
            raise RuntimeError("You must call reset() before step().")

        current_state = self.state.copy()
        pwm_raw = self._action_to_pwm(action)
        pwm_used = float(pwm_raw)
        delta_u = 0.0
        rn_residual = np.zeros(4, dtype=np.float64)
        rn_next_state_for_replay = None
        rn_obs_for_replay = None
        rn_reward_for_replay = None
        rn_terminated_for_replay = None
        rn_truncated_for_replay = None

        if self.rn_enabled and self.rn_mode != "none":
            rn_residual = self._predict_rn_residual(current_state, pwm_raw)

        # ============================================================
        # 1) Original dynamics, or action-corrected dynamics
        # ============================================================
        if self.rn_enabled and self.rn_mode == "action":
            pwm_used, delta_u = self._compute_rn_action_correction(current_state, pwm_raw, rn_residual)

        next_state_sim = rk4_step(
            state=current_state,
            pwm=pwm_used,
            dt=self.dt,
            p=self.params,
        )
        next_state_sim = self._wrap_state_angles(next_state_sim)

        # ============================================================
        # 2) Direct-dynamics correction: changes actual rollout state
        # ============================================================
        if self.rn_enabled and self.rn_mode == "direct_dynamics":
            next_state = self._apply_state_residual(next_state_sim, rn_residual)
        else:
            next_state = next_state_sim

        # Agent-facing state. Only this state is used for observation/reward/done.
        # Rollout dynamics and RN history remain based on the true simulation state.
        if self.use_lpf_velocity:
            if self.rn_enabled and self.rn_mode == "replay":
                raise NotImplementedError(
                    "rn_mode='replay' with use_lpf_velocity=True needs separate "
                    "measured-state handling for corrected replay transitions."
                )
            agent_current_state = (
                self.measured_state.copy()
                if self.measured_state is not None
                else self._reset_velocity_estimator(current_state)
            )
            agent_next_state = self._update_velocity_estimator(next_state)
        else:
            agent_current_state = current_state.copy()
            agent_next_state = next_state.copy()

        # ============================================================
        # 3) Replay-only correction: does NOT change rollout state
        # ============================================================
        if self.rn_enabled and self.rn_mode == "replay":
            # Here next_state_sim uses the original raw action because replay mode does not correct action.
            rn_next_state_for_replay = self._apply_state_residual(next_state_sim, rn_residual)
            rn_reward_for_replay = float(self.reward_fn(current_state, action, rn_next_state_for_replay, self))
            rn_terminated_for_replay, rn_truncated_for_replay, _rn_extra = self.done_fn(
                current_state, action, rn_next_state_for_replay, self
            )
            rn_terminated_for_replay = bool(rn_terminated_for_replay)
            rn_truncated_for_replay = bool(rn_truncated_for_replay)

            corrected_obs_state = self._apply_observation_noise(rn_next_state_for_replay)
            corrected_base_obs = self._state_to_base_obs(corrected_obs_state)

            # Construct the corrected next observation that should be stored in replay buffer.
            obs_hist_corr = list(self.obs_history)
            while len(obs_hist_corr) < self.obs_history_len:
                obs_hist_corr.insert(0, corrected_base_obs.copy())
            obs_hist_corr = obs_hist_corr[-self.obs_history_len:]
            if len(obs_hist_corr) > 0:
                obs_hist_corr = obs_hist_corr[1:] + [corrected_base_obs.copy()] if self.obs_history_len > 1 else [corrected_base_obs.copy()]

            act_hist_corr = list(self.act_history)
            if self.act_history_len > 0:
                act_feature = self._pwm_to_action_feature(float(pwm_raw))
                while len(act_hist_corr) < self.act_history_len:
                    act_hist_corr.insert(0, np.zeros((1,), dtype=np.float32))
                act_hist_corr = act_hist_corr[-self.act_history_len:]
                act_hist_corr = act_hist_corr[1:] + [act_feature] if self.act_history_len > 1 else [act_feature]

            rn_obs_for_replay = self._build_full_observation_from_histories(
                obs_hist_corr,
                act_hist_corr,
                corrected_base_obs,
            )

        # In replay mode, reward and done returned to SB3 should match the corrected transition.
        if self.rn_enabled and self.rn_mode == "replay":
            reward = float(rn_reward_for_replay)
            terminated = bool(rn_terminated_for_replay)
            truncated = bool(rn_truncated_for_replay)
            extra_info: Dict[str, Any] = {}
        else:
            reward = float(self.reward_fn(agent_current_state, action, agent_next_state, self))
            terminated, truncated, extra_info = self.done_fn(agent_current_state, action, agent_next_state, self)
            terminated = bool(terminated)
            truncated = bool(truncated)

        self.state = next_state
        self.step_count += 1
        self.episode_reward += reward

        self.max_abs_theta = max(self.max_abs_theta, abs(float(next_state[0])))
        self.max_abs_alpha = max(self.max_abs_alpha, abs(float(next_state[2])))

        info: Dict[str, Any] = {}
        if extra_info is not None:
            info.update(dict(extra_info))

        info["state"] = agent_next_state.copy()
        info["true_state"] = next_state.copy()
        info["measured_state"] = agent_next_state.copy()
        info["state_sim"] = next_state_sim.copy()
        info["use_lpf_velocity"] = bool(self.use_lpf_velocity)
        info["velocity_lpf"] = float(self.velocity_lpf)
        info["theta_dot_est"] = float(agent_next_state[1])
        info["alpha_dot_est"] = float(agent_next_state[3])
        info["pwm"] = float(pwm_used)
        info["episode_index"] = self.episode_idx
        info["step_count"] = self.step_count
        info["action_raw"] = action
        info["action_pwm"] = float(pwm_raw)
        info["action_pwm_used"] = float(pwm_used)

        if self.rn_enabled and self.rn_mode != "none":
            info["rn_enabled"] = True
            info["rn_mode"] = self.rn_mode
            info["rn_residual"] = rn_residual.copy()
            info["rn_lambda"] = float(self.rn_lambda)
            info["rn_delta_u"] = float(delta_u)
            if rn_next_state_for_replay is not None:
                info["rn_replay_next_state"] = rn_next_state_for_replay.copy()
            if rn_obs_for_replay is not None:
                # Custom ReplayBuffer reads this key and stores it as next_obs.
                info["rn_replay_next_obs"] = rn_obs_for_replay.copy()

        obs_state = self._apply_observation_noise(agent_next_state)
        base_obs = self._state_to_base_obs(obs_state)

        self.obs_history.append(base_obs.copy())
        if self.act_history_len > 0:
            # Observation history should reflect the action actually used by the rollout dynamics.
            self.act_history.append(self._pwm_to_action_feature(float(pwm_used)))

        obs = self._build_full_observation(base_obs)

        # Update RN history.  For replay mode and default action mode, this remains aligned with rollout.
        # In action mode, using raw pwm prevents repeated compensation in the RN input convention.
        if self.rn_enabled and self.rn_mode != "none":
            hist_state = next_state.copy()
            hist_pwm = float(pwm_used if self.rn_history_use_corrected else pwm_raw)
            self.rn_history.append((hist_state, hist_pwm))

        self.logger.log_step(
            episode=self.episode_idx,
            step=self.step_count,
            state=next_state,
            action=action,
            pwm=pwm_used,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )

        if terminated or truncated:
            self.logger.log_episode(
                episode=self.episode_idx,
                episode_reward=self.episode_reward,
                episode_length=self.step_count,
                max_abs_theta=self.max_abs_theta,
                max_abs_alpha=self.max_abs_alpha,
                final_state=next_state,
                done_reason=info.get("done_reason", None),
            )

        self.last_action = action
        self.last_pwm = float(pwm_used)

        return obs, reward, terminated, truncated, info

    def render(self):
        return None

    def close(self):
        self.logger.close()