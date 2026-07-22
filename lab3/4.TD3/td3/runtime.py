"""Shared runtime helpers and paper-aligned TD3 implementation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn.functional as F
import torch.nn.utils as torch_utils
from stable_baselines3.td3.td3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import config
from rip_env.register_envs import register_all_envs


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def dump_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def emit_event(event: str, **payload: Any) -> None:
    print(
        "[PANEL_JSON] "
        + json.dumps({"event": event, **payload}, ensure_ascii=False, default=str),
        flush=True,
    )


def config_snapshot() -> Dict[str, Any]:
    names = (
        "RUN", "ENV", "PHYSICAL_PARAMS", "DOMAIN_RANDOMIZATION", "TD3",
        "REWARD", "EVAL", "DISTILL", "TEST", "SMOKE", "PANEL",
    )
    return {name: getattr(config, name) for name in names}


class TD3ObservationContractWrapper(gym.ObservationWrapper):
    """Validate, but do not transform, the original seven-dimensional TD3 input.

    Required order:
      [sin(theta), cos(theta), theta_dot,
       sin(alpha), cos(alpha), alpha_dot, last_action_norm]
    """

    ORDER = (
        "sin_theta", "cos_theta", "theta_dot", "sin_alpha", "cos_alpha",
        "alpha_dot", "last_action_norm",
    )

    def __init__(self, env: gym.Env):
        super().__init__(env)
        shape = tuple(getattr(env.observation_space, "shape", ()))
        expected = (int(config.ENV["observation_dim"]),)
        if shape != expected:
            raise ValueError(
                f"Original-aligned TD3 observation must be {expected}, got {shape}. "
                f"Check observation_type={config.ENV['observation_type']!r}."
            )
        self.observation_space = env.observation_space
        self._validated = False

    def observation(self, observation):
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        if obs.shape != (7,):
            raise ValueError(f"TD3 observation changed shape: {obs.shape}")
        if not np.all(np.isfinite(obs)):
            raise ValueError("TD3 observation contains non-finite values")
        if not self._validated:
            if abs(float(obs[0] ** 2 + obs[1] ** 2) - 1.0) > 0.08:
                raise ValueError("theta sin/cos order check failed at indices 0,1")
            if abs(float(obs[3] ** 2 + obs[4] ** 2) - 1.0) > 0.08:
                raise ValueError("alpha sin/cos order check failed at indices 3,4")
            if abs(float(obs[6])) > 1.0001:
                raise ValueError("last_action_norm at index 6 is outside [-1,1]")
            self._validated = True
        return obs


class ActionRepeatWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, repeat: int):
        super().__init__(env)
        self.repeat = max(1, int(repeat))

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        info: Dict[str, Any] = {}
        obs = None
        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class TD3PaperAligned(TD3):
    """TD3 with separate actor/critic learning rates and gradient clipping.

    The update equations follow the uploaded original implementation.  The
    setup/load hooks are made robust so checkpoints can be loaded by the
    distillation and testing workers.
    """

    def __init__(
        self,
        *args,
        actor_learning_rate: float = 1e-4,
        critic_learning_rate: float = 1e-3,
        actor_grad_clip: float = 1.0,
        critic_grad_clip: float = 1.0,
        **kwargs,
    ):
        self.actor_learning_rate_custom = float(actor_learning_rate)
        self.critic_learning_rate_custom = float(critic_learning_rate)
        self.actor_grad_clip_custom = float(actor_grad_clip)
        self.critic_grad_clip_custom = float(critic_grad_clip)
        super().__init__(*args, **kwargs)
        if hasattr(self, "actor"):
            self._set_separate_optimizer_lrs()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._set_separate_optimizer_lrs()

    def _set_separate_optimizer_lrs(self) -> None:
        if hasattr(self, "actor") and hasattr(self.actor, "optimizer"):
            for group in self.actor.optimizer.param_groups:
                group["lr"] = float(self.actor_learning_rate_custom)
        if hasattr(self, "critic") and hasattr(self.critic, "optimizer"):
            for group in self.critic.optimizer.param_groups:
                group["lr"] = float(self.critic_learning_rate_custom)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        actor_losses = []
        critic_losses = []

        for _ in range(gradient_steps):
            self._n_updates += 1
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )

            with th.no_grad():
                noise = replay_data.actions.clone().data.normal_(
                    0, self.target_policy_noise
                )
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (
                    self.actor_target(replay_data.next_observations) + noise
                ).clamp(-1, 1)
                next_q_values = th.cat(
                    self.critic_target(replay_data.next_observations, next_actions),
                    dim=1,
                )
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (
                    1 - replay_data.dones
                ) * self.gamma * next_q_values

            current_q_values = self.critic(
                replay_data.observations, replay_data.actions
            )
            critic_loss = sum(
                F.mse_loss(current_q, target_q_values)
                for current_q in current_q_values
            )
            critic_losses.append(float(critic_loss.item()))

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            torch_utils.clip_grad_norm_(
                self.critic.parameters(), self.critic_grad_clip_custom
            )
            self.critic.optimizer.step()

            if self._n_updates % self.policy_delay == 0:
                actor_loss = -self.critic.q1_forward(
                    replay_data.observations,
                    self.actor(replay_data.observations),
                ).mean()
                actor_losses.append(float(actor_loss.item()))
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                torch_utils.clip_grad_norm_(
                    self.actor.parameters(), self.actor_grad_clip_custom
                )
                self.actor.optimizer.step()

                polyak_update(
                    self.critic.parameters(), self.critic_target.parameters(), self.tau
                )
                polyak_update(
                    self.actor.parameters(), self.actor_target.parameters(), self.tau
                )
                if hasattr(self, "critic_batch_norm_stats") and hasattr(
                    self, "critic_batch_norm_stats_target"
                ):
                    polyak_update(
                        self.critic_batch_norm_stats,
                        self.critic_batch_norm_stats_target,
                        1.0,
                    )
                if hasattr(self, "actor_batch_norm_stats") and hasattr(
                    self, "actor_batch_norm_stats_target"
                ):
                    polyak_update(
                        self.actor_batch_norm_stats,
                        self.actor_batch_norm_stats_target,
                        1.0,
                    )

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if actor_losses:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        if critic_losses:
            self.logger.record("train/critic_loss", np.mean(critic_losses))
        self._set_separate_optimizer_lrs()


def _set_log_dir(cfg, log_dir: Path) -> None:
    try:
        cfg.logging.log_dir = str(log_dir)
    except Exception:
        pass


def make_single_env_factory(
    *,
    randomization_level: float,
    seed: int,
    log_dir: str | Path,
    nominal: bool = False,
    max_physical_steps: Optional[int] = None,
) -> Callable[[], gym.Env]:
    def _factory() -> gym.Env:
        register_all_envs()
        cfg = config.build_env_config(
            randomization_level=randomization_level,
            nominal=nominal,
            max_physical_steps=max_physical_steps,
        )
        _set_log_dir(cfg, Path(log_dir))
        env = gym.make(
            str(config.ENV["env_id"]),
            env_config=cfg,
            reward_fn=config.build_reward_fn(),
            done_fn=config.build_done_fn(),
        )
        env = TD3ObservationContractWrapper(env)
        env = ActionRepeatWrapper(env, int(config.ENV["action_repeat"]))
        env = Monitor(env)
        env.reset(seed=int(seed))
        return env

    return _factory


def make_vec_env(
    *,
    n_envs: int,
    randomization_level: float,
    seed: int,
    log_dir: str | Path,
    max_physical_steps: Optional[int] = None,
):
    factories = [
        make_single_env_factory(
            randomization_level=randomization_level,
            seed=seed + index,
            log_dir=Path(log_dir) / f"env_{index}",
            max_physical_steps=max_physical_steps,
        )
        for index in range(int(n_envs))
    ]
    if (
        str(config.RUN.get("vec_env_type", "dummy")).lower() == "subproc"
        and int(n_envs) > 1
    ):
        return SubprocVecEnv(factories, start_method="spawn")
    return DummyVecEnv(factories)


def make_raw_env(
    *,
    randomization_level: float,
    seed: int = 0,
    max_physical_steps: Optional[int] = None,
    nominal: bool = False,
):
    return make_single_env_factory(
        randomization_level=randomization_level,
        seed=seed,
        log_dir=Path("./runs") / "eval_env",
        nominal=nominal,
        max_physical_steps=max_physical_steps,
    )()


def set_env_randomization_level(env, level: float) -> None:
    errors = []
    for target in (env, getattr(env, "venv", None)):
        if target is None:
            continue
        try:
            target.env_method("set_randomization_level", float(level))
            return
        except Exception as exc:
            errors.append(repr(exc))

    target = env
    visited = set()
    while target is not None and id(target) not in visited:
        visited.add(id(target))
        if hasattr(target, "set_randomization_level"):
            target.set_randomization_level(float(level))
            return
        target = getattr(target, "env", None)
    raise RuntimeError(
        f"Environment does not expose set_randomization_level({level}); "
        f"errors={errors}"
    )


def reset_vec_model_state(model) -> np.ndarray:
    obs = model.get_env().reset()
    model._last_obs = obs
    model._last_original_obs = None
    model._last_episode_starts = np.ones(
        (model.get_env().num_envs,), dtype=bool
    )
    if getattr(model, "action_noise", None) is not None:
        try:
            model.action_noise.reset()
        except Exception:
            pass
    return obs


def student_input(obs: np.ndarray, variant: str = "current") -> np.ndarray:
    if variant not in {"current", "both", "history", "no_history"}:
        raise ValueError(f"Unsupported variant: {variant}")
    arr = np.asarray(obs, dtype=np.float32)
    if arr.shape[-1] != 7:
        raise ValueError(f"TD3 teacher/student input must be 7-D, got {arr.shape}")
    return arr


def extract_physical_state(info: Dict[str, Any], obs: np.ndarray) -> np.ndarray:
    for key in ("state", "physical_state", "raw_state"):
        value = info.get(key)
        if value is not None:
            arr = np.asarray(value, dtype=float).reshape(-1)
            if arr.size >= 4:
                return arr[:4]
    o = np.asarray(obs, dtype=float).reshape(-1)
    return np.array(
        [
            float(np.arctan2(o[0], o[1])),
            float(o[2]),
            float(np.arctan2(o[3], o[4])),
            float(o[5]),
        ],
        dtype=float,
    )


def final_continuous_stable_entry(
    alpha_abs: Iterable[float],
    alpha_dot_abs: Iterable[float],
    *,
    capture_rad: float,
    alpha_dot_max: float,
    hold_steps: int,
):
    alpha = np.asarray(list(alpha_abs), dtype=float)
    alpha_dot = np.asarray(list(alpha_dot_abs), dtype=float)
    stable = (alpha <= capture_rad) & (alpha_dot <= alpha_dot_max)
    longest = 0
    current = 0
    first_entry = None
    for index, flag in enumerate(stable):
        if flag:
            if first_entry is None:
                first_entry = index
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    trailing = 0
    for flag in stable[::-1]:
        if not flag:
            break
        trailing += 1
    return longest >= hold_steps, first_entry, trailing, longest


def episode_metrics(
    *,
    rewards: list[float],
    states: list[np.ndarray],
    pwms: list[float],
    terminated: bool,
    dt_policy: float,
    capture_angle_deg: float,
    stable_alpha_dot_max: float,
    stable_hold_seconds: float,
) -> Dict[str, Any]:
    s = np.asarray(states if states else [np.zeros(4)], dtype=float)
    theta = np.abs(s[:, 0])
    theta_dot = np.abs(s[:, 1])
    alpha = np.abs(np.arctan2(np.sin(s[:, 2]), np.cos(s[:, 2])))
    alpha_dot = np.abs(s[:, 3])
    hold_steps = max(1, round(stable_hold_seconds / max(dt_policy, 1e-9)))
    success, entry, trailing, longest = final_continuous_stable_entry(
        alpha,
        alpha_dot,
        capture_rad=np.deg2rad(capture_angle_deg),
        alpha_dot_max=stable_alpha_dot_max,
        hold_steps=hold_steps,
    )
    pwm = np.abs(np.asarray(pwms if pwms else [0.0], dtype=float))
    return {
        "reward": float(np.sum(rewards)),
        "length": len(rewards),
        "duration_s": len(rewards) * dt_policy,
        "terminated": bool(terminated),
        "entered_capture": bool(np.any(alpha <= np.deg2rad(capture_angle_deg))),
        "stable_success": bool(success),
        "stable_entry_step": entry,
        "trailing_stable_steps": int(trailing),
        "longest_stable_steps": int(longest),
        "mean_abs_theta": float(np.mean(theta)),
        "mean_abs_theta_dot": float(np.mean(theta_dot)),
        "mean_abs_alpha": float(np.mean(alpha)),
        "mean_abs_alpha_dot": float(np.mean(alpha_dot)),
        "mean_abs_pwm": float(np.mean(pwm)),
    }


def aggregate_metrics(rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    def mean(key: str) -> float:
        return float(np.mean([float(row.get(key, 0.0)) for row in rows]))

    mean_reward = mean("reward")
    mean_length = mean("length")
    mean_theta = mean("mean_abs_theta")
    mean_alpha = mean("mean_abs_alpha")
    mean_pwm_norm = mean("mean_abs_pwm") / max(float(config.ENV["pwm_limit"]), 1e-9)
    done_rate = mean("terminated")
    # Same control-oriented score shape as the uploaded TD3 evaluator.
    control_score = (
        mean_reward
        + 0.02 * mean_length
        - 2.0 * mean_theta
        - 12.0 * mean_alpha
        - 2.0 * mean_pwm_norm
        - 20.0 * done_rate
    )
    return {
        "episodes": len(rows),
        "mean_reward": mean_reward,
        "mean_length": mean_length,
        "stable_success_rate": mean("stable_success"),
        "capture_rate": mean("entered_capture"),
        "termination_rate": done_rate,
        "mean_abs_theta": mean_theta,
        "mean_abs_theta_dot": mean("mean_abs_theta_dot"),
        "mean_abs_alpha": mean_alpha,
        "mean_abs_alpha_dot": mean("mean_abs_alpha_dot"),
        "mean_abs_pwm": mean("mean_abs_pwm"),
        "mean_pwm_abs_norm": mean_pwm_norm,
        "control_score": float(control_score),
    }
