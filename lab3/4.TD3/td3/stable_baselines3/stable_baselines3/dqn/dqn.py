import warnings
from typing import Any, ClassVar, TypeVar

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import LinearSchedule, get_parameters_by_name, polyak_update
from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy, QNetwork

SelfDQN = TypeVar("SelfDQN", bound="DQN")


class DQN(OffPolicyAlgorithm):
    """
    DQN / DDQN (minimal modified version)

    新增功能：
    1) use_double_dqn: 是否启用 DDQN target
    2) use_exponential_exploration: 是否启用指数衰减 epsilon
    3) exploration_decay: 指数衰减时间常数
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }

    exploration_schedule: Schedule
    q_net: QNetwork
    q_net_target: QNetwork
    policy: DQNPolicy

    def __init__(
        self,
        policy: str | type[DQNPolicy],
        env: GymEnv | str,
        learning_rate: float | Schedule = 1e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: int | tuple[int, str] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: type[ReplayBuffer] | None = None,
        replay_buffer_kwargs: dict[str, Any] | None = None,
        optimize_memory_usage: bool = False,
        n_steps: int = 1,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: th.device | str = "auto",
        _init_setup_model: bool = True,
        use_double_dqn: bool = True,
        use_exponential_exploration: bool = True,
        exploration_decay: float = 400000.0,
    ) -> None:
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=None,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            n_steps=n_steps,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            supported_action_spaces=(spaces.Discrete,),
            support_multi_env=True,
        )

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.target_update_interval = target_update_interval
        self._n_calls = 0
        self.max_grad_norm = max_grad_norm
        self.exploration_rate = 0.0

        self.use_double_dqn = use_double_dqn
        self.use_exponential_exploration = use_exponential_exploration
        self.exploration_decay = float(exploration_decay)

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()

        self.batch_norm_stats = get_parameters_by_name(self.q_net, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.q_net_target, ["running_"])

        self.exploration_schedule = LinearSchedule(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )

        if self.n_envs > 1 and self.n_envs > self.target_update_interval:
            warnings.warn(
                "The number of environments used is greater than the target network "
                f"update interval ({self.n_envs} > {self.target_update_interval}), "
                "therefore the target network will be updated after each call to env.step() "
                f"which corresponds to {self.n_envs} steps."
            )

    def _create_aliases(self) -> None:
        self.q_net = self.policy.q_net
        self.q_net_target = self.policy.q_net_target

    def _compute_exploration_rate(self) -> float:
        if self.use_exponential_exploration:
            decay = max(self.exploration_decay, 1.0)
            return float(
                self.exploration_final_eps
                + (self.exploration_initial_eps - self.exploration_final_eps)
                * np.exp(-float(self.num_timesteps) / decay)
            )
        return float(self.exploration_schedule(self._current_progress_remaining))

    def _on_step(self) -> None:
        """
        Update exploration rate and target network if needed.
        """
        self._n_calls += 1

        if self._n_calls % max(self.target_update_interval // self.n_envs, 1) == 0:
            polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)
            polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self.exploration_rate = self._compute_exploration_rate()
        self.logger.record("rollout/exploration_rate", self.exploration_rate)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []

        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                if self.use_double_dqn:
                    # DDQN:
                    # 1) online net 选动作
                    # 2) target net 评估该动作
                    next_q_values_online = self.q_net(replay_data.next_observations)
                    next_actions = next_q_values_online.argmax(dim=1, keepdim=True)

                    next_q_values_target_all = self.q_net_target(replay_data.next_observations)
                    next_q_values = th.gather(next_q_values_target_all, dim=1, index=next_actions)
                else:
                    # vanilla DQN
                    next_q_values_target_all = self.q_net_target(replay_data.next_observations)
                    next_q_values, _ = next_q_values_target_all.max(dim=1, keepdim=True)

                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values_all = self.q_net(replay_data.observations)
            current_q_values = th.gather(current_q_values_all, dim=1, index=replay_data.actions.long())

            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
        self.logger.record("train/is_double_dqn", int(self.use_double_dqn), exclude="tensorboard")

    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        if not deterministic and np.random.rand() < self.exploration_rate:
            if self.policy.is_vectorized_observation(observation):
                if isinstance(observation, dict):
                    n_batch = observation[next(iter(observation.keys()))].shape[0]
                else:
                    n_batch = observation.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array(self.action_space.sample())
        else:
            action, state = self.policy.predict(observation, state, episode_start, deterministic)
        return action, state

    def learn(
        self: SelfDQN,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "DQN",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDQN:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> list[str]:
        return [*super()._excluded_save_params(), "q_net", "q_net_target"]

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "policy.optimizer"]
        return state_dicts, []