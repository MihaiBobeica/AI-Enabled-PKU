"""
run.py

Single entry point for Stage-1 PPO sim-to-real balance training.

Default (opens the training panel):
    python run.py

Direct workers remain available:
    python run.py train
    python run.py smoke
    python run.py eval path/to/model.zip

Change training hyperparameters and randomization ranges in config.py, not here.
"""

from __future__ import annotations

import argparse
import copy
import csv
from collections import deque
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent

# Use the vendored Stable-Baselines3 source if requested in config.py.
# Gymnasium, torch, numpy and matplotlib remain normal Python runtime libraries.
try:
    import config
except Exception as exc:
    raise RuntimeError(f"Failed to import config.py: {exc}") from exc

# Prefer project-local runtime libraries when shipped.  This makes the package
# callable even when Gymnasium/SB3 are not globally installed in the active
# Python environment.  Heavy numerical libraries such as torch/numpy are still
# expected to come from the user's Python environment.
LOCAL_THIRD_PARTY = PROJECT_ROOT / "third_party"
if LOCAL_THIRD_PARTY.exists():
    sys.path.insert(0, str(LOCAL_THIRD_PARTY))

LOCAL_SB3_PARENT = PROJECT_ROOT / "stable_baselines3"
if bool(config.RUN.get("prefer_local_stable_baselines3", True)) and LOCAL_SB3_PARENT.exists():
    sys.path.insert(0, str(LOCAL_SB3_PARENT))
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import numpy as np
    import torch
    import gymnasium as gym
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    missing = exc.name
    print("\n[依赖缺失]")
    print(f"缺少 Python 包: {missing}")
    print("这个版本使用 Stable-Baselines3 + Gymnasium 的正式训练路线，不再使用自写最小 PPO。")
    print("项目已随包提供 Gymnasium 和 Stable-Baselines3 源码；这个错误通常表示 third_party/ 被删掉，")
    print("或者当前 Python 环境缺少 torch/numpy/matplotlib/pandas 等基础运行库。")
    print("如果是基础运行库缺失，再执行：")
    print("    pip install -r requirements.txt")
    raise

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, sync_envs_normalization
except Exception as exc:
    print("\n[Stable-Baselines3 导入失败]")
    print("本项目优先使用 ./stable_baselines3 中随包提供的 SB3 源码。")
    print(f"错误信息: {exc}")
    raise

from rip_env.register_envs import register_all_envs


# =============================================================================
# Utilities
# =============================================================================
def ensure_dir(path: str | Path) -> None:
    os.makedirs(path, exist_ok=True)


def dump_json(path: str | Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, int(config.RUN.get("torch_num_threads", 4))))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(config.RUN.get("torch_deterministic", False))
    torch.backends.cudnn.benchmark = not bool(config.RUN.get("torch_deterministic", False))


def is_vecnormalize(env) -> bool:
    return isinstance(env, VecNormalize)


# =============================================================================
# Wrapper: action repeat
# =============================================================================
class ActionRepeatWrapper(gym.Wrapper):
    """Apply one policy action for several simulator integration steps."""

    def __init__(self, env: gym.Env, repeat: int = 1):
        super().__init__(env)
        self.repeat = max(1, int(repeat))

    def step(self, action):
        total_reward = 0.0
        final_obs = None
        final_info: Dict[str, Any] = {}
        terminated_final = False
        truncated_final = False
        inner_steps = 0
        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            final_obs = obs
            final_info = dict(info)
            inner_steps += 1
            if terminated or truncated:
                terminated_final = bool(terminated)
                truncated_final = bool(truncated)
                break
        final_info["action_repeat"] = self.repeat
        final_info["inner_steps"] = inner_steps
        final_info["policy_reward_sum"] = total_reward
        return final_obs, total_reward, terminated_final, truncated_final, final_info


# =============================================================================
# Env factories
# =============================================================================
def make_single_env_factory(
    *,
    rank: int,
    env_cfg,
    reward_fn,
    done_fn,
    monitor_dir: Optional[str] = None,
) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        cfg = copy.deepcopy(env_cfg)
        # Step-level env logs in every parallel worker are too heavy.
        cfg.logging.enabled = bool(config.ENV.get("env_logging_enabled", True)) and rank == 0
        cfg.logging.save_step_log = bool(config.ENV.get("env_save_step_log", False)) and rank == 0
        cfg.logging.flush_every_step = bool(config.ENV.get("env_flush_every_step", False)) and rank == 0
        env = gym.make(
            str(config.ENV["env_id"]),
            env_config=cfg,
            reward_fn=reward_fn,
            done_fn=done_fn,
        )
        env = ActionRepeatWrapper(env, repeat=int(config.ENV["action_repeat"]))
        if monitor_dir is not None:
            ensure_dir(monitor_dir)
            env = Monitor(env, filename=os.path.join(monitor_dir, f"monitor_{rank}.csv"))
        else:
            env = Monitor(env)
        env.reset(seed=int(config.RUN["seed"]) + rank)
        return env
    return _init


def make_vec_env(*, n_envs: int, env_cfg, reward_fn, done_fn, monitor_dir: Optional[str], seed_offset: int = 0):
    factories = [
        make_single_env_factory(
            rank=seed_offset + i,
            env_cfg=env_cfg,
            reward_fn=reward_fn,
            done_fn=done_fn,
            monitor_dir=monitor_dir,
        )
        for i in range(int(n_envs))
    ]
    vec_type = str(config.RUN.get("vec_env_type", "dummy")).lower()
    if vec_type == "subproc" and int(n_envs) > 1:
        return SubprocVecEnv(factories, start_method="spawn")
    if vec_type != "dummy" and vec_type != "subproc":
        raise ValueError("RUN['vec_env_type'] must be 'dummy' or 'subproc'")
    return DummyVecEnv(factories)


def maybe_normalize_vec_env(vec_env, *, training: bool, norm_reward: bool):
    if not bool(config.PPO.get("normalize_obs", True)) and not bool(config.PPO.get("normalize_reward", True)):
        return vec_env
    return VecNormalize(
        vec_env,
        training=bool(training),
        norm_obs=bool(config.PPO.get("normalize_obs", True)),
        norm_reward=bool(norm_reward and config.PPO.get("normalize_reward", True)),
        clip_obs=float(config.PPO.get("clip_obs", 10.0)),
        clip_reward=float(config.PPO.get("clip_reward", 10.0)),
    )


# =============================================================================
# Callbacks
# =============================================================================


class TrainingProgressCallback(BaseCallback):
    """Emit stable progress lines for the GUI and save progress.json."""

    def __init__(self, run_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.run_dir = run_dir
        self.total = int(config.PPO["total_timesteps"])
        self.update_freq = max(1, int(getattr(config, "PANEL", {}).get("progress_update_freq", 5000)))
        self.last_report = -self.update_freq
        self.started_at = 0.0
        self.episode_reward_per_step_30 = deque(maxlen=30)
        self.episode_success_30 = deque(maxlen=30)
        self.training_metrics_path = os.path.join(self.run_dir, "training_metrics.csv")

    def _on_training_start(self) -> None:
        self.started_at = time.monotonic()
        with open(self.training_metrics_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timesteps", "episode_reward", "episode_length", "reward_per_step",
                "moving_mean_30", "episode_success", "success_rate_30",
            ])
        self._report(force=True)

    def _record_completed_episodes(self) -> None:
        infos = self.locals.get("infos", [])
        max_policy_steps = max(
            1,
            int(math.ceil(float(config.ENV["max_physical_steps"]) / max(int(config.ENV["action_repeat"]), 1))),
        )
        rows = []
        for info in infos:
            if not isinstance(info, dict):
                continue
            episode = info.get("episode")
            if not isinstance(episode, dict):
                continue
            reward = float(episode.get("r", 0.0))
            length = max(1, int(episode.get("l", 1)))
            reward_per_step = reward / float(length)
            success = 1.0 if length >= max_policy_steps else 0.0
            self.episode_reward_per_step_30.append(reward_per_step)
            self.episode_success_30.append(success)
            moving_mean = float(np.mean(self.episode_reward_per_step_30))
            success_rate = float(np.mean(self.episode_success_30))
            rows.append([
                int(self.num_timesteps), reward, length, reward_per_step, moving_mean, success, success_rate,
            ])
            if bool(getattr(config, "PANEL", {}).get("print_train_episodes", False)):
                print(
                    f"[TRAIN_EPISODE] timesteps={self.num_timesteps} reward={reward:.9g} "
                    f"length={length} reward_per_step={reward_per_step:.9g} "
                    f"moving_mean_30={moving_mean:.9g} success={int(success)} "
                    f"success_rate_30={success_rate:.9g}",
                    flush=True,
                )
        if rows:
            with open(self.training_metrics_path, "a", encoding="utf-8", newline="") as f:
                csv.writer(f).writerows(rows)

    def _report(self, force: bool = False) -> None:
        if not force and self.num_timesteps - self.last_report < self.update_freq:
            return
        self.last_report = int(self.num_timesteps)
        elapsed = max(time.monotonic() - self.started_at, 1e-9)
        fps = float(self.num_timesteps) / elapsed if self.num_timesteps > 0 else 0.0
        remaining = max(self.total - int(self.num_timesteps), 0)
        eta = float(remaining) / fps if fps > 1e-9 else 0.0
        percent = 100.0 * min(float(self.num_timesteps) / max(float(self.total), 1.0), 1.0)
        payload = {
            "algorithm": "PPO", "timesteps": int(self.num_timesteps), "total": self.total,
            "percent": percent, "fps": fps, "elapsed_s": elapsed, "eta_s": eta,
        }
        dump_json(os.path.join(self.run_dir, "training_progress.json"), payload)
        print(
            f"[PROGRESS] algorithm=PPO timesteps={self.num_timesteps} total={self.total} "
            f"percent={percent:.3f} fps={fps:.3f} elapsed_s={elapsed:.3f} eta_s={eta:.3f}",
            flush=True,
        )

    def _on_step(self) -> bool:
        self._record_completed_episodes()
        self._report(force=False)
        return True

    def _on_training_end(self) -> None:
        self._report(force=True)


class DomainRandomizationCurriculumCallback(BaseCallback):
    """Linearly increase domain randomization level during training."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.current_level = float(config.DOMAIN_RANDOMIZATION["dr_initial_level"])

    def level_at(self, timesteps: int) -> float:
        dr = config.DOMAIN_RANDOMIZATION
        if not bool(dr["enabled"]):
            return 0.0
        horizon = max(1.0, float(config.PPO["total_timesteps"]) * float(dr["dr_curriculum_fraction"]))
        x = float(np.clip(float(timesteps) / horizon, 0.0, 1.0))
        return float(dr["dr_initial_level"] + x * (dr["dr_final_level"] - dr["dr_initial_level"]))

    def _on_step(self) -> bool:
        new_level = self.level_at(self.num_timesteps)
        if abs(new_level - self.current_level) > 0.005:
            self.current_level = new_level
            self.training_env.env_method("set_randomization_level", self.current_level)
            if self.verbose:
                print(f"[DR] steps={self.num_timesteps} level={self.current_level:.3f}")
        return True


class RIPBalanceEvalCallback(BaseCallback):
    """Evaluate both nominal and randomized balance performance, then save best model."""

    def __init__(self, run_dir: str, eval_nominal_env, eval_random_env, verbose: int = 1):
        super().__init__(verbose)
        self.run_dir = run_dir
        self.eval_nominal_env = eval_nominal_env
        self.eval_random_env = eval_random_env
        self.eval_freq = int(config.EVAL["eval_freq"])
        self.n_eval_episodes = int(config.EVAL["n_eval_episodes"])
        self.max_eval_policy_steps = int(config.EVAL["max_eval_policy_steps"])
        self.best_score = -1e18
        self.history = []
        self.last_eval_timestep = 0
        ensure_dir(os.path.join(run_dir, "eval_logs"))
        ensure_dir(os.path.join(run_dir, "best_model"))
        self.csv_path = os.path.join(run_dir, "eval_logs", "eval_metrics.csv")
        with open(self.csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timesteps", "eval_type", "mean_reward", "mean_length", "reward_per_step",
                "success_rate", "mean_abs_theta", "mean_abs_alpha", "mean_abs_theta_dot",
                "mean_abs_alpha_dot", "mean_abs_pwm_norm", "terminated_rate", "score",
            ])

    def _on_step(self) -> bool:
        # num_timesteps advances by n_envs.  A modulo check can permanently miss
        # eval_freq when the two are not divisible (for example 25,001 with 16 envs).
        if self.num_timesteps - self.last_eval_timestep < self.eval_freq:
            return True
        self.last_eval_timestep = int(self.num_timesteps)
        if self.model.get_vec_normalize_env() is not None:
            sync_envs_normalization(self.training_env, self.eval_nominal_env)
            sync_envs_normalization(self.training_env, self.eval_random_env)

        nominal = self.evaluate(self.eval_nominal_env, seed_offset=10_000)
        randomized = self.evaluate(self.eval_random_env, seed_offset=20_000)
        self.history.append({"timesteps": self.num_timesteps, "nominal": nominal, "randomized": randomized})

        with open(self.csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            for name, metrics in [("nominal", nominal), ("randomized", randomized)]:
                writer.writerow([
                    self.num_timesteps, name, metrics["mean_reward"], metrics["mean_length"],
                    metrics["reward_per_step"], metrics["success_rate"], metrics["mean_abs_theta"],
                    metrics["mean_abs_alpha"], metrics["mean_abs_theta_dot"], metrics["mean_abs_alpha_dot"],
                    metrics["mean_abs_pwm_norm"], metrics["terminated_rate"], metrics["score"],
                ])

        if self.verbose:
            print(
                f"[EVAL] steps={self.num_timesteps} | "
                f"nominal score={nominal['score']:.2f}, term={nominal['terminated_rate']:.2f} | "
                f"random score={randomized['score']:.2f}, term={randomized['terminated_rate']:.2f}"
            )
            for eval_type, metrics in (("nominal", nominal), ("randomized", randomized)):
                print(
                    f"[EVAL_METRICS] timesteps={self.num_timesteps} eval_type={eval_type} "
                    f"reward_per_step={metrics['reward_per_step']:.9g} "
                    f"success_rate={metrics['success_rate']:.9g} "
                    f"mean_reward={metrics['mean_reward']:.9g} mean_length={metrics['mean_length']:.9g} "
                    f"mean_abs_alpha={metrics['mean_abs_alpha']:.9g} "
                    f"mean_abs_theta={metrics['mean_abs_theta']:.9g} "
                    f"terminated_rate={metrics['terminated_rate']:.9g}",
                    flush=True,
                )

        if randomized["score"] > self.best_score:
            self.best_score = randomized["score"]
            model_path = os.path.join(self.run_dir, "best_model", "best_model")
            self.model.save(model_path)
            vec_norm = self.model.get_vec_normalize_env()
            if vec_norm is not None:
                vec_norm.save(os.path.join(self.run_dir, "best_model", "vecnormalize.pkl"))
            print(f"[BEST] saved {model_path}.zip | randomized_score={self.best_score:.3f}")

        self.plot_history()
        return True

    def evaluate(self, env, seed_offset: int) -> Dict[str, float]:
        # VecEnv reset does not accept seed in SB3 VecEnv API.  Individual envs
        # were seeded when created; evaluation is deterministic policy-wise.
        ep_rewards, ep_lengths, ep_reward_per_step = [], [], []
        theta_abs_all, alpha_abs_all = [], []
        theta_dot_abs_all, alpha_dot_abs_all, pwm_abs_all = [], [], []
        terminated_flags = []

        for ep in range(self.n_eval_episodes):
            obs = env.reset()
            total_reward = 0.0
            length = 0
            terminated_early = False
            theta_abs, alpha_abs, theta_dot_abs, alpha_dot_abs, pwm_abs = [], [], [], [], []

            for _ in range(self.max_eval_policy_steps):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, infos = env.step(action)
                info = infos[0] if isinstance(infos, (list, tuple)) and len(infos) > 0 else {}
                total_reward += float(np.asarray(reward).reshape(-1)[0])
                length += 1

                st = info.get("true_state", info.get("state", None))
                if st is not None:
                    st = np.asarray(st, dtype=np.float64).reshape(4)
                    theta_abs.append(abs(float(config.wrap_to_pi(float(st[0])))))
                    theta_dot_abs.append(abs(float(st[1])))
                    alpha_abs.append(abs(float(config.wrap_to_pi(float(st[2])))))
                    alpha_dot_abs.append(abs(float(st[3])))
                pwm = info.get("effective_pwm", info.get("last_pwm", 0.0))
                pwm_abs.append(abs(float(pwm)) / max(float(config.ENV["pwm_limit"]), 1e-6))

                if bool(np.asarray(done).reshape(-1)[0]):
                    terminated_early = length < self.max_eval_policy_steps
                    break

            ep_rewards.append(total_reward)
            ep_lengths.append(length)
            ep_reward_per_step.append(total_reward / max(length, 1))
            theta_abs_all.append(np.mean(theta_abs) if theta_abs else np.nan)
            alpha_abs_all.append(np.mean(alpha_abs) if alpha_abs else np.nan)
            theta_dot_abs_all.append(np.mean(theta_dot_abs) if theta_dot_abs else np.nan)
            alpha_dot_abs_all.append(np.mean(alpha_dot_abs) if alpha_dot_abs else np.nan)
            pwm_abs_all.append(np.mean(pwm_abs) if pwm_abs else np.nan)
            terminated_flags.append(float(terminated_early))

        mean_reward = float(np.mean(ep_rewards))
        mean_length = float(np.mean(ep_lengths))
        reward_per_step = float(np.mean(ep_reward_per_step))
        mean_abs_alpha = float(np.nanmean(alpha_abs_all))
        mean_abs_theta = float(np.nanmean(theta_abs_all))
        mean_abs_theta_dot = float(np.nanmean(theta_dot_abs_all))
        mean_abs_alpha_dot = float(np.nanmean(alpha_dot_abs_all))
        mean_abs_pwm_norm = float(np.nanmean(pwm_abs_all))
        terminated_rate = float(np.mean(terminated_flags))
        success_rate = float(1.0 - terminated_rate)

        # Score is reward plus interpretable penalties; tune only if needed.
        score = (
            mean_reward
            + 0.02 * mean_length
            - 40.0 * mean_abs_alpha
            - 0.4 * mean_abs_alpha_dot
            - 0.1 * mean_abs_theta
            - 5.0 * terminated_rate
        )
        return {
            "mean_reward": mean_reward,
            "mean_length": mean_length,
            "reward_per_step": reward_per_step,
            "success_rate": success_rate,
            "mean_abs_theta": mean_abs_theta,
            "mean_abs_alpha": mean_abs_alpha,
            "mean_abs_theta_dot": mean_abs_theta_dot,
            "mean_abs_alpha_dot": mean_abs_alpha_dot,
            "mean_abs_pwm_norm": mean_abs_pwm_norm,
            "terminated_rate": terminated_rate,
            "score": float(score),
        }

    def plot_history(self) -> None:
        if len(self.history) < 1:
            return
        steps = [h["timesteps"] for h in self.history]
        nom = [h["nominal"]["score"] for h in self.history]
        rnd = [h["randomized"]["score"] for h in self.history]
        plt.figure(figsize=(8, 4.5))
        plt.plot(steps, nom, label="nominal")
        plt.plot(steps, rnd, label="randomized")
        plt.xlabel("timesteps")
        plt.ylabel("eval score")
        plt.title("PPO Sim-to-Real Balance Evaluation")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, "eval_logs", "eval_curve.png"), dpi=160)
        plt.close()


# =============================================================================
# Main routines
# =============================================================================
def apply_smoke_overrides() -> None:
    config.RUN["experiment_name"] = "smoke_" + str(config.RUN["experiment_name"])
    for key, value in config.SMOKE.items():
        if key in config.PPO:
            config.PPO[key] = value
        elif key in config.EVAL:
            config.EVAL[key] = value


def build_training_envs(run_dir: str):
    register_all_envs()
    reward_fn = config.build_reward_fn()
    done_fn = config.build_done_fn()

    train_cfg = config.build_env_config(randomization_level=float(config.DOMAIN_RANDOMIZATION["dr_initial_level"]), nominal=False)
    train_vec = make_vec_env(
        n_envs=int(config.PPO["n_envs"]),
        env_cfg=train_cfg,
        reward_fn=reward_fn,
        done_fn=done_fn,
        monitor_dir=os.path.join(run_dir, "monitor_train"),
    )
    train_vec = maybe_normalize_vec_env(train_vec, training=True, norm_reward=True)

    eval_nominal_cfg = config.build_env_config(nominal=True)
    eval_random_cfg = config.build_env_config(randomization_level=float(config.EVAL["eval_randomization_level"]), nominal=False)
    eval_nominal = make_vec_env(n_envs=1, env_cfg=eval_nominal_cfg, reward_fn=reward_fn, done_fn=done_fn, monitor_dir=None, seed_offset=1000)
    eval_random = make_vec_env(n_envs=1, env_cfg=eval_random_cfg, reward_fn=reward_fn, done_fn=done_fn, monitor_dir=None, seed_offset=2000)
    eval_nominal = maybe_normalize_vec_env(eval_nominal, training=False, norm_reward=False)
    eval_random = maybe_normalize_vec_env(eval_random, training=False, norm_reward=False)
    return train_vec, eval_nominal, eval_random, train_cfg


def train() -> None:
    run_dir = config.run_dir()
    ensure_dir(run_dir)
    ensure_dir(os.path.join(run_dir, "models"))
    print(f"[RUN_DIR] path={Path(run_dir).resolve()}", flush=True)
    dump_json(os.path.join(run_dir, "config_snapshot.json"), config.full_config_dict())

    set_global_seeds(int(config.RUN["seed"]))
    train_env, eval_nominal_env, eval_random_env, env_cfg = build_training_envs(run_dir)
    dump_json(os.path.join(run_dir, "env_config_snapshot.json"), config.env_config_to_dict(env_cfg))

    model = PPO(
        "MlpPolicy",
        train_env,
        **config.build_ppo_kwargs(),
    )

    # save_freq is in callback calls, not total timesteps; divide by n_envs.
    save_freq = max(int(config.EVAL["checkpoint_freq"]) // max(int(config.PPO["n_envs"]), 1), 1)
    callbacks = [
        TrainingProgressCallback(run_dir, verbose=0),
        DomainRandomizationCurriculumCallback(verbose=1),
        CheckpointCallback(
            save_freq=save_freq,
            save_path=os.path.join(run_dir, "models"),
            name_prefix="ppo_stage1_balance",
            save_replay_buffer=False,
            save_vecnormalize=True,
            verbose=1,
        ),
        RIPBalanceEvalCallback(run_dir, eval_nominal_env, eval_random_env, verbose=1),
    ]

    print("=" * 100)
    print("[TRAIN] Stage-1 PPO sim-to-real balance")
    print(f"run_dir = {run_dir}")
    print(f"env_id  = {config.ENV['env_id']}")
    print(f"SB3 PPO = {PPO}")
    print("All hyperparameters and randomization ranges are in config.py")
    print("=" * 100)

    model.learn(
        total_timesteps=int(config.PPO["total_timesteps"]),
        callback=CallbackList(callbacks),
        progress_bar=bool(config.PPO.get("progress_bar", False)),
    )

    final_path = os.path.join(run_dir, "final_model")
    model.save(final_path)
    vec_norm = model.get_vec_normalize_env()
    if vec_norm is not None:
        vec_norm.save(os.path.join(run_dir, "vecnormalize.pkl"))
    print(f"[DONE] final model saved to {final_path}.zip")
    print(f"[DONE] run_dir: {run_dir}")

    train_env.close()
    eval_nominal_env.close()
    eval_random_env.close()


def load_eval_env_and_model(model_path: str):
    if not model_path:
        raise ValueError("config.EVAL_MODEL_PATH is empty. Set it in config.py or run: python run.py eval path/to/model.zip")
    model_path = str(model_path)
    model_dir = os.path.dirname(model_path)
    vecnormalize_path = os.path.join(model_dir, "vecnormalize.pkl")

    register_all_envs()
    reward_fn = config.build_reward_fn()
    done_fn = config.build_done_fn()
    eval_cfg = config.build_env_config(randomization_level=float(config.EVAL["eval_randomization_level"]), nominal=False)
    env = make_vec_env(n_envs=1, env_cfg=eval_cfg, reward_fn=reward_fn, done_fn=done_fn, monitor_dir=None, seed_offset=3000)
    if os.path.exists(vecnormalize_path):
        env = VecNormalize.load(vecnormalize_path, env)
        env.training = False
        env.norm_reward = False
        print(f"[EVAL] loaded VecNormalize: {vecnormalize_path}")
    else:
        env = maybe_normalize_vec_env(env, training=False, norm_reward=False)
        print("[EVAL] vecnormalize.pkl not found next to model; using fresh eval normalization wrapper.")

    model = PPO.load(model_path, env=env, device=str(config.RUN["device"]))
    return env, model


def evaluate_model(model_path: str) -> None:
    env, model = load_eval_env_and_model(model_path)
    dummy_run_dir = config.run_dir()
    ensure_dir(dummy_run_dir)
    cb = RIPBalanceEvalCallback(dummy_run_dir, env, env, verbose=0)
    cb.model = model
    metrics = cb.evaluate(env, seed_offset=4000)
    print("=" * 100)
    print(f"[EVAL] model: {model_path}")
    for k, v in metrics.items():
        print(f"{k:>24s}: {v:.6f}")
    print("=" * 100)
    env.close()


def smoke() -> None:
    apply_smoke_overrides()
    train()


def parse_args():
    parser = argparse.ArgumentParser(description="Stage-1 SB3 PPO sim-to-real RIP balance")
    parser.add_argument("mode", nargs="?", choices=["train", "eval", "smoke"], default=None)
    parser.add_argument("model_path", nargs="?", default=None, help="Only used for eval mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode is None:
        from training_panel import TrainingPanel
        TrainingPanel().mainloop()
        return
    mode = str(args.mode).lower()
    if mode == "train":
        train()
    elif mode == "smoke":
        smoke()
    elif mode == "eval":
        evaluate_model(args.model_path or str(config.EVAL_MODEL_PATH))
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
