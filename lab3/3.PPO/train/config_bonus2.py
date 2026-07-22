"""
config_bonus2.py

Bonus-2 PPO single-policy swing-up + stabilize (no hybrid energy swing-up).

Usage:
    python run.py train --config config_bonus2
    python run.py smoke --config config_bonus2

Default hybrid balance training remains in config.py.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict

import numpy as np
from torch import nn

from rip_env.envs.types import (
    EnvConfig,
    RIPPhysicalParams,
    InitStateConfig,
    LimitConfig,
    NoiseConfig,
    LoggingConfig,
)
from rip_env.envs.sim2real_types import DomainRandomizationConfig
from rip_env.envs.done_fns import build_done_fn as build_done_fn_from_dict


# =============================================================================
# 0. One-file workflow settings
# =============================================================================
# Algorithm identity used by the optional training panel.
ALGORITHM_NAME = "PPO_BONUS2"

# Default mode when running `python run.py`.
# Valid values: "train", "eval", "smoke".
MODE = "train"

# Only used when MODE="eval".  Example:
# EVAL_MODEL_PATH = "runs/ppo_sb3_sim2real_balance_20260703_120000/best_model/best_model.zip"
EVAL_MODEL_PATH = ""


# =============================================================================
# 1. Run and logging settings
# =============================================================================
def _now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


RUN: Dict[str, Any] = {
    "experiment_name": f"ppo_bonus2_swingup_{_now_str()}",
    "root_log_dir": "./runs",
    "seed": 42,

    # local SB3 source is shipped under ./stable_baselines3; run.py injects it.
    "prefer_local_stable_baselines3": True,

    # Determinism improves reproducibility but can slow CUDA training.
    "torch_deterministic": False,
    "torch_num_threads": 2,
    "device": 'cpu',          # "auto", "cpu", "cuda"

    # Env parallelism.  DummyVecEnv is safer cross-platform; SubprocVecEnv is faster.
    "vec_env_type": 'dummy',   # "dummy" or "subproc"
}


# =============================================================================
# 2. Physical simulator and task settings
# =============================================================================
ENV: Dict[str, Any] = {
    "env_id": "RIPSim2RealBalance-v0",

    # Physical integration and policy output are both 200 Hz when action_repeat=1.
    "physical_dt": 0.005,
    "action_repeat": 1,
    "max_physical_steps": 2000,  # 10 s at 200 Hz
    "action_type": "continuous",
    "pwm_limit": 150.0,

    # 7-D observation matching TD3 / compact deploy.
    "observation_type": 'trig_hist1_act1',
    "clip_velocity_in_obs": True,

    # Hanging start for full swing-up (Bonus 2).
    "init_mode": "downward",
    "init_theta_mean_deg": 0.0,
    "init_theta_std_deg": 25.98076211353316,
    "init_theta_dot_mean": 0.0,
    "init_theta_dot_std": 0.0,
    "init_alpha_mean_deg": 180.0,
    "init_alpha_std_deg": 25.98076211353316,
    "init_alpha_dot_mean": 0.0,
    "init_alpha_dot_std": 0.0,

    # Safety/termination.  For balance, alpha_abs_limit_deg is deliberately not too loose.
    "theta_limit": 12.0 * np.pi,
    "theta_dot_limit": 45.0,
    "alpha_dot_limit": 40.0,
    "terminate_on_alpha_abs_deg": False,
    "alpha_abs_limit_deg": 45.0,

    # Fixed observation noise remains off; episode-level sensor randomization is below.
    "fixed_noise_enabled": False,
    "fixed_noise_theta_sigma": 0.0,
    "fixed_noise_alpha_sigma": 0.0,
    "fixed_noise_theta_dot_sigma": 0.0,
    "fixed_noise_alpha_dot_sigma": 0.0,

    # Internal env logging. Step logs are expensive; keep disabled for PPO training.
    "env_logging_enabled": True,
    "env_save_step_log": False,
    "env_flush_every_step": False,
}


# =============================================================================
# 3. Nominal physical parameters
# =============================================================================
# These are the simple simulator's nominal values.  The randomization table below
# samples multiplicative factors around these values at every episode reset.
PHYSICAL_PARAMS: Dict[str, float] = {
    "g": 9.8,
    "c_theta": 0.025,
    "c_alpha": 0.001,
    "k_t": 0.2310,
    "k_b": 0.1875,
    "k_u": 0.04706,
    "R": 4.2857,
    "m1": 0.20625,
    "m2": 0.15845,
    "l1cg": 0.080305,
    "l1": 0.151894,
    "l2cg": 0.066733,
    "I1z": 0.00049228,
    "I2x": 0.00036892,
    "I2y": 2.3641e-05,
    "I2z": 0.00036139,
}


# =============================================================================
# 4. Sim-to-real domain randomization settings
# =============================================================================
# The curriculum level in [0, 1] shrinks/expands these ranges around the nominal
# point.  Training starts at dr_initial_level and linearly grows to dr_final_level.
DOMAIN_RANDOMIZATION: Dict[str, Any] = {
    "enabled": True,
    "dr_initial_level": 0.0,
    "dr_final_level": 0.8,
    "dr_curriculum_fraction": 0.2,

    # Mechanics and motor parameters. Multiplicative log-uniform samples.
    # Wide friction/inertia ranges are intentional because those are usually the
    # most uncertain in real sim-to-real experiments.
    "param_scale_ranges": {
        "g_scale": (0.98, 1.02),
        "m1_scale": (0.9, 1.1),
        "m2_scale": (0.9, 1.1),
        "l1_scale": (0.90, 1.10),
        "l1cg_scale": (0.9, 1.1),
        "l2cg_scale": (0.9, 1.1),
        "I1z_scale": (0.9, 1.1),
        "I2x_scale": (0.9, 1.1),
        "I2y_scale": (0.9, 1.1),
        "I2z_scale": (0.9, 1.1),
        "c_theta_scale": (0.25, 3.5),
        "c_alpha_scale": (0.25, 4.5),
        "k_t_scale": (0.9, 1.1),
        "k_b_scale": (0.9, 1.1),
        "k_u_scale": (0.9, 1.1),
        "R_scale": (0.9, 1.1),
    },

    # Initial-state randomization curriculum for upright balance.
    "init_theta_std_deg_range": (5.0, 30.0),
    "init_alpha_std_deg_range": (8.0, 30.0),
    "init_theta_dot_std_range": (0.0, 1.5),
    "init_alpha_dot_std_range": (0.0, 2.5),

    # Actuator randomization: gain, bias, deadzone, lag, command delay, saturation.
    "pwm_limit_scale_range": (0.80, 1.05),
    "pwm_gain_range": (0.9, 1.1),
    "pwm_bias_range": (-4.0, 4.0),
    "pwm_deadzone_range": (0.0, 12.0),
    "pwm_noise_sigma_range": (0.0, 2.0),
    "actuator_tau_range": (0.0, 0.03),
    "action_delay_steps_range": (0, 0),

    # Sensor randomization: angle bias/noise, velocity noise, encoder quantization.
    "theta_bias_range": (-0.010, 0.010),
    "alpha_bias_range": (-0.012, 0.012),
    "theta_sigma_range": (0.0, 0.006),
    "alpha_sigma_range": (0.0, 0.008),
    "theta_dot_sigma_range": (0.0, 0.35),
    "alpha_dot_sigma_range": (0.0, 0.6),
    "encoder_quantization_rad_range": (0.0, 0.0015),

    # Optional low-pass velocity estimator.  This emulates real hardware where
    # velocities are usually computed from angle differences and filtered.
    "use_lpf_velocity_probability": 0.50,
    "velocity_lpf_range": (0.15, 0.45),

    # Keep process disturbance off for the first balance benchmark.  Later stages
    # can enable this after the core randomization/curriculum works.
    "process_theta_dot_sigma_range": (0.0, 0.0),
    "process_alpha_dot_sigma_range": (0.0, 0.0),
}


# =============================================================================
# 5. PPO hyperparameters, using Stable-Baselines3 PPO
# =============================================================================
PPO: Dict[str, Any] = {
    "total_timesteps": 3000000,
    "n_envs": 16,
    "n_steps": 1024,
    "batch_size": 1024,
    "n_epochs": 10,
    "learning_rate": 0.0003,
    "gamma": 0.9975,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "target_kl": 0.035,
    "ent_coef": 0.002,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "normalize_advantage": True,

    "net_arch_pi": (64, 64),
    "net_arch_vf": (64, 64),
    "activation_fn_name": 'ReLU',  # match TD3-style Bonus-2 deploy (ReLU/ReLU/Tanh)
    "log_std_init": -0.7,

    # VecNormalize is important for PPO stability under randomized observations/rewards.
    "normalize_obs": True,
    "normalize_reward": True,
    "clip_obs": 10.0,
    "clip_reward": 10.0,

    "verbose": 1,
    "progress_bar": False,
}


# =============================================================================
# 6. Balance reward
# =============================================================================
# TD3-aligned gated swing-up reward (Bonus 2).
REWARD: Dict[str, Any] = {
    "base_reward": 1.0,
    "a_theta": 0.001,
    "a_alpha": 8.0,
    "a_theta_dot": 0.001,
    "a_alpha_dot": 0.005,
    "a_u": 0.5,
    "a_du": 0.0,
    "gate_theta_max_rad": 1.0471975511965976,
    "gate_theta_dot_max": 5.0,
    "gate_alpha_max_deg": 12.0,
    "gate_alpha_dot_max": 2.0,
}


# =============================================================================
# 7. Evaluation and checkpoints
# =============================================================================
EVAL: Dict[str, Any] = {
    "eval_freq": 25001,
    "n_eval_episodes": 8,
    "max_eval_policy_steps": 2000,
    "checkpoint_freq": 50000,
    "save_best_model": True,
    "eval_randomization_level": 0.75,
}


# =============================================================================
# 8. Smoke mode settings
# =============================================================================
SMOKE: Dict[str, Any] = {
    "total_timesteps": 2048,
    "n_envs": 2,
    "n_steps": 128,
    "batch_size": 128,
    "eval_freq": 1024,
    "n_eval_episodes": 2,
    "max_eval_policy_steps": 300,
    "checkpoint_freq": 1024,
}


# =============================================================================
# 9. Optional Tkinter training panel
# =============================================================================
PANEL: Dict[str, Any] = {
    "progress_update_freq": 5000,
    "auto_save_before_run": True,
    "window_geometry": "1280x820",
    # Per-episode stdout is very noisy; metrics still go to training_metrics.csv.
    "print_train_episodes": False,
}


# =============================================================================
# Helper functions used by run.py
# =============================================================================
def run_dir() -> str:
    return os.path.join(str(RUN["root_log_dir"]), str(RUN["experiment_name"]))


def activation_fn():
    table = {
        "ReLU": nn.ReLU,
        "Tanh": nn.Tanh,
        "ELU": nn.ELU,
        "SiLU": nn.SiLU,
        "LeakyReLU": nn.LeakyReLU,
    }
    name = str(PPO["activation_fn_name"])
    if name not in table:
        raise ValueError(f"Unsupported activation function: {name}. Choose one of {sorted(table)}")
    return table[name]


def wrap_to_pi(x: float) -> float:
    while x > np.pi:
        x -= 2.0 * np.pi
    while x < -np.pi:
        x += 2.0 * np.pi
    return float(x)


def build_physical_params() -> RIPPhysicalParams:
    return RIPPhysicalParams(**PHYSICAL_PARAMS)


def build_randomization_config(level: float | None = None, *, enabled: bool | None = None) -> DomainRandomizationConfig:
    dr = DOMAIN_RANDOMIZATION
    return DomainRandomizationConfig(
        enabled=bool(dr["enabled"] if enabled is None else enabled),
        level=float(dr["dr_initial_level"] if level is None else level),
        param_scale_ranges=dict(dr["param_scale_ranges"]),
        init_theta_std_deg_range=tuple(dr["init_theta_std_deg_range"]),
        init_alpha_std_deg_range=tuple(dr["init_alpha_std_deg_range"]),
        init_theta_dot_std_range=tuple(dr["init_theta_dot_std_range"]),
        init_alpha_dot_std_range=tuple(dr["init_alpha_dot_std_range"]),
        pwm_limit_scale_range=tuple(dr["pwm_limit_scale_range"]),
        pwm_gain_range=tuple(dr["pwm_gain_range"]),
        pwm_bias_range=tuple(dr["pwm_bias_range"]),
        pwm_deadzone_range=tuple(dr["pwm_deadzone_range"]),
        pwm_noise_sigma_range=tuple(dr["pwm_noise_sigma_range"]),
        actuator_tau_range=tuple(dr["actuator_tau_range"]),
        action_delay_steps_range=tuple(dr["action_delay_steps_range"]),
        theta_bias_range=tuple(dr["theta_bias_range"]),
        alpha_bias_range=tuple(dr["alpha_bias_range"]),
        theta_sigma_range=tuple(dr["theta_sigma_range"]),
        alpha_sigma_range=tuple(dr["alpha_sigma_range"]),
        theta_dot_sigma_range=tuple(dr["theta_dot_sigma_range"]),
        alpha_dot_sigma_range=tuple(dr["alpha_dot_sigma_range"]),
        encoder_quantization_rad_range=tuple(dr["encoder_quantization_rad_range"]),
        use_lpf_velocity_probability=float(dr["use_lpf_velocity_probability"]),
        velocity_lpf_range=tuple(dr["velocity_lpf_range"]),
        process_theta_dot_sigma_range=tuple(dr["process_theta_dot_sigma_range"]),
        process_alpha_dot_sigma_range=tuple(dr["process_alpha_dot_sigma_range"]),
    )


def build_env_config(*, randomization_level: float | None = None, nominal: bool = False) -> EnvConfig:
    level = 0.0 if nominal else (DOMAIN_RANDOMIZATION["dr_initial_level"] if randomization_level is None else randomization_level)
    randomization_enabled = False if nominal else bool(DOMAIN_RANDOMIZATION["enabled"])

    cfg = EnvConfig(
        dt=float(ENV["physical_dt"]),
        max_steps=int(ENV["max_physical_steps"]),
        action_type=str(ENV["action_type"]),
        discrete_actions=[],
        continuous_pwm_limit=float(ENV["pwm_limit"]),
        observation_type=str(ENV["observation_type"]),
        clip_velocity_in_obs=bool(ENV["clip_velocity_in_obs"]),
        physical_params=build_physical_params(),
        init_state=InitStateConfig(
            mode=str(ENV["init_mode"]),
            theta_mean_deg=float(ENV["init_theta_mean_deg"]),
            theta_std_deg=float(ENV["init_theta_std_deg"]),
            theta_dot_mean=float(ENV["init_theta_dot_mean"]),
            theta_dot_std=float(ENV["init_theta_dot_std"]),
            alpha_mean_deg=float(ENV["init_alpha_mean_deg"]),
            alpha_std_deg=float(ENV["init_alpha_std_deg"]),
            alpha_dot_mean=float(ENV["init_alpha_dot_mean"]),
            alpha_dot_std=float(ENV["init_alpha_dot_std"]),
        ),
        limits=LimitConfig(
            theta_limit=float(ENV["theta_limit"]),
            theta_dot_limit=float(ENV["theta_dot_limit"]),
            alpha_dot_limit=float(ENV["alpha_dot_limit"]),
            terminate_on_alpha_abs_deg=bool(ENV["terminate_on_alpha_abs_deg"]),
            alpha_abs_limit_deg=float(ENV["alpha_abs_limit_deg"]),
        ),
        noise=NoiseConfig(
            enabled=bool(ENV["fixed_noise_enabled"]),
            theta_sigma=float(ENV["fixed_noise_theta_sigma"]),
            alpha_sigma=float(ENV["fixed_noise_alpha_sigma"]),
            theta_dot_sigma=float(ENV["fixed_noise_theta_dot_sigma"]),
            alpha_dot_sigma=float(ENV["fixed_noise_alpha_dot_sigma"]),
        ),
        logging=LoggingConfig(
            enabled=bool(ENV["env_logging_enabled"]),
            log_dir=os.path.join(run_dir(), "env_logs"),
            episode_csv_name="episode_log.csv",
            step_csv_name="step_log.csv",
            episode_jsonl_name="episode_log.jsonl",
            save_step_log=bool(ENV["env_save_step_log"]),
            flush_every_step=bool(ENV["env_flush_every_step"]),
            write_config_json=True,
        ),
    )

    # Dynamic attribute used by CartPoleRIPSim2RealEnv.
    cfg.randomization = build_randomization_config(level=float(level), enabled=randomization_enabled)
    return cfg


def build_done_fn():
    done_cfg = {
        "theta_limit": float(ENV["theta_limit"]),
        "theta_dot_limit": float(ENV["theta_dot_limit"]),
        "alpha_dot_limit": float(ENV["alpha_dot_limit"]),
        "terminate_on_alpha_abs_deg": bool(ENV["terminate_on_alpha_abs_deg"]),
        "alpha_abs_limit_deg": float(ENV["alpha_abs_limit_deg"]),
    }
    return build_done_fn_from_dict(done_cfg)


def _get_env_step_count(env) -> int:
    for name in (
        "step_count", "current_step", "_step_count", "_current_step",
        "elapsed_steps", "_elapsed_steps", "t", "_t",
    ):
        if hasattr(env, name):
            try:
                return int(getattr(env, name))
            except Exception:
                pass
    return 999999


def build_reward_fn():
    """TD3-style gated upright bonus minus quadratic costs (Bonus-2 swing-up)."""
    r = REWARD

    def _reward_fn(state, action, next_state, env) -> float:
        theta, theta_dot, alpha, alpha_dot = map(
            float, np.asarray(next_state, dtype=np.float64).reshape(4)
        )
        u = float(np.clip(np.asarray(action, dtype=float).reshape(-1)[0], -1.0, 1.0))
        step_count = _get_env_step_count(env)
        if (not hasattr(env, "bonus2_prev_u")) or step_count <= 1:
            env.bonus2_prev_u = 0.0
        du = u - float(env.bonus2_prev_u)
        env.bonus2_prev_u = u
        alpha_wrap = wrap_to_pi(alpha)
        gate = (
            abs(theta) < float(r["gate_theta_max_rad"])
            and abs(theta_dot) < float(r["gate_theta_dot_max"])
            and abs(alpha_wrap) < np.deg2rad(float(r["gate_alpha_max_deg"]))
            and abs(alpha_dot) < float(r["gate_alpha_dot_max"])
        )
        p_k = float(r["base_reward"]) if gate else 0.0
        cost = (
            float(r["a_theta"]) * theta * theta
            + float(r["a_alpha"]) * alpha_wrap * alpha_wrap
            + float(r["a_theta_dot"]) * theta_dot * theta_dot
            + float(r["a_alpha_dot"]) * alpha_dot * alpha_dot
            + float(r["a_u"]) * u * u
            + float(r["a_du"]) * du * du
        )
        return float(p_k - cost)

    return _reward_fn


def build_policy_kwargs() -> Dict[str, Any]:
    return {
        "net_arch": {
            "pi": list(PPO["net_arch_pi"]),
            "vf": list(PPO["net_arch_vf"]),
        },
        "activation_fn": activation_fn(),
        "log_std_init": float(PPO["log_std_init"]),
    }


def build_ppo_kwargs() -> Dict[str, Any]:
    kwargs = {
        "learning_rate": float(PPO["learning_rate"]),
        "n_steps": int(PPO["n_steps"]),
        "batch_size": int(PPO["batch_size"]),
        "n_epochs": int(PPO["n_epochs"]),
        "gamma": float(PPO["gamma"]),
        "gae_lambda": float(PPO["gae_lambda"]),
        "clip_range": float(PPO["clip_range"]),
        "ent_coef": float(PPO["ent_coef"]),
        "vf_coef": float(PPO["vf_coef"]),
        "max_grad_norm": float(PPO["max_grad_norm"]),
        "normalize_advantage": bool(PPO["normalize_advantage"]),
        "policy_kwargs": build_policy_kwargs(),
        "verbose": int(PPO["verbose"]),
        "seed": int(RUN["seed"]),
        "device": str(RUN["device"]),
    }
    if PPO["target_kl"] is not None:
        kwargs["target_kl"] = float(PPO["target_kl"])
    return kwargs


def full_config_dict() -> Dict[str, Any]:
    return {
        "ALGORITHM_NAME": ALGORITHM_NAME,
        "MODE": MODE,
        "EVAL_MODEL_PATH": EVAL_MODEL_PATH,
        "RUN": RUN,
        "ENV": ENV,
        "PHYSICAL_PARAMS": PHYSICAL_PARAMS,
        "DOMAIN_RANDOMIZATION": DOMAIN_RANDOMIZATION,
        "PPO": PPO,
        "REWARD": REWARD,
        "EVAL": EVAL,
        "SMOKE": SMOKE,
        "PANEL": PANEL,
    }


def env_config_to_dict(env_cfg: EnvConfig) -> Dict[str, Any]:
    d = env_cfg.to_dict() if hasattr(env_cfg, "to_dict") else asdict(env_cfg)
    if hasattr(env_cfg, "randomization"):
        d["randomization"] = env_cfg.randomization.to_dict()
    return d
