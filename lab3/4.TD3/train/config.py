"""Recovery-first TD3 configuration for the rotary inverted pendulum.

Production schedule
-------------------
- 0 .. 2,000,000 environment steps: exact nominal/paper-aligned TD3 setup.
- 2M .. 3M: fixed domain-randomization level 0.10.
- 3M .. 4M: fixed domain-randomization level 0.30.
- 4M .. 5M: fixed domain-randomization level 0.50.

The actor/critic architecture, observation convention, reward, action mapping,
and TD3 optimizer hyperparameters never change at stage boundaries.  Only the
environment distribution changes after the protected nominal recovery phase.
"""
from __future__ import annotations

import os
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


ALGORITHM_NAME = "TD3"

RUN: Dict[str, Any] = {
    "experiment_prefix": "td3_rip_original_2m_then_dr",
    "root_log_dir": './runs',
    "seed": 42,
    "prefer_local_stable_baselines3": True,
    "torch_deterministic": False,
    "torch_num_threads": 2,
    "device": 'cpu',            # CPU for this MLP + env-sim workload
    "vec_env_type": 'dummy',
}

ENV: Dict[str, Any] = {
    "env_id": "RIPSim2RealBalance-v0",
    "physical_dt": 0.005,
    "action_repeat": 1,
    "max_physical_steps": 1000,

    "action_type": "continuous",
    "pwm_limit": 150.0,
    # Kept only because EnvConfig requires it; TD3 does not use this table.
    "discrete_actions_compat": (-150, -120, -90, -60, -30, 30, 60, 90, 120, 150),

    # Exact original TD3 observation:
    # [sin(theta), cos(theta), theta_dot,
    #  sin(alpha), cos(alpha), alpha_dot, last_action_norm]
    "observation_type": 'trig_hist1_act1',
    "state_history_len": 1,
    "action_history_len": 1,
    "observation_dim": 7,
    "fixed_observation_scaling": False,
    "clip_velocity_in_obs": True,

    # Original reset approximation: uniform +/-45 deg represented by the
    # equivalent Gaussian standard deviation 45/sqrt(3).
    "init_mode": "downward",
    "init_theta_mean_deg": 0.0,
    "init_theta_std_deg": 25.98076211353316,
    "init_theta_dot_mean": 0.0,
    "init_theta_dot_std": 0.0,
    "init_alpha_mean_deg": 180.0,
    "init_alpha_std_deg": 25.98076211353316,
    "init_alpha_dot_mean": 0.0,
    "init_alpha_dot_std": 0.0,

    "theta_limit": 12.0 * np.pi,
    "theta_dot_limit": 45.0,
    "alpha_dot_limit": 40.0,
    "terminate_on_alpha_abs_deg": False,
    "alpha_abs_limit_deg": 45.0,

    "fixed_noise_enabled": False,
    "fixed_noise_theta_sigma": 0.0,
    "fixed_noise_alpha_sigma": 0.0,
    "fixed_noise_theta_dot_sigma": 0.0,
    "fixed_noise_alpha_dot_sigma": 0.0,

    # The uploaded original TD3 configuration used true simulated velocity.
    "use_lpf_velocity": False,
    "velocity_lpf": 0.25,

    "env_logging_enabled": True,
    "env_save_step_log": False,
    "env_flush_every_step": False,
}

# Exact defaults used by RIPPhysicalParams() in the uploaded original TD3 file.
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

DOMAIN_RANDOMIZATION: Dict[str, Any] = {
    "enabled": True,
    "dr_initial_level": 0.0,
    "dr_final_level": 0.50,
    "dr_curriculum_fraction": 1.0,

    "training_stage_names": ('nominal_original_alignment', 'randomization_0.10', 'randomization_0.30', 'randomization_0.50'),
    "training_stage_levels": (0.0, 0.1, 0.3, 0.5),
    "training_stage_steps": (2000000, 1000000, 1000000, 1000000),
    "training_stage_fractions": (0.4, 0.2, 0.2, 0.2),
    "nominal_recovery_steps": 2000000,

    "clear_replay_between_stages": True,
    "reset_optimizer_between_stages": True,
    "sync_target_between_stages": True,
    "reset_action_noise_between_stages": True,
    "stage_replay_warmup_steps": 25000,

    # The narrower user-edited randomization ranges are retained verbatim.
    "param_scale_ranges": {
        "g_scale": (0.98, 1.02),
        "m1_scale": (0.9, 1.1),
        "m2_scale": (0.9, 1.1),
        "l1_scale": (0.9, 1.1),
        "l1cg_scale": (0.9, 1.1),
        "l2cg_scale": (0.9, 1.1),
        "I1z_scale": (0.9, 1.1),
        "I2x_scale": (0.9, 1.1),
        "I2y_scale": (0.9, 1.1),
        "I2z_scale": (0.9, 1.1),
        "c_theta_scale": (0.5, 2.0),
        "c_alpha_scale": (0.5, 2.0),
        "k_t_scale": (0.9, 1.1),
        "k_b_scale": (0.9, 1.1),
        "k_u_scale": (0.9, 1.1),
        "R_scale": (0.9, 1.1),
    },
    "init_theta_std_deg_range": (2.0, 10.0),
    "init_alpha_std_deg_range": (4.0, 16.0),
    "init_theta_dot_std_range": (0.0, 1.5),
    "init_alpha_dot_std_range": (0.0, 2.5),
    "pwm_limit_scale_range": (0.8, 1.05),
    "pwm_gain_range": (0.9, 1.1),
    "pwm_bias_range": (-4.0, 4.0),
    "pwm_deadzone_range": (0.0, 10.0),
    "pwm_noise_sigma_range": (0.0, 2.0),
    "actuator_tau_range": (0.0, 0.03),
    "action_delay_steps_range": (0, 0),
    "theta_bias_range": (-0.010, 0.010),
    "alpha_bias_range": (-0.012, 0.012),
    "theta_sigma_range": (0.0, 0.006),
    "alpha_sigma_range": (0.0, 0.02),
    "theta_dot_sigma_range": (0.0, 0.35),
    "alpha_dot_sigma_range": (0.0, 0.6),
    "encoder_quantization_rad_range": (0.0, 0.0015),
    "use_lpf_velocity_probability": 0.5,
    "velocity_lpf_range": (0.2, 0.3),
    "process_theta_dot_sigma_range": (0.0, 0.0),
    "process_alpha_dot_sigma_range": (0.0, 0.0),
}

TD3: Dict[str, Any] = {
    "total_timesteps": 5000000,
    "n_envs": 1,

    # Uploaded paper-aligned baseline.
    "buffer_size": 1000000,
    "learning_starts": 10000,
    "batch_size": 128,
    "actor_learning_rate": 0.0001,
    "critic_learning_rate": 0.001,
    "gamma": 0.99,
    "tau": 0.005,
    "train_freq": 1,
    "gradient_steps": 1,
    "policy_delay": 2,
    "target_policy_noise": 0.2,
    "target_noise_clip": 0.5,
    "actor_grad_clip": 1.0,
    "critic_grad_clip": 1.0,
    "action_noise_sigma": 0.1,
    "net_arch_pi": (64, 64),
    "net_arch_qf": (64, 64),
    "activation_fn_name": 'ReLU',
    "optimize_memory_usage": False,
    "normalize_obs": False,
    "normalize_reward": False,
    "save_replay_buffer": False,
    "verbose": 1,
    "progress_bar": False,
}

# Exact executable reward values from the uploaded original TD3 configuration.
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

EVAL: Dict[str, Any] = {
    "eval_freq": 20000,
    "n_eval_episodes": 5,
    "max_eval_policy_steps": 1000,
    "checkpoint_freq": 100000,
    "save_best_model": True,
    "nominal_eval_randomization_level": 0.0,
    "eval_randomization_level": 0.5,
    "selected_model_min_success_rate": 0.6,
    "capture_angle_deg": 12.0,
    "stable_alpha_dot_max": 2.0,
    "stable_hold_seconds": 2.0,
    # Disabled by default to stay close to the original uninterrupted run.
    "early_stop_enabled": False,
    "early_stop_start_fraction": 0.85,
    "early_stop_patience_evals": 8,
    "early_stop_min_success_rate": 0.8,
    "early_stop_reward_min_delta": 1.0,
}

DISTILL: Dict[str, Any] = {
    "targets": "current",
    "student_hidden_sizes": (64, 64),
    "student_activation": "ReLU",
    "dagger_iterations": 6,
    "collect_steps_per_iter": 30000,
    "max_dataset_size": 250000,
    "first_iter_teacher_rollout": True,
    "student_action_probability": 0.5,
    "collect_randomization_levels": (0.0, 0.1, 0.3, 0.5),
    "epochs_per_iter": 25,
    "batch_size": 2048,
    "learning_rate": 0.001,
    "weight_decay": 1e-06,
    "grad_clip_norm": 1.0,
    "val_fraction": 0.10,
    "eval_randomization_level": 0.5,
    "eval_episodes": 16,
    "eval_max_policy_steps": 1000,
    "seed": 123,
    "device": "cpu",
}

TEST: Dict[str, Any] = {
    "duration_seconds": 30.0,
    "randomization_level": 0.5,
    "seed": 2026,
}

SMOKE: Dict[str, Any] = {
    "total_timesteps": 512,
    "stage_names": ("smoke_nominal", "smoke_randomization_0.10"),
    "stage_levels": (0.0, 0.10),
    "stage_steps": (256, 256),
    "stage_replay_warmup_steps": 16,
    "buffer_size": 2_000,
    "learning_starts": 32,
    "batch_size": 32,
    "train_freq": 16,
    "gradient_steps": 1,
    "eval_freq": 256,
    "n_eval_episodes": 1,
    "max_eval_policy_steps": 100,
    "checkpoint_freq": 256,
    "distill_collect_steps": 128,
    "distill_epochs": 1,
    "test_duration_seconds": 0.25,
}

PANEL: Dict[str, Any] = {
    "progress_update_freq": 2_000,
    "episode_curve_max_points": 2_000,
    "dr_audit_freq": 20_000,
    "save_dr_audit_csv": True,
    "auto_save_before_run": True,
    "window_geometry": "1420x900",
    "lock_config_while_running": True,
}


def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_run_dir() -> str:
    return os.path.join(
        str(RUN["root_log_dir"]),
        f"{RUN['experiment_prefix']}_{now_str()}",
    )


def observation_type() -> str:
    return str(ENV["observation_type"])


def activation_fn():
    table = {
        "ReLU": nn.ReLU,
        "Tanh": nn.Tanh,
        "ELU": nn.ELU,
        "SiLU": nn.SiLU,
        "LeakyReLU": nn.LeakyReLU,
    }
    name = str(TD3["activation_fn_name"])
    if name not in table:
        raise ValueError(f"Unsupported activation function: {name}")
    return table[name]


def build_physical_params() -> RIPPhysicalParams:
    return RIPPhysicalParams(**PHYSICAL_PARAMS)


def build_randomization_config(
    level: float | None = None,
    *,
    enabled: bool | None = None,
) -> DomainRandomizationConfig:
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


def build_env_config(
    *,
    randomization_level: float | None = None,
    nominal: bool = False,
    max_physical_steps: int | None = None,
) -> EnvConfig:
    level = 0.0 if nominal else (
        float(DOMAIN_RANDOMIZATION["dr_initial_level"])
        if randomization_level is None
        else float(randomization_level)
    )
    randomization_enabled = False if nominal else bool(DOMAIN_RANDOMIZATION["enabled"])

    cfg = EnvConfig(
        dt=float(ENV["physical_dt"]),
        max_steps=int(
            ENV["max_physical_steps"]
            if max_physical_steps is None
            else max_physical_steps
        ),
        action_type=str(ENV["action_type"]),
        discrete_actions=list(ENV["discrete_actions_compat"]),
        continuous_pwm_limit=float(ENV["pwm_limit"]),
        observation_type=observation_type(),
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
            log_dir="./runs/env_logs_pending",
            save_step_log=bool(ENV["env_save_step_log"]),
            flush_every_step=bool(ENV["env_flush_every_step"]),
            write_config_json=True,
        ),
    )
    cfg.use_lpf_velocity = bool(ENV["use_lpf_velocity"])
    cfg.velocity_lpf = float(ENV["velocity_lpf"])
    cfg.randomization = build_randomization_config(
        level=level,
        enabled=randomization_enabled,
    )
    return cfg


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


def _wrap_to_pi(x: float) -> float:
    return float((x + np.pi) % (2.0 * np.pi) - np.pi)


def build_reward_fn():
    """Build the executable reward used by the uploaded original TD3 code."""
    r = REWARD

    def _reward_fn(state, action, next_state, env) -> float:
        theta, theta_dot, alpha, alpha_dot = map(float, next_state)
        # Continuous TD3 action is already normalized to [-1, 1].  Reading it
        # directly avoids calling the sim2real actuator twice inside one step.
        u = float(np.clip(np.asarray(action, dtype=float).reshape(-1)[0], -1.0, 1.0))
        step_count = _get_env_step_count(env)
        if (not hasattr(env, "td3_prev_u")) or step_count <= 1:
            env.td3_prev_u = 0.0
        u_prev = float(env.td3_prev_u)
        du = u - u_prev
        env.td3_prev_u = u
        alpha_wrap = _wrap_to_pi(alpha)
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


def build_done_fn():
    return build_done_fn_from_dict({
        "theta_limit": ENV["theta_limit"],
        "theta_dot_limit": ENV["theta_dot_limit"],
        "alpha_dot_limit": ENV["alpha_dot_limit"],
        "terminate_on_alpha_abs_deg": ENV["terminate_on_alpha_abs_deg"],
        "alpha_abs_limit_deg": ENV["alpha_abs_limit_deg"],
    })


def build_td3_kwargs() -> Dict[str, Any]:
    return {
        # SB3 uses this for schedules/initial construction.  The custom TD3
        # class immediately assigns separate actor and critic rates.
        "learning_rate": float(TD3["actor_learning_rate"]),
        "buffer_size": int(TD3["buffer_size"]),
        "learning_starts": int(TD3["learning_starts"]),
        "batch_size": int(TD3["batch_size"]),
        "tau": float(TD3["tau"]),
        "gamma": float(TD3["gamma"]),
        "train_freq": (int(TD3["train_freq"]), "step"),
        "gradient_steps": int(TD3["gradient_steps"]),
        "policy_delay": int(TD3["policy_delay"]),
        "target_policy_noise": float(TD3["target_policy_noise"]),
        "target_noise_clip": float(TD3["target_noise_clip"]),
        "optimize_memory_usage": bool(TD3["optimize_memory_usage"]),
        "policy_kwargs": {
            "net_arch": {
                "pi": list(TD3["net_arch_pi"]),
                "qf": list(TD3["net_arch_qf"]),
            },
            "activation_fn": activation_fn(),
        },
        "verbose": int(TD3["verbose"]),
        "seed": int(RUN["seed"]),
        "device": str(RUN["device"]),
    }
