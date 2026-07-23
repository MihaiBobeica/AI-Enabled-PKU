"""Rotary inverted pendulum DQN configuration.

The first 2,000,000 environment steps intentionally reproduce RN_DQN(3).m:
- exact nominal physical parameters and RK4 equations;
- current 6-D trig observation with encoder-difference LPF velocities;
- 10 discrete PWM actions;
- 6 -> 64 -> 64 -> 10 vanilla DQN;
- N(0, 0.01) weight initialization and zero biases;
- Adam learning rate 5e-4;
- target network sync every 2,000 steps;
- behavior snapshot sync every 50 steps;
- Huber TD loss, gamma 0.95, replay 30,000, batch 512;
- exponential epsilon decay 1.0 -> 0.001 with time constant 400,000.

After 2,000,000 steps, the same policy continues through fixed domain-
randomization stages. Level 0 is mathematically the MATLAB nominal system.
"""
from __future__ import annotations

import math
import os
from datetime import datetime
from typing import Any, Dict

ALGORITHM_NAME = "DQN"

RUN: Dict[str, Any] = {
    "experiment_prefix": 'dqn',
    "root_log_dir": './runs',
    "seed": 42,
    "device": 'cpu',              # auto, cpu, mps, cuda
    "torch_num_threads": 2,
    "torch_deterministic": False,
    "prefer_local_stable_baselines3": True,
    "prefer_local_third_party": True,
}

ENV: Dict[str, Any] = {
    "physical_dt": 0.005,
    "max_physical_steps": 2000,
    "theta_limit": 37.69911184307752,
    "theta_dot_limit": 45.0,
    "alpha_dot_limit": 40.0,

    "pwm_limit": 150.0,
    "discrete_actions": (-150, -120, -90, -60, -30, 30, 60, 90, 120, 150),

    # MATLAB reset: theta ~ N(0, 5 deg), alpha ~ pi + N(0, 8 deg), zero velocities.
    "init_theta_mean_deg": 0.0,
    "init_theta_std_deg": 5.0,
    "init_theta_dot_mean": 0.0,
    "init_theta_dot_std": 0.0,
    "init_alpha_mean_deg": 180.0,
    "init_alpha_std_deg": 8.0,
    "init_alpha_dot_mean": 0.0,
    "init_alpha_dot_std": 0.0,

    # MATLAB observer: raw wrapped angle difference followed by LPF.
    "velocity_lpf": 0.25,
    "clip_velocity_in_observation": True,

    # The nominal phase must remain completely clean.
    "nominal_measurement_noise": False,
    "nominal_delay": False,
    "nominal_dynamics_residual_network": False,

    "observation_dim": 6,
    "observation_order": ('sin(theta)', 'cos(theta)', 'theta_dot_lpf', 'sin(alpha)', 'cos(alpha)', 'alpha_dot_lpf'),
}

# Exact constants from RN_DQN(3).m.
PHYSICAL_PARAMS: Dict[str, float] = {
    "g": 9.8,
    "c_theta": 0.025,
    "c_alpha": 0.001,
    "k_t": 0.231,
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
    "training_stage_names": ('nominal', 'dr_0.10', 'dr_0.30', 'dr_0.50'),
    "training_stage_levels": (0.0, 0.1, 0.3, 0.5),
    "training_stage_steps": (2000000, 1000000, 1000000, 1000000),
    "training_stage_fractions": (0.4, 0.2, 0.2, 0.2),
    "nominal_recovery_steps": 2000000,

    "clear_replay_between_stages": True,
    "reset_optimizer_between_stages": True,
    "sync_target_between_stages": True,
    "sync_behavior_snapshot_between_stages": True,
    "stage_replay_warmup_steps": 25000,

    # Full ranges are preserved. A stage level interpolates each range toward nominal.
    "param_scale_ranges": {
        "g_scale": (0.95, 1.05),
        "m1_scale": (0.75, 1.25),
        "m2_scale": (0.85, 1.15),
        "l1_scale": (0.9, 1.1),
        "l1cg_scale": (0.85, 1.15),
        "l2cg_scale": (0.75, 1.25),
        "I1z_scale": (0.55, 1.7),
        "I2x_scale": (0.8, 1.2),
        "I2y_scale": (0.5, 1.8),
        "I2z_scale": (0.5, 1.8),
        "c_theta_scale": (0.25, 3.5),
        "c_alpha_scale": (0.25, 4.5),
        "k_t_scale": (0.75, 1.25),
        "k_b_scale": (0.75, 1.25),
        "k_u_scale": (0.7, 1.3),
        "R_scale": (0.8, 1.25),
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
    "action_delay_steps_range": (0, 1),

    "theta_bias_range": (-0.01, 0.01),
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

DQN: Dict[str, Any] = {
    "total_timesteps": 5000000,
    "n_envs": 4,

    # Exact MATLAB values.
    "learning_rate": 0.0005,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_eps": 1e-08,
    "buffer_size": 30000,
    "learning_starts": 2000,
    "batch_size": 512,
    "gamma": 0.95,
    "train_freq": 4,
    "gradient_steps": 4,
    "target_update_interval": 2000,
    "behavior_snapshot_interval": 50,
    "weight_init_std": 0.01,
    "huber_delta": 1.0,
    "max_grad_norm": 0.0,          # 0 disables clipping, matching MATLAB.
    "use_double_dqn": True,  # stronger than course zip vanilla DQN; overnight default

    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.001,
    "exploration_decay": 400000.0,

    "net_arch": (64, 64),
    "activation_fn_name": 'ReLU',

    "save_every_steps": 20000,
    "save_replay_buffer": False,
    "progress_print_every_steps": 20000,

    # MATLAB stopping rule: last 10 episodes mean(max stable run) > 1400 steps.
    "stop_avg_window": 10,
    "stop_avg_stable_steps_threshold": 1400.0,
    "save_nominal_success_snapshot": True,
    "allow_early_nominal_stage_transition": False,

    # Post-2M optimization: restore some exploration after each distribution jump.
    "reset_exploration_each_randomized_stage": True,
    "randomized_stage_initial_eps": 0.1,
    "randomized_stage_final_eps": 0.01,
    "randomized_stage_exploration_decay": 300000.0,

    "direct_export_teacher": True,
}

REWARD: Dict[str, Any] = {
    "k_cos_alpha": 10.0,
    "k_alpha_dot": 0.001,
    "k_theta_dot": 0.0001,
    "k_theta": 0.0,
    "alpha_penalty_deg": 15.0,
    "alpha_penalty_value": 5.0,
    "action_l2": 0.0,
}

EVAL: Dict[str, Any] = {
    "eval_freq": 50000,
    "n_eval_episodes": 3,
    "max_eval_policy_steps": 2000,
    "checkpoint_freq": 100000,
    "save_best_model": True,

    "nominal_eval_randomization_level": 0.0,
    "randomized_eval_randomization_level": 0.5,

    "capture_angle_deg": 15.0,
    "stable_alpha_dot_max": 4.0,
    "stable_hold_steps": 1400,

    # Randomized-phase early stop only; nominal has the MATLAB stop rule above.
    "early_stop_enabled": True,
    "early_stop_start_fraction": 0.9,
    "early_stop_patience_evals": 8,
    "early_stop_min_success_rate": 0.6,
    "early_stop_reward_min_delta": 0.02,

    # Randomized best replaces nominal only if it reaches this threshold.
    "randomized_model_min_success_for_selection": 0.6,
}

# The trained DQN is already deployment-sized and is exported directly.

TEST: Dict[str, Any] = {
    "duration_seconds": 30.0,
    "randomization_level": 0.5,
    "seed": 2026,
    "capture_angle_deg": 15.0,
    "stable_alpha_dot_max": 4.0,
    "stable_hold_steps": 1400,
}

SMOKE: Dict[str, Any] = {
    "total_timesteps": 512,
    "stage_levels": (0.0, 0.1),
    "stage_steps": (256, 256),
    "learning_starts": 32,
    "stage_replay_warmup_steps": 32,
    "buffer_size": 2000,
    "batch_size": 32,
    "eval_freq": 256,
    "n_eval_episodes": 1,
    "max_eval_policy_steps": 100,
    "checkpoint_freq": 256,
    "test_duration_seconds": 0.25,
}

PANEL: Dict[str, Any] = {
    "progress_update_freq": 2000,
    "episode_curve_max_points": 2000,
    "save_dr_audit_csv": True,
    "auto_save_before_run": True,
    "window_geometry": '1460x920',
    "lock_config_while_running": True,
}


def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_run_dir() -> str:
    return os.path.join(
        str(RUN["root_log_dir"]),
        f"{RUN['experiment_prefix']}_{now_str()}",
    )


def config_sections() -> tuple[str, ...]:
    return (
        "RUN", "ENV", "PHYSICAL_PARAMS", "DOMAIN_RANDOMIZATION", "DQN",
        "REWARD", "EVAL", "TEST", "SMOKE", "PANEL",
    )
