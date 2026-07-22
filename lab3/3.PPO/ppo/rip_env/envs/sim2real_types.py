# rip_env/envs/sim2real_types.py
# Sim-to-real domain randomization config for rotary inverted pendulum balance.
# Added for Stage-1 PPO benchmark environment.

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Tuple, Any


Range = Tuple[float, float]


@dataclass
class DomainRandomizationConfig:
    """Episode-level randomization over dynamics, actuator and sensing.

    The ranges are intentionally expressed as multipliers around the nominal
    model whenever the key ends with ``_scale``.  A curriculum level in [0, 1]
    shrinks or expands each range around 1.0:
        level=0.0 -> no randomization
        level=1.0 -> use full configured range

    This implements the common sim-to-real idea: train on a distribution of
    simulators instead of a single clean simulator.
    """

    enabled: bool = True
    level: float = 0.65

    # Mechanical and motor-parameter multipliers.
    # Moderate defaults are deliberately not extreme; start with level 0.3-0.6,
    # then use 0.8-1.0 for robustness experiments.
    param_scale_ranges: Dict[str, Range] = field(default_factory=lambda: {
        # gravity/model convention
        "g_scale": (0.98, 1.02),

        # arm and pendulum masses/geometry
        "m1_scale": (0.75, 1.25),
        "m2_scale": (0.65, 1.35),
        "l1_scale": (0.90, 1.10),
        "l1cg_scale": (0.85, 1.15),
        "l2cg_scale": (0.75, 1.25),

        # inertias are often the most uncertain when hardware is changed
        "I1z_scale": (0.55, 1.70),
        "I2x_scale": (0.50, 1.80),
        "I2y_scale": (0.50, 1.80),
        "I2z_scale": (0.50, 1.80),

        # viscous friction is commonly poorly modeled; use wider ranges
        "c_theta_scale": (0.25, 3.50),
        "c_alpha_scale": (0.25, 4.50),

        # motor constants and PWM-voltage mapping
        "k_t_scale": (0.75, 1.25),
        "k_b_scale": (0.75, 1.25),
        "k_u_scale": (0.70, 1.30),
        "R_scale": (0.80, 1.25),
    })

    # Initial-state curriculum for upright balance.
    init_theta_std_deg_range: Range = (1.0, 8.0)
    init_alpha_std_deg_range: Range = (1.0, 12.0)
    init_theta_dot_std_range: Range = (0.0, 1.5)
    init_alpha_dot_std_range: Range = (0.0, 2.5)

    # Actuator randomization.
    pwm_limit_scale_range: Range = (0.80, 1.05)
    pwm_gain_range: Range = (0.75, 1.25)
    pwm_bias_range: Range = (-4.0, 4.0)          # PWM units
    pwm_deadzone_range: Range = (0.0, 12.0)      # PWM units
    pwm_noise_sigma_range: Range = (0.0, 2.0)    # PWM units
    actuator_tau_range: Range = (0.0, 0.030)     # seconds, first-order lag
    action_delay_steps_range: Tuple[int, int] = (0, 3)

    # Observation randomization.
    theta_bias_range: Range = (-0.010, 0.010)       # rad
    alpha_bias_range: Range = (-0.012, 0.012)       # rad
    theta_sigma_range: Range = (0.0, 0.006)         # rad
    alpha_sigma_range: Range = (0.0, 0.008)         # rad
    theta_dot_sigma_range: Range = (0.0, 0.35)      # rad/s
    alpha_dot_sigma_range: Range = (0.0, 0.60)      # rad/s
    encoder_quantization_rad_range: Range = (0.0, 0.0015)

    # Optional measurement filtering: real hardware usually estimates velocity
    # from angle differences.  Keep probability < 1 to avoid over-regularizing.
    use_lpf_velocity_probability: float = 0.50
    velocity_lpf_range: Range = (0.15, 0.45)

    # Small per-step process disturbance after integration, disabled by default
    # for Stage-1 balance reproducibility. Enable later if needed.
    process_theta_dot_sigma_range: Range = (0.0, 0.0)
    process_alpha_dot_sigma_range: Range = (0.0, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeRandomizationSnapshot:
    """Recorded parameters sampled at the start of an episode."""

    level: float = 0.0
    physical_param_scales: Dict[str, float] = field(default_factory=dict)
    pwm_limit: float = 150.0
    pwm_gain: float = 1.0
    pwm_bias: float = 0.0
    pwm_deadzone: float = 0.0
    pwm_noise_sigma: float = 0.0
    actuator_tau: float = 0.0
    action_delay_steps: int = 0
    theta_bias: float = 0.0
    alpha_bias: float = 0.0
    theta_sigma: float = 0.0
    alpha_sigma: float = 0.0
    theta_dot_sigma: float = 0.0
    alpha_dot_sigma: float = 0.0
    encoder_quantization_rad: float = 0.0
    use_lpf_velocity: bool = False
    velocity_lpf: float = 0.25

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
