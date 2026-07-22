# rl_env/envs/types.py

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional


@dataclass
class RIPPhysicalParams:
    """
    Furuta / Rotary Inverted Pendulum physical parameters.
    """
    g: float = 9.8

    c_theta: float = 0.025
    c_alpha: float = 0.001

    k_t: float = 0.2310
    k_b: float = 0.1875
    k_u: float = 0.04706
    R: float = 4.2857

    m1: float = 0.20625
    m2: float = 0.15845
    l1cg: float = 0.080305
    l1: float = 0.151894
    l2cg: float = 0.066733
    I1z: float = 0.00049228
    I2x: float = 0.00036892
    I2y: float = 2.3641e-05
    I2z: float = 0.00036139

    @property
    def K1(self) -> float:
        return self.k_t * self.k_u / self.R

    @property
    def K2(self) -> float:
        return self.k_t * self.k_b / self.R


@dataclass
class InitStateConfig:
    """
    Initial state distribution config.
    theta: rotary arm angle
    alpha: pendulum angle, upright is 0, downward is pi (after wrap)
    """
    mode: str = "downward"  # "downward", "upright", "custom"
    theta_mean_deg: float = 0.0
    theta_std_deg: float = 5.0

    theta_dot_mean: float = 0.0
    theta_dot_std: float = 0.0

    alpha_mean_deg: float = 180.0
    alpha_std_deg: float = 8.0

    alpha_dot_mean: float = 0.0
    alpha_dot_std: float = 0.0

    custom_state: Optional[List[float]] = None


@dataclass
class LimitConfig:
    theta_limit: float = 12.0 * 3.141592653589793
    theta_dot_limit: float = 45.0
    alpha_dot_limit: float = 40.0

    # Optional: if you want to terminate by angle range too, set terminate_on_alpha_abs_deg=True
    terminate_on_alpha_abs_deg: bool = False
    alpha_abs_limit_deg: float = 90.0


@dataclass
class NoiseConfig:
    enabled: bool = False
    theta_sigma: float = 0.0
    alpha_sigma: float = 0.0
    theta_dot_sigma: float = 0.0
    alpha_dot_sigma: float = 0.0


@dataclass
class LoggingConfig:
    enabled: bool = True
    log_dir: str = "./experiments/default_run"
    episode_csv_name: str = "episode_log.csv"
    step_csv_name: str = "step_log.csv"
    episode_jsonl_name: str = "episode_log.jsonl"
    save_step_log: bool = False
    flush_every_step: bool = False
    write_config_json: bool = True


@dataclass
class EnvConfig:
    dt: float = 0.005
    max_steps: int = 2000

    action_type: str = "discrete"  # "discrete" or "continuous"
    discrete_actions: List[float] = field(
        default_factory=lambda: [-150, -120, -90, -60, -30, 30, 60, 90, 120, 150]
    )
    continuous_pwm_limit: float = 150.0

    observation_type: str = "trig"  # "trig" or "raw"
    clip_velocity_in_obs: bool = True

    # Optional LPF velocity estimation.
    # Default False keeps old TD3/PPO/DQN behavior unchanged.
    # When True, observation/reward/done use velocities estimated from angle differences.
    use_lpf_velocity: bool = False
    velocity_lpf: float = 0.25

    physical_params: RIPPhysicalParams = field(default_factory=RIPPhysicalParams)
    init_state: InitStateConfig = field(default_factory=InitStateConfig)
    limits: LimitConfig = field(default_factory=LimitConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)