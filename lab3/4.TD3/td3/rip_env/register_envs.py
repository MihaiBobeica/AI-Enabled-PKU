# rip_env/register_envs.py
# Stage-1 only exposes the sim-to-real balance environment.

from gymnasium.envs.registration import register

_ALREADY_REGISTERED = False


def register_all_envs() -> None:
    global _ALREADY_REGISTERED
    if _ALREADY_REGISTERED:
        return

    register(
        id="RIPSim2RealBalance-v0",
        entry_point="rip_env.envs.cartpole_rip_sim2real:CartPoleRIPSim2RealEnv",
    )

    _ALREADY_REGISTERED = True
