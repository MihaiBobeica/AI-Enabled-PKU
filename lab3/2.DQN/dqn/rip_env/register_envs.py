from gymnasium.envs.registration import register, registry


def register_all_envs() -> None:
    entries = {
        "RIPSim2RealSwingUp-v0": "rip_env.envs.cartpole_rip_sim2real:CartPoleRIPSim2RealEnv",
        # Backward-compatible alias for older checkpoints/tools.
        "RIPSim2RealBalance-v0": "rip_env.envs.cartpole_rip_sim2real:CartPoleRIPSim2RealEnv",
    }
    for env_id, entry_point in entries.items():
        if env_id not in registry:
            register(id=env_id, entry_point=entry_point)
