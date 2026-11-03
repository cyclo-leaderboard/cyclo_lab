"""Register only the task-independent local WhiteRoom environment."""

import gymnasium as gym


gym.register(
    id="Cyclo-Real-WhiteRoom-FFW-SG2-v0",
    entry_point="cyclo_lab.manager_based.continuous_env:ContinuousManagerBasedEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:ContinuousWhiteRoomEnvCfg",
    },
    disable_env_checker=True,
)
