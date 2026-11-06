"""Continuous SG2 showroom task registration."""

import gymnasium as gym


gym.register(
    id="Cyclo-Real-Showroom-FFW-SG2-v0",
    entry_point="cyclo_lab.manager_based.continuous_env:ContinuousManagerBasedEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.platform.env_cfg:ContinuousShowroomEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Cyclo-Real-Showroom-Random-FFW-SG2-v0",
    entry_point="cyclo_lab.manager_based.continuous_env:ContinuousManagerBasedEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.platform.env_cfg:ContinuousRandomShowroomEnvCfg",
    },
    disable_env_checker=True,
)


# The catalog owns numbered task registration and runtime package paths.
from .tasks.catalog_runtime import register_catalog_task

for _task_id in ("000001", "000002", "000458", "000525"):
    register_catalog_task(_task_id)
