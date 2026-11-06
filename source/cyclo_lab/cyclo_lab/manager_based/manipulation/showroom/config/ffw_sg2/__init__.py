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

from .tasks.catalog_runtime import register_catalog_task

register_catalog_task("000458")

for _task_id in ("000001", "000002"):
    gym.register(
        id=f"Cyclo-Real-Showroom-Task{_task_id}-FFW-SG2-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.tasks.coffee_transport.task_{_task_id}.env_cfg:Task{_task_id}EnvCfg"
            ),
        },
        disable_env_checker=True,
    )
    gym.register(
        id=f"Cyclo-Real-Showroom-Task{_task_id}-Random-FFW-SG2-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.tasks.coffee_transport.task_{_task_id}.env_cfg:Task{_task_id}RandomEnvCfg"
            ),
        },
        disable_env_checker=True,
    )
    gym.register(
        id=(
            f"Cyclo-Real-Showroom-Task{_task_id}-Trajectory-Generation-"
            "FFW-SG2-v0"
        ),
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.tasks.coffee_transport.trajectory_generation_env:"
                f"Task{_task_id}TrajectoryGenerationEnvCfg"
            ),
        },
        disable_env_checker=True,
    )


register_catalog_task("000525")
