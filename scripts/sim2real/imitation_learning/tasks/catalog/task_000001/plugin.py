"""Task000001 Showroom environment registrations."""

TASK_ID = "000001"
ENVIRONMENT = "showroom"
_PACKAGE = "cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.coffee_transport"
_RUNTIME = "cyclo_lab/manager_based/manipulation/showroom/config/ffw_sg2/tasks/coffee_transport"


def register_environments() -> None:
    from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.catalog_runtime import (
        mount_task_package,
    )
    mount_task_package(TASK_ID, _PACKAGE, _RUNTIME)
    import gymnasium as gym

    prefix = f"Cyclo-Real-Showroom-Task{TASK_ID}"
    variants = {
        "": ("isaaclab.envs:ManagerBasedRLEnv", f"{_PACKAGE}.task_{TASK_ID}.env_cfg:Task{TASK_ID}EnvCfg"),
        "-Random": ("isaaclab.envs:ManagerBasedRLEnv", f"{_PACKAGE}.task_{TASK_ID}.env_cfg:Task{TASK_ID}RandomEnvCfg"),
        "-Trajectory-Generation": (
            "isaaclab.envs:ManagerBasedRLEnv",
            f"{_PACKAGE}.trajectory_generation_env:Task{TASK_ID}TrajectoryGenerationEnvCfg",
        ),
    }
    for suffix, (entry_point, env_cfg_entry_point) in variants.items():
        env_id = f"{prefix}{suffix}-FFW-SG2-v0"
        if env_id not in gym.registry:
            gym.register(
                id=env_id, entry_point=entry_point,
                kwargs={"env_cfg_entry_point": env_cfg_entry_point},
                disable_env_checker=True,
            )
