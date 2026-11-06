"""Task000525 Showroom registrations and runtime package."""

TASK_ID = "000525"
ENVIRONMENT = "showroom"
_PACKAGE = "cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.task_000525"
_TASKS = "cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks"
_RUNTIME = "cyclo_lab/manager_based/manipulation/showroom/config/ffw_sg2/tasks"


def register_environments() -> None:
    from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.catalog_runtime import (
        mount_task_package,
    )
    mount_task_package(TASK_ID, _TASKS, _RUNTIME)
    import gymnasium as gym

    prefix = f"Cyclo-Real-Showroom-Task{TASK_ID}"
    variants = {
        "": ("isaaclab.envs:ManagerBasedRLEnv", f"{_PACKAGE}.env_cfg:Task{TASK_ID}EnvCfg"),
        "-Random": ("isaaclab.envs:ManagerBasedRLEnv", f"{_PACKAGE}.env_cfg:Task{TASK_ID}RandomEnvCfg"),
        "-Trajectory-Generation": (
            f"{_PACKAGE}.locomanipulation_sdg_env:Task{TASK_ID}LocomanipulationSDGEnv",
            f"{_PACKAGE}.locomanipulation_sdg_env:Task{TASK_ID}TrajectoryGenerationEnvCfg",
        ),
        "-Locomanipulation-SDG": (
            f"{_PACKAGE}.locomanipulation_sdg_env:Task{TASK_ID}LocomanipulationSDGEnv",
            f"{_PACKAGE}.locomanipulation_sdg_env:Task{TASK_ID}LocomanipulationSDGEnvCfg",
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
