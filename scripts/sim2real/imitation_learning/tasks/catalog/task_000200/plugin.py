"""Local Task000200 registration and recording entry point."""

from __future__ import annotations

import importlib
from pathlib import Path


TASK_ID = "000200"
ENVIRONMENT = "white_room"
_ROOT = Path(__file__).resolve().parent


def add_cli_arguments(parser) -> None:
    parser.add_argument(
        "--task000200_center_seed",
        action="store_true",
        help="Record one nominal Task000200 episode with tape and plate centered.",
    )


def register_environments() -> None:
    from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.catalog_runtime import mount_task_package
    mount_task_package(TASK_ID, "cyclo_lab.assets.environments", "cyclo_lab/assets/environments")
    package = "cyclo_lab.manager_based.manipulation.white_room.config.ffw_sg2"
    mount_task_package(TASK_ID, package, "cyclo_lab/manager_based/manipulation/white_room/config/ffw_sg2")
    import gymnasium as gym

    gym.register(
        id="Cyclo-Real-WhiteRoom-Task000200-FFW-SG2-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": f"{package}.task_000200:Task000200EnvCfg"},
        disable_env_checker=True,
    )
    gym.register(
        id="Cyclo-Real-WhiteRoom-Task000200-Random-FFW-SG2-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": f"{package}.task_000200:Task000200RandomEnvCfg"},
        disable_env_checker=True,
    )
    for phase, cfg in (("Seed", "Task000200RightMimicSeedEnvCfg"), ("Generate", "Task000200RightMimicGenerateEnvCfg")):
        gym.register(
            id=f"Cyclo-Real-WhiteRoom-Task000200-Right-Mimic-{phase}-FFW-SG2-v0",
            entry_point=f"{package}.task_000200_mimic_env:Task000200MimicEnv",
            kwargs={"env_cfg_entry_point": f"{package}.task_000200_mimic_cfg:{cfg}"},
            disable_env_checker=True,
        )


def build_recording_factory(args):
    spec = importlib.util.spec_from_file_location("_cyclo_local_task_000200_recording", _ROOT / "recording.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load Task000200 recording controller.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_recording_factory(args)


def provenance() -> dict[str, str]:
    return {"id": TASK_ID, "environment": ENVIRONMENT}
