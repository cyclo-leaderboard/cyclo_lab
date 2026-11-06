"""WallRoom Task000400: the 1063 BG2 table and solid-wall scene."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
from pathlib import Path


TASK_ID = "000400"
ENVIRONMENT = "wall_room"
_ROOT = Path(__file__).resolve().parent
_TASK_300 = _ROOT.parent / "task_000300"


def add_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task000400_seed_ids", help="Comma-separated fixed seed IDs.")


def register_environments() -> None:
    from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.catalog_runtime import mount_task_package, register_scene_dependency
    # The Task000300 base configuration still imports shared WhiteRoom assets.
    register_scene_dependency("white_room")
    assets = "cyclo_lab.assets.environments"
    package = "cyclo_lab.manager_based.manipulation.white_room.config.ffw_sg2"
    for task_id in ("000300", TASK_ID):
        mount_task_package(task_id, assets, "cyclo_lab/assets/environments")
        mount_task_package(task_id, package, "cyclo_lab/manager_based/manipulation/white_room/config/ffw_sg2")

    import gymnasium as gym

    # The waypoint seed UI discovers literal registration IDs without Isaac Sim.
    gym.register(
        id="Cyclo-Real-WallRoom-Task000400-FFW-BG2-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": f"{package}.task_000400:Task000400EnvCfg"},
        disable_env_checker=True,
    )
    gym.register(
        id="Cyclo-Real-WallRoom-Task000400-Random-FFW-BG2-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": f"{package}.task_000400:Task000400RandomEnvCfg"},
        disable_env_checker=True,
    )
    for old_name, cfg_name in (
        ("Cyclo-Real-WhiteRoom-Task000400-FFW-BG2-v0", "Task000400EnvCfg"),
        ("Cyclo-Real-WhiteRoom-Task000400-Random-FFW-BG2-v0", "Task000400RandomEnvCfg"),
    ):
        gym.register(
            id=old_name,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={"env_cfg_entry_point": f"{package}.task_000400:{cfg_name}"},
            disable_env_checker=True,
        )


def build_recording_factory(args):
    path = _TASK_300 / "recording.py"
    spec = importlib.util.spec_from_file_location("_cyclo_local_task_000400_recording", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_recording_factory(args, TASK_ID)


def provenance() -> dict[str, str]:
    return {
        "id": TASK_ID,
        "environment": ENVIRONMENT,
        "source_task": "000300",
        "robot_model": "FFW-BG2",
        "robot_description_source": "1063 live robot_description 2026-10-22",
        "table_top_z_m": "0.735",
        "head_joint1_initial_rad": "0.83",
        "wall_reference": "1063 head camera 2026-10-22",
    }


def waypoint_robot_spec(environment: str) -> dict:
    if environment not in (
        "Cyclo-Real-WallRoom-Task000400-FFW-BG2-v0",
        "Cyclo-Real-WallRoom-Task000400-Random-FFW-BG2-v0",
    ):
        raise ValueError("Task000400 requires its BG2 environment; create a new BG2 plan.")
    return {
        "urdf_path": str(_ROOT / "reference/robot1063_20261022/live1063.urdf"),
        "robot_contract_id": "ffw_bg2_1063_fixed_base_19d_v1",
        "robot_model": "FFW-BG2",
    }
