"""Local Task000100 registration and recording entry point."""

from __future__ import annotations
import importlib
import importlib.util
from pathlib import Path

TASK_ID = "000100"
ENVIRONMENT = "white_room"
_ROOT = Path(__file__).resolve().parent

def add_cli_arguments(parser):
    parser.add_argument("--task000100_seed_ids", help="Comma-separated fixed seed IDs.")

def register_environments():
    from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.catalog_runtime import mount_task_package
    package = "cyclo_lab.manager_based.manipulation.white_room.config.ffw_sg2"
    mount_task_package(TASK_ID, package, "cyclo_lab/manager_based/manipulation/white_room/config/ffw_sg2")
    import gymnasium as gym
    gym.register(id="Cyclo-Real-WhiteRoom-Task000100-FFW-SG2-v0", entry_point="isaaclab.envs:ManagerBasedRLEnv", kwargs={"env_cfg_entry_point": f"{package}.task_000100:Task000100EnvCfg"}, disable_env_checker=True)
    gym.register(id="Cyclo-Real-WhiteRoom-Task000100-Random-FFW-SG2-v0", entry_point="isaaclab.envs:ManagerBasedRLEnv", kwargs={"env_cfg_entry_point": f"{package}.task_000100:Task000100RandomEnvCfg"}, disable_env_checker=True)
    for side in ("Left", "Right"):
        for phase in ("Seed", "Generate"):
            gym.register(id=f"Cyclo-Real-WhiteRoom-Task000100-{side}-Mimic-{phase}-FFW-SG2-v0", entry_point=f"{package}.mimic_env:Task000100MimicEnv", kwargs={"env_cfg_entry_point": f"{package}.mimic_cfg:Task000100{side}Mimic{phase}EnvCfg"}, disable_env_checker=True)

def build_recording_factory(args):
    spec = importlib.util.spec_from_file_location("_cyclo_local_task_000100_recording", _ROOT / "recording.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.build_recording_factory(args, TASK_ID)

def mimic_data_generator_class():
    package = "cyclo_lab.manager_based.manipulation.white_room.config.ffw_sg2"
    return importlib.import_module(f"{package}.mimic_datagen").Task000100DataGenerator

def provenance(): return {"id": TASK_ID, "environment": ENVIRONMENT}
