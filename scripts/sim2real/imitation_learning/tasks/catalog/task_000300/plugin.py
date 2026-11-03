"""Local Task000300 registration and recording entry point."""

from __future__ import annotations
import importlib
import importlib.util
from pathlib import Path

TASK_ID = "000300"
ENVIRONMENT = "white_room"
_ROOT = Path(__file__).resolve().parent
_RUNTIME = _ROOT / "runtime"

def add_cli_arguments(parser):
    parser.add_argument("--task000300_seed_ids", help="Comma-separated fixed seed IDs.")

def _append(module_name, relative):
    module = importlib.import_module(module_name)
    path = str(_RUNTIME / relative)
    if path not in module.__path__: module.__path__.append(path)

def register_environments():
    _append("cyclo_lab.assets.environments", "cyclo_lab/assets/environments")
    package = "cyclo_lab.manager_based.manipulation.white_room.config.ffw_sg2"
    _append(package, "cyclo_lab/manager_based/manipulation/white_room/config/ffw_sg2")
    import gymnasium as gym
    gym.register(id="Cyclo-Real-WhiteRoom-Task000300-FFW-SG2-v0", entry_point="isaaclab.envs:ManagerBasedRLEnv", kwargs={"env_cfg_entry_point": f"{package}.task_000300:Task000300EnvCfg"}, disable_env_checker=True)
    gym.register(id="Cyclo-Real-WhiteRoom-Task000300-Random-FFW-SG2-v0", entry_point="isaaclab.envs:ManagerBasedRLEnv", kwargs={"env_cfg_entry_point": f"{package}.task_000300:Task000300RandomEnvCfg"}, disable_env_checker=True)
    for side in ("Left", "Right"):
        for phase in ("Seed", "Generate"):
            gym.register(id=f"Cyclo-Real-WhiteRoom-Task000300-{side}-Mimic-{phase}-FFW-SG2-v0", entry_point=f"{package}.mimic_env:Task000300MimicEnv", kwargs={"env_cfg_entry_point": f"{package}.task_000300_mimic_cfg:Task000300{side}Mimic{phase}EnvCfg"}, disable_env_checker=True)

def build_recording_factory(args):
    spec = importlib.util.spec_from_file_location("_cyclo_local_task_000300_recording", _ROOT / "recording.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.build_recording_factory(args, TASK_ID)

def mimic_data_generator_class():
    package = "cyclo_lab.manager_based.manipulation.white_room.config.ffw_sg2"
    return importlib.import_module(f"{package}.mimic_datagen").Task000300DataGenerator

def provenance(): return {"id": TASK_ID, "environment": ENVIRONMENT}
