"""Local Showroom Task000668 registration."""

from __future__ import annotations
import importlib
from pathlib import Path

TASK_ID = "000668"
ENVIRONMENT = "showroom"
_ROOT = Path(__file__).resolve().parent

def add_cli_arguments(_parser): pass

def register_environments():
    tasks = importlib.import_module("cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks")
    path = str(_ROOT / "runtime")
    if path not in tasks.__path__: tasks.__path__.append(path)
    import gymnasium as gym
    package = "cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.task_000668"
    gym.register(id="Cyclo-Real-Showroom-Task000668-FFW-SG2-v0", entry_point="isaaclab.envs:ManagerBasedRLEnv", kwargs={"env_cfg_entry_point": f"{package}.env_cfg:Task000668EnvCfg"}, disable_env_checker=True)
    gym.register(id="Cyclo-Real-Showroom-Task000668-Random-FFW-SG2-v0", entry_point="isaaclab.envs:ManagerBasedRLEnv", kwargs={"env_cfg_entry_point": f"{package}.env_cfg:Task000668RandomEnvCfg"}, disable_env_checker=True)
    gym.register(id="Cyclo-Real-Showroom-Task000668-Trajectory-Generation-FFW-SG2-v0", entry_point="isaaclab.envs:ManagerBasedRLEnv", kwargs={"env_cfg_entry_point": f"{package}.trajectory_generation_env:Task000668TrajectoryGenerationEnvCfg"}, disable_env_checker=True)
    gym.register(id="Cyclo-Real-Showroom-Task000668-Visual-Replay-FFW-SG2-v0", entry_point="isaaclab.envs:ManagerBasedRLEnv", kwargs={"env_cfg_entry_point": f"{package}.visual_replay_env:Task000668VisualReplayEnvCfg"}, disable_env_checker=True)

def build_recording_factory(_args): return None
def provenance(): return {"id": TASK_ID, "environment": ENVIRONMENT}
