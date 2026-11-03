"""Head-camera-only scene-state replay environment for Task000668."""

from __future__ import annotations

from isaaclab.utils import configclass

from .env_cfg import Task000668EnvCfg


TASK000668_VISUAL_REPLAY_ENV_NAME = (
    "Cyclo-Real-Showroom-Task000668-Visual-Replay-FFW-SG2-v0"
)


@configclass
class Task000668VisualReplayEnvCfg(Task000668EnvCfg):
    """Render head and both wrist cameras for LeRobot visual replay."""

    env_name: str = TASK000668_VISUAL_REPLAY_ENV_NAME

    def __post_init__(self):
        super().__post_init__()
        self.env_name = TASK000668_VISUAL_REPLAY_ENV_NAME
