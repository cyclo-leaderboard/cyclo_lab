"""Trajectory-generation environment for fixed-platform Task000668."""

from __future__ import annotations

from isaaclab.utils import configclass

from cyclo_lab.manager_based.manipulation.common.ffw_sg2_mimic_action_cfg import (
    configure_ffw_sg2_mimic_ik_actions,
)

from .env_cfg import Task000668EnvCfg


TASK000668_TRAJECTORY_ENV_NAME = (
    "Cyclo-Real-Showroom-Task000668-Trajectory-Generation-FFW-SG2-v0"
)
TASK000668_PHYSICAL_TRAJECTORY_GENERATION_PROFILE_ID = (
    "task000668_target_xy_same_side_dual_eef16_fixed_platform_v1"
)


@configclass
class Task000668TrajectoryGenerationEnvCfg(Task000668EnvCfg):
    """Random target XY with dual-EEF16 control and a fixed platform.

    The target reset event inherited from :class:`Task000668EnvCfg` samples one
    left or right 15 x 10 cm box. Head, lift, and mobile base remain at their
    fixed task pose and are deliberately absent from both the action and policy
    state contracts.
    """

    env_name: str = TASK000668_TRAJECTORY_ENV_NAME
    trajectory_generation_profile_id: str = (
        TASK000668_PHYSICAL_TRAJECTORY_GENERATION_PROFILE_ID
    )

    def __post_init__(self):
        super().__post_init__()
        self.env_name = TASK000668_TRAJECTORY_ENV_NAME
        # Physical trajectory generation owns no render products. Removing the
        # observation term alone is insufficient because a spawned CameraCfg
        # still requires --enable_cameras and allocates render resources.
        self.observations.policy.cam_head = None
        self.scene.cam_head = None
        self.scene.cam_wrist_left = None
        self.scene.cam_wrist_right = None
        self.terminations.success = None
        # Keep failed states for diagnostics; the generator latches contact
        # failure per environment without auto-resetting the whole batch.
        self.terminations.cabinet_collision = None
        configure_ffw_sg2_mimic_ik_actions(
            self.actions,
            posture_bias_sides=("l", "r"),
            include_head=False,
            include_lift=False,
        )
