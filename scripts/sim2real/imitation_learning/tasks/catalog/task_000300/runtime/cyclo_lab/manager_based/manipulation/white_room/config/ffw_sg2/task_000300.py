"""Task000100 pick-and-place scene with the larger Task000300 cosmetic."""

import torch
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from cyclo_lab.assets.environments.white_room import WHITE_ROOM_USD_PATH
from cyclo_lab.assets.environments.white_room_layout import (
    BLOCK_ZONE_LATERAL_SIZE_M,
    BLOCK_ZONE_SIZE_M,
    BLOCK_ZONE_Y_OFFSET_M,
    SPAWN_CENTER_XY,
)
from cyclo_lab.assets.environments.white_room_task000300 import (
    COSMETIC_BODY_WIDTH_M,
    make_task000300_environment_cfg,
    make_task000300_navy_cosmetic_cfg,
    make_task000300_orange_tray_cfg,
)
from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.randomization.cfg import (
    ShowroomRandomizationCfg,
)
from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.common import (
    ShowroomTaskSpec,
)

from .env_cfg import WHITE_ROOM_RANDOMIZATION
from .task_000100 import Task000100EnvCfg


TASK_000300_SPEC = ShowroomTaskSpec(
    task_id="000300",
    env_name="Cyclo-Real-WhiteRoom-Task000300-FFW-SG2-v0",
    instruction="Pick up the navy cosmetic bottle from the table and place it in the orange tray.",
    target_object="navy_block",
    target_side="left_or_right",
    environment_usd_path=WHITE_ROOM_USD_PATH,
)

# Real Task000300 capture contract (green dot = bottle bottom-face center).
# Right-side points span table coordinates (20/40, 10/35) cm with the origin
# at the robot-near, robot-right corner.  The corresponding world rectangle is
# used directly so augmentation includes every measured center, including the
# 10 cm depth row that the older inset placement zone could not reach.
TASK000300_REAL_CENTER_X_BOUNDS_M = (-0.61012, -0.36012)
TASK000300_REAL_RIGHT_CENTER_Y_BOUNDS_M = (0.14, 0.34)
TASK000300_TABLE_CENTER_Y_M = -0.06


def randomize_task000300_cosmetic_pose(
    env, env_ids: torch.Tensor, side: str | None = None
) -> None:
    """Sample centers inside the same placement zone using the new footprint."""
    if side not in (None, "left", "right"):
        raise ValueError(f"Unsupported Task000300 spawn side: {side}")
    block = env.scene["navy_block"]
    root_pose = block.data.default_root_state[env_ids, :7].clone()
    root_pose[:, :3] += env.scene.env_origins[env_ids]
    samples = torch.rand((len(env_ids), 3), device=block.device)
    side_sign = (
        torch.where(samples[:, 0] < 0.5, -1.0, 1.0)
        if side is None
        else torch.full(
            (len(env_ids),),
            -1.0 if side == "left" else 1.0,
            dtype=samples.dtype,
            device=block.device,
        )
    )
    x_min, x_max = TASK000300_REAL_CENTER_X_BOUNDS_M
    right_y_min, right_y_max = TASK000300_REAL_RIGHT_CENTER_Y_BOUNDS_M
    right_y = right_y_min + samples[:, 2] * (right_y_max - right_y_min)
    left_y = 2.0 * TASK000300_TABLE_CENTER_Y_M - right_y
    root_pose[:, 0] = env.scene.env_origins[env_ids, 0] + x_min + samples[:, 1] * (x_max - x_min)
    root_pose[:, 1] = env.scene.env_origins[env_ids, 1] + torch.where(
        side_sign > 0.0, right_y, left_y
    )
    block.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    block.write_root_velocity_to_sim(
        torch.zeros((len(env_ids), 6), dtype=root_pose.dtype, device=block.device),
        env_ids=env_ids,
    )


@configclass
class Task000300EnvCfg(Task000100EnvCfg):
    """Fixed right-side seed scene with the Task000300 bottle and tray."""

    task_spec = TASK_000300_SPEC

    def __post_init__(self):
        super().__post_init__()
        # Keep robot links gravity-free in both waypoint recording and MimicGen.
        # World gravity and movable-object gravity remain enabled.
        self.scene.robot.spawn.rigid_props.disable_gravity = True
        self.scene.environment = make_task000300_environment_cfg()
        self.scene.navy_block = make_task000300_navy_cosmetic_cfg()
        self.scene.white_basket = make_task000300_orange_tray_cfg()


@configclass
class Task000300RandomEnvCfg(Task000300EnvCfg):
    """Random scene with footprint-safe positions for the longer cosmetic."""

    randomization: ShowroomRandomizationCfg = ShowroomRandomizationCfg(
        objects=WHITE_ROOM_RANDOMIZATION.objects
    )

    def __post_init__(self):
        super().__post_init__()
        self.env_name = "Cyclo-Real-WhiteRoom-Task000300-Random-FFW-SG2-v0"
        self.events.randomize_navy_block_pose = EventTerm(
            func=randomize_task000300_cosmetic_pose,
            mode="reset",
        )
