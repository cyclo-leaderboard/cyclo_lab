"""Episodic WhiteRoom seed task with the existing SG2 camera/action contract."""

from dataclasses import replace

from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm

from cyclo_lab.assets.environments.robotis_showroom import iter_robotis_showroom_object_cfgs
from cyclo_lab.assets.environments.white_room import (
    WHITE_ROOM_SG2_USD_PATH,
    WHITE_ROOM_USD_PATH,
    make_white_room_basket_cfg,
    make_white_room_block_cfg,
    make_white_room_environment_cfg,
    make_white_room_lighting_cfg,
    make_white_room_contact_operator_camera_cfg,
)
from cyclo_lab.assets.environments.white_room_layout import ROBOT_ROOT_XY
from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.platform.robot_cfg import (
    SG2_SHOWROOM_ROBOT_POS,
)
from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.randomization.cfg import (
    ShowroomRandomizationCfg,
)
from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.common import (
    EpisodicShowroomTaskEnvCfg,
    ShowroomTaskSpec,
)

from .env_cfg import (
    WHITE_ROOM_INITIAL_JOINT_POSITIONS,
    WHITE_ROOM_RANDOMIZATION,
    WhiteRoomResetEventsCfg,
)
from .events import randomize_navy_block_pose, reset_task000100_seed_objects


TASK_000100_SPEC = ShowroomTaskSpec(
    task_id="000100",
    env_name="Cyclo-Real-WhiteRoom-Task000100-FFW-SG2-v0",
    instruction="Pick up the navy cosmetic block from the table and place it in the white basket.",
    target_object="navy_block",
    target_side="left_or_right",
    environment_usd_path=WHITE_ROOM_USD_PATH,
)

# The soft-gripper distal meshes end at Z=-0.2365 m relative to arm_*_link7.
# Keep the former TCP's 5.5 mm clearance beyond the hard-gripper tip.
TASK000100_SOFT_GRIPPER_TCP_Z_M = -0.242


@configclass
class Task000100EnvCfg(EpisodicShowroomTaskEnvCfg):
    """19D B/N/R seed recording with a fixed robot and nine right-side points."""

    task_spec = TASK_000100_SPEC
    seed_side: str = "right"
    randomization: ShowroomRandomizationCfg = ShowroomRandomizationCfg(
        objects=replace(WHITE_ROOM_RANDOMIZATION.objects, enabled=False)
    )
    events: WhiteRoomResetEventsCfg = WhiteRoomResetEventsCfg()
    hold_reset_joint_targets: bool = True
    operator_camera_rows: tuple = (
        (
            ("cam_operator_contact_left", "Left View | Contact"),
            ("cam_operator_contact_right", "Right View | Contact"),
        ),
        (
            ("cam_wrist_left", "Wrist Left"),
            ("cam_head", "Head"),
            ("cam_wrist_right", "Wrist Right"),
        ),
    )

    def enable_operator_preview_cameras(self) -> None:
        """Attach the table-contact view for the operator dashboard."""
        self.scene.cam_operator_contact_left = make_white_room_contact_operator_camera_cfg("left")
        self.scene.cam_operator_contact_right = make_white_room_contact_operator_camera_cfg("right")

    def __post_init__(self):
        self.scene.white_basket = make_white_room_basket_cfg()
        self.scene.navy_block = make_white_room_block_cfg()
        super().__post_init__()

        for eef_frame in (self.scene.left_eef, self.scene.right_eef):
            eef_frame.target_frames[0].offset.pos = [
                0.0, 0.0, TASK000100_SOFT_GRIPPER_TCP_Z_M
            ]

        for object_name, _ in iter_robotis_showroom_object_cfgs():
            if hasattr(self.scene, object_name):
                delattr(self.scene, object_name)

        self.scene.ground = None
        self.scene.environment = make_white_room_environment_cfg()
        self.scene.ceiling_spots = make_white_room_lighting_cfg()
        self.scene.light.spawn.intensity = 0.5
        self.scene.robot.spawn.usd_path = WHITE_ROOM_SG2_USD_PATH
        self.scene.robot.init_state.pos = (
            ROBOT_ROOT_XY[0],
            ROBOT_ROOT_XY[1],
            SG2_SHOWROOM_ROBOT_POS[2],
        )
        self.scene.robot.init_state.joint_pos.update(WHITE_ROOM_INITIAL_JOINT_POSITIONS)
        self.events.set_robot_joint_pose.params = {
            **self.events.set_robot_joint_pose.params,
            "joint_positions": WHITE_ROOM_INITIAL_JOINT_POSITIONS.copy(),
        }
        self.events.randomize_navy_block_pose = EventTerm(
            func=reset_task000100_seed_objects,
            mode="reset",
            params={"side": self.seed_side},
        )


@configclass
class Task000100RandomEnvCfg(Task000100EnvCfg):
    """Augmentation reset with the same basket and 10-35 cm object zone."""

    randomization: ShowroomRandomizationCfg = ShowroomRandomizationCfg(
        objects=WHITE_ROOM_RANDOMIZATION.objects
    )

    def __post_init__(self):
        super().__post_init__()
        self.env_name = "Cyclo-Real-WhiteRoom-Task000100-Random-FFW-SG2-v0"
        self.events.randomize_navy_block_pose = EventTerm(
            func=randomize_navy_block_pose, mode="reset"
        )
