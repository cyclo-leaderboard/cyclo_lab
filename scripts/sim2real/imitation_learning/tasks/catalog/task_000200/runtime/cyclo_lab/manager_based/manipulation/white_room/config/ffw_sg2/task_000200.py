"""WhiteRoom Task000200 with a yellow plate and a blue tape roll."""

from dataclasses import replace

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from cyclo_lab.assets.environments.robotis_showroom import iter_robotis_showroom_object_cfgs
from cyclo_lab.assets.environments.white_room import (
    WHITE_ROOM_SG2_USD_PATH,
    WHITE_ROOM_USD_PATH,
    make_white_room_contact_operator_camera_cfg,
    make_white_room_environment_cfg,
    make_white_room_lighting_cfg,
)
from cyclo_lab.assets.environments.white_room_layout import ROBOT_ROOT_XY
from cyclo_lab.assets.environments.white_room_task000200 import (
    make_blue_tape_cfg,
    make_yellow_plate_cfg,
)
from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.platform.robot_cfg import (
    SG2_SHOWROOM_ROBOT_POS,
)
from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.platform.env_cfg import (
    DeterministicResetEventsCfg,
)
from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.randomization.cfg import (
    ShowroomRandomizationCfg,
)
from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.common import (
    EpisodicShowroomTaskEnvCfg,
    ShowroomTaskSpec,
)

from .env_cfg import WHITE_ROOM_INITIAL_JOINT_POSITIONS, WHITE_ROOM_RANDOMIZATION

SOFT_GRIPPER_TCP_Z_M = -0.242
from .task_000200_events import randomize_task000200_objects, set_task000200_seed_objects


TASK_000200_SPEC = ShowroomTaskSpec(
    task_id="000200",
    env_name="Cyclo-Real-WhiteRoom-Task000200-FFW-SG2-v0",
    instruction="Pick up the blue tape with the right gripper and place it on the yellow plate.",
    target_object="blue_tape",
    target_side="right",
    environment_usd_path=WHITE_ROOM_USD_PATH,
)


@configclass
class Task000200EnvCfg(EpisodicShowroomTaskEnvCfg):
    """Same WhiteRoom robot, camera and table contract with task-specific objects."""

    task_spec = TASK_000200_SPEC
    randomization: ShowroomRandomizationCfg = ShowroomRandomizationCfg(
        objects=replace(WHITE_ROOM_RANDOMIZATION.objects, enabled=False)
    )
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
        self.scene.cam_operator_contact_left = make_white_room_contact_operator_camera_cfg("left")
        self.scene.cam_operator_contact_right = make_white_room_contact_operator_camera_cfg("right")

    def __post_init__(self):
        self.scene.yellow_plate = make_yellow_plate_cfg()
        self.scene.blue_tape = make_blue_tape_cfg()
        super().__post_init__()

        for eef_frame in (self.scene.left_eef, self.scene.right_eef):
            eef_frame.target_frames[0].offset.pos = [
                0.0, 0.0, SOFT_GRIPPER_TCP_Z_M
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
        self.events.set_task000200_seed_objects = EventTerm(
            func=set_task000200_seed_objects,
            mode="reset",
        )


@configclass
class Task000200RandomEnvCfg(Task000200EnvCfg):
    """Both objects move independently within footprint-safe placement boxes."""

    events: DeterministicResetEventsCfg = DeterministicResetEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.env_name = "Cyclo-Real-WhiteRoom-Task000200-Random-FFW-SG2-v0"
        self.events.set_task000200_seed_objects = None
        self.events.randomize_task000200_objects = EventTerm(
            func=randomize_task000200_objects,
            mode="reset",
        )
