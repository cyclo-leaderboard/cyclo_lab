"""Continuous SG2 white-room environment with a movable white basket."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from cyclo_lab.assets.environments.white_room import (
    WHITE_ROOM_SG2_USD_PATH,
    make_white_room_environment_cfg,
    make_white_room_basket_cfg,
    make_white_room_block_cfg,
    make_white_room_lighting_cfg,
)
from cyclo_lab.assets.environments.white_room_layout import (
    BASKET_RANDOM_XY_MAX_M,
    ROBOT_ROOT_XY,
    ROBOT_XY_RANDOM_MAX_M,
    ROBOT_YAW_RANDOM_MAX_RAD,
)
from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.platform.env_cfg import (
    DeterministicResetEventsCfg,
)
from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.randomization.cfg import (
    ShowroomRandomizationCfg,
    ObjectPoseRandomizationCfg,
    RobotRootRandomizationCfg,
)
from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.randomization.event_cfg import (
    configure_profiled_reset_events,
)
from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.platform.robot_cfg import (
    SG2_SHOWROOM_ROBOT_POS,
    apply_sg2_showroom_camera_profile,
    make_sg2_showroom_robot_cfg,
    enable_sg2_showroom_ui_session_camera,
)
from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.platform.scene_cfg import (
    ShowroomEnvCfg,
)
from cyclo_lab.robot_specs.ffw.sg2 import (
    DEFAULT_SG2_CAMERA_PROFILE,
    FFW_SG2_ACTION_JOINT_NAMES,
)

from .events import randomize_navy_block_pose

WHITE_ROOM_RANDOMIZATION = ShowroomRandomizationCfg(
    robot_root=RobotRootRandomizationCfg(
        enabled=True,
        depth_x_max_m=ROBOT_XY_RANDOM_MAX_M,
        lateral_y_max_m=ROBOT_XY_RANDOM_MAX_M,
        yaw_max_rad=ROBOT_YAW_RANDOM_MAX_RAD,
    ),
    objects=ObjectPoseRandomizationCfg(
        enabled=True,
        object_names=("white_basket",),
        x_max_m=BASKET_RANDOM_XY_MAX_M,
        y_max_m=BASKET_RANDOM_XY_MAX_M,
        yaw_max_rad=0.0,
    ),
)
WHITE_ROOM_INITIAL_JOINT_POSITIONS = {
    "arm_l_joint1": 1.0,
    "arm_l_joint2": 0.0,
    "arm_l_joint3": 0.0,
    "arm_l_joint4": -2.2,
    "arm_l_joint5": 0.0,
    "arm_l_joint6": 0.0,
    "arm_l_joint7": 0.0,
    "gripper_l_joint1": 0.0,
    "arm_r_joint1": 1.0,
    "arm_r_joint2": 0.0,
    "arm_r_joint3": 0.0,
    "arm_r_joint4": -2.2,
    "arm_r_joint5": 0.0,
    "arm_r_joint6": 0.0,
    "arm_r_joint7": 0.0,
    "gripper_r_joint1": 0.0,
    "head_joint1": 0.7854,
    "head_joint2": 0.0,
    "lift_joint": -0.20,
    "left_wheel_steer": 0.0,
    "right_wheel_steer": 0.0,
    "rear_wheel_steer": 0.0,
}
WHITE_ROOM_IDLE_ACTION = tuple(
    WHITE_ROOM_INITIAL_JOINT_POSITIONS[name] for name in FFW_SG2_ACTION_JOINT_NAMES
) + (0.0, 0.0, 0.0)


@configclass
class WhiteRoomResetEventsCfg(DeterministicResetEventsCfg):
    randomize_navy_block_pose = EventTerm(func=randomize_navy_block_pose, mode="reset")


@configclass
class ContinuousWhiteRoomEnvCfg(ShowroomEnvCfg):
    """SG2 with its physical camera profile in the white-wall room."""

    env_name: str = "Cyclo-Real-WhiteRoom-FFW-SG2-v0"
    hold_reset_joint_targets: bool = True
    robot_profile: str = DEFAULT_SG2_CAMERA_PROFILE
    robot_profile_id: str = ""
    robot_profile_sha256: str = ""
    robot_profile_source: str = ""
    operator_camera_rows: tuple = ((
        ("cam_wrist_left", "Wrist Left"),
        ("cam_head", "Head"),
        ("cam_wrist_right", "Wrist Right"),
    ),)

    def enable_ui_session_camera(self) -> None:
        enable_sg2_showroom_ui_session_camera(self)

    idle_action: tuple[float, ...] = WHITE_ROOM_IDLE_ACTION

    def __post_init__(self):
        super().__post_init__()
        self.events = configure_profiled_reset_events(
            WhiteRoomResetEventsCfg(), WHITE_ROOM_RANDOMIZATION
        )
        self.events.set_robot_joint_pose.params = {
            **self.events.set_robot_joint_pose.params,
            "joint_positions": WHITE_ROOM_INITIAL_JOINT_POSITIONS.copy(),
        }
        robot_cfg = make_sg2_showroom_robot_cfg()
        robot_cfg.spawn.usd_path = WHITE_ROOM_SG2_USD_PATH
        robot_cfg.init_state.pos = (
            ROBOT_ROOT_XY[0],
            ROBOT_ROOT_XY[1],
            SG2_SHOWROOM_ROBOT_POS[2],
        )
        robot_cfg.init_state.joint_pos.update(WHITE_ROOM_INITIAL_JOINT_POSITIONS)
        self.scene.robot = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]
        self.scene.ground = None
        self.scene.environment = make_white_room_environment_cfg()
        self.scene.ceiling_spots = make_white_room_lighting_cfg()
        self.scene.light.spawn.intensity = 0.5
        self.scene.white_basket = make_white_room_basket_cfg()
        self.scene.navy_block = make_white_room_block_cfg()
        apply_sg2_showroom_camera_profile(self, self.robot_profile)
