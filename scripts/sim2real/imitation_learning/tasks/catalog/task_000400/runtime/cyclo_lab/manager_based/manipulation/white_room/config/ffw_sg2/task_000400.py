"""1063-matched WallRoom scene with the Task000300 bottle and tray."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from isaaclab.utils import configclass
from isaaclab.sensors import CameraCfg
from cyclo_lab.assets.robots.FFW_BG2 import FFW_BG2_CFG
from cyclo_lab.assets.sensors.ffw_sg2_cameras import make_ffw_sg2_head_camera_cfg, make_ffw_sg2_wrist_camera_cfg

from cyclo_lab.assets.environments.white_room_task000400 import (
    ROOM_USD_PATH,
    make_task000400_environment_cfg,
    make_task000400_lighting_cfg,
    make_task000400_navy_cosmetic_cfg,
    make_task000400_orange_tray_cfg,
)
from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.common import ShowroomTaskSpec
from cyclo_lab.robot_specs.ffw.sg2 import FFW_SG2_ACTION_JOINT_NAMES

from .task_000300 import Task000300EnvCfg, Task000300RandomEnvCfg


_TASK_ROOT = Path(__file__).resolve().parents[7]
_ROBOT_USD = _TASK_ROOT / "runtime/data/robots/bg2_1063/usd/bg2_1063.usd"
_CAMERA_INFO = _TASK_ROOT / "reference/robot1063_20261022/camera_info.json"


def _apply_bg2_robot(cfg, joint_positions):
    inherited = cfg.scene.robot.init_state
    robot = deepcopy(FFW_BG2_CFG)
    robot.prim_path = "{ENV_REGEX_NS}/Robot"
    robot.spawn.usd_path = str(_ROBOT_USD)
    robot.articulation_root_prim_path = "/root_joint"
    # The BG2 replacement must preserve Task000300's robot-only gravity policy.
    robot.spawn.rigid_props.disable_gravity = True
    robot.spawn.articulation_props.enabled_self_collisions = False
    robot.spawn.semantic_tags = [("class", "robot")]
    # Match the established table-scene lift hold controller.
    robot.actuators["lift"].stiffness = 250_000.0
    robot.actuators["lift"].damping = 5_000.0
    robot.actuators["lift"].effort_limit_sim = 5_000_000.0
    robot.init_state.pos = inherited.pos
    robot.init_state.rot = inherited.rot
    robot.init_state.joint_pos.update(joint_positions)
    robot.actuators["gripper_slave"].stiffness = 0.0
    robot.actuators["gripper_slave"].damping = 0.0
    cfg.scene.robot = robot

    for side in ("left", "right"):
        frame = getattr(cfg.scene, f"{side}_eef")
        frame.prim_path = "{ENV_REGEX_NS}/Robot/arm_base_link"
        frame.target_frames[0].prim_path = f"{{ENV_REGEX_NS}}/Robot/arm_{side[0]}_link7"
        # The deployed BG2 URDF defines its end_effector frame at -0.215 m.
        frame.target_frames[0].offset.pos = (0.0, 0.0, -0.215)

    cameras = json.loads(_CAMERA_INFO.read_text())
    head = cameras["head"]
    cfg.scene.cam_head = make_ffw_sg2_head_camera_cfg(
        width=head["width"], height=head["height"], intrinsic_matrix=head["k"],
        clipping_range=(0.05, 8.0)
    )
    cfg.scene.cam_head.prim_path = "{ENV_REGEX_NS}/Robot/head_link2/cam_head"
    # head_link2 -> zed_camera_link -> left optical frame, from live URDF.
    cfg.scene.cam_head.offset = CameraCfg.OffsetCfg(
        pos=(0.0238122, 0.02498203, -0.0109594),
        rot=(0.5, -0.5, 0.5, -0.5), convention="ros",
    )
    for side in ("left", "right"):
        calibration = cameras[f"wrist_{side}"]
        camera = make_ffw_sg2_wrist_camera_cfg(
            side, width=calibration["height"], height=calibration["width"],
            intrinsic_matrix=calibration["k"],
        )
        camera.prim_path = f"{{ENV_REGEX_NS}}/Robot/camera_{side}_link/cam_wrist_{side}"
        setattr(cfg.scene, f"cam_wrist_{side}", camera)
    cfg.robot_profile = cfg.robot_profile_id = "bg2_1063_20261022"
    cfg.robot_profile_source = str(_CAMERA_INFO)
    cfg.robot_profile_sha256 = hashlib.sha256(_CAMERA_INFO.read_bytes()).hexdigest()
    cfg.operator_camera_title = "BG2 Task000400"
    cfg.waypoint_operator_camera_parent = "{ENV_REGEX_NS}/Robot/base_link"
    cfg.sim2real_base_frame = "base_link"
    cfg.sim2real_base_body = "base_link"


TASK_000400_SPEC = ShowroomTaskSpec(
    task_id="000400",
    env_name="Cyclo-Real-WallRoom-Task000400-FFW-BG2-v0",
    instruction="Pick up the navy cosmetic bottle from the table and place it in the orange tray.",
    target_object="navy_block",
    target_side="left_or_right",
    environment_usd_path=ROOM_USD_PATH,
)


def _apply_task000400_scene(cfg, env_name: str) -> None:
    cfg.env_name = env_name
    cfg.scene.environment = make_task000400_environment_cfg()
    cfg.scene.ceiling_spots = make_task000400_lighting_cfg()
    cfg.scene.light = None
    cfg.scene.navy_block = make_task000400_navy_cosmetic_cfg()
    cfg.scene.white_basket = make_task000400_orange_tray_cfg()

    # Keep all other joints at the inherited WhiteRoom starting pose.
    joint_positions = {
        name: value for name, value in cfg.events.set_robot_joint_pose.params["joint_positions"].items()
        if name in FFW_SG2_ACTION_JOINT_NAMES
    }
    joint_positions["head_joint1"] = 0.83
    _apply_bg2_robot(cfg, joint_positions)
    cfg.events.set_robot_joint_pose.params["joint_positions"] = joint_positions
    if hasattr(cfg, "idle_action"):
        idle = list(cfg.idle_action)
        idle[FFW_SG2_ACTION_JOINT_NAMES.index("head_joint1")] = 0.83
        cfg.idle_action = tuple(idle)


@configclass
class Task000400EnvCfg(Task000300EnvCfg):
    task_spec = TASK_000400_SPEC

    def __post_init__(self):
        super().__post_init__()
        _apply_task000400_scene(self, TASK_000400_SPEC.env_name)


@configclass
class Task000400RandomEnvCfg(Task000300RandomEnvCfg):
    task_spec = TASK_000400_SPEC

    def __post_init__(self):
        super().__post_init__()
        _apply_task000400_scene(
            self, "Cyclo-Real-WallRoom-Task000400-Random-FFW-BG2-v0"
        )
