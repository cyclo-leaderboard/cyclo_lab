"""Runnable fixed-platform peanut-mix transport presets for Task000668."""

from __future__ import annotations

from copy import deepcopy

import torch
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from cyclo_lab.assets.environments.robotis_showroom import iter_robotis_showroom_object_cfgs
from cyclo_lab.assets.object import PEANUT_MIX_BAG_CFG
from cyclo_lab.assets.robots.FFW_SG2 import spawn_sg2_with_softgripper_physics
from cyclo_lab.manager_based.actions.ffw_sg2 import make_ffw_sg2_joint_position_action_cfg
from cyclo_lab.robot_specs.ffw.sg2 import (
    FFW_SG2_ACTION_JOINT_NAMES,
    FFW_SG2_LEFT_ARM_JOINT_NAMES,
    FFW_SG2_LEFT_GRIPPER_JOINT_NAMES,
    FFW_SG2_RIGHT_ARM_JOINT_NAMES,
    FFW_SG2_RIGHT_GRIPPER_JOINT_NAMES,
)

from ...platform import observations as showroom_obs
from ..coffee_transport.task_000001.env_cfg import Task000001EnvCfg
from .contract import (
    HEAD_PITCH_RAD,
    HEAD_YAW_RAD,
    LIFT_POSITION_M,
    LIFT_RANDOM_POSITION_RANGE_M,
    LOWER_SHELF_SUPPORT_Z_M,
    PEANUT_ORIGIN_ABOVE_SUPPORT_M,
    PEANUT_ROT_WXYZ,
    RIGHT_SPAWN_BOX,
    RIGHT_UPPER_CHESTNUT_NAMES,
    RIGHT_UPPER_PEANUT_NAMES,
    ROBOT_HOME_POS_M,
    ROBOT_HOME_ROT_WXYZ,
    SPAWN_BOXES_BY_SIDE,
    SPAWN_SIDES,
    TARGET_OBJECT,
    UPPER_SHELF_SUPPORT_Z_M,
)
from .spec import TASK_000668_NAME, TASK_000668_SPEC
from .collision import cabinet_collision, configure_cabinet_contact_sensors, reset_cabinet_collision


ARM_AND_GRIPPER_JOINT_NAMES = FFW_SG2_ACTION_JOINT_NAMES[:16]


@configclass
class Task000668ActionsCfg:
    """Absolute 16D actions: both arms and both soft grippers only."""

    arm_l_action: JointPositionActionCfg = make_ffw_sg2_joint_position_action_cfg(
        FFW_SG2_LEFT_ARM_JOINT_NAMES
    )
    gripper_l_action: JointPositionActionCfg = make_ffw_sg2_joint_position_action_cfg(
        FFW_SG2_LEFT_GRIPPER_JOINT_NAMES
    )
    arm_r_action: JointPositionActionCfg = make_ffw_sg2_joint_position_action_cfg(
        FFW_SG2_RIGHT_ARM_JOINT_NAMES
    )
    gripper_r_action: JointPositionActionCfg = make_ffw_sg2_joint_position_action_cfg(
        FFW_SG2_RIGHT_GRIPPER_JOINT_NAMES
    )


@configclass
class Task000668ObservationsCfg:
    """16D joint state with head and both wrist RGB policy cameras."""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=showroom_obs.last_action)
        joint_pos = ObsTerm(
            func=showroom_obs.joint_pos_name,
            params={"joint_names": ARM_AND_GRIPPER_JOINT_NAMES, "asset_name": "robot"},
        )
        joint_pos_target = ObsTerm(
            func=showroom_obs.joint_pos_target_name,
            params={"joint_names": ARM_AND_GRIPPER_JOINT_NAMES, "asset_name": "robot"},
        )
        left_eef_pose = ObsTerm(
            func=showroom_obs.eef_pose,
            params={
                "eef_cfg": SceneEntityCfg("left_eef"),
                "robot_cfg": SceneEntityCfg("robot"),
            },
        )
        right_eef_pose = ObsTerm(
            func=showroom_obs.eef_pose,
            params={
                "eef_cfg": SceneEntityCfg("right_eef"),
                "robot_cfg": SceneEntityCfg("robot"),
            },
        )
        cam_head = ObsTerm(
            func=showroom_obs.image,
            params={
                "sensor_cfg": SceneEntityCfg("cam_head"),
                "data_type": "rgb",
                "normalize": False,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


def make_task000668_target_cfg():
    """Create the single dynamic target bag shared by both spawn sides."""

    cfg = deepcopy(PEANUT_MIX_BAG_CFG)
    cfg.prim_path = f"{{ENV_REGEX_NS}}/{TARGET_OBJECT}"
    center_x, center_y = RIGHT_SPAWN_BOX.center_xy_m
    cfg.init_state.pos = (
        center_x,
        center_y,
        UPPER_SHELF_SUPPORT_Z_M + PEANUT_ORIGIN_ABOVE_SUPPORT_M,
    )
    cfg.init_state.rot = PEANUT_ROT_WXYZ
    cfg.spawn.semantic_tags = [
        ("class", "peanut_mix_bag"),
        ("instance", TARGET_OBJECT),
        ("task_side", "random_left_or_right"),
    ]
    return cfg


def iter_task000458_right_upper_inventory_cfgs():
    """Clone Task000458's authored six peanut and six chestnut placements."""

    families = (
        ("peanut_mix_bag", RIGHT_UPPER_PEANUT_NAMES),
        ("roasted_chestnut_bag", RIGHT_UPPER_CHESTNUT_NAMES),
    )
    authored_by_family = {family: [] for family, _ in families}
    for authored_name, authored_cfg in iter_robotis_showroom_object_cfgs():
        for family, _ in families:
            if authored_name == family or authored_name.startswith(f"{family}_"):
                authored_by_family[family].append((authored_name, authored_cfg))
                break

    for family, destination_names in families:
        authored = sorted(authored_by_family[family], key=lambda item: item[0])
        if len(authored) != len(destination_names):
            raise RuntimeError(
                f"Expected {len(destination_names)} authored {family} bags, "
                f"found {len(authored)}"
            )
        for destination_name, (_, authored_cfg) in zip(destination_names, authored):
            cfg = deepcopy(authored_cfg)
            cfg.prim_path = f"{{ENV_REGEX_NS}}/{destination_name}"
            cfg.spawn.semantic_tags = [
                ("class", family),
                ("instance", destination_name),
                ("role", "right_upper_shelf_distractor"),
                ("provenance", "task_000458_authored_pose"),
            ]
            yield destination_name, cfg


def randomize_task000668_target(env, env_ids: torch.Tensor) -> None:
    """Choose one side uniformly and sample the one target inside its 15x10 cm box."""

    asset = env.scene[TARGET_OBJECT]
    count = len(env_ids)
    sample = torch.rand((count, 2), device=asset.device)
    side_indices = torch.randint(len(SPAWN_SIDES), (count,), device=asset.device)
    boxes = tuple(SPAWN_BOXES_BY_SIDE[side] for side in SPAWN_SIDES)
    bounds = torch.tensor(
        [(box.x_min_m, box.x_max_m, box.y_min_m, box.y_max_m) for box in boxes],
        device=asset.device,
        dtype=sample.dtype,
    )
    selected = bounds[side_indices]
    pose = asset.data.default_root_state[env_ids, :7].clone()
    pose[:, 0] = selected[:, 0] + sample[:, 0] * (selected[:, 1] - selected[:, 0])
    pose[:, 1] = selected[:, 2] + sample[:, 1] * (selected[:, 3] - selected[:, 2])
    pose[:, 2] = UPPER_SHELF_SUPPORT_Z_M + PEANUT_ORIGIN_ABOVE_SUPPORT_M
    pose[:, :3] += env.scene.env_origins[env_ids]
    asset.write_root_pose_to_sim(pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim(
        torch.zeros((count, 6), device=asset.device), env_ids=env_ids
    )

    side_cache = getattr(env, "_task000668_spawn_sides", {})
    for env_id, side_index in zip(env_ids.tolist(), side_indices.tolist()):
        side_cache[int(env_id)] = SPAWN_SIDES[int(side_index)]
    env._task000668_spawn_sides = side_cache


def randomize_task000668_lift_start(
    env,
    env_ids: torch.Tensor,
    *,
    position_range_m: tuple[float, float] = LIFT_RANDOM_POSITION_RANGE_M,
) -> None:
    """Sample the inactive lift once per reset, set its target, and hold it."""

    lower, upper = position_range_m
    if lower > upper:
        raise ValueError(f"lift range must be ordered, got {position_range_m}")

    robot = env.scene["robot"]
    lift_id = robot.joint_names.index("lift_joint")
    sampled = lower + torch.rand(len(env_ids), device=robot.device) * (upper - lower)
    joint_pos = robot.data.joint_pos[env_ids].clone()
    joint_pos[:, lift_id] = sampled
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    robot.write_joint_state_to_sim(
        joint_pos,
        torch.zeros_like(joint_pos),
        env_ids=env_ids,
    )

    sample_cache = getattr(env, "_task000668_lift_start_positions_m", {})
    for env_id, position_m in zip(env_ids.tolist(), sampled.tolist()):
        sample_cache[int(env_id)] = float(position_m)
    env._task000668_lift_start_positions_m = sample_cache


def task000668_success(env, position_tolerance_m: float = 0.03) -> torch.Tensor:
    """Return true after the target reaches the lower shelf on its sampled side."""

    position = env.scene[TARGET_OBJECT].data.root_pos_w - env.scene.env_origins
    result = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    side_cache = getattr(env, "_task000668_spawn_sides", {})
    target_z = LOWER_SHELF_SUPPORT_Z_M + PEANUT_ORIGIN_ABOVE_SUPPORT_M
    for env_id in range(env.num_envs):
        side = side_cache.get(env_id)
        if side not in SPAWN_BOXES_BY_SIDE:
            continue
        box = SPAWN_BOXES_BY_SIDE[side]
        inside = (
            (position[env_id, 0] >= box.x_min_m - position_tolerance_m)
            & (position[env_id, 0] <= box.x_max_m + position_tolerance_m)
            & (position[env_id, 1] >= box.y_min_m - position_tolerance_m)
            & (position[env_id, 1] <= box.y_max_m + position_tolerance_m)
            & (torch.abs(position[env_id, 2] - target_z) <= position_tolerance_m)
        )
        result[env_id] = inside
    return result & ~cabinet_collision(env)


@configclass
class Task000668EnvCfg(Task000001EnvCfg):
    """Fixed base/lift/head task with bilateral 16D teleoperation."""

    cabinet_contact_sensor_names: tuple[str, ...] = ()
    task_spec = TASK_000668_SPEC
    task_name: str = TASK_000668_NAME
    sim2real_active_trajectory_groups: tuple[str, ...] = ("left_arm", "right_arm")
    lift_start_randomization_enabled: bool = False
    lift_start_position_range_m: tuple[float, float] = (
        LIFT_POSITION_M,
        LIFT_POSITION_M,
    )
    actions: Task000668ActionsCfg = Task000668ActionsCfg()
    observations: Task000668ObservationsCfg = Task000668ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.env_name = TASK_000668_SPEC.env_name
        self.task_name = TASK_000668_NAME

        self.scene.robot.spawn.func = spawn_sg2_with_softgripper_physics
        self.scene.robot.articulation_root_prim_path = None
        self.scene.robot.init_state.pos = ROBOT_HOME_POS_M
        self.scene.robot.init_state.rot = ROBOT_HOME_ROT_WXYZ
        self.scene.robot.init_state.joint_pos["head_joint1"] = HEAD_PITCH_RAD
        self.scene.robot.init_state.joint_pos["head_joint2"] = HEAD_YAW_RAD
        self.scene.robot.init_state.joint_pos["lift_joint"] = LIFT_POSITION_M
        # R/N reset through this event rather than reading init_state again.
        # Give it a task-local copy so head and lift always return to the
        # requested fixed pose without mutating the shared showroom defaults.
        self.events.set_robot_joint_pose.params["joint_positions"] = dict(
            self.scene.robot.init_state.joint_pos
        )

        for coffee_name in ("coffee_can_right", "coffee_can_left"):
            if hasattr(self.scene, coffee_name):
                delattr(self.scene, coffee_name)
        setattr(self.scene, TARGET_OBJECT, make_task000668_target_cfg())
        for object_name, object_cfg in iter_task000458_right_upper_inventory_cfgs():
            setattr(self.scene, object_name, object_cfg)

        for event_name in ("randomize_coffee_positions", "randomize_lift_start"):
            if hasattr(self.events, event_name):
                setattr(self.events, event_name, None)
        self.events.randomize_task000668_target = EventTerm(
            func=randomize_task000668_target,
            mode="reset",
        )
        # Placement and timers do not end an episode. Robot/cabinet contact
        # is a failure termination; generation retains failed states instead
        # of auto-resetting them (see trajectory_generation_env.py).
        self.terminations.success = None
        self.terminations.time_out = None
        configure_cabinet_contact_sensors(self)
        self.events.reset_cabinet_collision = EventTerm(func=reset_cabinet_collision, mode="reset")
        self.terminations.cabinet_collision = DoneTerm(func=cabinet_collision)


@configclass
class Task000668RandomEnvCfg(Task000668EnvCfg):
    """Recorder preset with random target pose and random held lift height."""

    env_name: str = "Cyclo-Real-Showroom-Task000668-Random-FFW-SG2-v0"
    lift_start_randomization_enabled: bool = True
    lift_start_position_range_m: tuple[float, float] = (
        LIFT_RANDOM_POSITION_RANGE_M
    )

    def __post_init__(self):
        requested_env_name = self.env_name
        super().__post_init__()
        self.env_name = requested_env_name
        self.events.randomize_task000668_lift_start = EventTerm(
            func=randomize_task000668_lift_start,
            mode="reset",
            params={"position_range_m": self.lift_start_position_range_m},
        )
