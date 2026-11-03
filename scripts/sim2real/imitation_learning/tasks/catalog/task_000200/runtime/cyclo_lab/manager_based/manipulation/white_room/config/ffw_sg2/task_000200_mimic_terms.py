"""Right-arm MimicGen phase signals for placing the tape on the plate."""

import torch

from cyclo_lab.assets.environments.white_room_task000200_layout import (
    PLATE_FLAT_RADIUS_M,
    TAPE_RADIUS_M,
)


def reset_metric_state(env, env_ids=None):
    if not hasattr(env, "_task000200_lifted"):
        env._task000200_lifted = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._task000200_placed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    ids = (torch.arange(env.num_envs, device=env.device)
           if env_ids is None else torch.as_tensor(env_ids, device=env.device).reshape(-1))
    env._task000200_lifted[ids] = False
    env._task000200_placed[ids] = False


def _state(env):
    if not hasattr(env, "_task000200_lifted"):
        reset_metric_state(env)
    tape = env.scene["blue_tape"]
    plate = env.scene["yellow_plate"]
    robot = env.scene["robot"]
    gripper_index = robot.joint_names.index("gripper_r_joint1")
    gripper_pos = robot.data.joint_pos[:, gripper_index]
    plate_pos = plate.data.root_pos_w
    tape_pos = tape.data.root_pos_w
    lifted = (tape_pos[:, 2] - plate_pos[:, 2] >= 0.020) & (gripper_pos >= 0.5)
    env._task000200_lifted |= lifted

    delta_xy = tape_pos[:, :2] - plate_pos[:, :2]
    centered = torch.linalg.vector_norm(delta_xy, dim=1) <= PLATE_FLAT_RADIUS_M - TAPE_RADIUS_M
    bottom_z = tape_pos[:, 2] - plate_pos[:, 2]
    supported_height = (bottom_z >= 0.001) & (bottom_z <= 0.010)
    released = gripper_pos <= 0.2
    placed = centered & supported_height & released
    env._task000200_placed |= env._task000200_lifted & placed
    return (
        env._task000200_lifted.clone(),
        env._task000200_placed.clone(),
        env._task000200_lifted & placed,
    )


def tape_lifted(env):
    return _state(env)[0]


def tape_placed_on_plate(env):
    return _state(env)[1]


def task_success(env):
    return _state(env)[2]
