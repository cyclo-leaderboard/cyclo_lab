"""MimicGen subtask and success signals for WhiteRoom Task000100."""

from __future__ import annotations

import torch

from cyclo_lab.assets.environments.white_room_layout import (
    BASKET_SIZE_XY,
    BLOCK_FOOTPRINT_XY,
    TABLE_TOP_Z,
)
from .contact_failure import invalid_contact


def _ids(env, env_ids) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    return torch.as_tensor(env_ids, dtype=torch.long, device=env.device).reshape(-1)


def _ensure_buffers(env) -> None:
    if hasattr(env, "_task000100_grasp_latched"):
        return
    env._task000100_grasp_latched = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    env._task000100_release_latched = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    env._task000100_mimic_metrics = None


def reset_metric_state(env, env_ids=None) -> None:
    """Clear per-episode latches after reset or reset_to."""
    _ensure_buffers(env)
    ids = _ids(env, env_ids)
    env._task000100_grasp_latched[ids] = False
    env._task000100_release_latched[ids] = False
    env._task000100_mimic_metrics = None


def _metrics(env, side: str) -> dict[str, torch.Tensor]:
    if side not in ("left", "right"):
        raise ValueError(f"Unsupported Task000100 Mimic side: {side}")
    _ensure_buffers(env)
    step = int(env.common_step_counter)
    cached = env._task000100_mimic_metrics
    if cached is not None and cached["step"] == step and cached["side"] == side:
        return cached

    block = env.scene["navy_block"]
    basket = env.scene["white_basket"]
    robot = env.scene["robot"]
    gripper_index = robot.joint_names.index(f"gripper_{side[0]}_joint1")
    gripper_closed = robot.data.joint_pos[:, gripper_index] >= 0.5
    gripper_open = robot.data.joint_pos[:, gripper_index] <= 0.2

    # The seed starts with the cosmetic block lying on the table. A modest
    # center-height increase is a stable, geometry-independent grasp boundary.
    lifted = block.data.root_pos_w[:, 2] >= TABLE_TOP_Z + 0.035
    env._task000100_grasp_latched |= gripper_closed & lifted

    delta = block.data.root_pos_w - basket.data.root_pos_w
    half_x = 0.5 * (BASKET_SIZE_XY[0] - BLOCK_FOOTPRINT_XY[0])
    half_y = 0.5 * (BASKET_SIZE_XY[1] - BLOCK_FOOTPRINT_XY[1])
    inside_xy = (delta[:, 0].abs() <= half_x) & (delta[:, 1].abs() <= half_y)
    # Existing 2 mm-floor seeds rest about 16 mm above the basket root;
    # the strengthened 5 mm floor rests the block about 19 mm above it.
    # Reject a block below either floor while retaining those source seeds.
    # Basket walls end 50 mm above their root. A block center above that rim
    # is still in the air, even if its XY center projects over the basket.
    inside_z = (delta[:, 2] >= 0.012) & (delta[:, 2] <= 0.05)
    inside_basket = inside_xy & inside_z
    released_in_basket = (
        env._task000100_grasp_latched & gripper_open & inside_basket
    )
    env._task000100_release_latched |= released_in_basket

    metrics = {
        "step": step,
        "side": side,
        "block_lifted": env._task000100_grasp_latched.clone(),
        "released_in_basket": env._task000100_release_latched.clone(),
        # The task outcome is final physical containment. Grasp/release latches
        # divide source subtasks, but must not reject a placed block. Robot
        # contact with the table or gripper contact with the basket fails.
        "success": inside_basket & ~invalid_contact(env),
    }
    env._task000100_mimic_metrics = metrics
    return metrics


def block_lifted(env, side: str) -> torch.Tensor:
    """Latched end of the approach/grasp segment."""
    return _metrics(env, side)["block_lifted"]


def released_in_basket(env, side: str) -> torch.Tensor:
    """Latched end of the carry/release segment."""
    return _metrics(env, side)["released_in_basket"]


def task_success(env, side: str) -> torch.Tensor:
    """Require current block containment and no forbidden robot contact."""
    return _metrics(env, side)["success"]
