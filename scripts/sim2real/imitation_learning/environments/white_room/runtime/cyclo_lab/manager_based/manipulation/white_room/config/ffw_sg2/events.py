"""White-room-only reset events."""

import torch

from cyclo_lab.assets.environments.white_room_layout import (
    BLOCK_ZONE_Y_OFFSET_M,
    SPAWN_CENTER_XY,
    block_center_half_ranges,
    right_mimic_center_y_bounds,
    task000100_seed_positions,
)


def randomize_navy_block_pose(
    env, env_ids: torch.Tensor, side: str | None = None, right_outer_trim_m: float = 0.0
) -> None:
    if side not in (None, "left", "right"):
        raise ValueError(f"Unsupported Task000100 spawn side: {side}")
    if right_outer_trim_m and side != "right":
        raise ValueError("Right-side outer trim requires side='right'")
    block = env.scene["navy_block"]
    root_pose = block.data.default_root_state[env_ids, :7].clone()
    root_pose[:, :3] += env.scene.env_origins[env_ids]

    samples = torch.rand((len(env_ids), 3), device=block.device)
    if side is None:
        side_sign = torch.where(samples[:, 0] < 0.5, -1.0, 1.0)
    else:
        side_sign = torch.full(
            (len(env_ids),),
            -1.0 if side == "left" else 1.0,
            dtype=samples.dtype,
            device=block.device,
        )
    x_half, y_half = block_center_half_ranges()
    root_pose[:, 0] = (
        env.scene.env_origins[env_ids, 0]
        + SPAWN_CENTER_XY[0]
        + (2.0 * samples[:, 1] - 1.0) * x_half
    )
    if right_outer_trim_m:
        y_min, y_max = right_mimic_center_y_bounds(right_outer_trim_m)
        root_pose[:, 1] = env.scene.env_origins[env_ids, 1] + y_min + samples[:, 2] * (y_max - y_min)
    else:
        root_pose[:, 1] = (
            env.scene.env_origins[env_ids, 1]
            + SPAWN_CENTER_XY[1]
            + side_sign * BLOCK_ZONE_Y_OFFSET_M
            + (2.0 * samples[:, 2] - 1.0) * y_half
        )

    block.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    block.write_root_velocity_to_sim(
        torch.zeros((len(env_ids), 6), dtype=root_pose.dtype, device=block.device),
        env_ids=env_ids,
    )


def set_task000100_seed_block_pose(
    env, env_ids: torch.Tensor, side: str = "right"
) -> None:
    """Spawn the current fixed grid point; recorder advances it only after N."""
    positions = task000100_seed_positions(side=side)
    index = int(getattr(env, "_task000100_seed_index", 0))
    if not 0 <= index < len(positions):
        raise ValueError(f"Task000100 seed index is out of range: {index}")
    position = positions[index]
    block = env.scene["navy_block"]
    root_pose = block.data.default_root_state[env_ids, :7].clone()
    root_pose[:, :3] += env.scene.env_origins[env_ids]
    root_pose[:, 0] = env.scene.env_origins[env_ids, 0] + position.x_m
    root_pose[:, 1] = env.scene.env_origins[env_ids, 1] + position.y_m
    block.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    block.write_root_velocity_to_sim(
        torch.zeros((len(env_ids), 6), dtype=root_pose.dtype, device=block.device),
        env_ids=env_ids,
    )


def reset_task000100_seed_objects(env, env_ids: torch.Tensor, side: str = "right") -> None:
    """Restore both physical objects to the seed pose with zero velocity."""
    basket = env.scene["white_basket"]
    basket_state = basket.data.default_root_state[env_ids].clone()
    basket_state[:, :3] += env.scene.env_origins[env_ids]
    basket.write_root_pose_to_sim(basket_state[:, :7], env_ids=env_ids)
    basket.write_root_velocity_to_sim(
        torch.zeros((len(env_ids), 6), dtype=basket_state.dtype, device=basket_state.device),
        env_ids=env_ids,
    )
    set_task000100_seed_block_pose(env, env_ids, side=side)
