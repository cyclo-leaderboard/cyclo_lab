"""Independent, footprint-safe Task000200 plate and tape resets."""

import torch

from cyclo_lab.assets.environments.white_room_task000200_layout import (
    PLATE_BOX,
    PLATE_RADIUS_M,
    TAPE_BOX,
    TAPE_RADIUS_M,
    task000200_tape_seed_positions,
)


def set_task000200_seed_objects(env, env_ids: torch.Tensor) -> None:
    """Keep the plate centered and select one of the tape's four safe corners."""
    seeds = task000200_tape_seed_positions()
    index = int(getattr(env, "_task000200_seed_index", 0))
    if not 0 <= index < len(seeds):
        raise ValueError(f"Invalid Task000200 tape seed index: {index}")
    positions = (
        ("yellow_plate", PLATE_BOX.nominal_center()),
        ("blue_tape", (seeds[index].x_m, seeds[index].y_m)),
    )
    for name, (x_m, y_m) in positions:
        obj = env.scene[name]
        pose = obj.data.default_root_state[env_ids, :7].clone()
        pose[:, :3] += env.scene.env_origins[env_ids]
        pose[:, 0] = env.scene.env_origins[env_ids, 0] + x_m
        pose[:, 1] = env.scene.env_origins[env_ids, 1] + y_m
        obj.write_root_pose_to_sim(pose, env_ids=env_ids)
        obj.write_root_velocity_to_sim(
            torch.zeros((len(env_ids), 6), device=obj.device, dtype=pose.dtype),
            env_ids=env_ids,
        )


def randomize_task000200_objects(env, env_ids: torch.Tensor) -> None:
    """Randomize each full circular footprint inside its own 30 x 30 cm box."""
    for name, box, radius in (
        ("yellow_plate", PLATE_BOX, PLATE_RADIUS_M),
        ("blue_tape", TAPE_BOX, TAPE_RADIUS_M),
    ):
        obj = env.scene[name]
        pose = obj.data.default_root_state[env_ids, :7].clone()
        pose[:, :3] += env.scene.env_origins[env_ids]
        x_min, x_max, y_min, y_max = box.center_bounds(radius)
        sample = torch.rand((len(env_ids), 2), device=obj.device, dtype=pose.dtype)
        pose[:, 0] = env.scene.env_origins[env_ids, 0] + x_min + (x_max - x_min) * sample[:, 0]
        pose[:, 1] = env.scene.env_origins[env_ids, 1] + y_min + (y_max - y_min) * sample[:, 1]
        obj.write_root_pose_to_sim(pose, env_ids=env_ids)
        obj.write_root_velocity_to_sim(
            torch.zeros((len(env_ids), 6), device=obj.device, dtype=pose.dtype),
            env_ids=env_ids,
        )
