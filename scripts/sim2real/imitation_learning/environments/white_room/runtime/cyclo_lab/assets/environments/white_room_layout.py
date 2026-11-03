"""World-space layout and reset bounds for the SG2 white room."""

import math
from dataclasses import dataclass


WALL_HEIGHT_M = 1.20
WALL_CLEARANCE_M = 0.03
PARTITION_WALL_THICKNESS_M = 0.08
PARTITION_PANEL_WIDTH_M = 0.40
PARTITION_PANEL_GAP_M = 0.003
PARTITION_PANEL_COUNT = 10
PARTITION_WALL_SIZE_Y_M = (
    PARTITION_PANEL_COUNT * PARTITION_PANEL_WIDTH_M
    + (PARTITION_PANEL_COUNT - 1) * PARTITION_PANEL_GAP_M
)
PARTITION_PANEL_CENTERS_Y_M = (
    -1.8735, -1.4705, -1.0675, -0.6645, -0.2615,
    0.1415, 0.5445, 0.9475, 1.3505, 1.7535,
)

TABLE_CENTER_XY = (-0.55862, -0.06)
TABLE_SIZE_XY = (0.597, 1.20)
TABLE_TOP_Z = 0.685
PARTITION_WALL_INNER_FACE_X_M = (
    TABLE_CENTER_XY[0] - TABLE_SIZE_XY[0] / 2.0 - WALL_CLEARANCE_M
)
PARTITION_WALL_CENTER_X_M = (
    PARTITION_WALL_INNER_FACE_X_M - PARTITION_WALL_THICKNESS_M / 2.0
)
ROBOT_SIDE_TABLE_EDGE_X_M = TABLE_CENTER_XY[0] + TABLE_SIZE_XY[0] / 2
SPAWN_DEPTH_MIN_M = 0.10
SPAWN_DEPTH_MAX_M = 0.35
SPAWN_CENTER_DEPTH_M = (SPAWN_DEPTH_MIN_M + SPAWN_DEPTH_MAX_M) / 2
SPAWN_CENTER_XY = (ROBOT_SIDE_TABLE_EDGE_X_M - SPAWN_CENTER_DEPTH_M, TABLE_CENTER_XY[1])
BASKET_CENTER_DEPTH_M = 0.25
BASKET_CENTER_XY = (ROBOT_SIDE_TABLE_EDGE_X_M - BASKET_CENTER_DEPTH_M, TABLE_CENTER_XY[1])

BASKET_SIZE_XY = (0.14, 0.21)
BASKET_RANDOM_XY_MAX_M = 0.02

ROBOT_ROOT_XY = (0.0, -0.05909)
ROBOT_XY_RANDOM_MAX_M = 0.01
ROBOT_YAW_RANDOM_MAX_RAD = math.radians(1.0)

BLOCK_SIZE_LOCAL_XYZ = (0.036, 0.028, 0.067)
# Lay the 67 x 36 mm face on the table, with the 28 mm axis vertical.
BLOCK_FOOTPRINT_XY = (BLOCK_SIZE_LOCAL_XYZ[2], BLOCK_SIZE_LOCAL_XYZ[0])
BLOCK_ZONE_SIZE_M = SPAWN_DEPTH_MAX_M - SPAWN_DEPTH_MIN_M
BLOCK_ZONE_LATERAL_SIZE_M = 0.30
BLOCK_ZONE_Y_OFFSET_M = 0.35
RIGHT_MIMIC_OUTER_TRIM_M = 0.0
TASK000100_SEED_DEPTH_OFFSETS_M = (-0.075, 0.0, 0.075)
TASK000100_SEED_LATERAL_OFFSETS_M = (-0.10, 0.0, 0.10)


@dataclass(frozen=True)
class Task000100SeedPosition:
    label: str
    side: str
    x_m: float
    y_m: float


def task000100_seed_positions(side: str | None = None) -> tuple[Task000100SeedPosition, ...]:
    """Nine row-major fixed block centers per selected side, viewed from the robot."""
    if side not in (None, "left", "right"):
        raise ValueError(f"Unsupported Task000100 seed side: {side}")
    return tuple(
        Task000100SeedPosition(
            label=f"{prefix}{row * 3 + col + 1}",
            side=grid_side,
            x_m=round(SPAWN_CENTER_XY[0] + x_offset, 4),
            y_m=round(SPAWN_CENTER_XY[1] + side_sign * BLOCK_ZONE_Y_OFFSET_M + y_offset, 4),
        )
        for grid_side, prefix, side_sign in (("left", "L", -1), ("right", "R", 1))
        if side is None or grid_side == side
        for row, x_offset in enumerate(TASK000100_SEED_DEPTH_OFFSETS_M)
        for col, y_offset in enumerate(TASK000100_SEED_LATERAL_OFFSETS_M)
    )


def block_center_half_ranges() -> tuple[float, float]:
    """Keep the full laid-down block inside the 25x30 cm placement zone."""
    return (
        (BLOCK_ZONE_SIZE_M - BLOCK_FOOTPRINT_XY[0]) / 2.0,
        (BLOCK_ZONE_LATERAL_SIZE_M - BLOCK_FOOTPRINT_XY[1]) / 2.0,
    )


def right_mimic_center_y_bounds(outer_trim_m: float = RIGHT_MIMIC_OUTER_TRIM_M) -> tuple[float, float]:
    """Right-side block center bounds with an optional outer-edge trim."""
    if not 0.0 <= outer_trim_m < BLOCK_ZONE_LATERAL_SIZE_M - BLOCK_FOOTPRINT_XY[1]:
        raise ValueError(f"Invalid right-side outer trim: {outer_trim_m}")
    center_y = SPAWN_CENTER_XY[1] + BLOCK_ZONE_Y_OFFSET_M
    _, half_range = block_center_half_ranges()
    return center_y - half_range, center_y + half_range - outer_trim_m
