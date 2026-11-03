"""Geometry and data constants for showroom Task000668."""

from __future__ import annotations

from dataclasses import dataclass


TARGET_OBJECT = "peanut_mix_bag"
TASK_INSTRUCTION = (
    "Move the peanut mix from the upper shelf to the lower shelf using the "
    "gripper on its side."
)

ROBOT_HOME_POS_M = (-1.47138, 0.80011813, 0.0)
ROBOT_HOME_ROT_WXYZ = (0.0, 0.0, 0.0, 1.0)
HEAD_PITCH_RAD = 0.3
HEAD_YAW_RAD = 0.0
LIFT_POSITION_M = -0.10
LIFT_RANDOM_POSITION_RANGE_M = (-0.15, -0.05)

UPPER_SHELF_SUPPORT_Z_M = 1.305128932
LOWER_SHELF_SUPPORT_Z_M = 1.017119288
PEANUT_ORIGIN_ABOVE_SUPPORT_M = 0.057353368
PEANUT_ROT_WXYZ = (0.70710677, 0.70710677, 0.0, 0.0)

RIGHT_UPPER_PEANUT_NAMES = tuple(
    f"right_upper_peanut_mix_bag_{index:02d}" for index in range(6)
)
RIGHT_UPPER_CHESTNUT_NAMES = tuple(
    f"right_upper_roasted_chestnut_bag_{index:02d}" for index in range(6)
)
RIGHT_UPPER_INVENTORY_NAMES = RIGHT_UPPER_PEANUT_NAMES + RIGHT_UPPER_CHESTNUT_NAMES


@dataclass(frozen=True)
class SpawnBox:
    """Closed world-aligned range for the target bag root center."""

    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float

    @property
    def center_xy_m(self) -> tuple[float, float]:
        return (
            0.5 * (self.x_min_m + self.x_max_m),
            0.5 * (self.y_min_m + self.y_max_m),
        )


# Each box is exactly 10 cm deep (X) by 15 cm wide (Y). One side is chosen
# uniformly per reset; two target bags are never spawned simultaneously.
LEFT_SPAWN_BOX = SpawnBox(
    x_min_m=-2.24137,
    x_max_m=-2.14137,
    y_min_m=0.46542,
    y_max_m=0.61542,
)
RIGHT_SPAWN_BOX = SpawnBox(
    x_min_m=-2.24137,
    x_max_m=-2.14137,
    y_min_m=0.87894,
    y_max_m=1.02894,
)

SPAWN_SIDES = ("left", "right")
SPAWN_BOXES_BY_SIDE = {
    "left": LEFT_SPAWN_BOX,
    "right": RIGHT_SPAWN_BOX,
}
