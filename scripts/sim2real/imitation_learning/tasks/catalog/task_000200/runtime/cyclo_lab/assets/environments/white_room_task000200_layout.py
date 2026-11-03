"""Task000200 tabletop placement boxes in WhiteRoom world coordinates."""

from dataclasses import dataclass

from .white_room_layout import ROBOT_SIDE_TABLE_EDGE_X_M, TABLE_CENTER_XY, TABLE_SIZE_XY


PLATE_RADIUS_M = 0.095
PLATE_FLAT_RADIUS_M = 0.075
TAPE_RADIUS_M = 0.054
BOX_SIZE_M = 0.30
NEAR_DEPTH_M = 0.10
FAR_DEPTH_M = 0.40
RIGHT_EDGE_MARGIN_M = 0.15
BETWEEN_BOXES_M = 0.05


@dataclass(frozen=True)
class TabletopBox:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def center_bounds(self, radius: float) -> tuple[float, float, float, float]:
        """Return centers for which the whole circular footprint stays inside."""
        if radius <= 0 or 2 * radius > min(self.x_max - self.x_min, self.y_max - self.y_min):
            raise ValueError(f"Object radius {radius} does not fit in the tabletop box")
        return (
            self.x_min + radius,
            self.x_max - radius,
            self.y_min + radius,
            self.y_max - radius,
        )

    def nominal_center(self) -> tuple[float, float]:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)


_RIGHT_TABLE_EDGE_Y = TABLE_CENTER_XY[1] + TABLE_SIZE_XY[1] / 2
_TAPE_Y_MAX = _RIGHT_TABLE_EDGE_Y - RIGHT_EDGE_MARGIN_M
_TAPE_Y_MIN = _TAPE_Y_MAX - BOX_SIZE_M
_PLATE_Y_MAX = _TAPE_Y_MIN - BETWEEN_BOXES_M
_PLATE_Y_MIN = _PLATE_Y_MAX - BOX_SIZE_M
_X_MIN = ROBOT_SIDE_TABLE_EDGE_X_M - FAR_DEPTH_M
_X_MAX = ROBOT_SIDE_TABLE_EDGE_X_M - NEAR_DEPTH_M

PLATE_BOX = TabletopBox(_X_MIN, _X_MAX, _PLATE_Y_MIN, _PLATE_Y_MAX)
TAPE_BOX = TabletopBox(_X_MIN, _X_MAX, _TAPE_Y_MIN, _TAPE_Y_MAX)


@dataclass(frozen=True)
class TapeSeedPosition:
    label: str
    x_m: float
    y_m: float


def task000200_tape_seed_positions() -> tuple[TapeSeedPosition, ...]:
    """Clockwise corners of the footprint-safe tape-center rectangle."""
    far_x, near_x, left_y, right_y = TAPE_BOX.center_bounds(TAPE_RADIUS_M)
    return (
        TapeSeedPosition("T1", far_x, left_y),
        TapeSeedPosition("T2", far_x, right_y),
        TapeSeedPosition("T3", near_x, right_y),
        TapeSeedPosition("T4", near_x, left_y),
    )


def validate_layout() -> None:
    table_x_min = TABLE_CENTER_XY[0] - TABLE_SIZE_XY[0] / 2
    table_x_max = TABLE_CENTER_XY[0] + TABLE_SIZE_XY[0] / 2
    table_y_min = TABLE_CENTER_XY[1] - TABLE_SIZE_XY[1] / 2
    table_y_max = TABLE_CENTER_XY[1] + TABLE_SIZE_XY[1] / 2
    for box, radius in ((PLATE_BOX, PLATE_RADIUS_M), (TAPE_BOX, TAPE_RADIUS_M)):
        if not (table_x_min <= box.x_min < box.x_max <= table_x_max
                and table_y_min <= box.y_min < box.y_max <= table_y_max):
            raise ValueError("Task000200 spawn box exceeds the table")
        box.center_bounds(radius)
    if TAPE_BOX.y_min - PLATE_BOX.y_max < BETWEEN_BOXES_M - 1e-9:
        raise ValueError("Task000200 object boxes overlap")


validate_layout()
