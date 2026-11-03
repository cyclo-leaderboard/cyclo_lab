"""Task000300 objects on the 1063 Task000400 table and room."""

from .white_room_task000300 import (
    COSMETIC_SPAWN_CLEARANCE_M,
    COSMETIC_TOTAL_LENGTH_M,
    make_task000300_navy_cosmetic_cfg,
    make_task000300_orange_tray_cfg,
)
from .wall_room import (
    WALL_ROOM_USD_PATH,
    make_wall_room_environment_cfg,
    make_wall_room_lighting_cfg,
)


TABLE_TOP_Z_M = 0.735
TABLE_HEIGHT_DELTA_M = 0.05
ROOM_USD_PATH = WALL_ROOM_USD_PATH


def make_task000400_environment_cfg():
    return make_wall_room_environment_cfg()


def make_task000400_lighting_cfg():
    return make_wall_room_lighting_cfg()


def make_task000400_navy_cosmetic_cfg():
    cfg = make_task000300_navy_cosmetic_cfg()
    x, y, _ = cfg.init_state.pos
    cfg.init_state.pos = (
        x,
        y,
        TABLE_TOP_Z_M + COSMETIC_TOTAL_LENGTH_M / 2.0 + COSMETIC_SPAWN_CLEARANCE_M,
    )
    return cfg


def make_task000400_orange_tray_cfg():
    cfg = make_task000300_orange_tray_cfg()
    x, y, z = cfg.init_state.pos
    cfg.init_state.pos = (x, y, z + TABLE_HEIGHT_DELTA_M)
    return cfg
