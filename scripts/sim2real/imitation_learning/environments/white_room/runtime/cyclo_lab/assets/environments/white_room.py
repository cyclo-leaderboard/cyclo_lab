"""White-wall room using the showroom shell and floor geometry."""

import math
from pathlib import Path

from .physics import spawn_environment_with_friction_once
from .white_room_layout import (
    BASKET_CENTER_XY,
    BLOCK_SIZE_LOCAL_XYZ,
    BLOCK_ZONE_Y_OFFSET_M,
    SPAWN_CENTER_XY,
    TABLE_CENTER_XY,
    TABLE_SIZE_XY,
    TABLE_TOP_Z,
)


WHITE_ROOM_USD_PATH = str(
    Path(__file__).resolve().parents[3] / "data/environments/white_room/white_room.usda"
)
WHITE_ROOM_SG2_USD_PATH = str(
    Path(__file__).resolve().parents[3] / "data/environments/white_room/white_room_sg2.usda"
)
WHITE_ROOM_LIGHTING_USD_PATH = str(
    Path(__file__).resolve().parents[3]
    / "data/environments/white_room/white_room_lighting.usda"
)

_SPAWN_WHITE_ROOM = None


def _spawn_white_room_once(prim_path, cfg, translation=None, orientation=None, **kwargs):
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import UsdPhysics

    prim = spawn_environment_with_friction_once(
        prim_path, cfg, translation, orientation, **kwargs
    )
    floor = get_current_stage().GetPrimAtPath(
        f"{prim_path}/robotis_showroom/ShowroomShell/Floor"
    )
    if not floor.IsValid():
        raise RuntimeError(f"Missing white-room floor under {prim_path}")
    if not floor.HasAPI(UsdPhysics.CollisionAPI):
        raise RuntimeError(f"White-room floor has no collision API under {prim_path}")
    UsdPhysics.CollisionAPI(floor).CreateCollisionEnabledAttr(True).Set(True)
    return prim


def spawn_white_room(prim_path, cfg, translation=None, orientation=None, **kwargs):
    global _SPAWN_WHITE_ROOM
    if _SPAWN_WHITE_ROOM is None:
        from isaaclab.sim.utils import clone

        _SPAWN_WHITE_ROOM = clone(_spawn_white_room_once)
    return _SPAWN_WHITE_ROOM(prim_path, cfg, translation, orientation, **kwargs)


def make_white_room_environment_cfg():
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WhiteRoom",
        spawn=sim_utils.UsdFileCfg(
            func=spawn_white_room,
            usd_path=WHITE_ROOM_USD_PATH,
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.003,
                rest_offset=0.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )


def make_white_room_lighting_cfg():
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WhiteRoomRailLights",
        spawn=sim_utils.UsdFileCfg(usd_path=WHITE_ROOM_LIGHTING_USD_PATH),
    )


def _world_look_at_quaternion(
    position: tuple[float, float, float],
    target: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    """Return a world-camera quaternion whose local +X points at target."""
    dx, dy, dz = (target[index] - position[index] for index in range(3))
    yaw = math.atan2(dy, dx)
    pitch = math.atan2(-dz, math.hypot(dx, dy))
    half_yaw = 0.5 * yaw
    half_pitch = 0.5 * pitch
    return (
        math.cos(half_yaw) * math.cos(half_pitch),
        -math.sin(half_yaw) * math.sin(half_pitch),
        math.cos(half_yaw) * math.sin(half_pitch),
        math.sin(half_yaw) * math.cos(half_pitch),
    )


def make_white_room_contact_operator_camera_cfg(side: str):
    """Look straight across the tabletop from its left or right edge."""
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    if side not in ("left", "right"):
        raise ValueError(f"Unsupported contact camera side: {side}")

    # Place each optical center 8 cm outside a table side and 1 cm above the
    # tabletop. Keep camera and target at the table's mid-depth so the view
    # crosses the table laterally instead of looking diagonally from the robot.
    lateral_sign = -1.0 if side == "left" else 1.0
    table_side_y = TABLE_CENTER_XY[1] + lateral_sign * TABLE_SIZE_XY[1] / 2.0
    camera_pos = (
        TABLE_CENTER_XY[0],
        table_side_y + lateral_sign * 0.08,
        TABLE_TOP_Z + 0.01,
    )
    table_target = (
        TABLE_CENTER_XY[0],
        TABLE_CENTER_XY[1],
        TABLE_TOP_Z + 0.08,
    )
    return CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/cam_operator_contact_{side}",
        update_period=0.0,
        height=540,
        width=960,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=1.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 5.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=camera_pos,
            rot=_world_look_at_quaternion(camera_pos, table_target),
            convention="world",
        ),
    )


WHITE_ROOM_BASKET_USD_PATH = str(
    Path(__file__).resolve().parents[3] / "data/environments/white_room/white_basket.usda"
)


def make_white_room_basket_cfg():
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg

    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WhiteBasket",
        spawn=sim_utils.UsdFileCfg(
            usd_path=WHITE_ROOM_BASKET_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=1.0,
                angular_damping=1.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.003,
                rest_offset=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.13),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(BASKET_CENTER_XY[0], BASKET_CENTER_XY[1], 0.686),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


def make_white_room_block_cfg():
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg

    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/NavyBlock",
        spawn=sim_utils.CuboidCfg(
            size=BLOCK_SIZE_LOCAL_XYZ,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_linear_velocity=0.5,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.003,
                rest_offset=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.020),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.025, 0.045, 0.085),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(
                SPAWN_CENTER_XY[0],
                SPAWN_CENTER_XY[1] - BLOCK_ZONE_Y_OFFSET_M,
                TABLE_TOP_Z + BLOCK_SIZE_LOCAL_XYZ[1] / 2.0 + 0.0005,
            ),
            # Local z (67 mm) points along world x; local x (36 mm) along world y.
            rot=(0.5, 0.5, 0.5, 0.5),
        ),
    )
