"""Cosmetic bottle and orange tray for WhiteRoom Task000300."""

from pathlib import Path

from .white_room_layout import TABLE_TOP_Z, task000100_seed_positions

COSMETIC_BODY_WIDTH_M = 0.053
COSMETIC_BODY_HEIGHT_M = 0.109
COSMETIC_LID_DIAMETER_M = 0.045
COSMETIC_LID_HEIGHT_M = 0.036
COSMETIC_TOTAL_LENGTH_M = COSMETIC_BODY_HEIGHT_M + COSMETIC_LID_HEIGHT_M
ORANGE_TRAY_SIZE_XY_M = (0.262, 0.195)
ORANGE_TRAY_HEIGHT_M = 0.020
ORANGE_TRAY_WALL_M = 0.006
ORANGE_TRAY_FLOOR_M = 0.006
ORANGE_TRAY_CORNER_RADIUS_M = 0.030
# Start clear of the table contact shell when resetting an upright rigid body
# to a new XY position. It settles onto the tabletop in the first physics step.
COSMETIC_SPAWN_CLEARANCE_M = 0.0065
TASK000300_TABLE_STATIC_FRICTION = 2.5
TASK000300_TABLE_DYNAMIC_FRICTION = 2.2
TASK000300_COSMETIC_MASS_KG = 0.20

NAVY_COSMETIC_USD_PATH = str(
    Path(__file__).resolve().parents[3] / "data/object/task_000300/navy_cosmetic.usda"
)
ORANGE_TRAY_USD_PATH = str(
    Path(__file__).resolve().parents[3] / "data/object/task_000300/orange_tray.usda"
)
FIRST_RIGHT_SEED_POSITION = task000100_seed_positions(side="right")[0]

_SPAWN_TASK000300_WHITE_ROOM = None


def _spawn_task000300_white_room_once(
    prim_path, cfg, translation=None, orientation=None, **kwargs
):
    """Spawn WhiteRoom and override only the Task000300 tabletop friction."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material

    from .white_room import _spawn_white_room_once

    prim = _spawn_white_room_once(
        prim_path, cfg, translation, orientation, **kwargs
    )
    material_path = f"{prim_path}/Task000300TablePhysicsMaterial"
    material_cfg = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="max",
        restitution_combine_mode="min",
        static_friction=TASK000300_TABLE_STATIC_FRICTION,
        dynamic_friction=TASK000300_TABLE_DYNAMIC_FRICTION,
        restitution=0.0,
    )
    material_cfg.func(material_path, material_cfg)
    bind_physics_material(f"{prim_path}/WhiteTable/Top", material_path)
    return prim


def spawn_task000300_white_room(
    prim_path, cfg, translation=None, orientation=None, **kwargs
):
    """Clone the Task000300 WhiteRoom with its tabletop-only material."""
    global _SPAWN_TASK000300_WHITE_ROOM
    if _SPAWN_TASK000300_WHITE_ROOM is None:
        from isaaclab.sim.utils import clone

        _SPAWN_TASK000300_WHITE_ROOM = clone(_spawn_task000300_white_room_once)
    return _SPAWN_TASK000300_WHITE_ROOM(
        prim_path, cfg, translation, orientation, **kwargs
    )


def make_task000300_environment_cfg():
    """Return WhiteRoom with Task000300-specific tabletop friction."""
    from .white_room import make_white_room_environment_cfg

    cfg = make_white_room_environment_cfg()
    cfg.spawn.func = spawn_task000300_white_room
    return cfg


def make_task000300_orange_tray_cfg():
    """Reuse the basket scene key and placement with Task000300 tray geometry."""
    from .white_room import make_white_room_basket_cfg

    cfg = make_white_room_basket_cfg()
    cfg.spawn.usd_path = ORANGE_TRAY_USD_PATH
    return cfg


def make_task000300_navy_cosmetic_cfg():
    """Keep the Task000100 scene key and stand the cosmetic upright."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg

    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/NavyBlock",
        spawn=sim_utils.UsdFileCfg(
            usd_path=NAVY_COSMETIC_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_linear_velocity=0.5,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.003,
                rest_offset=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=TASK000300_COSMETIC_MASS_KG
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(
                FIRST_RIGHT_SEED_POSITION.x_m,
                FIRST_RIGHT_SEED_POSITION.y_m,
                TABLE_TOP_Z + COSMETIC_TOTAL_LENGTH_M / 2.0 + COSMETIC_SPAWN_CLEARANCE_M,
            ),
            # Local +Z points up: body on the table, lid above it.
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
