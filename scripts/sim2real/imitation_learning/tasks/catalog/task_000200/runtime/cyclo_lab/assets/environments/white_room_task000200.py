"""Physical objects for the separate WhiteRoom Task000200 scene."""

from pathlib import Path

from .white_room_layout import TABLE_TOP_Z
from .white_room_task000200_layout import PLATE_BOX, TAPE_BOX


_OBJECT_DIR = Path(__file__).resolve().parents[3] / "data/object/task_000200"
YELLOW_PLATE_USD_PATH = str(_OBJECT_DIR / "yellow_plate.usda")
BLUE_TAPE_USD_PATH = str(_OBJECT_DIR / "blue_tape.usda")

_BLUE_TAPE_EXPECTED_COLLIDER_COUNT = 96
_BLUE_TAPE_STATIC_FRICTION = 2.25
_BLUE_TAPE_DYNAMIC_FRICTION = 2.0
_SPAWN_BLUE_TAPE_WITH_FRICTION = None

# Spawn defaults; the seed reset selects one of four safe tape corners.
# Random generation samples both footprint-safe center bounds.
PLATE_CENTER_XY = PLATE_BOX.nominal_center()
TAPE_CENTER_XY = TAPE_BOX.nominal_center()


def _make_object_cfg(prim_name: str, usd_path: str, xy: tuple[float, float], mass: float):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg

    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.5,
                angular_damping=0.5,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(xy[0], xy[1], TABLE_TOP_Z + 0.0005),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


def make_yellow_plate_cfg():
    return _make_object_cfg("YellowPlate", YELLOW_PLATE_USD_PATH, PLATE_CENTER_XY, 0.058)


def _spawn_blue_tape_with_friction_once(
    prim_path,
    cfg,
    translation=None,
    orientation=None,
    **kwargs,
):
    """Spawn the tape and bind high-friction material to every convex sector."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.from_files import from_files
    from isaaclab.sim.utils import bind_physics_material
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import Usd, UsdPhysics

    prim = from_files.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    stage = get_current_stage()
    collision_paths = [
        str(child.GetPath())
        for child in Usd.PrimRange(stage.GetPrimAtPath(prim_path))
        if child.HasAPI(UsdPhysics.CollisionAPI)
        and UsdPhysics.CollisionAPI(child).GetCollisionEnabledAttr().Get() is not False
    ]
    if len(collision_paths) != _BLUE_TAPE_EXPECTED_COLLIDER_COUNT:
        raise RuntimeError(
            "Task000200 blue tape must have "
            f"{_BLUE_TAPE_EXPECTED_COLLIDER_COUNT} enabled colliders, got {len(collision_paths)}."
        )

    material_path = f"{prim_path}/tapePhysicsMaterial"
    physics_material = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="max",
        restitution_combine_mode="min",
        static_friction=_BLUE_TAPE_STATIC_FRICTION,
        dynamic_friction=_BLUE_TAPE_DYNAMIC_FRICTION,
        restitution=0.0,
    )
    physics_material.func(material_path, physics_material)
    for collision_path in collision_paths:
        bind_physics_material(collision_path, material_path)
    print(
        "[Task000200 tape physics] bound high-friction material to "
        f"{len(collision_paths)} convex colliders."
    )
    return prim


def spawn_blue_tape_with_friction(
    prim_path,
    cfg,
    translation=None,
    orientation=None,
    **kwargs,
):
    """Clone and spawn the tape with validated high-friction colliders."""
    global _SPAWN_BLUE_TAPE_WITH_FRICTION
    if _SPAWN_BLUE_TAPE_WITH_FRICTION is None:
        from isaaclab.sim.utils import clone

        _SPAWN_BLUE_TAPE_WITH_FRICTION = clone(_spawn_blue_tape_with_friction_once)
    return _SPAWN_BLUE_TAPE_WITH_FRICTION(
        prim_path,
        cfg,
        translation,
        orientation,
        **kwargs,
    )


def make_blue_tape_cfg():
    cfg = _make_object_cfg("BlueTape", BLUE_TAPE_USD_PATH, TAPE_CENTER_XY, 0.09)
    cfg.spawn.func = spawn_blue_tape_with_friction
    return cfg
