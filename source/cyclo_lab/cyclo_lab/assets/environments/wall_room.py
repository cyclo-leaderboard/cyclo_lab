"""Independent BG2 wall-room scene and lighting assets."""

from pathlib import Path

from .physics import spawn_environment_with_friction_once


_DATA = Path(__file__).resolve().parents[3] / "data/environments/wall_room"
WALL_ROOM_USD_PATH = str(_DATA / "wall_room.usda")
WALL_ROOM_LIGHTING_USD_PATH = str(_DATA / "wall_room_lighting.usda")
_SPAWN_WALL_ROOM = None


def _spawn_wall_room_once(prim_path, cfg, translation=None, orientation=None, **kwargs):
    from isaaclab.sim.utils import bind_physics_material
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import UsdPhysics
    import isaaclab.sim as sim_utils

    prim = spawn_environment_with_friction_once(
        prim_path, cfg, translation, orientation, **kwargs
    )
    stage = get_current_stage()
    for name in ("Floor", "Walls/Rear", "Walls/CameraLeft", "Walls/CameraRight"):
        wall = stage.GetPrimAtPath(f"{prim_path}/{name}")
        if not wall.IsValid() or not wall.HasAPI(UsdPhysics.CollisionAPI):
            raise RuntimeError(f"WallRoom collision is missing: {prim_path}/{name}")
    material_path = f"{prim_path}/TablePhysicsMaterial"
    material = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="max", restitution_combine_mode="min",
        static_friction=2.5, dynamic_friction=2.2, restitution=0.0,
    )
    material.func(material_path, material)
    bind_physics_material(f"{prim_path}/WhiteTable/Top", material_path)
    return prim


def spawn_wall_room(prim_path, cfg, translation=None, orientation=None, **kwargs):
    global _SPAWN_WALL_ROOM
    if _SPAWN_WALL_ROOM is None:
        from isaaclab.sim.utils import clone

        _SPAWN_WALL_ROOM = clone(_spawn_wall_room_once)
    return _SPAWN_WALL_ROOM(prim_path, cfg, translation, orientation, **kwargs)


def make_wall_room_environment_cfg():
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WallRoom",
        spawn=sim_utils.UsdFileCfg(
            func=spawn_wall_room,
            usd_path=WALL_ROOM_USD_PATH,
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.003, rest_offset=0.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )


def make_wall_room_lighting_cfg():
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WallRoomLights",
        spawn=sim_utils.UsdFileCfg(usd_path=WALL_ROOM_LIGHTING_USD_PATH),
    )
