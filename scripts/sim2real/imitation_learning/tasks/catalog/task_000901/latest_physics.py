"""Task-local latest replay physics; never changes shared SG2 defaults."""
import json
import math
from pathlib import Path

from pxr import Usd, UsdPhysics
from isaaclab.sim.utils import clone
from _cyclo_task000901_contact import apply_candidate

PROFILE = json.loads((Path(__file__).parent/'physics_profile.json').read_text())


def apply_latest_robot_physics(result):
    stage = result.GetStage()
    root = str(result.GetPath())
    material = stage.GetPrimAtPath(root+'/distalGripperPhysicsMaterial')
    if not material or not material.HasAPI(UsdPhysics.MaterialAPI):
        raise ValueError('Missing SG2 distal material')
    api = UsdPhysics.MaterialAPI(material)
    api.CreateStaticFrictionAttr().Set(PROFILE['distal_friction'][0])
    api.CreateDynamicFrictionAttr().Set(PROFILE['distal_friction'][1])
    audit = apply_candidate(stage, root, PROFILE['contact_mode'])
    filtered = 0
    pieces = 0
    joints = []
    for child in Usd.PrimRange(result):
        if '_cyclo_softgripper_proxy' in str(child.GetPath()):
            if child.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(child).GetCollisionEnabledAttr().Get() is not False:
                pieces += 1
            if child.HasAPI(UsdPhysics.FilteredPairsAPI):
                targets = UsdPhysics.FilteredPairsAPI(child).GetFilteredPairsRel().GetTargets()
                if not all('_cyclo_softgripper_proxy' in str(t) for t in targets):
                    raise ValueError('Unexpected non-tip exclusion')
                filtered += len(targets)
        if child.GetName() in {f'gripper_{s}_joint{j}' for s in 'lr' for j in (2, 4)}:
            child.GetAttribute('physics:upperLimit').Set(math.degrees(PROFILE['joint24_upper_limit_rad']))
            # Retain the original USD mimic frequency/damping, as in automation.
            joints.append(str(child.GetPath()))
    if (pieces, filtered, len(joints)) != (20, 50, 4):
        raise ValueError(f'Unexpected latest SG2 contact layout: {pieces}, {filtered}, {joints}')
    result.SetCustomDataByKey('task901:physicsProfile', PROFILE['id'])
    return audit


@clone
def spawn_latest_robot(path, cfg, translation=None, orientation=None, **kwargs):
    from cyclo_lab.assets.robots.FFW_SG2 import spawn_sg2_with_softgripper_physics
    result = spawn_sg2_with_softgripper_physics(path, cfg, translation, orientation, **kwargs)
    apply_latest_robot_physics(result)
    return result
