"""Task000901 rigid-contact profile selected from the real-to-sim A/B runs."""

import math
import os
import re

import numpy as np
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics


PROFILE = {
    "name": "rigid_contact_recess1mm",
    "tip_recess_m": 0.001,
    "tip_static_friction": 2.0,
    "tip_dynamic_friction": 1.4,
    "plate_static_friction": 0.6,
    "plate_dynamic_friction": 0.45,
    "holder_static_friction": 0.25,
    "holder_dynamic_friction": 0.20,
    "joint24_upper_limit_rad": 1.1,
    "joint24_mimic_natural_frequency": 100.0,
    "joint24_mimic_damping_ratio": 1.0,
    "robot_solver_velocity_iterations": 4,
    "opposing_tip_filtered_pairs_retained": 50,
    "table_height_offset_m": -0.003,
}

# Replay-only override: placements, camera settings and geometry stay current.
if os.environ.get("TASK901_PHYSICS_PROFILE") == "test4":
    PROFILE.update({"holder_static_friction": 0.5, "holder_dynamic_friction": 0.5})


def apply_candidate(stage, robot_path, mode):
    """Apply the rigid rounded pocket used by the selected real-to-sim profile."""
    if mode not in ("baseline", "recess1mm", "recess3mm", "recess5mm"):
        raise ValueError(mode)
    recess_m = {
        "baseline": 0.0, "recess1mm": 0.001,
        "recess3mm": 0.003, "recess5mm": 0.005,
    }[mode]
    material = stage.GetPrimAtPath(robot_path + "/distalGripperPhysicsMaterial")
    physx = PhysxSchema.PhysxMaterialAPI.Apply(material)
    physx.CreateCompliantContactStiffnessAttr().Set(0.0)
    physx.CreateCompliantContactDampingAttr().Set(0.0)
    physx.CreateCompliantContactAccelerationSpringAttr().Set(False)
    report = {
        "mode": mode,
        "stiffness_N_per_m": 0.0,
        "damping_Ns_per_m": 0.0,
        "recess_max_m": recess_m,
        "visual_mesh_changed": False,
        "pieces": [],
    }
    groups = {}
    cache = UsdGeom.XformCache()
    for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_path)):
        path = str(prim.GetPath())
        match = re.search(r"/gripper_([lr])_[^/]*_([lr]2)/", path)
        if not (
            match and "_cyclo_softgripper_proxy/" in path
            and prim.IsA(UsdGeom.Mesh)
            and prim.HasAPI(UsdPhysics.CollisionAPI)
        ):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        transform = cache.GetLocalToWorldTransform(prim)
        world = np.asarray([transform.Transform(Gf.Vec3d(*point)) for point in points])
        groups.setdefault(match.groups(), []).append((prim, mesh, points, world, transform))
    if len(groups) != 4 or any(len(group) != 5 for group in groups.values()):
        raise RuntimeError("Expected four distal fingers with five convex pieces each")
    centers = {
        key: np.concatenate([item[3] for item in group]).mean(axis=0)
        for key, group in groups.items()
    }
    for (side, finger), group in groups.items():
        all_world = np.concatenate([item[3] for item in group])
        center = all_world.mean(axis=0)
        _, _, axes = np.linalg.svd(all_world - center, full_matrices=False)
        long_axis = axes[0]
        longitudinal = all_world @ long_axis
        long_min, long_max = float(longitudinal.min()), float(longitudinal.max())
        other = (side, "r2" if finger == "l2" else "l2")
        inward = centers[other] - centers[(side, finger)]
        inward -= np.dot(inward, long_axis) * long_axis
        inward /= np.linalg.norm(inward)
        for prim, mesh, points, world, transform in group:
            projection = world @ inward
            span = np.ptp(projection)
            if span <= 0.001:
                raise RuntimeError(f"Invalid distal collision span: {prim.GetPath()}")
            weight = np.clip((projection - projection.min()) / span * 2 - 1, 0, 1)
            weight = weight * weight * (3 - 2 * weight)
            fraction = np.clip(
                (world @ long_axis - long_min) / (long_max - long_min), 0, 1
            )
            shift = recess_m * weight * np.sin(np.pi * fraction) ** 2
            moved = world - shift[:, None] * inward
            inverse = transform.GetInverse()
            local = np.asarray([inverse.Transform(Gf.Vec3d(*point)) for point in moved])
            if recess_m:
                mesh.GetPointsAttr().Set([Gf.Vec3f(*point) for point in local])
                mesh.GetExtentAttr().Set(
                    [Gf.Vec3f(*local.min(axis=0)), Gf.Vec3f(*local.max(axis=0))]
                )
            report["pieces"].append({
                "path": str(prim.GetPath()),
                "vertex_count": len(points),
                "max_world_shift_m": float(shift.max()),
            })
    return report


def apply_robot_profile(stage, robot_path):
    """Apply all task-local gripper settings without changing visual meshes."""
    material = stage.GetPrimAtPath(robot_path + "/distalGripperPhysicsMaterial")
    if not material or not material.HasAPI(UsdPhysics.MaterialAPI):
        raise RuntimeError("Task000901 requires the SG2 distal gripper material")
    api = UsdPhysics.MaterialAPI(material)
    api.CreateStaticFrictionAttr().Set(PROFILE["tip_static_friction"])
    api.CreateDynamicFrictionAttr().Set(PROFILE["tip_dynamic_friction"])
    PhysxSchema.PhysxMaterialAPI.Apply(material).CreateFrictionCombineModeAttr().Set("max")
    filtered = 0
    joints = 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_path)):
        if "_cyclo_softgripper_proxy" in str(prim.GetPath()) and prim.HasAPI(
            UsdPhysics.FilteredPairsAPI
        ):
            targets = UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets()
            if not all("_cyclo_softgripper_proxy" in str(target) for target in targets):
                raise RuntimeError("Unexpected non-tip collision exclusion")
            filtered += len(targets)
        if prim.GetName() in {f"gripper_{side}_joint{index}" for side in "lr" for index in (2, 4)}:
            prim.GetAttribute("physics:upperLimit").Set(
                math.degrees(PROFILE["joint24_upper_limit_rad"])
            )
            prim.GetAttribute("physxMimicJoint:rotX:naturalFrequency").Set(
                PROFILE["joint24_mimic_natural_frequency"]
            )
            prim.GetAttribute("physxMimicJoint:rotX:dampingRatio").Set(
                PROFILE["joint24_mimic_damping_ratio"]
            )
            joints += 1
    if filtered != PROFILE["opposing_tip_filtered_pairs_retained"]:
        raise RuntimeError(f"Expected 50 opposing-tip exclusions, got {filtered}")
    if joints != 4:
        raise RuntimeError(f"Expected four joint2/4 mimic joints, got {joints}")
    # Test4's replay renderer authors the pocket once, after robot spawning.
    if os.environ.get("TASK901_PHYSICS_PROFILE") == "test4":
        return {"mode": "recess1mm", "recess_max_m": 0.001, "deferred_to_renderer": True}
    audit = apply_candidate(stage, robot_path, "recess1mm")
    if abs(audit["recess_max_m"] - PROFILE["tip_recess_m"]) > 1e-12:
        raise RuntimeError("Task000901 gripper recess mismatch")
    return audit
