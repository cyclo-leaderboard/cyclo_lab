"""World-frame rigid-object pose controls for UI-launched simulations."""

from __future__ import annotations

import math
import re


OBJECT_REQUEST_FILE = "/tmp/cyclo_lab_ui_object_request.json"


def validate_object_pose(name: str, pose: object) -> dict[str, list[float]]:
    """Accept only a named rigid prop and bounded XYZ/RPY values."""
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,99}", name) or name == "robot":
        raise ValueError("Select a scene object, not the robot")
    if not isinstance(pose, dict):
        raise ValueError("Object pose must contain xyz_m and rpy_deg")
    result = {}
    for key, limit in (("xyz_m", 10.0), ("rpy_deg", 360.0)):
        values = pose.get(key)
        if not isinstance(values, list) or len(values) != 3 or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or abs(value) > limit for value in values
        ):
            raise ValueError(f"{key} must have three finite values within +/-{limit:g}")
        result[key] = [float(value) for value in values]
    return result


def snapshot_object_poses(env) -> dict[str, dict]:
    from isaaclab.utils.math import euler_xyz_from_quat

    result = {}
    for name, obj in env.scene.rigid_objects.items():
        pose = obj.data.root_state_w[0, :7].detach()
        roll, pitch, yaw = euler_xyz_from_quat(pose[3:7].unsqueeze(0))
        angles = [
            ((math.degrees(float(axis[0])) + 180.0) % 360.0) - 180.0
            for axis in (roll, pitch, yaw)
        ]
        result[name] = {"xyz_m": pose[:3].cpu().tolist(), "rpy_deg": angles}
    return result


def apply_object_pose(env, name: str, pose: object) -> dict[str, list[float]]:
    import torch
    from isaaclab.utils.math import quat_from_euler_xyz

    validated = validate_object_pose(name, pose)
    if name not in env.scene.rigid_objects:
        raise ValueError(f"Unknown scene object: {name}")
    obj = env.scene.rigid_objects[name]
    root_pose = obj.data.root_state_w[0:1, :7].detach().clone()
    root_pose[0, :3] = torch.as_tensor(validated["xyz_m"], dtype=root_pose.dtype, device=root_pose.device)
    angles = torch.as_tensor(validated["rpy_deg"], dtype=root_pose.dtype, device=root_pose.device) * (math.pi / 180.0)
    root_pose[0, 3:7] = quat_from_euler_xyz(angles[0:1], angles[1:2], angles[2:3])[0]
    obj.write_root_pose_to_sim(root_pose)
    obj.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=root_pose.dtype, device=root_pose.device))
    return validated
