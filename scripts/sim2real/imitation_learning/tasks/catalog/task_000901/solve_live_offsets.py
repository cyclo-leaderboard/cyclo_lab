"""Convert Task901 image-fitted world camera poses to live SG2 link offsets.

The archived fit used USD xforms after an extra sim.reset(), when the robot
joints were zero. Here the desired world camera poses are preserved, while the
parent links are evaluated at the actual 1050 joint values from PhysX.
"""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path("/tmp/task901_calibration")
new = json.loads((ROOT / "validation.json").read_text())
head = json.loads((ROOT / "head_extrinsic.json").read_text())
wrists = json.loads((ROOT / "wrist_extrinsics.json").read_text())


def row_pose(pos, quat_wxyz):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(
        [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    ).as_matrix().T
    matrix[3, :3] = pos
    return matrix


def quat_wxyz(matrix):
    x, y, z, w = Rotation.from_matrix(matrix[:3, :3].T).as_quat()
    return [float(w), float(x), float(y), float(z)]


mapping = {
    "head": ("head_link2", "sim_head_parent_usd_matrix", head),
    "left": ("arm_l_link7", "sim_wrist_left_parent_usd_matrix", wrists["cameras"]["left"]),
    "right": ("arm_r_link7", "sim_wrist_right_parent_usd_matrix", wrists["cameras"]["right"]),
}
result = {}
for name, (body_name, parent_key, fit) in mapping.items():
    parent_rest = np.array(new[parent_key], dtype=float)
    body_rest = np.array(new["sim_body_usd_matrices"][body_name], dtype=float)
    body = new["sim_body_poses"][body_name]
    body_live = row_pose(body["pos_w"], body["quat_w"])
    parent_live = parent_rest @ np.linalg.inv(body_rest) @ body_live

    fit_local = row_pose(fit["offset_opengl_pos_m"], fit["offset_opengl_quat_wxyz"])
    desired_world = fit_local @ parent_rest
    local_live = desired_world @ np.linalg.inv(parent_live)
    body_local = desired_world @ np.linalg.inv(body_live)
    assert np.max(np.abs(local_live @ parent_live - desired_world)) < 1e-8
    result[name] = {
        "offset_opengl_pos_m": local_live[3, :3].tolist(),
        "offset_opengl_quat_wxyz": quat_wxyz(local_live),
        "body_local_opengl_pos_m": body_local[3, :3].tolist(),
        "body_local_opengl_quat_wxyz": quat_wxyz(body_local),
        "desired_eye_world_m": desired_world[3, :3].tolist(),
        "body_live_pos_world_m": body["pos_w"],
        "fit_reprojection_rms_px": fit["reprojection_rms_px"],
    }

(ROOT / "live_offsets.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
