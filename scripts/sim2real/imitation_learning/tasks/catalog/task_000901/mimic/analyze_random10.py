"""Audit Task901 random Mimic attempts for spacing and spatial pickup order.

This script is intentionally Isaac-free.  Run it with the dependency-bearing
Isaac Python environment because the host Python does not necessarily provide
NumPy or h5py.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import h5py
import numpy as np


SLOT_OBJECTS = ("plate_blue", "plate_yellow", "plate_pink")
SLOT_NAMES = ("lower_left", "lower_right", "upper_center")
COLORS = ("blue", "yellow", "pink")
MIN_CENTER_DISTANCE_M = 0.221
MAX_SLOT_OFFSET_M = 0.0101
LIFT_THRESHOLD_M = 0.025


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def demo_names(handle: h5py.File) -> list[str]:
    return sorted(handle["data"], key=lambda name: int(name.rsplit("_", 1)[1]))


def root_pose(group: h5py.Group, object_name: str) -> np.ndarray:
    initial = group[f"initial_state/rigid_object/{object_name}/root_pose"][:]
    states = group[f"states/rigid_object/{object_name}/root_pose"][:]
    return np.concatenate((initial, states), axis=0)


def first_lift_frame(pose: np.ndarray) -> int | None:
    lifted = np.flatnonzero(pose[:, 2] >= pose[0, 2] + LIFT_THRESHOLD_M)
    return int(lifted[0]) if len(lifted) else None


def analyze_demo(
    group: h5py.Group,
    canonical_xy: dict[str, np.ndarray],
    expected_slot_xyz: dict[str, np.ndarray],
) -> dict:
    poses = {name: root_pose(group, name) for name in SLOT_OBJECTS}
    initial_xy = np.stack([poses[name][0, :2] for name in SLOT_OBJECTS])
    pairwise = {
        f"{SLOT_NAMES[i]}__{SLOT_NAMES[j]}": float(np.linalg.norm(initial_xy[i] - initial_xy[j]))
        for i, j in itertools.combinations(range(3), 2)
    }
    offsets = {
        slot: (poses[name][0, :2] - canonical_xy[name]).tolist()
        for slot, name in zip(SLOT_NAMES, SLOT_OBJECTS)
    }
    lift_frames = {
        slot: first_lift_frame(poses[name])
        for slot, name in zip(SLOT_NAMES, SLOT_OBJECTS)
    }
    lift_values = list(lift_frames.values())
    pickup_order_ok = all(value is not None for value in lift_values) and lift_values == sorted(lift_values)
    spacing_ok = min(pairwise.values()) >= MIN_CENTER_DISTANCE_M
    camera_region_proxy_ok = all(
        np.max(np.abs(offsets[slot])) <= MAX_SLOT_OFFSET_M for slot in SLOT_NAMES
    )
    final_checks = {}
    for slot, name in zip(SLOT_NAMES, SLOT_OBJECTS):
        final_pose = poses[name][-1]
        final_velocity = group[f"states/rigid_object/{name}/root_velocity"][-1]
        quaternion = final_pose[3:7]
        quaternion = quaternion / np.linalg.norm(quaternion)
        _, qx, qy, _ = quaternion
        normal_z_abs = abs(1.0 - 2.0 * (qx * qx + qy * qy))
        target_error = float(np.linalg.norm(final_pose[:3] - expected_slot_xyz[name]))
        linear_speed = float(np.linalg.norm(final_velocity[:3]))
        angular_speed = float(np.linalg.norm(final_velocity[3:]))
        final_checks[slot] = {
            "object": name,
            "target_error_m": target_error,
            "normal_z_abs": float(normal_z_abs),
            "linear_speed_mps": linear_speed,
            "angular_speed_radps": angular_speed,
            "position_ok": bool(target_error < 0.035),
            "upright_in_rack_ok": bool(normal_z_abs < 0.55),
            "slow_ok": bool(linear_speed < 0.03 and angular_speed < 0.5),
        }
    return {
        "frames": int(group.attrs.get("num_samples", len(group["actions"]))),
        "initial_xy_m": {slot: initial_xy[i].tolist() for i, slot in enumerate(SLOT_NAMES)},
        "offset_from_camera_validated_reference_xy_m": offsets,
        "pairwise_center_distance_m": pairwise,
        "minimum_center_distance_m": float(min(pairwise.values())),
        "spacing_ok": bool(spacing_ok),
        "camera_region_proxy_ok": bool(camera_region_proxy_ok),
        "first_lift_frame": lift_frames,
        "pickup_order": list(SLOT_NAMES),
        "pickup_order_ok": bool(pickup_order_ok),
        "final_checks": final_checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--success", type=Path, required=True)
    parser.add_argument("--failed", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    manifest_path = args.reference.parent / "input/input_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    canonical_xy = {
        name: np.asarray(manifest["plate_initial_xy_m"][name], dtype=float)
        for name in SLOT_OBJECTS
    }
    provenance = json.loads((args.reference.parent / "provenance.json").read_text())
    expected_slot_xyz = {
        name: np.asarray(provenance["expected_slot_xyz"][name], dtype=float)
        for name in SLOT_OBJECTS
    }
    with h5py.File(args.success, "r") as success, h5py.File(args.failed, "r") as failed:
        success_names = demo_names(success)
        failed_names = demo_names(failed)
        episodes = []
        permutations = list(itertools.permutations(COLORS))
        for index, name in enumerate(success_names):
            row = analyze_demo(success[f"data/{name}"], canonical_xy, expected_slot_xyz)
            row.update(
                outcome="success",
                demo=name,
                visual_slot_colors=dict(zip(SLOT_NAMES, permutations[index % len(permutations)])),
            )
            episodes.append(row)
        for name in failed_names:
            row = analyze_demo(failed[f"data/{name}"], canonical_xy, expected_slot_xyz)
            row.update(outcome="failed", demo=name)
            episodes.append(row)

    result = {
        "schema": "cyclo.task901.random10_spatial_order_audit.v1",
        "attempts": len(episodes),
        "successes": len(success_names),
        "failures": len(failed_names),
        "success_rate": len(success_names) / len(episodes) if episodes else 0.0,
        "required_pickup_order": list(SLOT_NAMES),
        "slot_object_identity": dict(zip(SLOT_NAMES, SLOT_OBJECTS)),
        "color_note": (
            "Plate geometry, mass, and friction are identical. Slot identity stays fixed for Mimic transforms; "
            "the listed color permutation is applied to the saved-state camera replay."
        ),
        "minimum_allowed_center_distance_m": MIN_CENTER_DISTANCE_M,
        "maximum_reference_xy_offset_m": MAX_SLOT_OFFSET_M,
        "success_file": str(args.success),
        "success_sha256": sha256(args.success),
        "failed_file": str(args.failed),
        "failed_sha256": sha256(args.failed),
        "all_successes_spacing_ok": all(e["spacing_ok"] for e in episodes if e["outcome"] == "success"),
        "all_successes_camera_region_proxy_ok": all(
            e["camera_region_proxy_ok"] for e in episodes if e["outcome"] == "success"
        ),
        "all_successes_pickup_order_ok": all(
            e["pickup_order_ok"] for e in episodes if e["outcome"] == "success"
        ),
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "attempts", "successes", "failures", "success_rate",
        "all_successes_spacing_ok", "all_successes_camera_region_proxy_ok",
        "all_successes_pickup_order_ok",
    )}, indent=2))


if __name__ == "__main__":
    main()
