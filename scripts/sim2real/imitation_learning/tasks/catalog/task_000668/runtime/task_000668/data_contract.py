"""HDF5 and policy-array contracts for the Task000668 split pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from cyclo_lab.robot_specs.ffw.sg2 import FFW_SG2_ACTION_JOINT_NAMES

from .contract import TARGET_OBJECT, TASK_INSTRUCTION


JOINT16_NAMES = tuple(FFW_SG2_ACTION_JOINT_NAMES[:16])
LEROBOT_STATE17_NAMES = (*JOINT16_NAMES, "lift_joint")
EEF16_CONTRACT_ID = "ffw_sg2_task000668_dual_eef16_v1"
JOINT16_CONTRACT_ID = "ffw_sg2_rev1_dual_arm_16d_v1"
STAGING_SCHEMA = "cyclo.isaac_action_replay_staging.v1"
STAGING_ACTION_SEMANTICS = "pre_step_raw_absolute_joint_position_command"
CAMERA_MAP = {
    "cam_head": "cam_left_head",
    "cam_wrist_left": "cam_left_wrist",
    "cam_wrist_right": "cam_right_wrist",
}
CAMERA_SHAPES = {
    "cam_head": (376, 672),
    "cam_wrist_left": (640, 480),
    "cam_wrist_right": (640, 480),
}
CAMERA_ROTATION_DEG = {
    "cam_head": 0,
    "cam_wrist_left": 0,
    "cam_wrist_right": 0,
}


class Task000668DataError(RuntimeError):
    """Raised when Task000668 data crosses a pipeline boundary incorrectly."""


def jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    return value


def parse_names(value: Any, label: str) -> list[str]:
    value = jsonable(value)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise Task000668DataError(f"{label} must be a JSON string list") from error
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise Task000668DataError(f"{label} must be a string list")
    return value


def dataset_paths(group: h5py.Group, prefix: str = "") -> list[str]:
    result: list[str] = []
    for name, value in group.items():
        path = f"{prefix}/{name}" if prefix else name
        if isinstance(value, h5py.Dataset):
            result.append(path)
        elif isinstance(value, h5py.Group):
            result.extend(dataset_paths(value, path))
    return result


def image_dataset_paths(group: h5py.Group) -> list[str]:
    """Return image-like datasets; physical Task000668 must have none."""

    return [
        path
        for path in dataset_paths(group)
        if path.startswith("obs/cam_")
        or "/rgb" in path.lower()
        or "/image" in path.lower()
    ]


def _require_shape(group: h5py.Group, path: str, shape: tuple[int, ...]) -> None:
    if path not in group:
        raise Task000668DataError(f"{group.name}: missing {path}")
    actual = tuple(group[path].shape)
    if actual != shape:
        raise Task000668DataError(f"{group.name}/{path}: {actual} != {shape}")


def _require_complete_scene_state(group: h5py.Group, frames: int) -> None:
    """Require every initial articulation/object field throughout replay."""

    initial = group["initial_state"]
    states = group["states"]
    initial_paths = set(dataset_paths(initial))
    state_paths = set(dataset_paths(states))
    if initial_paths != state_paths:
        raise Task000668DataError(
            f"{group.name}: initial/state scene fields differ: "
            f"missing_states={sorted(initial_paths - state_paths)}, "
            f"missing_initial={sorted(state_paths - initial_paths)}"
        )
    for scene_path in sorted(initial_paths):
        initial_shape = tuple(initial[scene_path].shape)
        state_shape = tuple(states[scene_path].shape)
        if not initial_shape or initial_shape[0] != 1:
            raise Task000668DataError(
                f"{group.name}/initial_state/{scene_path}: expected one initial row"
            )
        expected = (frames, *initial_shape[1:])
        if state_shape != expected:
            raise Task000668DataError(
                f"{group.name}/states/{scene_path}: {state_shape} != {expected}"
            )


def validate_hdf5(
    path: str | Path,
    *,
    representation: str,
    require_success: bool = True,
) -> dict[str, Any]:
    """Validate the physical EEF16 or converted joint16 replay boundary."""

    expected_contract = {
        "eef": EEF16_CONTRACT_ID,
        "joint": JOINT16_CONTRACT_ID,
    }.get(representation)
    if expected_contract is None:
        raise ValueError("representation must be 'eef' or 'joint'")
    source_path = Path(path).resolve()
    with h5py.File(source_path, "r") as source:
        if "data" not in source:
            raise Task000668DataError("HDF5 is missing /data")
        data = source["data"]
        task_id = str(jsonable(data.attrs.get("task_id", "")))
        if task_id not in ("000668", "668"):
            raise Task000668DataError(f"unexpected task_id {task_id!r}")
        contract_id = str(jsonable(data.attrs.get("robot_contract_id", "")))
        if contract_id != expected_contract:
            raise Task000668DataError(
                f"robot_contract_id {contract_id!r} != {expected_contract!r}"
            )
        action_names = parse_names(data.attrs.get("action_names"), "action_names")
        state_names = parse_names(
            data.attrs.get("observation_state_names"),
            "observation_state_names",
        )
        if len(action_names) != 16 or state_names != list(JOINT16_NAMES):
            raise Task000668DataError("Task000668 action/state name width is not 16")
        if any(name.startswith(("head_", "lift_", "linear_", "angular_")) for name in action_names):
            raise Task000668DataError("head/lift/base leaked into Task000668 actions")
        names = sorted(data.keys(), key=lambda item: int(item.rsplit("_", 1)[-1]))
        if not names:
            raise Task000668DataError("HDF5 has no episodes")
        frame_count = 0
        all_image_paths: dict[str, list[str]] = {}
        for name in names:
            group = data[name]
            if require_success and not bool(group.attrs.get("success", False)):
                raise Task000668DataError(f"{name} is not marked successful")
            frames = int(group["actions"].shape[0])
            frame_count += frames
            for required, shape in {
                "actions": (frames, 16),
                "obs/actions": (frames, 16),
                "obs/joint_pos": (frames, 16),
                "obs/joint_pos_target": (frames, 16),
                "obs/left_eef_pose": (frames, 7),
                "obs/right_eef_pose": (frames, 7),
                "states/articulation/robot/root_pose": (frames, 7),
                f"states/rigid_object/{TARGET_OBJECT}/root_pose": (frames, 7),
                "initial_state/articulation/robot/root_pose": (1, 7),
                f"initial_state/rigid_object/{TARGET_OBJECT}/root_pose": (1, 7),
            }.items():
                _require_shape(group, required, shape)
            _require_complete_scene_state(group, frames)
            images = image_dataset_paths(group)
            if images:
                all_image_paths[name] = images
            if representation == "joint":
                action = np.asarray(group["actions"], dtype=np.float32)
                target = np.asarray(group["obs/joint_pos_target"], dtype=np.float32)
                if not np.array_equal(action, target):
                    raise Task000668DataError(
                        f"{name}: converted joint16 actions differ from joint_pos_target"
                    )
        if all_image_paths:
            raise Task000668DataError(
                f"physical HDF5 contains forbidden image datasets: {all_image_paths}"
            )
        control_hz = float(data.attrs.get("control_hz", 0.0))
        if control_hz <= 0.0 or not control_hz.is_integer():
            raise Task000668DataError(f"invalid control_hz {control_hz}")
    return {
        "path": str(source_path),
        "representation": representation,
        "episodes": len(names),
        "frames": frame_count,
        "fps": int(control_hz),
        "action_names": action_names,
        "state_names": state_names,
        "task_instruction": TASK_INSTRUCTION,
        "rgb_datasets": [],
        "size_bytes": source_path.stat().st_size,
    }


def policy_arrays(group: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    """Return a 17D state with held lift and the untouched 16D action."""

    state = np.asarray(group["obs/joint_pos"], dtype=np.float32)
    action = np.asarray(group["actions"], dtype=np.float32)
    if (
        state.ndim != 2
        or action.ndim != 2
        or state.shape != (action.shape[0], len(JOINT16_NAMES))
        or action.shape[1] != len(JOINT16_NAMES)
    ):
        raise Task000668DataError(
            f"{group.name}: expected 16D source state/action, got {state.shape}/{action.shape}"
        )
    lift_key = next(
        (
            key
            for key in ("recording_initial_lift_position_m", "lift_start_measured_m")
            if key in group.attrs
        ),
        None,
    )
    if lift_key is None:
        raise Task000668DataError(
            f"{group.name}: missing recording_initial_lift_position_m or "
            "lift_start_measured_m required for the LeRobot state"
        )
    try:
        lift_m = float(group.attrs[lift_key])
    except (TypeError, ValueError, OverflowError) as error:
        raise Task000668DataError(f"{group.name}: invalid {lift_key}") from error
    if not np.isfinite(lift_m):
        raise Task000668DataError(f"{group.name}: non-finite {lift_key}")
    state = np.concatenate(
        (state, np.full((state.shape[0], 1), lift_m, dtype=np.float32)), axis=1
    )
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise Task000668DataError(f"{group.name}: non-finite policy arrays")
    return state, action
