"""Preview an SG2 q_des plan in a newly launched Cyclo Lab environment.

Requires the Cyclo Lab Isaac Sim Python environment. It starts Isaac only
with an explicit --execute flag. --record-hdf5 waits for UI acceptance.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from uuid import uuid4

from seed_plan import LAB_ROOT, discover_environments, joint_contract, validate_plan, validate_object_poses


STATE_FILE = LAB_ROOT / ".runtime/joint_waypoint_seed_ui/live_state.json"
CAMERA_DIR = STATE_FILE.parent / "cameras"
CONTROL_DIR = LAB_ROOT / ".runtime/joint_waypoint_seed_ui"
IK_REQUEST_FILE = CONTROL_DIR / "ik_request.json"
IK_RESULT_FILE = CONTROL_DIR / "ik_result.json"
PREVIEW_REQUEST_FILE = CONTROL_DIR / "preview_request.json"
PREVIEW_RESULT_FILE = CONTROL_DIR / "preview_result.json"
RESET_REQUEST_FILE = CONTROL_DIR / "reset_request.json"
RESET_RESULT_FILE = CONTROL_DIR / "reset_result.json"
OBJECT_REQUEST_FILE = CONTROL_DIR / "object_request.json"
OBJECT_RESULT_FILE = CONTROL_DIR / "object_result.json"
STOP_REQUEST_FILE = CONTROL_DIR / "stop_request.json"
STOP_RESULT_FILE = CONTROL_DIR / "stop_result.json"
ACCEPT_REQUEST_FILE = CONTROL_DIR / "accept_request.json"
LEADER_LIVE_FILE = CONTROL_DIR / "leader_live_mini.json"
LEADER_TELEOP_FILE = CONTROL_DIR / "leader_teleop.json"
TELEOP_CAPTURE_REQUEST_FILE = CONTROL_DIR / "teleop_capture_request.json"
TELEOP_CAPTURE_RESULT_FILE = CONTROL_DIR / "teleop_capture_result.json"
PREVIEW_HZ = 15.0
CAMERA_NAMES = {
    "overhead_left": "cam_overhead_left",
    "overhead_center": "cam_overhead_center",
    "overhead_right": "cam_overhead_right",
    "head": "cam_head",
    "wrist_left": "cam_wrist_left",
    "wrist_right": "cam_wrist_right",
}
EXPECTED_ACTION_TERMS = (
    "arm_l_action", "gripper_l_action", "arm_r_action",
    "gripper_r_action", "head_action", "lift_action",
)
TRAY_OBJECT_CANDIDATES = ("white_basket", "orange_tray", "tray")


class StopRequested(Exception):
    """Raised after a matching UI disconnect request is received."""

    def __init__(self, request: dict):
        super().__init__("UI requested environment disconnect")
        self.request = request


def apply_control_rate(cfg, control_hz: float) -> tuple[float, int]:
    """Apply a plan rate to the env decimation without editing Cyclo Lab source."""
    physics_hz = 1.0 / float(cfg.sim.dt)
    ratio = physics_hz / float(control_hz)
    rounded = round(ratio)
    if control_hz > physics_hz or not abs(ratio - rounded) < 1.0e-9:
        raise ValueError(
            f"control_hz={control_hz:g} must divide the env physics rate {physics_hz:g} Hz "
            "into an integer decimation."
        )
    cfg.decimation = int(rounded)
    if hasattr(cfg, "control_hz"):
        cfg.control_hz = float(control_hz)
    if hasattr(cfg, "recording_control_hz"):
        cfg.recording_control_hz = float(control_hz)
    return physics_hz, cfg.decimation


def object_pose_snapshot(env) -> dict:
    """Expose named rigid scene props; articulated robots are never spawn controls."""
    from isaaclab.utils.math import euler_xyz_from_quat
    result = {}
    for name, obj in env.scene.rigid_objects.items():
        pose = obj.data.root_state_w[0, :7].detach()
        rpy = euler_xyz_from_quat(pose[3:7].unsqueeze(0))
        angles = [((math.degrees(float(v[0])) + 180.0) % 360.0) - 180.0 for v in rpy]
        result[name] = {"xyz_m": pose[:3].cpu().tolist(), "rpy_deg": angles,
                        "quaternion_wxyz": pose[3:7].cpu().tolist()}
    return result


def initial_object_poses(env, plan: dict) -> dict:
    """Keep authored spawn poses fixed while live rigid bodies move."""
    poses = object_pose_snapshot(env)
    for name, pose in plan.get("object_poses", {}).items():
        if name in poses:
            poses[name] = {**poses[name], "xyz_m": list(pose["xyz_m"]), "rpy_deg": list(pose["rpy_deg"])}
    target = getattr(env.cfg, "target_object", "")
    if plan.get("object_position_xyz_m") is not None and target in poses:
        poses[target]["xyz_m"] = list(plan["object_position_xyz_m"])
    return poses


def write_state(
    env, plan: dict, step: int, q_des=None, *,
    phase: str, initial_joint_pos: dict, camera_frame_step: int, cameras: list[str], run_id: str,
    plan_name: str,
    status: str = "running",
) -> None:
    """Publish measured world positions from the selected env, never guessed UI poses."""
    preview_worker = getattr(env, "_live_camera_previews", None)
    if preview_worker is not None:
        camera_frame_step, cameras = preview_worker.snapshot()
    scene = env.scene
    tcp = {}
    for side in ("left", "right"):
        key = f"{side}_eef"
        if key in scene.sensors:
            tcp[side] = scene[key].data.target_pos_w[0, 0, :].detach().cpu().tolist()
    target_name = getattr(env.cfg, "target_object", "")
    object_root = None
    if target_name and target_name in scene.rigid_objects:
        object_root = scene[target_name].data.root_pos_w[0].detach().cpu().tolist()
    tray_name = next((name for name in TRAY_OBJECT_CANDIDATES if name in scene.rigid_objects), None)
    tray_center = None
    if tray_name is not None:
        tray_center = scene[tray_name].data.root_pos_w[0].detach().cpu().tolist()
    payload = {
        "status": status,
        "updated_at_ms": round(time.time() * 1000),
        "environment": plan["environment"],
        "plan_name": plan_name,
        "run_id": run_id,
        "step": step,
        "phase": phase,
        "recording_mode": bool(getattr(env, "_joint_seed_recording", False)),
        "teleop_recording": bool(getattr(env, "_teleop_recording", False)),
        "sim_control_hz": 1.0 / float(env.step_dt),
        "physics_hz": 1.0 / float(env.cfg.sim.dt),
        "decimation": int(env.cfg.decimation),
        "frame": "world",
        "tcp_xyz_m": tcp,
        "object_name": target_name or None,
        "objects": object_pose_snapshot(env),
        "initial_object_poses": getattr(env, "_waypoint_initial_object_poses", {}),
        "object_poses": plan.get("object_poses", {}),
        "object_root_xyz_m": object_root,
        "tray_name": tray_name,
        "tray_center_xyz_m": tray_center,
        "q_des": None if q_des is None else q_des.detach().cpu().tolist(),
        "measured_joint_pos": dict(zip(
            plan["joint_order"],
            scene["robot"].data.joint_pos[0, env._waypoint_joint_ids].detach().cpu().tolist(),
        )),
        "initial_joint_pos": initial_joint_pos,
        "camera_frame_step": camera_frame_step,
        "cameras": cameras,
        "preview_timing_ms": getattr(env, "_live_preview_timing_ms", {}),
        "gripper_velocity_limits": getattr(env, "_gripper_velocity_limits", {}),
    }
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE_FILE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    os.replace(temporary, STATE_FILE)


def write_camera_previews(env, rotations: dict[str, int]) -> list[str]:
    """Encode the existing Isaac camera RGB tensors like Cyclo Lab's publisher."""
    if not getattr(env, "_joint_seed_recording", False):
        from live_camera_previews import LiveCameraPreviews

        worker = getattr(env, "_live_camera_previews", None)
        if worker is None:
            worker = env._live_camera_previews = LiveCameraPreviews(CAMERA_DIR)
        render_start = time.perf_counter()
        env.sim.render()
        capture_start = time.perf_counter()
        images = {}
        for label, sensor_name in CAMERA_NAMES.items():
            sensor = env.scene.sensors.get(sensor_name)
            if sensor is None:
                continue
            # Refresh annotator buffers after this wall-clock-scheduled render.
            sensor.update(0.0, force_recompute=True)
            data = sensor.data
            frame = int(sensor.frame[0].item())
            if worker.last_frames.get(label) == frame:
                continue
            # Own the CPU buffer so later simulator updates cannot overwrite it.
            image = data.output["rgb"][0, ..., :3].detach().cpu().numpy().copy()
            limit = {"cam_head": (480, 270), "cam_wrist_left": (180, 240),
                     "cam_wrist_right": (180, 240)}.get(sensor_name)
            images[label] = (image, int(rotations.get(sensor_name, 0)) % 4, limit)
            worker.last_frames[label] = frame
        if images:
            worker.submit(images)
        env._live_preview_timing_ms = {
            "render": (capture_start - render_start) * 1000,
            "capture_submit": (time.perf_counter() - capture_start) * 1000,
        }
        return worker.snapshot()[1]

    import cv2
    import torch

    CAMERA_DIR.mkdir(parents=True, exist_ok=True)
    available = []
    for label, sensor_name in CAMERA_NAMES.items():
        if sensor_name not in env.scene.sensors:
            continue
        image = env.scene.sensors[sensor_name].data.output["rgb"][0].detach()
        turns = int(rotations.get(sensor_name, 0)) % 4
        if turns:
            image = torch.rot90(image, k=turns, dims=(0, 1))
        image = image[..., :3].contiguous().cpu().numpy()
        if image.dtype != "uint8":
            if image.size and float(image.max()) <= 1.0:
                image = image * 255.0
            image = image.clip(0, 255).astype("uint8")
        # Keep the export monitor at native resolution. The live teleop runner
        # alone uses smaller policy-camera images; CSS keeps tile sizes fixed.
        display_limit = ({
            "cam_head": (480, 270),
            "cam_wrist_left": (180, 240),
            "cam_wrist_right": (180, 240),
        }.get(sensor_name) if not getattr(env, "_joint_seed_recording", False) else None)
        if display_limit is not None:
            max_width, max_height = display_limit
            scale = min(1.0, max_width / image.shape[1], max_height / image.shape[0])
            if scale < 1.0:
                image = cv2.resize(
                    image,
                    (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
                    interpolation=cv2.INTER_AREA,
                )
        jpeg_options = ([cv2.IMWRITE_JPEG_QUALITY, 70] if display_limit is not None else [])
        success, buffer = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), jpeg_options)
        if not success:
            raise RuntimeError(f"Could not encode {sensor_name} preview")
        destination = CAMERA_DIR / f"{label}.jpg"
        temporary = CAMERA_DIR / f"{label}.jpg.tmp"
        temporary.write_bytes(buffer.tobytes())
        os.replace(temporary, destination)
        available.append(label)
    return available


def main() -> None:
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--plan", type=Path)
    probe.add_argument("--execute", action="store_true")
    preview_args, _unused = probe.parse_known_args()
    if not preview_args.execute:
        if preview_args.plan is None:
            probe.error("--plan is required")
        plan = validate_plan(json.loads(preview_args.plan.read_text(encoding="utf-8")), discover_environments())
        total_steps = sum(w["steps"] for w in plan["waypoints"])
        print(f"Validated {preview_args.plan}: {total_steps} waypoint steps; add --execute to preview.")
        return
    parser = argparse.ArgumentParser(description=__doc__)
    plan_source = parser.add_mutually_exclusive_group(required=True)
    plan_source.add_argument("--plan", type=Path)
    plan_source.add_argument("--plan-json", help="Inline connection plan from the UI server.")
    plan_source.add_argument("--trajectory", type=Path, help="Saved timestamped teleop trajectory.")
    parser.add_argument("--execute", action="store_true", help="Launch a new Isaac Sim process and replay the plan.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ik-wait-seconds", type=float, default=0,
                        help="After playback, hold the final q_des this long for browser XYZ IK queries.")
    parser.add_argument("--hold-final", action="store_true",
                        help="After playback, hold the final q_des until Isaac Sim is closed.")
    parser.add_argument("--record-hdf5", action="store_true",
                        help="Record to a pending HDF5 file and wait for UI acceptance after playback.")
    parser.add_argument("--auto-accept", action="store_true",
                        help="Finalize a recorded HDF5 immediately after successful playback.")
    parser.add_argument("--accept-wait-seconds", type=float, default=300,
                        help="How long to wait for UI approval after recorded playback.")
    parser.add_argument("--dataset-file", type=Path,
                        help="Final .hdf5 path; defaults to a unique file under Cyclo Lab datasets.")
    # Keep the standard Cyclo Lab launcher options, including --device and --headless.
    # Pinocchio must be imported before AppLauncher so Isaac Lab's compatible
    # distribution wins over any copy bundled with Isaac Sim.
    try:
        import pinocchio  # noqa: F401
    except ImportError:
        pass
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not 0 <= args.ik_wait_seconds <= 3600:
        parser.error("--ik-wait-seconds must be between 0 and 3600")
    if not 1 <= args.accept_wait_seconds <= 3600:
        parser.error("--accept-wait-seconds must be between 1 and 3600")
    if args.dataset_file is not None and not args.record_hdf5:
        parser.error("--dataset-file requires --record-hdf5")
    if args.auto_accept and not args.record_hdf5:
        parser.error("--auto-accept requires --record-hdf5")
    trajectory = None
    setup_plan = None
    if args.trajectory is not None:
        from teleop_trajectory import validate_trajectory, resample_15hz
        trajectory = json.loads(args.trajectory.read_text(encoding="utf-8"))
        plan = validate_plan(trajectory["source_plan"], discover_environments())
        validate_trajectory(trajectory, plan["joint_order"])
        if "FFW-BG2" not in plan["environment"]:
            parser.error("Teleop trajectory replay currently supports BG2.")
        setup_plan = plan
        plan = {**plan, "control_hz": 15.0}
        args.plan = args.trajectory
    elif args.plan_json is not None:
        plan = validate_plan(json.loads(args.plan_json), discover_environments())
        args.plan = Path("ui_connection")
    else:
        plan = validate_plan(json.loads(args.plan.read_text(encoding="utf-8")), discover_environments())
    if plan["preview"] != "head":
        parser.error("This starter supports live RGB preview; offline video rendering is still a separate step.")

    imitation_root = LAB_ROOT / "scripts/sim2real/imitation_learning"
    sys.path.insert(0, str(imitation_root))
    sys.path.insert(0, str(imitation_root / "recorder"))
    from local_plugins import load_for_task

    plugin = load_for_task(plan["environment"])
    plugin.configure_launcher_args(args)
    args.enable_cameras = True
    app = AppLauncher(vars(args)).app
    env = None
    dashboard = None
    dataset_file = None
    pending_file = None
    accepted = False
    completed = False
    run_id = uuid4().hex
    initial_pose_hold_steps = max(1, round(plan["initial_pose_hold_s"] * plan["control_hz"]))
    replay_steps = list(resample_15hz(trajectory["samples"])) if trajectory else None
    try:
        import gymnasium as gym
        import torch
        from isaaclab.managers import DatasetExportMode, RecorderManagerBaseCfg
        from isaaclab_tasks.utils import parse_env_cfg
        import cyclo_lab  # noqa: F401  registers built-in Gym tasks
        from ik_preview import estimate_q_des
        from cyclo_lab.robot_specs.ffw.sg2 import FFW_SG2_ACTION_JOINT_NAMES
        if args.record_hdf5:
            from recorder_manager.recorder_manager import StreamingRecorderManager
            from cyclo_lab.robot_specs.ffw.sg2 import hdf5_contract_metadata

        plugin.register_environments()
        cfg = parse_env_cfg(plan["environment"], device=args.device, num_envs=1)
        cfg.init_action_cfg("record")
        # Use the same uncapped arm/gripper dynamics for live control and seed
        # replay. None would inherit a speed cap from the USD joint instead.
        for actuator in cfg.scene.robot.actuators.values():
            if any(name.startswith(("arm_", "gripper_")) for name in actuator.joint_names_expr):
                actuator.velocity_limit_sim = 1.0e9
                actuator.velocity_limit = None
        physics_hz, initial_decimation = apply_control_rate(cfg, plan["control_hz"])
        if not args.record_hdf5:
            cfg.camera_hz = PREVIEW_HZ
            render_steps = physics_hz / PREVIEW_HZ
            if not math.isclose(render_steps, round(render_steps)):
                raise ValueError("Live camera rate must divide the physics rate.")
            cfg.sim.render_interval = int(round(render_steps))
        print(
            f"Control rate: {plan['control_hz']:g} Hz = {physics_hz:g} Hz physics / "
            f"decimation {initial_decimation}",
            flush=True,
        )
        cfg.seed = args.seed
        if hasattr(cfg.terminations, "success"):
            cfg.terminations.success = None
        if hasattr(cfg.terminations, "time_out"):
            cfg.terminations.time_out = None
        cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_NONE
        if not args.record_hdf5:
            # EXPORT_NONE skips file export, but recorder terms still retain RGB
            # tensors each step. Preview sessions must not accumulate episodes.
            cfg.recorders = RecorderManagerBaseCfg(dataset_export_mode=DatasetExportMode.EXPORT_NONE)
        if args.record_hdf5:
            dataset_file = (args.dataset_file or (
                LAB_ROOT / "datasets/waypoint_seed" /
                f"{args.plan.stem}_{run_id[:8]}.hdf5"
            )).resolve()
            if dataset_file.suffix != ".hdf5":
                raise ValueError("--dataset-file must end in .hdf5")
            pending_file = dataset_file.with_name(dataset_file.stem + ".pending.hdf5")
            if dataset_file.exists() or pending_file.exists():
                raise FileExistsError(f"HDF5 output already exists: {dataset_file}")
            dataset_file.parent.mkdir(parents=True, exist_ok=True)
            cfg.recorders.dataset_export_dir_path = str(dataset_file.parent)
            cfg.recorders.dataset_filename = pending_file.stem
        # Use the original SG2 operator dashboard's three robot-following
        # overhead views. Task000300's task-specific contact cameras are not
        # the operator views requested in this UI.
        from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.platform.robot_cfg import (
            enable_sg2_showroom_operator_cameras,
        )
        from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.platform.scene_cfg import (
            OPERATOR_CAMERA_ROWS, OPERATOR_CAMERA_ROTATIONS,
        )
        enable_sg2_showroom_operator_cameras(cfg)
        operator_parent = getattr(cfg, "waypoint_operator_camera_parent", None)
        if operator_parent:
            for side in ("left", "center", "right"):
                camera = getattr(cfg.scene, f"cam_overhead_{side}")
                camera.prim_path = f"{operator_parent}/cam_overhead_{side}"
        if not args.record_hdf5:
            # Keep the three external operator views at their authored size.
            # This runner only supplies the UI during live preview. Seed HDF5
            # replay and policy inference use separate native camera configs.
            for sensor_name in ("cam_head", "cam_wrist_left", "cam_wrist_right"):
                camera = getattr(cfg.scene, sensor_name, None)
                if camera is not None:
                    camera.width = max(1, camera.width * 3 // 8)
                    camera.height = max(1, camera.height * 3 // 8)
        elif "FFW-BG2" in plan["environment"]:
            head_size = (cfg.scene.cam_head.width, cfg.scene.cam_head.height)
            if head_size != (1280, 720):
                raise RuntimeError(f"BG2 seed export requires a 1280x720 head camera, got {head_size}.")
        cfg.operator_camera_rows = OPERATOR_CAMERA_ROWS
        cfg.operator_camera_rotations = OPERATOR_CAMERA_ROTATIONS
        rows = getattr(cfg, "operator_camera_rows", None)
        if not rows:
            raise ValueError(f"No operator/head camera dashboard in {plan['environment']}")
        # Spawn saved props at their authored poses from the first physics frame.
        # A fresh GPU scene can otherwise warm up at its USD/default transforms
        # before a post-reset teleport takes effect, unlike an already-running preview.
        spawn_poses = dict(plan.get("object_poses", {}))
        legacy_xyz = plan.get("object_position_xyz_m")
        if legacy_xyz is not None and cfg.target_object not in spawn_poses:
            getattr(cfg.scene, cfg.target_object).init_state.pos = tuple(legacy_xyz)
        from isaaclab.utils.math import quat_from_euler_xyz
        for object_name, pose in spawn_poses.items():
            object_cfg = getattr(cfg.scene, object_name, None)
            if object_cfg is None or not hasattr(object_cfg, "init_state"):
                raise ValueError(f"Saved object is unavailable: {object_name}")
            angles = torch.tensor(pose["rpy_deg"], dtype=torch.float32) * (math.pi / 180.0)
            quat = quat_from_euler_xyz(angles[0:1], angles[1:2], angles[2:3])[0]
            object_cfg.init_state.pos = tuple(pose["xyz_m"])
            object_cfg.init_state.rot = tuple(float(v) for v in quat)
        env = gym.make(plan["environment"], cfg=cfg).unwrapped
        env._joint_seed_recording = bool(args.record_hdf5)
        env._gripper_velocity_limits = {
            name: actuator.velocity_limit_sim[0].detach().cpu().tolist()
            for name, actuator in env.scene["robot"].actuators.items()
            if name.startswith("gripper_")
        }
        applied_control_hz = 1.0 / float(env.step_dt)
        if abs(applied_control_hz - plan["control_hz"]) > 1.0e-6:
            raise RuntimeError(
                f"Requested {plan['control_hz']:g} Hz but the created env applies {applied_control_hz:g} Hz."
            )
        if args.record_hdf5:
            del env.recorder_manager
            cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
            env.recorder_manager = StreamingRecorderManager(cfg.recorders, env)
            # Keep export chunks small, with a nonempty last chunk for finalization.
            total_steps = (len(replay_steps) if trajectory else initial_pose_hold_steps + sum(waypoint["steps"] for waypoint in plan["waypoints"]))
            flush_steps = 30
            while total_steps % flush_steps == 0:
                flush_steps += 1
            env.recorder_manager.flush_steps = flush_steps
            env.recorder_manager.compression = "lzf"
            env.recorder_manager.recording_enabled = False
            metadata = {
                "schema_version": "cyclo_lab_hdf5_v1",
                "dataset_origin": "isaaclab_hdf5_seed",
                "task_env_name": plan["environment"],
                "task_id": str(getattr(cfg, "task_id", "")),
                "task_instruction": str(getattr(cfg, "task_instruction", "")),
                "target_object_name": str(getattr(cfg, "target_object", "")),
                "control_hz": plan["control_hz"],
                "camera_hz": float(getattr(cfg, "camera_hz", plan["control_hz"])),
                "initial_pose_hold_s": 0.0 if trajectory else initial_pose_hold_steps / plan["control_hz"],
                "initial_pose_hold_steps": 0 if trajectory else initial_pose_hold_steps,
                "observation_semantics": "pre_step",
                "scene_state_semantics": "post_step",
                "success_criterion_id": "manual_operator_acceptance",
                "joint_waypoint_plan_json": json.dumps(plan, ensure_ascii=False),
                "teleop_trajectory_json": json.dumps(trajectory, ensure_ascii=False) if trajectory else "",
                "source_plan_file": str(args.plan.resolve()),
                "run_id": run_id,
            }
            metadata.update(hdf5_contract_metadata(cfg.actions))
            from seed_plan import local_robot_spec
            robot_spec = local_robot_spec(plan["environment"])
            if robot_spec:
                import hashlib
                metadata.update({key: value for key, value in robot_spec.items() if key != "urdf_path"})
                metadata["robot_urdf_sha256"] = hashlib.sha256(Path(robot_spec["urdf_path"]).read_bytes()).hexdigest()
            env.recorder_manager._dataset_file_handler.set_dataset_metadata(metadata)
            env.recorder_manager.cfg.dataset_export_mode = DatasetExportMode.EXPORT_ALL
        if tuple(env.action_manager.active_terms) != EXPECTED_ACTION_TERMS or env.action_manager.total_action_dim != 19:
            raise ValueError(f"Environment does not expose the SG2 19D record action: {env.action_manager.active_terms}")
        names = tuple(FFW_SG2_ACTION_JOINT_NAMES)
        if names != tuple(plan["joint_order"]):
            raise ValueError("Plan joint order differs from the active Cyclo Lab SG2 contract.")
        robot = env.scene["robot"]
        joint_ids, resolved = robot.find_joints(list(names), preserve_order=True)
        if tuple(resolved) != names:
            raise ValueError(f"Selected robot joint mapping differs from q_des order: {resolved}")
        env._waypoint_joint_ids = joint_ids
        target_object_name = str(getattr(cfg, "target_object", ""))

        def set_object_poses(poses) -> None:
            from isaaclab.utils.math import quat_from_euler_xyz
            poses = validate_object_poses(poses)
            missing = set(poses).difference(env.scene.rigid_objects)
            if missing:
                raise ValueError(f"Objects are not available in this environment: {sorted(missing)}")
            for name, pose in poses.items():
                obj = env.scene[name]
                root_pose = obj.data.root_state_w[0:1, :7].detach().clone()
                root_pose[0, :3] = torch.tensor(pose["xyz_m"], dtype=root_pose.dtype, device=root_pose.device)
                angles = torch.tensor(pose["rpy_deg"], dtype=root_pose.dtype, device=root_pose.device) * (math.pi / 180.0)
                root_pose[0, 3:7] = quat_from_euler_xyz(angles[0:1], angles[1:2], angles[2:3])[0]
                obj.write_root_pose_to_sim(root_pose)
                obj.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=root_pose.dtype, device=root_pose.device))

        def apply_plan_object_poses(active_plan) -> None:
            poses = dict(active_plan.get("object_poses", {}))
            # Existing v3 plans changed only the target XYZ and kept its reset orientation.
            if active_plan.get("object_position_xyz_m") is not None and target_object_name not in poses:
                current_pose = object_pose_snapshot(env).get(target_object_name)
                if current_pose is None:
                    raise ValueError("The selected environment does not expose its target rigid object.")
                poses[target_object_name] = {"xyz_m": active_plan["object_position_xyz_m"], "rpy_deg": current_pose["rpy_deg"]}
            set_object_poses(poses)

        required_cameras = {name for row in rows for name, _label in row}
        missing = required_cameras.difference(env.scene.sensors)
        if missing:
            raise ValueError(f"Missing operator cameras: {sorted(missing)}")
        if not args.headless:
            from cyclo_lab.runtime.viewers import CameraDashboard

            dashboard = CameraDashboard(
                env,
                rows=rows,
                panel_rotations=dict(getattr(cfg, "operator_camera_rotations", ())),
                window_title=getattr(cfg, "operator_camera_title", "Camera Dashboard"),
                window_size=getattr(cfg, "operator_camera_window_size", 1800),
                window_position=(20, 20),
            )
        env.reset()
        apply_plan_object_poses(plan)
        env._waypoint_initial_object_poses = initial_object_poses(env, plan)
        # Reset joint state is applied before the cameras publish their next
        # rendered frame. Hold the measured init pose for one camera period
        # while recording is disabled so HDF5 frame 0 starts after reset.
        current = robot.data.joint_pos[0, joint_ids].detach().clone()
        camera_hz = float(getattr(cfg, "camera_hz", plan["control_hz"]))
        camera_warmup_steps = max(1, math.ceil(plan["control_hz"] / camera_hz))
        with torch.inference_mode():
            for _ in range(camera_warmup_steps):
                env.step(current.unsqueeze(0))
        current = robot.data.joint_pos[0, joint_ids].detach().clone()
        if args.record_hdf5 and trajectory is None:
            env.recorder_manager.recording_enabled = True
            env.recorder_manager.record_post_reset(torch.arange(env.num_envs, device=env.device))
        if dashboard is not None:
            dashboard.update()
        initial_joint_pos = dict(zip(names, current.cpu().tolist()))
        camera_rotations = dict(getattr(cfg, "operator_camera_rotations", ()))
        handled_ik_id = None
        handled_preview_id = None
        handled_reset_id = None
        handled_object_id = None
        handled_stop_id = None

        def stop_request() -> dict | None:
            nonlocal handled_stop_id
            try:
                request = json.loads(STOP_REQUEST_FILE.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None
            request_id = request.get("id")
            if (
                not request_id or request_id == handled_stop_id
                or request.get("environment") != plan["environment"]
                or request.get("run_id") not in (None, run_id)
            ):
                return None
            handled_stop_id = request_id
            return request

        def process_ik_request() -> None:
            nonlocal handled_ik_id
            try:
                request = json.loads(IK_REQUEST_FILE.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return
            request_id = request.get("id")
            if (
                not request_id or request_id == handled_ik_id
                or request.get("run_id") != run_id
                or request.get("environment") != plan["environment"]
            ):
                return
            handled_ik_id = request_id
            try:
                result = estimate_q_des(
                    env,
                    request["side"],
                    request["xyz_m"],
                    solver=request.get("solver", "iterative_lm"),
                )
                result.update({"id": request_id, "status": "ok"})
            except (ValueError, KeyError, RuntimeError, TypeError, IndexError) as exc:
                result = {"id": request_id, "status": "error", "error": str(exc)}
            IK_RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
            temporary = IK_RESULT_FILE.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
            os.replace(temporary, IK_RESULT_FILE)

        def process_object_request() -> None:
            nonlocal handled_object_id, source_plan
            try:
                request = json.loads(OBJECT_REQUEST_FILE.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return
            request_id = request.get("id")
            if (
                not request_id or request_id == handled_object_id
                or request.get("run_id") != run_id
                or request.get("environment") != plan["environment"]
            ):
                return
            handled_object_id = request_id
            try:
                if teleop_draft is not None:
                    raise ValueError("Save or discard the teleop recording before moving an object.")
                object_name = request.get("object_name") or target_object_name
                current_pose = object_pose_snapshot(env).get(object_name)
                if current_pose is None:
                    raise ValueError(f"Unknown scene object: {object_name}")
                pose = {"xyz_m": request["xyz_m"], "rpy_deg": request.get("rpy_deg", current_pose["rpy_deg"])}
                set_object_poses({object_name: pose})
                plan.setdefault("object_poses", {})[object_name] = pose
                source_plan.setdefault("object_poses", {})[object_name] = pose
                env._waypoint_initial_object_poses[object_name] = pose
                if object_name == target_object_name:
                    plan["object_position_xyz_m"] = None
                result = {"id": request_id, "status": "ok", "object_name": object_name, **pose}
            except (ValueError, KeyError, RuntimeError, TypeError, IndexError) as exc:
                result = {"id": request_id, "status": "error", "error": str(exc)}
            OBJECT_RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
            temporary = OBJECT_RESULT_FILE.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
            os.replace(temporary, OBJECT_RESULT_FILE)

        available_cameras = write_camera_previews(env, camera_rotations)
        camera_frame_step = 0
        last_camera_time = time.monotonic()
        write_state(
            env, plan, 0, phase="reset", initial_joint_pos=initial_joint_pos,
            camera_frame_step=camera_frame_step, cameras=available_cameras, run_id=run_id,
            plan_name=args.plan.name,
        )
        name_to_index = {name: index for index, name in enumerate(names)}
        _, joint_limits = joint_contract(plan["environment"])
        global_step = 0
        deadline = time.monotonic()
        teleop_draft = None
        teleop_start = None
        teleop_last_time = 0.0
        handled_capture_id = None
        source_plan = plan.copy()

        def process_capture_request() -> None:
            nonlocal teleop_draft, teleop_start, teleop_last_time, handled_capture_id
            try:
                request = json.loads(TELEOP_CAPTURE_REQUEST_FILE.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return
            request_id = request.get("id")
            if (not request_id or request_id == handled_capture_id
                    or request.get("run_id") != run_id
                    or request.get("environment") != plan["environment"]):
                return
            handled_capture_id = request_id
            try:
                action = request.get("action")
                if action == "record":
                    if teleop_draft is not None:
                        raise ValueError("Teleop recording is already active.")
                    from teleop_trajectory import validate_recording_start
                    validate_recording_start(source_plan, initial_joint_pos, names, current.cpu().tolist())
                    teleop_start = time.monotonic()
                    teleop_last_time = 0.0
                    teleop_draft = CONTROL_DIR / f"teleop_{run_id}_{request_id}.jsonl"
                    header = {"schema_version": "teleop_waypoint_v1", "run_id": run_id,
                              "source_plan": source_plan, "joint_order": list(names),
                              "environment": plan["environment"]}
                    teleop_draft.write_text(
                        json.dumps(header) + "\n"
                        + json.dumps({"time_s": 0.0, "q_des": current.cpu().tolist()}) + "\n",
                        encoding="utf-8",
                    )
                    env._teleop_recording = True
                    result = {"id": request_id, "status": "ok", "action": action}
                elif action in ("save", "discard"):
                    if teleop_draft is None:
                        raise ValueError("No active teleop recording.")
                    draft = teleop_draft
                    if action == "save":
                        teleop_last_time = max(time.monotonic() - teleop_start, teleop_last_time + 1.0e-6)
                        with draft.open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps({"time_s": teleop_last_time, "q_des": current.cpu().tolist()}) + "\n")
                    else:
                        draft.unlink(missing_ok=True)
                    teleop_draft = None
                    teleop_start = None
                    env._teleop_recording = False
                    result = {"id": request_id, "status": "ok", "action": action,
                              "draft": draft.name if action == "save" else None}
                else:
                    raise ValueError("Unknown teleop recording action.")
            except (OSError, ValueError) as exc:
                result = {"id": request_id, "status": "error", "error": str(exc)}
            temporary = TELEOP_CAPTURE_RESULT_FILE.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(result) + "\n", encoding="utf-8")
            os.replace(temporary, TELEOP_CAPTURE_RESULT_FILE)

        def run_segment(
            start, target, steps: int, phase: str, status: str = "running", *,
            interpolation: str = "linear",
            active_plan: dict = plan, plan_name: str = args.plan.name,
        ) -> None:
            nonlocal global_step, deadline, last_camera_time, available_cameras, camera_frame_step, teleop_last_time
            apply_control_rate(env.cfg, active_plan["control_hz"])
            for segment_step in range(1, steps + 1):
                requested_stop = stop_request()
                if requested_stop is not None:
                    raise StopRequested(requested_stop)
                if not app.is_running():
                    raise RuntimeError("Isaac Sim closed before the plan completed.")
                fraction = segment_step / steps
                if interpolation == "smoothstep":
                    fraction = fraction * fraction * (3.0 - 2.0 * fraction)
                q_des = torch.lerp(start, target, fraction)
                if args.record_hdf5 and env.recorder_manager.recording_enabled:
                    timestamp_s = global_step / active_plan["control_hz"]
                    env.recorder_manager.add_to_episodes("obs/timestamp_s", torch.tensor([[timestamp_s]], dtype=torch.float64, device=env.device))
                    env.recorder_manager.add_to_episodes("obs/step_index", torch.tensor([[global_step]], dtype=torch.int64, device=env.device))
                capture_time = None
                if teleop_draft is not None and phase in ("leader_teleop", "leader_waiting", "final_hold"):
                    # Timestamp command application, not the end of a slow
                    # physics/render step, so replay preserves input timing.
                    capture_time = max(time.monotonic() - teleop_start, teleop_last_time + 1.0e-6)
                if args.record_hdf5:
                    _obs, _reward, terminated, truncated, _extras = env.step(q_des.unsqueeze(0))
                else:
                    # Keep native physics timing, but render live cameras only
                    # on the wall-clock preview schedule below. The temporary
                    # divisor is above every physics counter in this step.
                    render_interval = env.cfg.sim.render_interval
                    env.cfg.sim.render_interval = env._sim_step_counter + env.cfg.decimation + 1
                    try:
                        _obs, _reward, terminated, truncated, _extras = env.step(q_des.unsqueeze(0))
                    finally:
                        env.cfg.sim.render_interval = render_interval
                if bool(terminated.any().item()) or bool(truncated.any().item()):
                    raise RuntimeError(f"Environment terminated at step {global_step + 1}.")
                global_step += 1
                if capture_time is not None:
                    teleop_last_time = capture_time
                    with teleop_draft.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({"time_s": teleop_last_time, "q_des": q_des.cpu().tolist()}) + "\n")
                process_ik_request()
                process_object_request()
                if dashboard is not None:
                    dashboard.update()
                now = time.monotonic()
                if now - last_camera_time >= 1.0 / PREVIEW_HZ:
                    available_cameras = write_camera_previews(env, camera_rotations)
                    camera_frame_step = global_step
                    # Keep the wall-clock schedule instead of restarting the
                    # interval on each control step. Skip missed slots without
                    # queuing old frames or running catch-up captures.
                    elapsed_slots = max(1, math.floor((now - last_camera_time) * PREVIEW_HZ))
                    last_camera_time += elapsed_slots / PREVIEW_HZ
                write_state(
                    env, active_plan, global_step, q_des,
                    phase=phase,
                    initial_joint_pos=initial_joint_pos,
                    camera_frame_step=camera_frame_step,
                    cameras=available_cameras,
                    run_id=run_id,
                    plan_name=plan_name,
                    status=status,
                )
                deadline += 1.0 / active_plan["control_hz"]
                time.sleep(max(0.0, deadline - time.monotonic()))

        def process_preview_request() -> bool:
            nonlocal handled_preview_id, current, initial_joint_pos, deadline
            nonlocal available_cameras, camera_frame_step, last_camera_time
            nonlocal source_plan, teleop_draft, teleop_start
            try:
                request = json.loads(PREVIEW_REQUEST_FILE.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return False
            request_id = request.get("id")
            if (
                not request_id or request_id == handled_preview_id
                or request.get("run_id") != run_id
                or request.get("environment") != plan["environment"]
            ):
                return False
            handled_preview_id = request_id
            started_at = time.monotonic()
            try:
                if teleop_draft is not None:
                    raise ValueError("Save or discard the teleop recording before previewing.")
                preview_plan = validate_plan(request.get("plan"), [plan["environment"]])
                source_plan = preview_plan
                teleop_draft = None
                teleop_start = None
                if tuple(preview_plan["joint_order"]) != names:
                    raise ValueError("Preview joint order differs from the active SG2 contract.")
                plan["object_position_xyz_m"] = preview_plan.get("object_position_xyz_m")
                plan["object_poses"] = preview_plan.get("object_poses", {})
                env.reset()
                if getattr(env, "_live_camera_previews", None) is not None:
                    env._live_camera_previews.last_frames.clear()
                apply_plan_object_poses(preview_plan)
                env._waypoint_initial_object_poses = initial_object_poses(env, preview_plan)
                if dashboard is not None:
                    dashboard.update()
                current = robot.data.joint_pos[0, joint_ids].detach().clone()
                initial_joint_pos = dict(zip(names, current.cpu().tolist()))
                deadline = time.monotonic()
                last_camera_time = deadline
                available_cameras = write_camera_previews(env, camera_rotations)
                camera_frame_step += 1
                write_state(
                    env, preview_plan, global_step, current, phase="preview_reset",
                    initial_joint_pos=initial_joint_pos,
                    camera_frame_step=camera_frame_step, cameras=available_cameras,
                    run_id=run_id, plan_name="Unsaved UI preview",
                )
                for waypoint_index, waypoint in enumerate(preview_plan["waypoints"], start=1):
                    if not waypoint["enabled"]:
                        continue
                    target = current.clone()
                    for name, value in waypoint["q_des"].items():
                        target[name_to_index[name]] = value
                    run_segment(
                        current, target, waypoint["steps"], f"preview_point_{waypoint_index}",
                        interpolation=waypoint["interpolation"],
                        active_plan=preview_plan, plan_name="Unsaved UI preview",
                    )
                    current = target
                result = {
                    "id": request_id,
                    "status": "ok",
                    "steps": sum(waypoint["steps"] for waypoint in preview_plan["waypoints"]),
                    "duration_s": sum(waypoint["actual_duration_s"] for waypoint in preview_plan["waypoints"]),
                    "wall_time_s": time.monotonic() - started_at,
                    "sim_control_hz": 1.0 / float(env.step_dt),
                    "physics_hz": 1.0 / float(env.cfg.sim.dt),
                    "decimation": int(env.cfg.decimation),
                }
            except (ValueError, KeyError, RuntimeError, TypeError, IndexError) as exc:
                result = {"id": request_id, "status": "error", "error": str(exc)}
            PREVIEW_RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
            temporary = PREVIEW_RESULT_FILE.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
            os.replace(temporary, PREVIEW_RESULT_FILE)
            return True

        def process_reset_request() -> bool:
            nonlocal handled_reset_id, current, initial_joint_pos, deadline
            nonlocal available_cameras, camera_frame_step, last_camera_time
            nonlocal source_plan, teleop_draft, teleop_start
            try:
                request = json.loads(RESET_REQUEST_FILE.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return False
            request_id = request.get("id")
            if (
                not request_id or request_id == handled_reset_id
                or request.get("run_id") != run_id
                or request.get("environment") != plan["environment"]
            ):
                return False
            handled_reset_id = request_id
            try:
                if teleop_draft is not None:
                    raise ValueError("Save or discard the teleop recording before resetting.")
                env.reset()
                if getattr(env, "_live_camera_previews", None) is not None:
                    env._live_camera_previews.last_frames.clear()
                apply_plan_object_poses(plan)
                env._waypoint_initial_object_poses = initial_object_poses(env, plan)
                if dashboard is not None:
                    dashboard.update()
                current = robot.data.joint_pos[0, joint_ids].detach().clone()
                initial_joint_pos = dict(zip(names, current.cpu().tolist()))
                source_plan = {**plan, "waypoints": [{
                    "duration_s": 1.0 / plan["control_hz"], "q_des": initial_joint_pos,
                    "enabled": True, "interpolation": "linear", "memo": "reset hold",
                }]}
                teleop_draft = None
                teleop_start = None
                deadline = time.monotonic()
                last_camera_time = deadline
                available_cameras = write_camera_previews(env, camera_rotations)
                camera_frame_step += 1
                write_state(
                    env, plan, global_step, current, phase="final_hold",
                    initial_joint_pos=initial_joint_pos,
                    camera_frame_step=camera_frame_step, cameras=available_cameras,
                    run_id=run_id, plan_name="Reset to init pose",
                )
                result = {"id": request_id, "status": "ok", "initial_joint_pos": initial_joint_pos}
            except (ValueError, KeyError, RuntimeError, TypeError, IndexError) as exc:
                result = {"id": request_id, "status": "error", "error": str(exc)}
            RESET_RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
            temporary = RESET_RESULT_FILE.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
            os.replace(temporary, RESET_RESULT_FILE)
            return True

        try:
            with torch.inference_mode():
                if args.record_hdf5 and trajectory is None:
                    run_segment(current, current, initial_pose_hold_steps, "initial_pose_hold")
                if trajectory is not None and args.record_hdf5:
                    # Recreate the teleop start state without recording its setup motion.
                    for waypoint_index, waypoint in enumerate(plan["waypoints"], start=1):
                        if not waypoint["enabled"]:
                            continue
                        target = current.clone()
                        for name, value in waypoint["q_des"].items():
                            target[name_to_index[name]] = value
                        run_segment(current, target, waypoint["steps"], f"setup_{waypoint_index}", status="preparing", interpolation=waypoint["interpolation"], active_plan=setup_plan)
                        current = target
                    if max(abs(a - b) for a, b in zip(current.cpu().tolist(), trajectory["samples"][0]["q_des"])) > 0.05:
                        raise RuntimeError("Teleop start pose differs from replayed setup.")
                    global_step = 0
                    env.recorder_manager.recording_enabled = True
                    env.recorder_manager.record_post_reset(torch.arange(env.num_envs, device=env.device))
                    for command in replay_steps:
                        target = torch.tensor(command, device=env.device, dtype=current.dtype)
                        run_segment(current, target, 1, "teleop_replay")
                        current = target
                for waypoint_index, waypoint in enumerate(() if trajectory else plan["waypoints"], start=1):
                    if not waypoint["enabled"]:
                        continue
                    target = current.clone()
                    for name, value in waypoint["q_des"].items():
                        target[name_to_index[name]] = value
                    run_segment(current, target, waypoint["steps"], f"waypoint_{waypoint_index}", interpolation=waypoint["interpolation"])
                    current = target
                if args.record_hdf5:
                    env.recorder_manager.recording_enabled = False
                    env.recorder_manager.cfg.dataset_export_mode = DatasetExportMode.EXPORT_NONE
                    if args.auto_accept:
                        accepted = True
                        print("Playback complete. Finalizing the browser-requested HDF5 export.", flush=True)
                    else:
                        wait_until = time.monotonic() + args.accept_wait_seconds
                        print(f"Playback complete. Review the UI and approve HDF5 within {args.accept_wait_seconds:.0f} s.", flush=True)
                        while time.monotonic() < wait_until and app.is_running():
                            run_segment(current, current, 1, "awaiting_acceptance", status="awaiting_acceptance")
                            try:
                                request = json.loads(ACCEPT_REQUEST_FILE.read_text(encoding="utf-8"))
                            except (OSError, ValueError):
                                request = None
                            if request is not None and request.get("run_id") == run_id:
                                accepted = True
                                break
                    if accepted:
                        write_state(
                            env, plan, global_step, current, phase="finalizing",
                            initial_joint_pos=initial_joint_pos,
                            camera_frame_step=camera_frame_step, cameras=available_cameras,
                            run_id=run_id, plan_name=args.plan.name, status="finalizing",
                        )
                        for episode in getattr(env.recorder_manager, "_episodes", {}).values():
                            if episode is not None and not episode.is_empty():
                                episode.success = True
                                episode.metadata = {"operator_accepted": True, "run_id": run_id}
                        env.recorder_manager.cfg.dataset_export_mode = DatasetExportMode.EXPORT_ALL
                        env.recorder_manager.export_episodes(from_step=False)
                        if env.recorder_manager.exported_successful_episode_count < 1:
                            raise RuntimeError("Recorder did not export a successful HDF5 episode.")
                        completed = True
                    else:
                        print("HDF5 approval timed out; pending recording will be discarded.", flush=True)
                if (args.hold_final or args.ik_wait_seconds) and not args.record_hdf5:
                    wait_until = None if args.hold_final else time.monotonic() + args.ik_wait_seconds
                    active_teleop_id = None
                    active_reader_token = None
                    leader_origin = {}
                    sim_origin = current.clone()
                    while (wait_until is None or time.monotonic() < wait_until) and app.is_running():
                        requested_stop = stop_request()
                        if requested_stop is not None:
                            raise StopRequested(requested_stop)
                        process_capture_request()
                        if process_reset_request():
                            continue
                        if process_preview_request():
                            continue
                        try:
                            teleop = json.loads(LEADER_TELEOP_FILE.read_text(encoding="utf-8"))
                        except (OSError, ValueError):
                            teleop = {}
                        if (
                            teleop.get("enabled") and teleop.get("run_id") == run_id
                            and teleop.get("environment") == plan["environment"]
                            and teleop.get("side") in ("left", "right", "both")
                        ):
                            try:
                                live = json.loads(LEADER_LIVE_FILE.read_text(encoding="utf-8"))
                            except (OSError, ValueError):
                                live = {}
                            sides = ("left", "right") if teleop["side"] == "both" else (teleop["side"],)
                            samples = {side: live.get("values", {}).get(side, {}) for side in sides}
                            values = {name: value for sample in samples.values() for name, value in sample.get("q_des", {}).items()}
                            joint_names = [name for side in sides for name in
                                           ([f"arm_{side[0]}_joint{i}" for i in range(1, 8)] + [f"gripper_{side[0]}_joint1"])]
                            fresh = all(
                                round(time.time() * 1000) - sample.get("captured_at_ms", 0) <= 500
                                for sample in samples.values()
                            ) and all(name in values and math.isfinite(values[name]) for name in joint_names)
                            if fresh:
                                if active_teleop_id != teleop["id"] or active_reader_token != live.get("run_id"):
                                    active_teleop_id = teleop["id"]
                                    active_reader_token = live.get("run_id")
                                    leader_origin = {name: values[name] for name in joint_names}
                                    sim_origin = current.clone()
                                target = current.clone()
                                rate = float(teleop["control_hz"])
                                for name in joint_names:
                                    index = name_to_index[name]
                                    desired = float(sim_origin[index]) + values[name] - leader_origin[name]
                                    lower, upper = joint_limits[name]
                                    desired = max(lower, min(upper, desired))
                                    target[index] = desired
                                run_segment(current, target, 1, "leader_teleop", active_plan={**plan, "control_hz": rate})
                                current = target
                                continue
                            active_teleop_id = None
                            current = robot.data.joint_pos[0, joint_ids].detach().clone()
                            run_segment(current, current, 1, "leader_waiting")
                            continue
                        if active_teleop_id is not None:
                            current = robot.data.joint_pos[0, joint_ids].detach().clone()
                        active_teleop_id = None
                        run_segment(current, current, 1, "final_hold")
            print(f"Completed preview: {global_step} commands. HDF5 {'approved' if completed else 'not saved'}.")
        except StopRequested as exc:
            write_state(
                env, plan, global_step, current, phase="stopped",
                initial_joint_pos=initial_joint_pos,
                camera_frame_step=camera_frame_step, cameras=available_cameras,
                run_id=run_id, plan_name=args.plan.name, status="disconnected",
            )
            result = {
                "id": exc.request["id"], "status": "ok", "environment": plan["environment"],
                "run_id": run_id,
            }
            STOP_RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
            temporary = STOP_RESULT_FILE.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(result) + "\n", encoding="utf-8")
            os.replace(temporary, STOP_RESULT_FILE)
            print(f"Disconnected environment by UI request: {plan['environment']}", flush=True)
    except BaseException as exc:
        # SimulationApp.close() can suppress the Python traceback during Kit
        # shutdown, so publish the real startup/runtime error first.
        try:
            failure = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "environment": plan["environment"],
                "recording_mode": bool(args.record_hdf5),
                "run_id": run_id,
                "updated_at_ms": round(time.time() * 1000),
            }
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            temporary = STATE_FILE.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(failure, ensure_ascii=False) + "\n", encoding="utf-8")
            os.replace(temporary, STATE_FILE)
            print(f"Seed runner failed: {failure['error']}", flush=True)
        except Exception as state_exc:
            print(f"Seed runner failed: {type(exc).__name__}: {exc}; state update failed: {state_exc}", flush=True)
        raise
    finally:
        if env is not None and getattr(env, "_live_camera_previews", None) is not None:
            env._live_camera_previews.close()
        if "teleop_draft" in locals() and teleop_draft is not None:
            teleop_draft.unlink(missing_ok=True)
        if dashboard is not None:
            dashboard.close()
        if env is not None:
            env.close()
        if pending_file is not None:
            if completed:
                try:
                    if not pending_file.is_file():
                        raise RuntimeError(f"Recorder did not create HDF5 output: {pending_file}")
                    import h5py

                    expected_steps = len(replay_steps) if trajectory else initial_pose_hold_steps + sum(waypoint["steps"] for waypoint in plan["waypoints"])
                    with h5py.File(pending_file, "r") as dataset:
                        episode = dataset["data"]["demo_0"]
                        recorded_steps = int(episode.attrs["num_samples"])
                        action_steps = int(episode["actions"].shape[0])
                        if "FFW-BG2" in plan["environment"]:
                            head_shape = episode["obs"]["cam_head"].shape
                            if head_shape[1:3] != (720, 1280):
                                raise RuntimeError(f"BG2 seed HDF5 head camera must be 720x1280, got {head_shape}.")
                        if trajectory:
                            timestamps = episode["obs"]["timestamp_s"][:].reshape(-1)
                            if len(timestamps) != expected_steps or any(
                                abs(float(value) - index / 15.0) > 1.0e-6
                                for index, value in enumerate(timestamps)
                            ):
                                raise RuntimeError("Teleop HDF5 timestamps are not 15 Hz.")
                            cameras = {
                                name: int(episode["obs"][name].shape[0])
                                for name in ("cam_head", "cam_wrist_left", "cam_wrist_right")
                            }
                            if any(cameras.get(name) != expected_steps for name in
                                   ("cam_head", "cam_wrist_left", "cam_wrist_right")):
                                raise RuntimeError(f"Camera frame counts differ from {expected_steps} steps: {cameras}")
                    if recorded_steps != expected_steps or action_steps != expected_steps:
                        raise RuntimeError(
                            f"HDF5 action length mismatch: expected {expected_steps}, "
                            f"episode {recorded_steps}, actions {action_steps}"
                        )
                    os.link(pending_file, dataset_file)
                    pending_file.unlink()
                except Exception as exc:
                    state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
                    state.update({"status": "failed", "error": str(exc), "pending_file": str(pending_file), "updated_at_ms": round(time.time() * 1000)})
                    temporary = STATE_FILE.with_suffix(".json.tmp")
                    temporary.write_text(json.dumps(state, ensure_ascii=False) + "\n", encoding="utf-8")
                    os.replace(temporary, STATE_FILE)
                    app.close()
                    raise
                state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
                state.update({"status": "saved", "dataset_file": str(dataset_file), "updated_at_ms": round(time.time() * 1000)})
                temporary = STATE_FILE.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(state, ensure_ascii=False) + "\n", encoding="utf-8")
                os.replace(temporary, STATE_FILE)
                print(f"Saved HDF5 seed: {dataset_file}", flush=True)
            elif pending_file.is_file():
                pending_file.unlink()
                if STATE_FILE.is_file():
                    state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
                    if state.get("run_id") == run_id and state.get("status") == "awaiting_acceptance":
                        state.update({"status": "discarded", "updated_at_ms": round(time.time() * 1000)})
                        temporary = STATE_FILE.with_suffix(".json.tmp")
                        temporary.write_text(json.dumps(state, ensure_ascii=False) + "\n", encoding="utf-8")
                        os.replace(temporary, STATE_FILE)
        # Kit shutdown may end the Python process, so finalize files first.
        app.close()


if __name__ == "__main__":
    main()
