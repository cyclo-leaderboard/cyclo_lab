"""Build a Task901 Mimic seed HDF5 from one manually approved replay episode.

The selected MCAP is used only to identify and verify an already completed
Test4 rigid-contact physics replay by default.  Actions in the seed are EEF targets derived from
that measured-state PD replay; they are not relabeled as original teleop
commands.  The command refuses to mark a seed successful unless the caller
passes ``--approved-success`` after reviewing the comparison video.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import h5py
import numpy as np

from prepare_reference import PHASES, ROOT_POS, ROOT_ROT, TCP_Z, quat_matrix, world_to_root_pose


HOME = Path("/home/robotis-ai")
REPLAY_ROOT = HOME / "agents/experiments/2026-11-03_09-56-43_task000901_real_sim_replay"
AUTOMATED = REPLAY_ROOT / "automated"
MIMIC_SEEDS = REPLAY_ROOT / "mimic_seeds"
WORKSPACE = HOME / "cyclo_intelligence/docker/workspace"
SEED_ID = "Cyclo-Real-WhiteRoom-Task000901-Bimanual-Mimic-Seed-FFW-SG2-v0"
DEFAULT_VARIANT = "test4"
CANONICAL_END_S = 57.6
CANONICAL_PHASE_CUTS_S = [float(p[2]) for p in PHASES[:-1]]
VARIANTS = {
    "test4": "Test4_Rigid_Task_000901_Insert_Plate_Real2Sim_MCAP",
    "test5": "Test5_RubberLike_Task_000901_Insert_Plate_Real2Sim_MCAP",
}
CONTACT_SCRIPTS = {
    "test4": Path(__file__).resolve().parent.parent / "test4_contact.py",
    "test5": REPLAY_ROOT / "automation/variants/compliant_contact_1mm_f2p0_1p4_mimic100/contact_candidate.py",
}
HOLDER_POSE_AT_TABLE_070 = np.array(
    [-.5464056504055486, .15655533467208427, .7002,
     .7006398248933365, 0., 0., .7135151265204086], dtype=float
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_episode(raw: str) -> tuple[Path, int, Path]:
    path = Path(raw).expanduser()
    if path.is_file():
        path = path.parent
    if path.is_absolute() and str(path).startswith("/workspace/"):
        path = WORKSPACE / path.relative_to("/workspace")
    path = path.resolve()
    if not path.is_dir() or not path.name.isdigit():
        raise ValueError(f"Expected a numbered MCAP episode directory, got {path}")
    episode = int(path.name)
    mcap = path / f"{episode}_0.mcap"
    if not mcap.is_file():
        raise FileNotFoundError(mcap)
    return path, episode, mcap


def completed_run(episode_dir: Path, episode: int, variant: str) -> Path:
    root = AUTOMATED / VARIANTS[variant] / f"episode_{episode:03d}"
    candidates = []
    for run in root.glob("run_*"):
        status_path = run / "status.json"
        provenance_path = run / "automation_provenance.json"
        if not status_path.is_file() or not provenance_path.is_file():
            continue
        status = json.loads(status_path.read_text())
        if status.get("status") != "complete":
            continue
        if Path(status.get("source", "")).resolve() != episode_dir:
            continue
        video = Path(status.get("video", ""))
        if not video.is_file():
            continue
        candidates.append(run)
    if not candidates:
        raise FileNotFoundError(
            f"No completed {variant} replay for episode {episode}: {root}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def parse_phase_cuts(value: str | None, time_end: float) -> tuple[list[float], str]:
    if value:
        cuts = [float(item.strip()) for item in value.split(",") if item.strip()]
        method = "caller supplied seconds"
    else:
        scale = time_end / CANONICAL_END_S
        cuts = [value * scale for value in CANONICAL_PHASE_CUTS_S]
        method = f"episode6 canonical cuts scaled by trajectory duration ({scale:.9g}x)"
    if len(cuts) != len(PHASES) - 1:
        raise ValueError(f"Expected {len(PHASES)-1} phase cuts, got {len(cuts)}")
    if cuts != sorted(cuts) or cuts[0] <= 0 or cuts[-1] >= time_end:
        raise ValueError(f"Invalid phase cuts for time range 0..{time_end}: {cuts}")
    return cuts, method


def build_seed(episode_dir: Path, mcap: Path, run: Path, variant: str,
               output: Path, phase_cuts_text: str | None) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    inp, render = run / "input", run / "render"
    status = json.loads((run / "status.json").read_text())
    source_camera_video = Path(status["video"])
    if not source_camera_video.is_file():
        raise FileNotFoundError(source_camera_video)
    required = [
        inp / "aligned_replay.npz", inp / "input_manifest.json",
        inp / "scene_snapshot.py", render / "physics_states.npz",
        render / "gripper_kinematics.npz", render / "physics_result.json",
        render / "all_plates_hold_3s.npy", render / "contact_candidate_audit.json",
        render / "tip_contact_audit.json", render / "gripper_friction_audit.json",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing replay files:\n  " + "\n  ".join(map(str, missing)))

    manifest = json.loads((inp / "input_manifest.json").read_text())
    result = json.loads((render / "physics_result.json").read_text())
    if int(manifest["episode_index"]) != int(episode_dir.name):
        raise ValueError("Replay episode does not match selected MCAP")
    if manifest["mcap_sha256"] != sha256(mcap):
        raise ValueError("Replay MCAP fingerprint does not match selected MCAP")
    if result.get("object_pose_writes_after_initialization") != 0:
        raise ValueError("Replay wrote object poses after initialization")

    data = np.load(inp / "aligned_replay.npz")
    physics = np.load(render / "physics_states.npz")
    kin = np.load(render / "gripper_kinematics.npz")
    count = len(data["time_sec"])
    if len(kin["time_sec"]) != count or not np.allclose(kin["time_sec"], data["time_sec"]):
        raise ValueError("Replay and gripper kinematic clocks do not match")
    if len(physics["object_states"]) != count:
        raise ValueError("Replay object-state clock does not match aligned replay")

    extra = 45  # terminal 3 s at 15 Hz
    def extend(array):
        return np.concatenate((array, np.repeat(array[-1:], extra, axis=0)))

    eef, eef_pose7 = {}, {}
    for side in ("left", "right"):
        index = kin["body_names"].tolist().index(f"arm_{side[0]}_link7")
        world = kin["body_pose_xyzw"][:, index].copy()
        world[:, 3:7] = world[:, [6, 3, 4, 5]]
        world[:, :3] += quat_matrix(world[:, 3:7]) @ np.array([0., 0., TCP_Z])
        eef[side + "_arm"] = extend(world_to_root_pose(world))
        root_quat = world[:, [6, 5, 4, 3]] * np.array([-1., -1., 1., 1.])
        eef_pose7[side] = extend(
            np.concatenate(((world[:, :3] - ROOT_POS) @ ROOT_ROT, root_quat), axis=1)
        )
    target = {name: np.concatenate((poses[1:], poses[-1:])) for name, poses in eef.items()}

    names = data["joint_names"].tolist()
    ordered = (
        [f"arm_l_joint{i}" for i in range(1, 8)] + ["gripper_l_joint1"]
        + [f"arm_r_joint{i}" for i in range(1, 8)]
        + ["gripper_r_joint1", "head_joint1", "head_joint2", "lift_joint"]
    )
    absent = [name for name in ordered if name not in names]
    if absent:
        raise ValueError(f"Missing required measured joints: {absent}")
    joint_targets = extend(
        data["reference_position"][:, [names.index(name) for name in ordered]]
    ).astype(np.float32)
    actions = joint_targets.copy()
    for side, span in (("left", slice(0, 7)), ("right", slice(8, 15))):
        poses = eef_pose7[side]
        actions[:, span] = np.concatenate((poses[1:], poses[-1:]))

    object_poses = {
        str(name): extend(world_to_root_pose(physics["object_states"][:, index, :7]))
        for index, name in enumerate(physics["plate_names"])
    }
    table_offset = float(manifest.get("table_height_offset_m", 0.0))
    holder_pose = HOLDER_POSE_AT_TABLE_070.copy()
    holder_pose[2] += table_offset
    object_poses["plate_holder"] = world_to_root_pose(
        np.repeat(holder_pose[None], count + extra, axis=0)
    )

    cut_seconds, cut_method = parse_phase_cuts(phase_cuts_text, float(data["time_sec"][-1]))
    cut_indices = [int(np.searchsorted(data["time_sec"], seconds)) for seconds in cut_seconds]

    output.mkdir(parents=True)
    shutil.copytree(inp, output / "input")
    (output / "render").mkdir()
    for path in required:
        if path.parent == render:
            shutil.copy2(path, output / "render" / path.name)
    shutil.copy2(CONTACT_SCRIPTS[variant], output / "contact_candidate.py")
    camera_video = output / "camera_real_sim.mp4"
    shutil.copy2(source_camera_video, camera_video)

    hdf5_path = output / "seed.hdf5"
    with h5py.File(hdf5_path, "x") as file:
        group = file.create_group("data")
        group.attrs["env_args"] = json.dumps({"env_name": SEED_ID, "type": 2, "env_kwargs": {}})
        group.attrs["total"] = count + extra
        group.attrs["action_representation"] = "eef19_wxyz"
        group.attrs["source_action_provenance"] = (
            "EEF targets derived from measured-position PD replay; not original teleop actions"
        )
        demo = group.create_group("demo_0")
        demo.attrs["num_samples"] = count + extra
        demo.attrs["success"] = True
        demo.attrs["success_approval"] = "manual --approved-success CLI attestation"
        demo.create_dataset("actions", data=actions)
        demo.create_dataset("source_joint_targets", data=joint_targets)
        info = demo.create_group("obs/datagen_info")
        for name in eef:
            info.create_dataset("eef_pose/" + name, data=eef[name])
            info.create_dataset("target_eef_pose/" + name, data=target[name])
        for name, poses in object_poses.items():
            info.create_dataset("object_pose/" + name, data=poses)
        for phase, cut in zip(PHASES[:-1], cut_indices):
            info.create_dataset(
                "subtask_term_signals/" + phase[0],
                data=(np.arange(count + extra) >= cut - 1).astype(np.int32),
            )

    with h5py.File(hdf5_path, "r") as file:
        demo = file["data/demo_0"]
        if demo["actions"].shape != (count + extra, 19):
            raise ValueError(f"Unexpected action shape: {demo['actions'].shape}")
        if not bool(demo.attrs["success"]):
            raise ValueError("Seed was not marked successful")

    contact = json.loads((render / "contact_candidate_audit.json").read_text())
    tip = json.loads((render / "tip_contact_audit.json").read_text())
    friction = json.loads((render / "gripper_friction_audit.json").read_text())
    provenance = {
        "schema": "task901.selected-success-mimic-seed.v1",
        "source_episode_host": str(episode_dir),
        "source_run": str(run),
        "source_variant": variant,
        "source_video": str(source_camera_video),
        "camera_video": str(camera_video),
        "camera_video_sha256": sha256(camera_video),
        "manual_success_approved": True,
        "mcap_sha256": manifest["mcap_sha256"],
        "source_hashes": {str(path.relative_to(run)): sha256(path) for path in required},
        "original_teleop_actions_used": False,
        "action_provenance": "measured-state PD replay converted to EEF19 targets",
        "control_hz": 15,
        "physics_hz": float(result["physics_hz"]),
        "source_robot_sha256": result["robot_usd_sha256"],
        "table_height_offset_m": table_offset,
        "source_frames": count,
        "seed_frames": count + extra,
        "phase_cut_method": cut_method,
        "phase_cuts_seconds": cut_seconds,
        "phase_cuts_indices": cut_indices,
        "phase_review_required_before_generation": True,
        "physics_profile": {
            "contact_mode": contact["mode"],
            "tip_recess_m": contact["recess_max_m"],
            "contact_stiffness_N_per_m": contact["stiffness_N_per_m"],
            "contact_damping_Ns_per_m": contact["damping_Ns_per_m"],
            "tip_static_friction": friction["static_friction"],
            "tip_dynamic_friction": friction["dynamic_friction"],
            "plate_static_friction": 0.6,
            "plate_dynamic_friction": 0.45,
            "use_source_scene_holder_material": True,
            "tip_contact": {key: tip.get(key) for key in (
                "opposing_tip_filtered_pairs_retained", "joint24_upper_limit_rad",
                "joint24_mimic_mode", "joint24_mimic_natural_frequency",
                "joint24_mimic_damping_ratio", "robot_solver_velocity_iterations",
            )},
        },
        "expected_slot_xyz": {
            str(name): physics["object_states"][-1, index, :3].tolist()
            for index, name in enumerate(physics["plate_names"])
        },
        "hdf5": str(hdf5_path),
        "hdf5_sha256": sha256(hdf5_path),
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_path", help="MCAP episode directory or *_0.mcap")
    parser.add_argument(
        "--variant", choices=sorted(VARIANTS), default=DEFAULT_VARIANT,
        help=f"replay physics profile to preserve (default: {DEFAULT_VARIANT})",
    )
    parser.add_argument(
        "--approved-success", action="store_true",
        help="Required manual attestation that the selected real/sim replay succeeded",
    )
    parser.add_argument(
        "--phase-cuts",
        help="Optional eight comma-separated cut times in simulation seconds",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.approved_success and not args.dry_run:
        parser.error("Refusing to create a successful seed without --approved-success")

    episode_dir, episode, mcap = resolve_episode(args.episode_path)
    run = completed_run(episode_dir, episode, args.variant)
    output = args.output or MIMIC_SEEDS / args.variant / f"episode_{episode:03d}" / run.name
    summary = {
        "episode": episode,
        "variant": args.variant,
        "mcap": str(mcap),
        "completed_replay": str(run),
        "output": str(output),
    }
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        print("DRY_RUN_OK")
        return 0
    provenance = build_seed(episode_dir, mcap, run, args.variant, output, args.phase_cuts)
    print(json.dumps(provenance, indent=2))
    print(f"HDF5_READY {provenance['hdf5']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
