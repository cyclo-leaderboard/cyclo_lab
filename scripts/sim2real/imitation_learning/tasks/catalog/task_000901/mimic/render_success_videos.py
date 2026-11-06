"""Render Task901 successful Mimic episodes through head and wrist cameras.

The replay restores the recorded initial/post-action states without advancing
physics.  It therefore visualizes exactly the accepted trajectory while keeping
the generation result unchanged.  Visual plate colors are permuted per episode;
all plate physics are color-independent and remain tied to spatial slot identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--audit", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=2)
parser.add_argument("--limit-episodes", type=int, default=None)
parser.add_argument("--limit-frames", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import h5py  # noqa: E402
import imageio_ffmpeg  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from pxr import Gf, Usd  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


REPO = Path("/workspace/cyclo_lab")
IMITATION_ROOT = REPO / "scripts/sim2real/imitation_learning"
sys.path.insert(0, str(IMITATION_ROOT))
from local_plugins import load_for_task  # noqa: E402


CAMERAS = ("cam_head", "cam_wrist_left", "cam_wrist_right")
CAMERA_SHAPES = {"cam_head": (720, 1280), "cam_wrist_left": (640, 480), "cam_wrist_right": (640, 480)}
SLOT_OBJECTS = ("plate_blue", "plate_yellow", "plate_pink")
SLOT_PRIMS = {"lower_left": "PlateBlue", "lower_right": "PlateYellow", "upper_center": "PlatePink"}
COLOR_RGB = {
    "blue": (0.40, 0.95, 1.00),
    "yellow": (1.00, 1.00, 0.22),
    "pink": (0.94, 0.72, 1.00),
}
FPS = 15
LABEL_H = 52
CANVAS_W = 1280
CANVAS_H = LABEL_H + 720 + LABEL_H + 640


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nested(target: dict, path: str, value: torch.Tensor) -> None:
    keys = path.split("/")
    for key in keys[:-1]:
        target = target.setdefault(key, {})
    target[keys[-1]] = value


def load_trajectory(group: h5py.Group) -> dict[str, np.ndarray]:
    paths = []
    group["initial_state"].visititems(
        lambda name, obj: paths.append(name) if isinstance(obj, h5py.Dataset) else None
    )
    return {
        path: np.concatenate((group[f"initial_state/{path}"][:], group[f"states/{path}"][:]), axis=0)
        for path in paths
    }


def state_at(trajectories: list[dict[str, np.ndarray]], step: int, device: str) -> dict:
    state = {}
    for path in trajectories[0]:
        values = np.stack([trajectory[path][min(step, len(trajectory[path]) - 1)] for trajectory in trajectories])
        nested(state, path, torch.as_tensor(values, device=device))
    return state


def find_color_attribute(stage, env_index: int, prim_name: str):
    root = stage.GetPrimAtPath(f"/World/envs/env_{env_index}/{prim_name}")
    matches = []
    for prim in Usd.PrimRange(root):
        attr = prim.GetAttribute("inputs:diffuseColor")
        if attr and attr.Get() is not None:
            matches.append(attr)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one color shader under {root.GetPath()}, got {len(matches)}")
    return matches[0]


def apply_slot_colors(env, env_index: int, slot_colors: dict[str, str]) -> None:
    for slot, prim_name in SLOT_PRIMS.items():
        color = COLOR_RGB[slot_colors[slot]]
        find_color_attribute(env.sim.stage, env_index, prim_name).Set(Gf.Vec3f(*color))


def capture(env, env_ids: torch.Tensor) -> dict[str, np.ndarray]:
    env.scene.write_data_to_sim()
    for name in CAMERAS:
        env.scene[name].reset(env_ids)
    env.sim.render()
    frames = {}
    for name in CAMERAS:
        camera = env.scene[name]
        camera.update(env.sim.get_physics_dt(), force_recompute=True)
        rgb = camera.data.output["rgb"][..., :3].detach().cpu().numpy()
        if rgb.dtype != np.uint8:
            if rgb.size and float(rgb.max()) <= 1.0:
                rgb *= 255.0
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        if rgb.shape[1:] != (*CAMERA_SHAPES[name], 3):
            raise RuntimeError(f"{name} unexpected RGB shape: {rgb.shape}")
        frames[name] = np.ascontiguousarray(rgb)
    return frames


def compose(frames: dict[str, np.ndarray], env_index: int, title: str, step: int, length: int) -> np.ndarray:
    canvas = np.full((CANVAS_H, CANVAS_W, 3), 20, np.uint8)
    head = frames["cam_head"][env_index]
    left = frames["cam_wrist_left"][env_index]
    right = frames["cam_wrist_right"][env_index]
    canvas[LABEL_H:LABEL_H + 720] = head
    wrist_y = LABEL_H + 720 + LABEL_H
    canvas[wrist_y:wrist_y + 640, 80:560] = left
    canvas[wrist_y:wrist_y + 640, 720:1200] = right
    cv2.putText(canvas, f"{title}  frame {step + 1}/{length}", (16, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(canvas, "HEAD", (16, LABEL_H + 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(canvas, "WRIST LEFT", (80, LABEL_H + 720 + 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(canvas, "WRIST RIGHT", (720, LABEL_H + 720 + 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (245, 245, 245), 2, cv2.LINE_AA)
    return canvas


def count_decoded(path: Path) -> tuple[int, float, str]:
    capture_handle = cv2.VideoCapture(str(path))
    count = 0
    while True:
        ok, frame = capture_handle.read()
        if not ok:
            break
        if frame.shape[:2] != (CANVAS_H, CANVAS_W):
            raise RuntimeError(f"Bad frame shape in {path}: {frame.shape}")
        count += 1
    fps = float(capture_handle.get(cv2.CAP_PROP_FPS))
    codec = int(capture_handle.get(cv2.CAP_PROP_FOURCC))
    fourcc = bytes((codec >> shift) & 255 for shift in (0, 8, 16, 24)).decode("ascii")
    capture_handle.release()
    return count, fps, fourcc


def run() -> None:
    audit = json.loads(args.audit.read_text())
    success_rows = [row for row in audit["episodes"] if row["outcome"] == "success"]
    if args.limit_episodes is not None:
        success_rows = success_rows[:args.limit_episodes]
    if not success_rows:
        raise ValueError("No successful episodes to render")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)

    with h5py.File(args.input, "r") as source:
        task = json.loads(source["data"].attrs["env_args"])["env_name"]
    plugin = load_for_task(task)
    plugin.register_environments()
    from _cyclo_task000901_scene import make_task000901_head_camera_cfg, make_task000901_wrist_camera_cfg

    env_count = min(args.num_envs, len(success_rows))
    cfg = parse_env_cfg(task, device=args.device, num_envs=env_count)
    cfg.scene.replicate_physics = False
    cfg.scene.cam_head = make_task000901_head_camera_cfg(calibrated=True)
    cfg.scene.cam_wrist_left = make_task000901_wrist_camera_cfg("left", calibrated=True)
    cfg.scene.cam_wrist_right = make_task000901_wrist_camera_cfg("right", calibrated=True)
    cfg.events = None
    cfg.recorders = None
    cfg.rerender_on_reset = False
    env = gym.make(task, cfg=cfg).unwrapped
    env_ids = torch.arange(env_count, device=env.device, dtype=torch.long)
    env.reset()
    manifest = {
        "schema": "cyclo.task901.success_camera_replay.v1",
        "source": str(args.input),
        "source_sha256": sha256(args.input),
        "audit": str(args.audit),
        "audit_sha256": sha256(args.audit),
        "fps": FPS,
        "cameras": list(CAMERAS),
        "canvas_wh": [CANVAS_W, CANVAS_H],
        "render_method": "saved initial and every post-action state; no physics stepping",
        "color_method": "per-episode shader color permutation on identical plate physics",
        "episodes": [],
    }
    try:
        with h5py.File(args.input, "r") as source:
            for batch_start in range(0, len(success_rows), env_count):
                batch = success_rows[batch_start:batch_start + env_count]
                padded = batch + [batch[-1]] * (env_count - len(batch))
                groups = [source[f"data/{row['demo']}"] for row in padded]
                trajectories = [load_trajectory(group) for group in groups]
                full_lengths = [len(group["actions"]) + 1 for group in groups]
                lengths = [min(length, args.limit_frames or length) for length in full_lengths]
                for local_index, row in enumerate(padded):
                    apply_slot_colors(env, local_index, row["visual_slot_colors"])
                env.scene.reset_to(state_at(trajectories, 0, env.device), env_ids, is_relative=True)
                env.sim.forward()
                for _ in range(16 if batch_start == 0 else 4):
                    first_frames = capture(env, env_ids)
                writers = []
                paths = []
                for row in batch:
                    path = args.output / f"success_{row['demo']}_head_wrist.mp4"
                    writer = imageio_ffmpeg.write_frames(
                        str(path), (CANVAS_W, CANVAS_H), fps=FPS, codec="libx264",
                        pix_fmt_in="rgb24", pix_fmt_out="yuv420p", quality=7,
                        macro_block_size=2, ffmpeg_log_level="error",
                        output_params=["-movflags", "+faststart"],
                    )
                    writer.send(None)
                    writers.append(writer)
                    paths.append(path)
                try:
                    for step in range(max(lengths[:len(batch)])):
                        if step:
                            env.scene.reset_to(state_at(trajectories, step, env.device), env_ids, is_relative=True)
                            env.sim.forward()
                        frames = first_frames if step == 0 else capture(env, env_ids)
                        for local_index, row in enumerate(batch):
                            if step < lengths[local_index]:
                                colors = row["visual_slot_colors"]
                                title = (
                                    f"{row['demo']} | lower-left {colors['lower_left']} -> "
                                    f"lower-right {colors['lower_right']} -> upper-center {colors['upper_center']}"
                                )
                                writers[local_index].send(
                                    compose(frames, local_index, title, step, lengths[local_index])
                                )
                finally:
                    for writer in writers:
                        writer.close()
                for local_index, (row, path) in enumerate(zip(batch, paths)):
                    decoded, fps, codec = count_decoded(path)
                    if decoded != lengths[local_index] or abs(fps - FPS) > 0.01:
                        raise RuntimeError(f"Video validation failed for {path}: {decoded}, {fps}, {codec}")
                    preview = args.output / f"success_{row['demo']}_first_frame.png"
                    cv2.imwrite(str(preview), cv2.cvtColor(
                        compose(first_frames, local_index, row["demo"], 0, lengths[local_index]),
                        cv2.COLOR_RGB2BGR,
                    ))
                    manifest["episodes"].append({
                        "demo": row["demo"],
                        "slot_colors": row["visual_slot_colors"],
                        "frames": decoded,
                        "full_source_frames": full_lengths[local_index],
                        "fps": fps,
                        "codec": codec,
                        "video": path.name,
                        "video_sha256": sha256(path),
                        "first_frame": preview.name,
                        "visibility_evidence": (
                            "Use the composite first-frame PNG and decoded video; no geometric pixel projection "
                            "is asserted because Isaac camera-frame conventions differ by sensor backend."
                        ),
                    })
                    print(f"TASK901_RENDER {row['demo']} {decoded} frames {path}", flush=True)
                (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        run()
    finally:
        simulation_app.close()
