"""Render one recorded Task000901 Mimic episode by replaying saved simulator states.

This is a visual audit, not a second physics rollout.  The script writes the
recorded robot and plate states into an isolated simulator and captures a fixed
review camera.  It never connects to ROS, Zenoh, or a physical robot.
"""
import argparse
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", type=Path, required=True)
parser.add_argument("--episode", default="demo_0")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--label", default="Task000901 MimicGen saved-state replay")
parser.add_argument("--stride", type=int, default=2)
from isaaclab.app import AppLauncher

args = parser.parse_args()
if args.stride < 1:
    parser.error("--stride must be positive")
if args.output.exists():
    parser.error(f"Refusing to overwrite {args.output}")
app = AppLauncher(headless=True, enable_cameras=True).app

env = None
writer = None
failed = False
try:
    import cv2
    import gymnasium as gym
    import imageio_ffmpeg
    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import Camera, CameraCfg
    from isaaclab.utils.datasets import HDF5DatasetFileHandler
    from isaaclab_tasks.utils import parse_env_cfg

    imitation_root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(imitation_root))
    from local_plugins import load_for_task

    task = "Cyclo-Real-WhiteRoom-Task000901-Bimanual-Mimic-Generate-FFW-SG2-v0"
    load_for_task(task).register_environments()
    cfg = parse_env_cfg(task, device="cuda:0", num_envs=1)
    cfg.plate_xy_half_range_m = 0.0
    cfg.recorders = {}
    cfg.terminations = {}
    env = gym.make(task, cfg=cfg).unwrapped
    env.reset()

    preview = Camera(CameraCfg(
        prim_path="/World/Task901MimicReviewCamera",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=30,
            horizontal_aperture=36,
            clipping_range=(0.02, 100.0),
        ),
    ))
    env.sim.reset()
    env.reset()
    preview.set_world_poses_from_view(
        torch.tensor([[-0.35, -0.25, 2.00]], device=env.device),
        torch.tensor([[-0.53, -0.04, 0.74]], device=env.device),
    )
    for _ in range(20):
        env.sim.render()

    handler = HDF5DatasetFileHandler()
    handler.open(str(args.dataset))
    names = list(handler.get_episode_names())
    if args.episode not in names:
        raise KeyError(f"Episode {args.episode!r} not in {names}")
    episode = handler.load_episode(args.episode, env.device)
    frame_count = int(episode.data["actions"].shape[0])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    writer = subprocess.Popen([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x720",
        "-r", "15", "-i", "-", "-an", "-c:v", "libx264",
        "-threads", "4", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output),
    ], stdin=subprocess.PIPE)

    env_ids = torch.tensor([0], device=env.device, dtype=torch.long)
    rendered = 0
    for index in range(0, frame_count, args.stride):
        state = episode.get_state(index)
        env.scene.reset_to(state, env_ids=env_ids, is_relative=True)
        env.sim.forward()
        env.scene.update(0.0)
        env.sim.render()
        preview.update(1 / 15, force_recompute=True)
        rgb = preview.data.output["rgb"][0, :, :, :3].cpu().numpy().copy()
        if rgb.shape != (720, 1280, 3) or rgb.std() < 3:
            raise ValueError(f"Invalid rendered frame {index}: {rgb.shape}, std={rgb.std()}")
        cv2.rectangle(rgb, (0, 0), (1280, 68), (20, 24, 31), -1)
        cv2.putText(rgb, args.label, (24, 29), cv2.FONT_HERSHEY_SIMPLEX,
                    0.72, (245, 247, 250), 2, cv2.LINE_AA)
        cv2.putText(rgb, f"saved state {index:04d}/{frame_count - 1:04d} | {args.stride}x temporal preview",
                    (24, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (176, 205, 235), 1, cv2.LINE_AA)
        writer.stdin.write(rgb.tobytes())
        rendered += 1
        if rendered % 100 == 0:
            print(f"RENDER_PROGRESS {rendered}", flush=True)
    writer.stdin.close()
    return_code = writer.wait()
    writer = None
    if return_code != 0:
        raise RuntimeError(f"ffmpeg exited {return_code}")
    print(f"RENDER_DONE frames={rendered} output={args.output}", flush=True)
except BaseException:
    failed = True
    import traceback
    traceback.print_exc()
finally:
    if writer is not None:
        if writer.stdin:
            writer.stdin.close()
        writer.wait()
    if env is not None:
        env.close()
    app.close(wait_for_replicator=False, skip_cleanup=True)
sys.stdout.flush()
sys.stderr.flush()
raise SystemExit(1 if failed else 0)
