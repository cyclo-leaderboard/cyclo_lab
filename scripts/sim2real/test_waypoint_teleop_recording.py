"""Regression checks for timestamped teleop recording without launching Isaac."""

from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

UI_ROOT = Path(__file__).resolve().parent / "imitation_learning/waypoint_seed_ui"
sys.path.insert(0, str(UI_ROOT))
sys.path.insert(0, str(UI_ROOT.parent))
from teleop_trajectory import resample_15hz, validate_recording_start, validate_trajectory
from preview_video import is_complete_mp4


def trajectory():
    return {"schema_version": "teleop_waypoint_v1", "environment": "test",
            "source_plan": {"environment": "test"}, "joint_order": ["joint"],
            "samples": [{"time_s": 0.0, "q_des": [0.0]},
                        {"time_s": 0.1, "q_des": [1.0]},
                        {"time_s": 0.3, "q_des": [1.0]}]}


class TrajectoryTest(unittest.TestCase):
    def test_irregular_samples_preserve_motion_and_hold(self):
        raw = validate_trajectory(trajectory(), ["joint"])
        samples = list(resample_15hz(raw["samples"]))
        self.assertEqual(len(samples), 5)
        self.assertAlmostEqual(samples[1][0], 2 / 3)
        self.assertEqual(samples[2:], [[1.0], [1.0], [1.0]])

    def test_invalid_time_and_joint_values(self):
        for field, value in (("time_s", True), ("time_s", float("nan")),
                             ("time_s", 0), ("time_s", 3601),
                             ("q_des", [True]), ("q_des", [float("inf")]),
                             ("q_des", [])):
            with self.subTest(field=field, value=value):
                raw = trajectory()
                raw["samples"][1][field] = value
                with self.assertRaises(ValueError):
                    validate_trajectory(raw, ["joint"])

    def test_wrong_environment_or_joint_order(self):
        for key, value in (("environment", "other"), ("joint_order", ["other"])):
            raw = trajectory()
            raw[key] = value
            with self.assertRaises(ValueError):
                validate_trajectory(raw, ["joint"])

    def test_start_pose_uses_enabled_setup_only(self):
        plan = {"waypoints": [{"enabled": True, "q_des": {"joint": 0.5}},
                              {"enabled": False, "q_des": {"joint": 1.0}}]}
        validate_recording_start(plan, {"joint": 0}, ["joint"], [0.5])
        with self.assertRaisesRegex(ValueError, "before Record"):
            validate_recording_start(plan, {"joint": 0}, ["joint"], [1.0])


class VideoTest(unittest.TestCase):
    def test_partial_mp4_is_not_published(self):
        def box(name, data):
            return (8 + len(data)).to_bytes(4, "big") + name + data
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "preview.mp4"
            payload = box(b"ftyp", b"isom") + box(b"mdat", b"frame")
            path.write_bytes(payload)
            self.assertFalse(is_complete_mp4(path))
            payload += box(b"moov", b"metadata")
            path.write_bytes(payload)
            self.assertTrue(is_complete_mp4(path))
            path.write_bytes(payload[:-1])
            self.assertFalse(is_complete_mp4(path))


class LifecycleTest(unittest.TestCase):
    def test_unowned_reader_shutdown_does_not_modify_control_files(self):
        import serve
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(serve, "CONTROL_DIR", root), \
                 patch.object(serve, "LEADER_LIVE_FILE", root / "leader.json"), \
                 patch.object(serve, "LEADER_TELEOP_FILE", root / "teleop.json"):
                serve.stop_leader_reader()
                self.assertEqual(list(root.iterdir()), [])


class PreviewWorkerTest(unittest.TestCase):
    def test_slow_encoder_keeps_only_latest_pending_image(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("Run in the Isaac Python environment for OpenCV/numpy")
        from live_camera_previews import LiveCameraPreviews
        with tempfile.TemporaryDirectory() as folder:
            worker = LiveCameraPreviews(Path(folder))
            entered, release = threading.Event(), threading.Event()
            encode = cv2.imencode
            def slow_encode(*args, **kwargs):
                entered.set()
                if not release.wait(3):
                    raise RuntimeError("Encoder test timed out")
                return encode(*args, **kwargs)
            try:
                with patch.object(cv2, "imencode", slow_encode):
                    worker.submit({"head": (np.full((8, 8, 3), 10, np.uint8), 0, None)})
                    self.assertTrue(entered.wait(3))
                    for value in range(20, 100):
                        worker.submit({"head": (np.full((8, 8, 3), value, np.uint8), 0, None)})
                    self.assertEqual(len(worker._pending), 1)
                    release.set()
                    deadline = time.monotonic() + 3
                    while worker.snapshot()[0] < 2 and time.monotonic() < deadline:
                        time.sleep(.01)
                    self.assertEqual(worker.snapshot(), (2, ["head"]))
                    self.assertAlmostEqual(cv2.imread(str(Path(folder) / "head.jpg")).mean(), 99, delta=2)
            finally:
                release.set()
                worker.close()
            self.assertFalse(worker._thread.is_alive())


if __name__ == "__main__":
    unittest.main()
