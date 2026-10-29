"""Local browser UI for drafting plans and controlling its isolated Isaac runner."""

from __future__ import annotations

import argparse
import atexit
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import subprocess
import threading
import time
from urllib.parse import parse_qs, quote, unquote, urlsplit
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preview_video import is_complete_mp4

from seed_plan import (
    IK_SOLVER_LABELS,
    LAB_ROOT,
    SG2_URDF_FILE,
    discover_environments,
    joint_contract,
    local_robot_spec,
    materialize_plan_targets,
    validate_plan,
    validate_object_poses,
)
from teleop_trajectory import validate_trajectory


HERE = Path(__file__).resolve().parent
PLANS = LAB_ROOT / ".runtime/joint_waypoint_seed_ui/plans"
TELEOP_PLANS = PLANS / "teleop"
STATE_FILE = LAB_ROOT / ".runtime/joint_waypoint_seed_ui/live_state.json"
CAMERA_DIR = STATE_FILE.parent / "cameras"
DATASET_ROOT = LAB_ROOT / "datasets/waypoint_seed"
CAMERA_LABELS = (
    "overhead_left", "overhead_center", "overhead_right",
    "wrist_left", "head", "wrist_right",
)
# The Isaac container writes datasets as root. Keep browser-to-runner control
# messages in a directory created by the host UI server so both sides can
# atomically replace files without changing dataset ownership.
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
RUNNER_SG2_URDF = "/tmp/waypoint_seed_sg2.urdf"
RUNNER_BG2_URDF = "/tmp/waypoint_seed_bg2.urdf"
RUNNER_CONTAINER = os.environ.get("SEED_RUNNER_CONTAINER", "cyclo_lab")
RUNNER_LAB_ROOT = os.environ.get("SEED_RUNNER_LAB_ROOT", "/workspace/cyclo_lab")
RUNNER_DIR = os.environ.get(
    "SEED_RUNNER_DIR", f"{RUNNER_LAB_ROOT}/scripts/sim2real/imitation_learning/waypoint_seed_ui"
)
RUNNER_PYTHON = os.environ.get(
    "SEED_RUNNER_PYTHON", f"{RUNNER_LAB_ROOT}/third_party/IsaacLab/_isaac_sim/python.sh"
)
CONNECT_LOCK = threading.RLock()
EXPORT_LOCK = threading.Lock()
EXPORT_STATUS: dict = {}
RENDER_LOCK = threading.Lock()
LAUNCH_STATE = {"environment": None, "started_at_ms": 0}
RENDER_JOBS: dict[str, dict] = {}
LEADER_LOCK = threading.RLock()
LEADER_READERS = {
    profile: {"process": None, "values": {}, "error": "", "last_start": 0.0, "token": ""}
    for profile in ("a3", "mini")
}
LEADER_LIVE_FILE = CONTROL_DIR / "leader_live_mini.json"
LEADER_TELEOP_FILE = CONTROL_DIR / "leader_teleop.json"
TELEOP_CAPTURE_REQUEST_FILE = CONTROL_DIR / "teleop_capture_request.json"
TELEOP_CAPTURE_RESULT_FILE = CONTROL_DIR / "teleop_capture_result.json"
TELEOP_CAPTURE_LOCK = threading.Lock()
LEADER_ACTIVE_ARMS: set[str] = set()


def stop_leader_reader(profile: str | None = None) -> None:
    for selected in ((profile,) if profile else LEADER_READERS):
        with LEADER_LOCK:
            reader = LEADER_READERS[selected]
            process, token = reader["process"], reader["token"]
            reader.update(process=None, token="", values={}, error="")
        if token:
            (CONTROL_DIR / f"leader_stop_{selected}.txt").write_text(token, encoding="utf-8")
        if selected == "mini" and token:
            LEADER_LIVE_FILE.unlink(missing_ok=True)
            write_json_atomic(LEADER_TELEOP_FILE, {"enabled": False})
            LEADER_ACTIVE_ARMS.clear()
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.terminate()


def _read_leader_lines(profile: str, process) -> None:
    for line in process.stdout:
        try:
            if line.startswith("LEADER_STATE="):
                value = json.loads(line.removeprefix("LEADER_STATE="))
                with LEADER_LOCK:
                    reader = LEADER_READERS[profile]
                    if reader["process"] is process:
                        reader["values"][value["side"]] = value
                        reader["error"] = ""
                        if profile == "mini":
                            write_json_atomic(LEADER_LIVE_FILE, {"run_id": reader["token"], "values": reader["values"]})
            elif line.startswith("LEADER_ERROR="):
                with LEADER_LOCK:
                    reader = LEADER_READERS[profile]
                    if reader["process"] is process:
                        reader["error"] = json.loads(line.removeprefix("LEADER_ERROR=")).get("error", "Leader reader failed.")
            elif profile == "mini" and line.startswith("LEADER_COMMAND="):
                command = json.loads(line.removeprefix("LEADER_COMMAND="))
                with LEADER_LOCK:
                    current_reader = LEADER_READERS[profile]["process"] is process
                if current_reader:
                    handle_leader_command(command)
        except (ValueError, KeyError, TypeError):
            continue
    process.wait()
    with LEADER_LOCK:
        reader = LEADER_READERS[profile]
        if reader["process"] is process:
            reader["error"] = reader["error"] or "Leader reader stopped. Click Read Leader to reconnect."


def ensure_leader_reader(profile: str) -> None:
    with LEADER_LOCK:
        reader = LEADER_READERS[profile]
        if reader["process"] is not None and reader["process"].poll() is None:
            return
        if time.monotonic() - reader["last_start"] < 3:
            return
        reader["last_start"] = time.monotonic()
        reader["values"].clear()
        reader["error"] = ""
        reader["token"] = uuid4().hex
        domain, port = ("30", "7447") if profile == "a3" else ("31", "7448")
        reader["process"] = subprocess.Popen([
            "docker", "exec", "-e", f"ROS_DOMAIN_ID={domain}",
            "-e", "ZENOH_ROUTER_IP=127.0.0.1", "-e", f"ZENOH_ROUTER_PORT={port}",
            "-w", RUNNER_DIR, RUNNER_CONTAINER,
            RUNNER_PYTHON, "leader_snapshot.py", "--watch",
            "--stop-file", f"{RUNNER_LAB_ROOT}/.runtime/joint_waypoint_seed_ui/leader_stop_{profile}.txt",
            "--token", reader["token"],
        ], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1)
        threading.Thread(target=_read_leader_lines, args=(profile, reader["process"]), daemon=True).start()


def live_state() -> dict:
    state = {"status": "idle"}
    if STATE_FILE.is_file():
        try:
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            state = {"status": "unavailable"}
    now_ms = round(time.time() * 1000)
    if state.get("status") in ("running", "awaiting_acceptance") and now_ms - state.get("updated_at_ms", 0) > 3000:
        state["status"] = "stale"
    with CONNECT_LOCK:
        launched_at = int(LAUNCH_STATE["started_at_ms"] or 0)
        launched_env = LAUNCH_STATE["environment"]
        if state.get("status") in ("running", "awaiting_acceptance") and state.get("environment") == launched_env:
            LAUNCH_STATE.update(environment=None, started_at_ms=0)
        elif (
            state.get("status") in ("failed", "saved", "disconnected")
            and state.get("environment") == launched_env
            and state.get("updated_at_ms", 0) >= launched_at
        ):
            LAUNCH_STATE.update(environment=None, started_at_ms=0)
        elif launched_env and now_ms - launched_at < 180_000:
            return {
                "status": "starting", "environment": launched_env,
                "updated_at_ms": launched_at, "phase": "launching",
            }
        elif launched_env:
            LAUNCH_STATE.update(environment=None, started_at_ms=0)
    return state


def handle_leader_command(command: dict) -> None:
    """Follow Mini Leader joystick toggles only in the connected BG2 preview."""
    side, data = command.get("side"), command.get("data")
    if side not in ("left", "right") or not isinstance(data, int):
        return
    if round(time.time() * 1000) - command.get("captured_at_ms", 0) > 1000:
        return
    state = live_state()
    if (
        state.get("status") != "running" or state.get("recording_mode")
        or state.get("phase") not in ("final_hold", "leader_teleop", "leader_waiting")
        or "FFW-BG2" not in state.get("environment", "")
    ):
        return
    with LEADER_LOCK:
        if data == 2:
            if side in LEADER_ACTIVE_ARMS:
                LEADER_ACTIVE_ARMS.remove(side)
            else:
                LEADER_ACTIVE_ARMS.add(side)
        elif data == 1:
            LEADER_ACTIVE_ARMS.add(side)
        else:
            LEADER_ACTIVE_ARMS.discard(side)
        arms = tuple(sorted(LEADER_ACTIVE_ARMS))
    request = {
        "id": uuid4().hex, "enabled": bool(arms),
        "side": "both" if len(arms) == 2 else (arms[0] if arms else None),
        "run_id": state["run_id"], "environment": state["environment"],
        "control_hz": 60.0, "source": "mini_joystick",
    }
    write_json_atomic(LEADER_TELEOP_FILE, request)



def explicit_waypoint_plan(raw: object, state: dict | None = None) -> dict:
    """Normalize browser plans before writing saved, preview or export JSON."""
    plan = validate_plan(raw, discover_environments())
    state = live_state() if state is None else state
    initial = state.get("initial_joint_pos") if state.get("environment") == plan["environment"] else None
    return materialize_plan_targets(plan, initial)


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def request_teleop_capture(action: str) -> dict:
    state = live_state()
    if (state.get("status") != "running" or state.get("recording_mode")
            or "FFW-BG2" not in state.get("environment", "")
            or state.get("phase") not in ("final_hold", "leader_teleop", "leader_waiting")):
        raise ValueError("Connect a BG2 teleop preview before recording.")
    if not TELEOP_CAPTURE_LOCK.acquire(blocking=False):
        raise ValueError("Another teleop recording action is in progress.")
    try:
        request = {"id": uuid4().hex, "action": action, "run_id": state["run_id"],
                   "environment": state["environment"]}
        write_json_atomic(TELEOP_CAPTURE_REQUEST_FILE, request)
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            try:
                result = json.loads(TELEOP_CAPTURE_RESULT_FILE.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                result = {}
            if result.get("id") == request["id"]:
                if result.get("status") != "ok":
                    raise ValueError(result.get("error", "Teleop recording failed."))
                return result
            time.sleep(0.05)
        raise ValueError("Teleop runner did not acknowledge the recording action.")
    finally:
        TELEOP_CAPTURE_LOCK.release()


def runner_urdf_env(environment: str) -> list[str]:
    """Read the selected robot URDF and pass it to the Isaac container."""
    joint_contract(environment)
    spec = local_robot_spec(environment)
    source = Path(spec["urdf_path"]) if spec else SG2_URDF_FILE
    target = RUNNER_BG2_URDF if spec else RUNNER_SG2_URDF
    variable = "BG2_URDF_PATH" if spec else "SG2_URDF_PATH"
    if not source.is_file():
        raise ValueError(f"Robot URDF is missing: {source}")
    subprocess.run(
        ["docker", "cp", str(source), f"{RUNNER_CONTAINER}:{target}"],
        check=True, capture_output=True, text=True, timeout=30,
    )
    return ["-e", f"{variable}={target}"]


def validate_control_rate(control_hz: float, physics_hz: float = 120.0) -> tuple[float, int]:
    ratio = physics_hz / float(control_hz)
    decimation = round(ratio)
    if control_hz > physics_hz or decimation < 1 or abs(ratio - decimation) > 1.0e-9:
        supported = "1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120"
        raise ValueError(
            f"{control_hz:g} Hz cannot be applied with {physics_hz:g} Hz physics. "
            f"Supported integer rates: {supported}"
        )
    return physics_hz, int(decimation)


def validate_folder_name(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("Enter a Seed Export folder name.")
    value = value.strip()
    if not value or len(value) > 80 or not re.fullmatch(r"[\w-]+", value, flags=re.UNICODE):
        raise ValueError("Use up to 80 letters, numbers, underscores, or hyphens for the folder name.")
    return value


def validate_plan_filename(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("Enter a Save Waypoints filename.")
    name = value.strip()
    if name.endswith(".json"):
        name = name[:-5]
    if not name or len(name) > 80 or not re.fullmatch(r"[\w-]+", name, flags=re.UNICODE):
        raise ValueError("Use up to 80 letters, numbers, underscores, or hyphens for the filename.")
    return f"{name}.json"


def seed_path(relative: str, *, suffix: str | None = None) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("A seed path is required.")
    path = (DATASET_ROOT / relative).resolve()
    root = DATASET_ROOT.resolve()
    if path == root or root not in path.parents or (suffix and path.suffix != suffix):
        raise ValueError("The seed path is invalid.")
    return path


def export_status(payload: dict) -> None:
    global EXPORT_STATUS
    EXPORT_STATUS = {**payload, "updated_at_ms": round(time.time() * 1000)}


def start_export_job(job_id: str, plan: dict, folder: str, trajectory: dict | None = None) -> None:
    try:
        export_status({"id": job_id, "status": "stopping_preview", "folder": folder})
        state = live_state()
        if state.get("status") == "awaiting_acceptance" or (
            state.get("recording_mode")
            and state.get("status") in ("running", "starting", "finalizing", "stale")
        ):
            raise ValueError("A Seed Export is already recording.")
        if state.get("status") in ("running", "starting"):
            if state.get("environment") != plan["environment"]:
                raise ValueError(f"Another environment is running: {state.get('environment')}")
            request = {
                "id": uuid4().hex, "run_id": state.get("run_id"),
                "environment": state["environment"], "requested_at_ms": round(time.time() * 1000),
            }
            write_json_atomic(STOP_REQUEST_FILE, request)
            deadline = time.monotonic() + 90.0
            while time.monotonic() < deadline:
                current = live_state()
                if current.get("status") not in ("running", "starting", "awaiting_acceptance"):
                    break
                time.sleep(0.5)
            else:
                raise ValueError("The preview runner did not stop within 90 seconds. Check Disconnect status.")

        export_status({"id": job_id, "status": "launching_recorder", "folder": folder})
        STOP_REQUEST_FILE.unlink(missing_ok=True)
        STOP_RESULT_FILE.unlink(missing_ok=True)
        ACCEPT_REQUEST_FILE.unlink(missing_ok=True)
        # A folder is a seed collection. Always add a uniquely named HDF5 so
        # the same Seed Export folder can be reused without overwriting files.
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        dataset_name = f"seed_{stamp}.hdf5"
        dataset_stem = Path(dataset_name).stem
        container_folder = f"{RUNNER_LAB_ROOT}/datasets/waypoint_seed/{folder}"
        container_dataset = f"{container_folder}/{dataset_name}"
        container_plan = f"{container_folder}/{dataset_stem}_plan.json"
        subprocess.run(
            ["docker", "exec", RUNNER_CONTAINER, "mkdir", "-p", container_folder],
            check=True, capture_output=True, text=True, timeout=30,
        )
        subprocess.run(
            ["docker", "exec", "-i", RUNNER_CONTAINER, "python3", "-c",
             "from pathlib import Path; import sys; Path(sys.argv[1]).write_text(sys.stdin.read(), encoding='utf-8')",
             container_plan],
            input=json.dumps(trajectory if trajectory else plan, ensure_ascii=False) + "\n",
            check=True, capture_output=True, text=True, timeout=30,
        )
        urdf_env = runner_urdf_env(plan["environment"])
        subprocess.run([
            "docker", "exec", "-d", "-e", f"CYCLO_LAB_ROOT={RUNNER_LAB_ROOT}", *urdf_env,
            "-w", RUNNER_DIR, RUNNER_CONTAINER, RUNNER_PYTHON, "run_seed.py",
            "--trajectory" if trajectory else "--plan", container_plan, "--execute", "--headless", "--record-hdf5", "--auto-accept",
            "--dataset-file", container_dataset,
        ], check=True, capture_output=True, text=True, timeout=30)
        launched_at_ms = round(time.time() * 1000)
        with CONNECT_LOCK:
            LAUNCH_STATE.update(environment=plan["environment"], started_at_ms=launched_at_ms)
        export_status({"id": job_id, "status": "loading_environment", "folder": folder})
        deadline = time.monotonic() + 120.0
        current = None
        while time.monotonic() < deadline:
            current = live_state()
            if (
                current.get("environment") == plan["environment"]
                and current.get("recording_mode")
                and current.get("status") in ("running", "finalizing", "saved")
            ):
                applied_hz = float(current.get("sim_control_hz", 0))
                if abs(applied_hz - plan["control_hz"]) > 1.0e-6:
                    request = {
                        "id": uuid4().hex, "run_id": current.get("run_id"),
                        "environment": plan["environment"], "requested_at_ms": round(time.time() * 1000),
                    }
                    write_json_atomic(STOP_REQUEST_FILE, request)
                    raise ValueError(
                        f"Requested {plan['control_hz']:g} Hz, but {applied_hz:g} Hz was applied. Recording stopped."
                    )
                break
            if (
                current.get("environment") == plan["environment"]
                and current.get("status") == "failed"
                and int(current.get("updated_at_ms", 0)) >= launched_at_ms
            ):
                raise ValueError(current.get("error") or "The HDF5 recorder failed to start.")
            time.sleep(0.25)
        else:
            raise ValueError("The recorder did not report its applied control rate within 120 seconds.")
        export_status({
            "id": job_id, "status": "recording_started", "folder": folder,
            "dataset": f"{folder}/{dataset_name}", "control_hz": float(current["sim_control_hz"]),
            "physics_hz": float(current["physics_hz"]), "decimation": int(current["decimation"]),
        })
    except Exception as exc:
        with CONNECT_LOCK:
            LAUNCH_STATE.update(environment=None, started_at_ms=0)
        export_status({"id": job_id, "status": "error", "folder": folder, "error": str(exc)})
    finally:
        if EXPORT_LOCK.locked():
            EXPORT_LOCK.release()


def render_seed_job(job_id: str, dataset: Path, relative: str, fps: float) -> None:
    try:
        with RENDER_LOCK:
            RENDER_JOBS[job_id] = {"id": job_id, "status": "rendering", "dataset": relative}
        relative_container = dataset.relative_to(DATASET_ROOT).as_posix()
        container_dataset = f"{RUNNER_LAB_ROOT}/datasets/waypoint_seed/{relative_container}"
        renderer = f"{RUNNER_LAB_ROOT}/scripts/sim2real/imitation_learning/render_hdf5_cameras.py"
        subprocess.run([
            "docker", "exec", RUNNER_CONTAINER, RUNNER_PYTHON, renderer,
            "--input-file", container_dataset, "--fps", f"{fps:g}", "--overwrite",
        ], check=True, capture_output=True, text=True, timeout=600)
        video = dataset.with_name(f"{dataset.stem}_camera_previews") / "demo_0_three_cameras.mp4"
        if not is_complete_mp4(video):
            raise RuntimeError(f"Rendered video is missing or incomplete: {video}")
        url = f"/api/seed-video?path={quote(video.relative_to(DATASET_ROOT).as_posix())}"
        with RENDER_LOCK:
            RENDER_JOBS[job_id] = {"id": job_id, "status": "ready", "dataset": relative, "video_url": url}
    except Exception as exc:
        with RENDER_LOCK:
            RENDER_JOBS[job_id] = {"id": job_id, "status": "error", "dataset": relative, "error": str(exc)}


class Handler(BaseHTTPRequestHandler):
    def _camera_stream(self, run_id: str) -> None:
        state = live_state()
        if state.get("run_id") != run_id or state.get("status") not in ("running", "awaiting_acceptance"):
            self._json(404, {"error": "Live camera stream unavailable"})
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=camera-frame")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.connection.settimeout(2.0)
        self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
        previous = {}
        deadline = time.monotonic()
        try:
            while True:
                state = live_state()
                if state.get("run_id") != run_id or state.get("status") not in ("running", "awaiting_acceptance"):
                    return
                for label in CAMERA_LABELS:
                    if label not in state.get("cameras", []):
                        continue
                    try:
                        # The writer atomically replaces files; stat and read
                        # the same open file so metadata matches these bytes.
                        with (CAMERA_DIR / f"{label}.jpg").open("rb") as frame:
                            stamp = os.fstat(frame.fileno()).st_mtime_ns
                            if previous.get(label) == stamp:
                                continue
                            payload = frame.read()
                    except OSError:
                        continue
                    header = (f"--camera-frame\r\nContent-Type: image/jpeg\r\n"
                              f"Content-Length: {len(payload)}\r\nX-Camera: {label}\r\n"
                              f"X-Published-Ms: {stamp / 1_000_000:.3f}\r\n\r\n").encode("ascii")
                    self.wfile.write(header + payload + b"\r\n")
                    previous[label] = stamp
                self.wfile.flush()
                deadline += 1.0 / 15.0
                now = time.monotonic()
                if deadline < now:
                    deadline += (math.floor((now - deadline) * 15.0) + 1) / 15.0
                time.sleep(max(0.0, deadline - time.monotonic()))
        except (OSError, TimeoutError):
            pass
        finally:
            self.close_connection = True

    def _reply(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: object) -> None:
        self._reply(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _video(self, path: Path) -> None:
        size = path.stat().st_size
        start, end, status = 0, max(0, size - 1), 200
        header = self.headers.get("Range", "")
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", header) if header else None
        if match:
            if match.group(1):
                start = int(match.group(1))
                end = min(end, int(match.group(2))) if match.group(2) else end
            elif match.group(2):
                start = max(0, size - int(match.group(2)))
            if start > end or start >= size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            status = 206
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as stream:
            stream.seek(start)
            remaining = length
            while remaining:
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_GET(self) -> None:
        request = urlsplit(self.path)
        if request.path == "/":
            self._reply(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif request.path == "/api/catalog":
            joints, limits = joint_contract()
            self._json(200, {
                "environments": discover_environments(), "joints": joints, "limits": limits,
                "limits_source": str(SG2_URDF_FILE) if SG2_URDF_FILE.is_file() else "Cyclo Lab URDF-derived 19D limits",
                "ik_solvers": IK_SOLVER_LABELS,
                "environment_limits": {
                    name: joint_contract(name)[1] for name in discover_environments()
                    if "FFW-BG2" in name
                },
            })
        elif request.path == "/api/plans":
            mode = parse_qs(request.query).get("mode", ["waypoint"])[0]
            directory = TELEOP_PLANS if mode == "teleop" else PLANS
            self._json(200, {"plans": sorted((path.name for path in directory.glob("*.json") if path.is_file()), reverse=True)})
        elif request.path.startswith("/api/plans/"):
            name = unquote(request.path.removeprefix("/api/plans/"))
            try:
                valid_name = validate_plan_filename(name)
            except ValueError:
                self._json(404, {"error": "Plan not found"})
                return
            if valid_name != name:
                self._json(404, {"error": "Plan not found"})
                return
            try:
                mode = parse_qs(request.query).get("mode", ["waypoint"])[0]
                if mode == "teleop":
                    trajectory = json.loads((TELEOP_PLANS / name).read_text(encoding="utf-8"))
                    joint_names, _ = joint_contract(trajectory["environment"])
                    validate_trajectory(trajectory, joint_names)
                    self._json(200, {"trajectory": trajectory})
                    return
                plan = validate_plan(json.loads((PLANS / name).read_text(encoding="utf-8")), discover_environments())
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                self._json(400, {"error": str(exc)})
                return
            self._json(200, {"name": name, "plan": plan})
        elif request.path == "/api/state":
            state = live_state()
            if state.get("status") == "running" and "FFW-BG2" in state.get("environment", ""):
                ensure_leader_reader("mini")
            self._json(200, state)
        elif request.path == "/api/leader/snapshot":
            side = parse_qs(request.query).get("side", [""])[0]
            environment = parse_qs(request.query).get("environment", [""])[0]
            profile = parse_qs(request.query).get("profile", ["a3"])[0]
            if profile not in ("a3", "mini") or side not in ("left", "right") or environment not in discover_environments() or not any(
                robot in environment for robot in ("FFW-SG2", "FFW-BG2")
            ):
                self._json(400, {"error": "Select an SG2 or BG2 environment and a left or right arm."})
                return
            try:
                ensure_leader_reader(profile)
                with LEADER_LOCK:
                    result = dict(LEADER_READERS[profile]["values"].get(side, {}))
                    error = LEADER_READERS[profile]["error"]
                age_ms = round(time.time() * 1000) - result.get("captured_at_ms", 0)
                if age_ms > 1000:
                    self._json(200, {"status": "waiting", "side": side, "profile": profile, "error": error or "Waiting for live leader joint states."})
                    return
                _, limits = joint_contract(environment)
                result["outside_limits"] = [
                    name for name, value in result["q_des"].items()
                    if name not in limits or not limits[name][0] <= value <= limits[name][1]
                ]
                result["status"] = "live"
                result["profile"] = profile
                self._json(200, result)
            except (OSError, ValueError) as exc:
                self._json(503, {"error": str(exc)})
        elif request.path == "/api/leader/teleop":
            try:
                self._json(200, json.loads(LEADER_TELEOP_FILE.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                self._json(200, {"enabled": False})
        elif request.path == "/api/ik-result":
            request_id = parse_qs(request.query).get("id", [""])[0]
            try:
                result = json.loads(IK_RESULT_FILE.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                result = None
            if result is None or result.get("id") != request_id or not request_id:
                self._json(202, {"status": "pending"})
            else:
                self._json(200, result)
        elif request.path == "/api/preview-result":
            request_id = parse_qs(request.query).get("id", [""])[0]
            try:
                result = json.loads(PREVIEW_RESULT_FILE.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                result = None
            if result is None or result.get("id") != request_id or not request_id:
                self._json(202, {"status": "pending"})
            else:
                self._json(200, result)
        elif request.path == "/api/reset-result":
            request_id = parse_qs(request.query).get("id", [""])[0]
            try:
                result = json.loads(RESET_RESULT_FILE.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                result = None
            if result is None or result.get("id") != request_id or not request_id:
                self._json(202, {"status": "pending"})
            else:
                self._json(200, result)
        elif request.path == "/api/object-result":
            request_id = parse_qs(request.query).get("id", [""])[0]
            try:
                result = json.loads(OBJECT_RESULT_FILE.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                result = None
            if result is None or result.get("id") != request_id or not request_id:
                self._json(202, {"status": "pending"})
            else:
                self._json(200, result)
        elif request.path == "/api/export-status":
            request_id = parse_qs(request.query).get("id", [""])[0]
            result = EXPORT_STATUS
            if result is None or result.get("id") != request_id or not request_id:
                self._json(202, {"status": "pending"})
            else:
                self._json(200, result)
        elif request.path == "/api/seeds":
            seeds = []
            if DATASET_ROOT.is_dir():
                for path in sorted(DATASET_ROOT.rglob("*.hdf5"), key=lambda item: item.stat().st_mtime, reverse=True):
                    if path.name.endswith(".pending.hdf5"):
                        continue
                    relative = path.relative_to(DATASET_ROOT).as_posix()
                    metadata = {}
                    try:
                        metadata_path = path.with_suffix(".json")
                        if not metadata_path.is_file():
                            metadata_path = path.with_name(f"{path.stem}_plan.json")
                        if not metadata_path.is_file():
                            metadata_path = path.parent / "export.json"
                        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    except (OSError, ValueError, TypeError):
                        pass
                    video = path.with_name(f"{path.stem}_camera_previews") / "demo_0_three_cameras.mp4"
                    seeds.append({
                        "relative": relative,
                        "folder": path.parent.relative_to(DATASET_ROOT).as_posix(),
                        "size_bytes": path.stat().st_size,
                        "modified_at_ms": round(path.stat().st_mtime * 1000),
                        "control_hz": metadata.get("control_hz"),
                        "environment": metadata.get("environment"),
                        "video_url": (
                            f"/api/seed-video?path={quote(video.relative_to(DATASET_ROOT).as_posix())}"
                            if is_complete_mp4(video) else None
                        ),
                    })
            self._json(200, {"seeds": seeds})
        elif request.path == "/api/render-status":
            request_id = parse_qs(request.query).get("id", [""])[0]
            if not request_id or request_id == "undefined":
                self._json(200, {
                    "status": "error",
                    "error": "Render status expired. Click View again.",
                })
                return
            with RENDER_LOCK:
                result = RENDER_JOBS.get(request_id)
            self._json(200 if result is not None else 202, result or {"status": "pending"})
        elif request.path == "/api/seed-video":
            try:
                path = seed_path(parse_qs(request.query).get("path", [""])[0], suffix=".mp4")
                if not path.is_file():
                    raise OSError("video missing")
            except (ValueError, OSError):
                self._json(404, {"error": "Seed video not found"})
                return
            self._video(path)
        elif request.path == "/api/camera-stream":
            self._camera_stream(parse_qs(request.query).get("run", [""])[0])
        elif request.path.startswith("/api/camera/"):
            label = request.path.removeprefix("/api/camera/").removesuffix(".jpg")
            state = live_state()
            frame = parse_qs(request.query).get("frame", [None])[0]
            if (
                request.path != f"/api/camera/{label}.jpg"
                or label not in CAMERA_LABELS
                or state.get("status") not in ("running", "awaiting_acceptance")
                or label not in state.get("cameras", [])
                or frame != str(state.get("camera_frame_step"))
            ):
                self._json(404, {"error": "Live camera frame unavailable"})
                return
            path = CAMERA_DIR / f"{label}.jpg"
            try:
                payload = path.read_bytes()
            except OSError:
                self._json(404, {"error": "Live camera frame unavailable"})
                return
            self._reply(200, payload, "image/jpeg")
        else:
            self._json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if self.path not in (
            "/api/plans", "/api/ik-requests", "/api/preview-requests", "/api/reset-requests", "/api/object-requests",
            "/api/runner/connect", "/api/runner/disconnect", "/api/recording/accept",
            "/api/recording/start", "/api/seeds/render",
            "/api/leader/stop", "/api/teleop/recording",
        ):
            self._json(404, {"error": "Not found"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= (128_000 if self.path in ("/api/plans", "/api/preview-requests", "/api/recording/start") else 2_000):
                raise ValueError("Request size is invalid.")
            raw = json.loads(self.rfile.read(size))
            if self.path == "/api/teleop/recording":
                action = raw.get("action") if isinstance(raw, dict) else None
                if action not in ("record", "discard"):
                    raise ValueError("Choose Record or Discard.")
                self._json(200, request_teleop_capture(action))
                return
            if self.path == "/api/leader/stop":
                profile = raw.get("profile")
                if profile not in LEADER_READERS:
                    raise ValueError("Select an A3 or Mini leader.")
                stop_leader_reader(profile)
                self._json(200, {"status": "stopped"})
                return
            if self.path == "/api/runner/connect":
                self._post_runner_connect(raw)
                return
            if self.path == "/api/runner/disconnect":
                self._post_runner_disconnect(raw)
                return
            if self.path == "/api/ik-requests":
                self._post_ik_request(raw)
                return
            if self.path == "/api/preview-requests":
                self._post_preview_request(raw)
                return
            if self.path == "/api/reset-requests":
                self._post_reset_request(raw)
                return
            if self.path == "/api/object-requests":
                self._post_object_request(raw)
                return
            if self.path == "/api/recording/accept":
                self._post_accept(raw)
                return
            if self.path == "/api/recording/start":
                self._post_recording_start(raw)
                return
            if self.path == "/api/seeds/render":
                self._post_seed_render(raw)
                return
            if not isinstance(raw, dict):
                raise ValueError("Save Waypoints request must be an object.")
            name = validate_plan_filename(raw.get("filename"))
            mode = raw.get("mode", "waypoint")
            if mode == "teleop":
                path = TELEOP_PLANS / name
                if path.exists():
                    self._json(409, {"error": f"A saved teleop waypoint named {name} already exists."})
                    return
                result = request_teleop_capture("save")
                draft = CONTROL_DIR / result["draft"]
                with draft.open(encoding="utf-8") as stream:
                    header = json.loads(stream.readline())
                    samples = [json.loads(line) for line in stream if line.strip()]
                trajectory = {**header, "samples": samples}
                joint_names, _ = joint_contract(trajectory["environment"])
                validate_trajectory(trajectory, joint_names)
                plan = trajectory
                directory = TELEOP_PLANS
            elif mode == "waypoint":
                plan = explicit_waypoint_plan(raw.get("plan"))
                directory = PLANS
            else:
                raise ValueError("Unknown save mode.")
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / name
            if path.exists():
                self._json(409, {"error": f"A saved waypoint file named {name} already exists. Choose another filename."})
                return
            with path.open("x", encoding="utf-8") as stream:
                json.dump(plan, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            if mode == "teleop":
                try:
                    draft.unlink(missing_ok=True)
                except OSError:
                    pass
            self._json(201, {
                "saved_path": str(path),
                "steps": len(plan["samples"]) if mode == "teleop" else sum(w["steps"] for w in plan["waypoints"]),
                "duration_s": plan["samples"][-1]["time_s"] if mode == "teleop" else sum(w["actual_duration_s"] for w in plan["waypoints"]),
            })
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._json(400, {"error": str(exc)})
        except OSError as exc:
            self._json(500, {"error": f"Control file error: {exc}"})

    def _post_recording_start(self, raw: dict) -> None:
        if not isinstance(raw, dict):
            raise ValueError("Seed Export request must be an object.")
        folder = validate_folder_name(raw.get("folder_name"))
        trajectory = None
        if raw.get("mode") == "teleop":
            name = validate_plan_filename(raw.get("trajectory_name"))
            trajectory = json.loads((TELEOP_PLANS / name).read_text(encoding="utf-8"))
            plan = validate_plan(trajectory["source_plan"], discover_environments())
            validate_trajectory(trajectory, plan["joint_order"])
            plan = {**plan, "control_hz": 15.0}
        else:
            plan = explicit_waypoint_plan(raw.get("plan"))
        physics_hz, decimation = validate_control_rate(plan["control_hz"])
        if not EXPORT_LOCK.acquire(blocking=False):
            raise ValueError("Another Seed Export is starting.")
        job_id = uuid4().hex
        try:
            thread = threading.Thread(
                target=start_export_job, args=(job_id, plan, folder, trajectory),
                name=f"seed-export-{job_id[:8]}", daemon=True,
            )
            thread.start()
        except Exception:
            EXPORT_LOCK.release()
            raise
        self._json(202, {
            "status": "stopping_preview", "id": job_id, "folder": folder,
            "control_hz": plan["control_hz"], "physics_hz": physics_hz, "decimation": decimation,
        })

    def _post_seed_render(self, raw: dict) -> None:
        if not isinstance(raw, dict):
            raise ValueError("Seed viewer request must be an object.")
        relative = raw.get("path")
        dataset = seed_path(relative, suffix=".hdf5")
        if not dataset.is_file() or dataset.name.endswith(".pending.hdf5"):
            raise ValueError("Select a completed HDF5 seed.")
        video = dataset.with_name(f"{dataset.stem}_camera_previews") / "demo_0_three_cameras.mp4"
        if is_complete_mp4(video):
            self._json(200, {
                "status": "ready", "dataset": relative,
                "video_url": f"/api/seed-video?path={quote(video.relative_to(DATASET_ROOT).as_posix())}",
            })
            return
        fps = 15.0
        try:
            metadata_path = dataset.with_suffix(".json")
            if not metadata_path.is_file():
                metadata_path = dataset.with_name(f"{dataset.stem}_plan.json")
            if not metadata_path.is_file():
                metadata_path = dataset.parent / "export.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            fps = float(metadata.get("control_hz", fps))
        except (OSError, ValueError, TypeError):
            pass
        if not math.isfinite(fps) or fps <= 0:
            fps = 15.0
        job_id = uuid4().hex
        with RENDER_LOCK:
            for active_id, job in RENDER_JOBS.items():
                if job.get("dataset") == relative and job.get("status") in ("queued", "rendering"):
                    self._json(202, {**job, "id": active_id})
                    return
            RENDER_JOBS[job_id] = {"id": job_id, "status": "queued", "dataset": relative}
        threading.Thread(
            target=render_seed_job, args=(job_id, dataset, relative, fps),
            name=f"seed-render-{job_id[:8]}", daemon=True,
        ).start()
        self._json(202, {"status": "queued", "id": job_id, "dataset": relative})

    def _post_runner_connect(self, raw: dict) -> None:
        if not isinstance(raw, dict):
            raise ValueError("Connect request must be an object.")
        environments = discover_environments()
        plan = validate_plan({
            "environment": raw.get("environment"),
            "control_hz": raw.get("control_hz", 15),
            "object_position_xyz_m": raw.get("object_position_xyz_m"),
            "object_poses": raw.get("object_poses", {}),
            "preview": "head",
            "waypoints": [{"duration_s": 0.1, "memo": "UI connection hold", "q_des": {}}],
        }, environments)
        validate_control_rate(plan["control_hz"])
        with CONNECT_LOCK:
            state = live_state()
            if state.get("status") in ("running", "awaiting_acceptance", "starting"):
                if state.get("environment") == plan["environment"]:
                    self._json(200, {"status": "already_connected", "environment": plan["environment"]})
                    return
                raise ValueError(f"Disconnect the running environment first: {state.get('environment')}")
            urdf_env = runner_urdf_env(plan["environment"])
            write_json_atomic(LEADER_TELEOP_FILE, {"enabled": False})
            with LEADER_LOCK:
                LEADER_ACTIVE_ARMS.clear()
            STOP_REQUEST_FILE.unlink(missing_ok=True)
            STOP_RESULT_FILE.unlink(missing_ok=True)
            try:
                subprocess.run([
                    "docker", "exec", "-d",
                    "-e", f"CYCLO_LAB_ROOT={RUNNER_LAB_ROOT}", *urdf_env,
                    "-w", RUNNER_DIR,
                    RUNNER_CONTAINER,
                    RUNNER_PYTHON, "run_seed.py",
                    "--plan-json", json.dumps(plan, ensure_ascii=False),
                    "--execute", "--headless", "--hold-final",
                ], check=True, capture_output=True, text=True, timeout=30)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                detail = getattr(exc, "stderr", "") or str(exc)
                raise ValueError(f"Could not start the preview runner: {detail.strip()}") from exc
            LAUNCH_STATE.update(environment=plan["environment"], started_at_ms=round(time.time() * 1000))
        self._json(202, {"status": "starting", "environment": plan["environment"]})

    def _post_runner_disconnect(self, raw: dict) -> None:
        if not isinstance(raw, dict) or not isinstance(raw.get("environment"), str):
            raise ValueError("Select an environment to disconnect.")
        with CONNECT_LOCK:
            state = live_state()
            if state.get("status") not in ("running", "awaiting_acceptance", "starting"):
                self._json(200, {"status": "disconnected", "environment": raw["environment"]})
                return
            if state.get("environment") != raw["environment"]:
                raise ValueError(f"The running environment differs from the selection: {state.get('environment')}")
            write_json_atomic(LEADER_TELEOP_FILE, {"enabled": False})
            with LEADER_LOCK:
                LEADER_ACTIVE_ARMS.clear()
            request = {
                "id": uuid4().hex,
                "run_id": state.get("run_id"),
                "environment": state["environment"],
                "requested_at_ms": round(time.time() * 1000),
            }
            write_json_atomic(STOP_REQUEST_FILE, request)
            LAUNCH_STATE.update(environment=None, started_at_ms=0)
        self._json(202, {"status": "stopping", "id": request["id"]})

    def _post_accept(self, raw: dict) -> None:
        state = live_state()
        if not isinstance(raw, dict) or state.get("status") != "awaiting_acceptance":
            raise ValueError("No HDF5 run is awaiting approval.")
        if not isinstance(raw.get("run_id"), str) or raw["run_id"] != state.get("run_id"):
            raise ValueError("The run ID does not match.")
        if raw.get("environment") != state.get("environment"):
            raise ValueError("The running environment differs from the selection.")
        request = {"run_id": raw["run_id"], "accepted_at_ms": round(time.time() * 1000)}
        ACCEPT_REQUEST_FILE.parent.mkdir(parents=True, exist_ok=True)
        temporary = ACCEPT_REQUEST_FILE.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(request) + "\n", encoding="utf-8")
        os.replace(temporary, ACCEPT_REQUEST_FILE)
        self._json(202, {"status": "accepted"})

    def _post_ik_request(self, raw: dict) -> None:
        if not isinstance(raw, dict):
            raise ValueError("IK request must be an object")
        state = live_state()
        if state.get("status") != "running" or not state.get("run_id"):
            raise ValueError("The selected environment's preview runner must be active.")
        if raw.get("environment") != state.get("environment"):
            raise ValueError("The running environment differs from the UI selection.")
        if raw.get("side") not in ("left", "right"):
            raise ValueError("Arm must be left or right.")
        if state.get("phase") not in ("ik_wait", "final_hold"):
            raise ValueError("Wait until Preview Motion finishes before calculating TCP IK.")
        solver = raw.get("solver", "iterative_lm")
        if solver not in IK_SOLVER_LABELS:
            raise ValueError(f"Unsupported IK solver: {solver}")
        xyz = raw.get("xyz_m")
        if not isinstance(xyz, list) or len(xyz) != 3 or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or abs(value) > 10 for value in xyz
        ):
            raise ValueError("XYZ must contain three finite world-frame values in metres.")
        request = {
            "id": uuid4().hex,
            "run_id": state["run_id"],
            "environment": state["environment"],
            "side": raw["side"],
            "xyz_m": xyz,
            "solver": solver,
        }
        IK_REQUEST_FILE.parent.mkdir(parents=True, exist_ok=True)
        temporary = IK_REQUEST_FILE.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(request) + "\n", encoding="utf-8")
        os.replace(temporary, IK_REQUEST_FILE)
        self._json(202, {"status": "pending", "id": request["id"]})

    def _post_preview_request(self, raw: dict) -> None:
        write_json_atomic(LEADER_TELEOP_FILE, {"enabled": False})
        with LEADER_LOCK:
            LEADER_ACTIVE_ARMS.clear()
        state = live_state()
        if (
            state.get("status") != "running"
            or state.get("phase") not in ("ik_wait", "final_hold")
            or not state.get("run_id")
        ):
            raise ValueError("The preview runner must be ready while holding its final pose.")
        plan = explicit_waypoint_plan(raw, state)
        validate_control_rate(plan["control_hz"])
        if plan["environment"] != state.get("environment"):
            raise ValueError("The running environment differs from the UI selection.")
        request = {
            "id": uuid4().hex,
            "run_id": state["run_id"],
            "environment": state["environment"],
            "plan": plan,
        }
        PREVIEW_REQUEST_FILE.parent.mkdir(parents=True, exist_ok=True)
        temporary = PREVIEW_REQUEST_FILE.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(request, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, PREVIEW_REQUEST_FILE)
        self._json(202, {
            "status": "pending",
            "id": request["id"],
            "steps": sum(waypoint["steps"] for waypoint in plan["waypoints"]),
            "duration_s": sum(waypoint["actual_duration_s"] for waypoint in plan["waypoints"]),
        })

    def _post_reset_request(self, raw: dict) -> None:
        write_json_atomic(LEADER_TELEOP_FILE, {"enabled": False})
        with LEADER_LOCK:
            LEADER_ACTIVE_ARMS.clear()
        state = live_state()
        if (
            not isinstance(raw, dict)
            or state.get("status") != "running"
            or state.get("phase") not in ("ik_wait", "final_hold")
            or not state.get("run_id")
        ):
            raise ValueError("The preview runner must be ready while holding its final pose.")
        if raw.get("environment") != state.get("environment"):
            raise ValueError("The running environment differs from the UI selection.")
        request = {
            "id": uuid4().hex,
            "run_id": state["run_id"],
            "environment": state["environment"],
        }
        RESET_REQUEST_FILE.parent.mkdir(parents=True, exist_ok=True)
        temporary = RESET_REQUEST_FILE.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(request) + "\n", encoding="utf-8")
        os.replace(temporary, RESET_REQUEST_FILE)
        self._json(202, {"status": "pending", "id": request["id"]})

    def _post_object_request(self, raw: dict) -> None:
        state = live_state()
        if (
            not isinstance(raw, dict)
            or state.get("status") != "running"
            or state.get("phase") not in ("ik_wait", "final_hold")
            or not state.get("run_id")
        ):
            raise ValueError("The environment must be connected and holding its pose.")
        if raw.get("environment") != state.get("environment"):
            raise ValueError("The running environment differs from the UI selection.")
        object_name = raw.get("object_name") or state.get("object_name")
        available = state.get("objects", {})
        if object_name not in available:
            raise ValueError("Select an available scene object; reconnect to refresh the object list.")
        pose = validate_object_poses({object_name: {
            "xyz_m": raw.get("xyz_m"),
            "rpy_deg": raw.get("rpy_deg", available[object_name]["rpy_deg"]),
        }})[object_name]
        request = {
            "id": uuid4().hex,
            "run_id": state["run_id"],
            "environment": state["environment"],
            "object_name": object_name,
            **pose,
        }
        write_json_atomic(OBJECT_REQUEST_FILE, request)
        self._json(202, {"status": "pending", "id": request["id"]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.10.193")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    atexit.register(stop_leader_reader)
    print(f"Open http://{args.host}:{args.port} (Ctrl+C to stop)", flush=True)
    try:
        ensure_leader_reader("mini")
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_leader_reader()
        server.server_close()


if __name__ == "__main__":
    main()
