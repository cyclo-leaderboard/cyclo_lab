"""Read one SG2 leader arm pose without publishing robot commands."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import threading
import time

from cyclo_lab.runtime.transport.ros2_zenoh import JOINT_STATE, UINT8, close_endpoint, create_subscriber


def joint_names(side: str) -> list[str]:
    prefix = "l" if side == "left" else "r"
    return [f"arm_{prefix}_joint{i}" for i in range(1, 8)] + [f"gripper_{prefix}_joint1"]


def joint_values(message, side: str) -> dict | None:
    values = dict(zip(message.name, message.position))
    expected = joint_names(side)
    if any(name not in values or not math.isfinite(float(values[name])) for name in expected):
        return None
    result = {name: float(values[name]) for name in expected}
    # Both A3 and Mini Leader configurations add 0.1 rad to the gripper
    # target; arm joints have no offsets or reversed axes.
    result[expected[-1]] += 0.1
    return result


def read_leader(side: str, timeout_s: float) -> dict:
    if side not in ("left", "right"):
        raise ValueError("Arm must be left or right.")
    ready = threading.Event()
    snapshot = {}

    def on_joint_state(message) -> None:
        values = joint_values(message, side)
        if values is None:
            return
        snapshot.update(values)
        ready.set()

    topic = "/leader/joint_states"
    subscriber = create_subscriber(topic, JOINT_STATE, on_joint_state)
    try:
        if not ready.wait(timeout_s):
            raise TimeoutError(f"No complete leader joint state arrived on {topic} within {timeout_s:g} s.")
        return {"side": side, "topic": topic, "q_des": snapshot, "captured_at_ms": round(time.time() * 1000)}
    finally:
        close_endpoint(subscriber)


def watch_leaders(stop_file: Path, token: str) -> None:
    """Stream only validated leader observations; never publish commands."""
    lock = threading.Lock()
    latest = {}
    commands = []
    subscribers = []

    def callback(side: str, message) -> None:
        values = joint_values(message, side)
        if values is None:
            return
        with lock:
            latest[side] = {"side": side, "q_des": values, "captured_at_ms": round(time.time() * 1000)}

    try:
        def on_joint_state(message) -> None:
            for side in ("left", "right"):
                callback(side, message)

        subscribers.append(create_subscriber("/leader/joint_states", JOINT_STATE, on_joint_state))
        for side in ("left", "right"):
            def on_command(message, arm=side) -> None:
                with lock:
                    commands.append({"side": arm, "data": int(message.data), "captured_at_ms": round(time.time() * 1000)})
            subscribers.append(create_subscriber(f"/leader/{side}_command", UINT8, on_command))
        while True:
            time.sleep(1 / 30)
            try:
                if stop_file.read_text(encoding="utf-8").strip() == token:
                    break
            except OSError:
                pass
            with lock:
                pending = tuple(latest.values())
                latest.clear()
                pending_commands = tuple(commands)
                commands.clear()
            for result in pending:
                print("LEADER_STATE=" + json.dumps(result, allow_nan=False), flush=True)
            for result in pending_commands:
                print("LEADER_COMMAND=" + json.dumps(result, allow_nan=False), flush=True)
    finally:
        for subscriber in subscribers:
            close_endpoint(subscriber)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=("left", "right"))
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--token")
    args = parser.parse_args()
    try:
        if args.watch:
            if args.stop_file is None or not args.token:
                parser.error("--watch requires --stop-file and --token")
            watch_leaders(args.stop_file, args.token)
        else:
            if args.side is None:
                parser.error("--side is required without --watch")
            result = read_leader(args.side, args.timeout)
    except Exception as exc:
        result = {"error": str(exc)}
        if args.watch:
            print("LEADER_ERROR=" + json.dumps(result, allow_nan=False), flush=True)
    if not args.watch:
        print("LEADER_SNAPSHOT=" + json.dumps(result, allow_nan=False), flush=True)
