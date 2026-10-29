"""Validation and 15 Hz sampling for timestamped leader teleop commands."""

import math


def validate_recording_start(plan, initial_joint_pos, joint_names, current):
    """Reject an unreplayable start before recording, rather than during export."""
    expected = dict(initial_joint_pos)
    for waypoint in plan["waypoints"]:
        if waypoint["enabled"]:
            expected.update(waypoint["q_des"])
    if any(abs(float(value) - expected[name]) > 0.05 for name, value in zip(joint_names, current)):
        raise ValueError(
            "The robot moved beyond the saved setup before Record. "
            "Reset or preview the desired waypoint setup, then Record before teleoperating."
        )


def validate_trajectory(raw, joint_names):
    if not isinstance(raw, dict) or raw.get("schema_version") != "teleop_waypoint_v1":
        raise ValueError("Unsupported teleop trajectory.")
    source_plan = raw.get("source_plan")
    if not isinstance(source_plan, dict) or not isinstance(raw.get("environment"), str):
        raise ValueError("Teleop trajectory needs an environment and source plan.")
    if source_plan.get("environment") != raw["environment"]:
        raise ValueError("Teleop trajectory and source plan environments differ.")
    samples = raw.get("samples")
    if not isinstance(samples, list) or len(samples) < 2:
        raise ValueError("Teleop trajectory needs at least two samples.")
    previous = -1.0
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("Invalid teleop sample.")
        t = sample.get("time_s")
        values = sample.get("q_des")
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t <= previous:
            raise ValueError("Teleop sample times must increase.")
        if not isinstance(values, list) or len(values) != len(joint_names) or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values
        ):
            raise ValueError("Teleop sample joint values are invalid.")
        previous = t
    if samples[0]["time_s"] != 0 or previous > 3600:
        raise ValueError("Teleop duration must be between 0 and 3600 seconds.")
    if raw.get("joint_order") != list(joint_names):
        raise ValueError("Teleop joint order differs from the environment contract.")
    return raw


def resample_15hz(samples):
    """Return commands at 15 Hz; include the last partial interval."""
    duration = samples[-1]["time_s"]
    count = max(1, math.ceil(duration * 15))
    index = 0
    for step in range(count):
        t = step / 15.0
        while index + 1 < len(samples) - 1 and samples[index + 1]["time_s"] < t:
            index += 1
        left, right = samples[index], samples[index + 1]
        span = right["time_s"] - left["time_s"]
        alpha = min(1.0, max(0.0, (t - left["time_s"]) / span))
        yield [(1 - alpha) * a + alpha * b for a, b in zip(left["q_des"], right["q_des"])]
