"""Validated, Task901-only manual calibration overlay."""
from __future__ import annotations

import json
import math
from pathlib import Path

PROFILE = Path(__file__).with_name("applied_profile.json")
CAMERAS = ("head", "wrist_left", "wrist_right")
RAILS = ("main", "right")


def defaults():
    return {
        "schema": "task901.manual-calibration.v1",
        "robot": {"x_m": 0.0, "y_m": 0.0, "yaw_deg": 0.0},
        "cameras": {name: {"x_m": 0.0, "y_m": 0.0, "z_m": 0.0,
                           "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0,
                           "r": 1.0, "g": 1.0, "b": 1.0, "exposure": 1.0,
                           "highlight_knee": 0.0, "highlight_shoulder": 0.0}
                    for name in CAMERAS},
        "lights": {name: {"x_m": 0.0, "y_m": 0.0, "z_m": 0.0,
                          "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0,
                          "intensity": 1.0} for name in RAILS},
    }


def validate(raw):
    expected = defaults()
    if not isinstance(raw, dict) or set(raw) != set(expected) or raw["schema"] != expected["schema"]:
        raise ValueError("Invalid Task901 profile schema")
    out = defaults()
    limits = {"x_m": (-0.2, 0.2), "y_m": (-0.2, 0.2), "z_m": (-0.2, 0.2),
              "roll_deg": (-20, 20), "pitch_deg": (-20, 20), "yaw_deg": (-20, 20),
              "r": (0.3, 2), "g": (0.3, 2), "b": (0.3, 2),
              "exposure": (0.3, 2), "intensity": (0.2, 3),
              "highlight_knee": (-0.3, 0.3), "highlight_shoulder": (-0.3, 0.3)}
    for group in ("robot", "cameras", "lights"):
        source = raw[group]
        if set(source) != set(expected[group]):
            raise ValueError(f"Invalid {group} keys")
        entries = {"robot": {"robot": source}, "cameras": source, "lights": source}[group]
        targets = {"robot": {"robot": out["robot"]}, "cameras": out["cameras"], "lights": out["lights"]}[group]
        for name, values in entries.items():
            if set(values) != set(targets[name]):
                raise ValueError(f"Invalid {group}.{name} keys")
            for key, value in values.items():
                if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                    raise ValueError(f"Invalid {group}.{name}.{key}")
                lo, hi = limits[key]
                if not lo <= value <= hi:
                    raise ValueError(f"Out of range {group}.{name}.{key}: {value}")
                targets[name][key] = float(value)
    return out


def load():
    return validate(json.loads(PROFILE.read_text())) if PROFILE.exists() else defaults()


def apply_config(cfg, profile):
    """Apply deltas to a freshly constructed Cyclo Lab Task901 env config."""
    from scipy.spatial.transform import Rotation
    from isaaclab.sensors import CameraCfg
    robot = cfg.scene.robot.init_state
    delta = profile["robot"]
    robot.pos = (robot.pos[0] + delta["x_m"], robot.pos[1] + delta["y_m"], robot.pos[2])
    w, x, y, z = robot.rot
    q = Rotation.from_euler("z", delta["yaw_deg"], degrees=True) * Rotation.from_quat([x, y, z, w])
    x, y, z, w = q.as_quat()
    robot.rot = (w, x, y, z)
    for name in CAMERAS:
        camera = getattr(cfg.scene, "cam_" + name)
        base = camera.offset
        values = profile["cameras"][name]
        w, x, y, z = base.rot
        turn = Rotation.from_quat([x, y, z, w]) * Rotation.from_euler(
            "xyz", [values["roll_deg"], values["pitch_deg"], values["yaw_deg"]], degrees=True)
        x, y, z, w = turn.as_quat()
        camera.offset = CameraCfg.OffsetCfg(
            pos=tuple(base.pos[i] + values[key] for i, key in enumerate(("x_m", "y_m", "z_m"))),
            rot=(w, x, y, z), convention=base.convention)
    return cfg


def apply_color(profiles, profile):
    for name in CAMERAS:
        values = profile["cameras"][name]
        source = profiles[name]
        source["matrix"] = tuple(tuple(v * values[color] for v in row)
                                 for row, color in zip(source["matrix"], ("r", "g", "b")))
        source["tone_scale"] *= values["exposure"]
        source["highlight_knee"] += values["highlight_knee"]
        source["highlight_shoulder"] += values["highlight_shoulder"]


def apply_lights(stage, profile):
    from pxr import Gf, Sdf
    for name, prim_name in (("main", "Rail_Main"), ("right", "Rail_Right")):
        prim = next((p for p in stage.Traverse() if p.GetName() == prim_name
                     and "/RailLights/" in str(p.GetPath())), None)
        if prim is None or not prim.IsValid():
            raise RuntimeError(f"Light rail not found: {prim_name}")
        values = profile["lights"][name]
        translation = prim.GetAttribute("xformOp:translate")
        current = translation.Get()
        translation.Set(Gf.Vec3d(*(current[i] + values[key] for i, key in enumerate(("x_m", "y_m", "z_m")))))
        angles = prim.CreateAttribute("xformOp:rotateXYZ", Sdf.ValueTypeNames.Float3)
        angles.Set(Gf.Vec3f(values["roll_deg"], values["pitch_deg"], values["yaw_deg"]))
        order = prim.GetAttribute("xformOpOrder")
        tokens = list(order.Get()) if order.IsValid() else []
        if "xformOp:rotateXYZ" not in tokens:
            order.Set(tokens + ["xformOp:rotateXYZ"])
        for child in prim.GetChildren():
            intensity = child.GetAttribute("inputs:intensity")
            if intensity.IsValid():
                intensity.Set(float(intensity.Get()) * values["intensity"])
