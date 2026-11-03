"""Closure-dependent effective TCP for the SG2 soft gripper.

The cubic is fitted to the midpoint of the closest opposing distal-mesh
vertices over gripper joint1 in [0, 1.1] rad.  Only its relative displacement
is used: q=1.0 rad is anchored to the existing closed-grasp TCP at -0.242 m.
"""

from __future__ import annotations

import torch


GRIPPER_MIN_RAD = 0.0
GRIPPER_MAX_RAD = 1.1
CLOSED_REFERENCE_RAD = 1.0
CLOSED_REFERENCE_TCP_Z_M = -0.242
GEOMETRY_Z_POLY3 = (
    0.0044541985215427364,
    0.017153610974329686,
    -0.05009301601645295,
    -0.17446067344287108,
)


def _poly(value):
    a, b, c, d = GEOMETRY_Z_POLY3
    return ((a * value + b) * value + c) * value + d


_REFERENCE_GEOMETRY_Z_M = _poly(CLOSED_REFERENCE_RAD)


def gripper_dependent_tcp_z(gripper_rad: torch.Tensor | float):
    """Return link7-local TCP z for the measured/commanded gripper angle."""
    if isinstance(gripper_rad, torch.Tensor):
        bounded = torch.clamp(gripper_rad, GRIPPER_MIN_RAD, GRIPPER_MAX_RAD)
    else:
        bounded = min(max(float(gripper_rad), GRIPPER_MIN_RAD), GRIPPER_MAX_RAD)
    return CLOSED_REFERENCE_TCP_Z_M + _poly(bounded) - _REFERENCE_GEOMETRY_Z_M
