"""Isaac-free policy-control eligibility for UI-launched environments."""

from __future__ import annotations

from collections.abc import Iterable


_POLICY_TERMS = frozenset((
    "arm_l_action", "gripper_l_action", "arm_r_action", "gripper_r_action",
))
_PREVIEW_ONLY_MARKERS = (
    "-Mimic-", "-Trajectory-Generation-", "-Visual-Replay-", "-Locomanipulation-SDG-",
)


def supports_policy_control(gym_id: str, active_terms: Iterable[str], action_dim: int) -> bool:
    """Keep generation/IK environments available for preview without action routing."""

    return (
        ("FFW-SG2" in gym_id or "FFW-BG2" in gym_id)
        and not any(marker in gym_id for marker in _PREVIEW_ONLY_MARKERS)
        and _POLICY_TERMS.issubset(active_terms)
        and action_dim in (16, 19, 22)
    )
