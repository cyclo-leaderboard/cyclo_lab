"""Pure source-demo contract helpers for Task000668 trajectory generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contract import (
    LEFT_SPAWN_BOX,
    LOWER_SHELF_SUPPORT_Z_M,
    PEANUT_ORIGIN_ABOVE_SUPPORT_M,
    RIGHT_SPAWN_BOX,
)


GRIPPER_INDEX_BY_SIDE = {"left": 7, "right": 15}


@dataclass(frozen=True)
class PickPlacePhase:
    """Inferred active-side gripper boundaries in one successful seed."""

    side: str
    close_onset_step: int
    grasp_step: int
    release_step: int
    open_value: float
    closed_value: float


def infer_spawn_side(x_m: float, y_m: float, tolerance_m: float = 0.03) -> str:
    """Map a target root position to exactly one Task000668 spawn box."""

    matches = []
    for side, box in (("left", LEFT_SPAWN_BOX), ("right", RIGHT_SPAWN_BOX)):
        if (
            box.x_min_m - tolerance_m <= x_m <= box.x_max_m + tolerance_m
            and box.y_min_m - tolerance_m <= y_m <= box.y_max_m + tolerance_m
        ):
            matches.append(side)
    if len(matches) != 1:
        raise ValueError(
            f"target XY ({x_m:.5f}, {y_m:.5f}) does not identify one spawn side"
        )
    return matches[0]


def _first_sustained(mask: np.ndarray, start: int, count: int) -> int | None:
    for step in range(max(0, start), len(mask) - count + 1):
        if bool(np.all(mask[step : step + count])):
            return step
    return None


def infer_pick_place_phase(
    actions: np.ndarray,
    target_post_step_poses: np.ndarray,
    *,
    side: str,
    stable_steps: int = 4,
    minimum_excursion: float = 0.10,
) -> PickPlacePhase:
    """Infer close/grasp/release without relying on the optional G key.

    Task000668 seeds are stored with B/N, so G markers are intentionally not
    required. The active soft-gripper excursion supplies close/open timing and
    the target reaching lower-shelf height disambiguates release from startup.
    """

    actions = np.asarray(actions, dtype=np.float64)
    target = np.asarray(target_post_step_poses, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 16:
        raise ValueError(f"Task000668 source actions must have shape (T, 16), got {actions.shape}")
    if target.ndim != 2 or target.shape[1] < 3 or len(target) != len(actions):
        raise ValueError("target post-step poses must align one-to-one with actions")
    if len(actions) < max(20, stable_steps * 4):
        raise ValueError("Task000668 source demo is too short to infer pick/place phases")
    if side not in GRIPPER_INDEX_BY_SIDE:
        raise ValueError(f"unknown side {side!r}")

    values = actions[:, GRIPPER_INDEX_BY_SIDE[side]]
    if not np.isfinite(values).all() or not np.isfinite(target[:, :3]).all():
        raise ValueError("source gripper/target trajectory contains non-finite values")
    open_window = min(10, max(stable_steps, len(values) // 20))
    open_value = float(np.median(values[:open_window]))
    signed_excursion = values - open_value
    peak_step = int(np.argmax(np.abs(signed_excursion)))
    peak = float(signed_excursion[peak_step])
    if abs(peak) < minimum_excursion:
        raise ValueError("active soft gripper has no clear close excursion")
    direction = 1.0 if peak > 0.0 else -1.0
    progress = direction * signed_excursion
    closed_value = float(np.median(values[peak_step : min(len(values), peak_step + stable_steps)]))
    excursion = direction * (closed_value - open_value)
    if excursion < minimum_excursion:
        excursion = float(np.max(progress))
        closed_value = open_value + direction * excursion

    grasp_mask = progress >= 0.80 * excursion
    grasp_step = _first_sustained(grasp_mask, 1, stable_steps)
    if grasp_step is None:
        raise ValueError("active soft gripper never reaches a stable closed command")
    onset_candidates = np.flatnonzero(progress[: grasp_step + 1] >= 0.15 * excursion)
    if not len(onset_candidates):
        raise ValueError("cannot locate active soft-gripper close onset")
    close_onset_step = int(onset_candidates[0])

    lower_z = LOWER_SHELF_SUPPORT_Z_M + PEANUT_ORIGIN_ABOVE_SUPPORT_M
    on_lower_shelf = np.abs(target[:, 2] - lower_z) <= 0.07
    released = progress <= 0.30 * excursion
    release_mask = on_lower_shelf & released
    release_step = _first_sustained(release_mask, grasp_step + stable_steps, stable_steps)
    if release_step is None:
        raise ValueError(
            "source must open the active gripper after placing the target on the lower shelf"
        )
    if len(actions) - release_step < stable_steps * 2:
        raise ValueError("source must include a post-release return toward HOME")

    return PickPlacePhase(
        side=side,
        close_onset_step=close_onset_step,
        grasp_step=grasp_step,
        release_step=release_step,
        open_value=open_value,
        closed_value=closed_value,
    )
