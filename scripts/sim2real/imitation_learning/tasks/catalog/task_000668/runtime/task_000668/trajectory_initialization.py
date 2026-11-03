"""Deterministic initial-lift sampling and task/home frame interpolation."""

from __future__ import annotations

import numpy as np

from .contract import LIFT_RANDOM_POSITION_RANGE_M


def sample_lift_positions(
    episode_seeds: list[int],
    position_range_m: tuple[float, float] = LIFT_RANDOM_POSITION_RANGE_M,
) -> np.ndarray:
    """Use a separate RNG stream from target XY; never sample during replay."""

    lower, upper = position_range_m
    if not np.isfinite([lower, upper]).all() or lower > upper:
        raise ValueError(f"invalid lift range: {position_range_m}")
    return np.asarray(
        [
            np.random.default_rng(np.random.SeedSequence([seed, 668, 1])).uniform(lower, upper)
            for seed in episode_seeds
        ],
        dtype=np.float32,
    )


def active_lift_frame_weight(
    step: int, grasp_step: int, release_step: int, unwind_steps: int
) -> float:
    """Start at lifted HOME, keep shelf-frame manipulation, return to HOME.

    Only HOME/approach/return are lift-relative. Grasp-to-release remains in
    the shelf frame because cabinet shelf heights do not move with the lift.
    """

    if step < grasp_step:
        return max(0.0, 1.0 - step / max(1, grasp_step))
    if step <= release_step:
        return 0.0
    return min(1.0, (step - release_step) / max(1, unwind_steps))
