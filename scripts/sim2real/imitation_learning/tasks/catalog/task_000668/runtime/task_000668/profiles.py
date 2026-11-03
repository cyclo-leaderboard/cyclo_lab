"""Authoritative visual-only replay profile for Task000668."""

from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.randomization.cfg import (
    CameraRandomizationCfg,
    LightingRandomizationCfg,
    ShelfAppearanceRandomizationCfg,
    ShowroomGenerationRandomizationCfg,
)


# This profile deliberately leaves wall and every physical axis disabled. A
# visual replay must retain the recorded robot/object state and joint16 labels.
TASK000668_VISUAL_REPLAY_AUGMENTATION = ShowroomGenerationRandomizationCfg(
    lighting=LightingRandomizationCfg(enabled=True),
    shelf=ShelfAppearanceRandomizationCfg(enabled=True),
    camera=CameraRandomizationCfg(
        enabled=True,
        camera_names=("cam_head",),
        local_pitch_max_rad=0.0,
    ),
)

# Render the exact environment defaults for the original 100 trajectories.
TASK000668_VISUAL_REPLAY_BASELINE = ShowroomGenerationRandomizationCfg()
