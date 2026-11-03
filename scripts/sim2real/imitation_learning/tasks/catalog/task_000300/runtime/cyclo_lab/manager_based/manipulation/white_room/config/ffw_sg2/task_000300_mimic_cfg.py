"""Left- and right-arm MimicGen environments for WhiteRoom Task000300."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from cyclo_lab.assets.environments.white_room_task000300 import (
    make_task000300_environment_cfg,
    make_task000300_navy_cosmetic_cfg,
    make_task000300_orange_tray_cfg,
)
from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.randomization.cfg import (
    ShowroomRandomizationCfg,
)

from . import task_000300_mimic_terms
from .env_cfg import WHITE_ROOM_RANDOMIZATION
from .mimic_cfg import (
    Task000100LeftMimicSeedEnvCfg,
    Task000100RightMimicSeedEnvCfg,
)
from .task_000300 import TASK_000300_SPEC, randomize_task000300_cosmetic_pose


@configclass
class Task000300RightMimicSeedEnvCfg(Task000100RightMimicSeedEnvCfg):
    task_spec = TASK_000300_SPEC
    env_name: str = "Cyclo-Real-WhiteRoom-Task000300-Right-Mimic-Seed-FFW-SG2-v0"

    def __post_init__(self):
        requested_env_name = self.env_name
        super().__post_init__()
        self.env_name = requested_env_name
        # This adapter inherits Task000100 directly, so mirror the seed robot policy.
        self.scene.robot.spawn.rigid_props.disable_gravity = True
        self.scene.environment = make_task000300_environment_cfg()
        self.scene.navy_block = make_task000300_navy_cosmetic_cfg()
        self.scene.white_basket = make_task000300_orange_tray_cfg()
        params = {"side": self.mimic_side}
        self.observations.subtask_terms.block_lifted = ObsTerm(
            func=task_000300_mimic_terms.block_lifted, params=params
        )
        self.observations.subtask_terms.released_in_basket = ObsTerm(
            func=task_000300_mimic_terms.released_in_basket, params=params
        )
        self.terminations.success = DoneTerm(
            func=task_000300_mimic_terms.task_success, params=params
        )
        self.datagen_config.name = "task000300_GENERATED_right_white_room_v1"
        # Bound diagnostics and production runs by attempts, retaining failures separately.
        self.datagen_config.generation_guarantee = False
        for subtask in self.subtask_configs["right_arm"]:
            subtask.num_interpolation_steps = 0


@configclass
class Task000300RightMimicGenerateEnvCfg(Task000300RightMimicSeedEnvCfg):
    env_name: str = "Cyclo-Real-WhiteRoom-Task000300-Right-Mimic-Generate-FFW-SG2-v0"
    randomization: ShowroomRandomizationCfg = WHITE_ROOM_RANDOMIZATION

    def __post_init__(self):
        requested_env_name = self.env_name
        super().__post_init__()
        self.env_name = requested_env_name
        self.events.randomize_navy_block_pose = EventTerm(
            func=randomize_task000300_cosmetic_pose,
            mode="reset",
            params={"side": "right"},
        )


@configclass
class Task000300LeftMimicSeedEnvCfg(Task000100LeftMimicSeedEnvCfg):
    """Left-arm Task000300 adapter for mirrored right-arm source motions."""

    task_spec = TASK_000300_SPEC
    env_name: str = "Cyclo-Real-WhiteRoom-Task000300-Left-Mimic-Seed-FFW-SG2-v0"

    def __post_init__(self):
        requested_env_name = self.env_name
        super().__post_init__()
        self.env_name = requested_env_name
        # This adapter inherits Task000100 directly, so mirror the seed robot policy.
        self.scene.robot.spawn.rigid_props.disable_gravity = True
        self.scene.environment = make_task000300_environment_cfg()
        self.scene.navy_block = make_task000300_navy_cosmetic_cfg()
        self.scene.white_basket = make_task000300_orange_tray_cfg()
        params = {"side": self.mimic_side}
        self.observations.subtask_terms.block_lifted = ObsTerm(
            func=task_000300_mimic_terms.block_lifted, params=params
        )
        self.observations.subtask_terms.released_in_basket = ObsTerm(
            func=task_000300_mimic_terms.released_in_basket, params=params
        )
        self.terminations.success = DoneTerm(
            func=task_000300_mimic_terms.task_success, params=params
        )
        self.datagen_config.name = "task000300_GENERATED_left_white_room_v1"
        self.datagen_config.generation_guarantee = False
        for subtask in self.subtask_configs["left_arm"]:
            subtask.num_interpolation_steps = 0


@configclass
class Task000300LeftMimicGenerateEnvCfg(Task000300LeftMimicSeedEnvCfg):
    env_name: str = "Cyclo-Real-WhiteRoom-Task000300-Left-Mimic-Generate-FFW-SG2-v0"
    randomization: ShowroomRandomizationCfg = WHITE_ROOM_RANDOMIZATION

    def __post_init__(self):
        requested_env_name = self.env_name
        super().__post_init__()
        self.env_name = requested_env_name
        self.events.randomize_navy_block_pose = EventTerm(
            func=randomize_task000300_cosmetic_pose,
            mode="reset",
            params={"side": "left"},
        )
