"""Isaac Lab Mimic environments for WhiteRoom Task000100."""

from __future__ import annotations

from isaaclab.envs import mdp as isaaclab_mdp
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from cyclo_lab.manager_based.manipulation.common import (
    configure_ffw_sg2_mimic_ik_actions,
)
from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.common import (
    EpisodicShowroomObservationsCfg,
)
from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.randomization.cfg import (
    ShowroomRandomizationCfg,
)

from . import mimic_terms
from .contact_failure import configure_contact_failure_sensors
from .env_cfg import WHITE_ROOM_RANDOMIZATION, WhiteRoomResetEventsCfg
from .events import randomize_navy_block_pose
from .task_000100 import Task000100EnvCfg, TASK000100_SOFT_GRIPPER_TCP_Z_M
from cyclo_lab.assets.environments.white_room_layout import RIGHT_MIMIC_OUTER_TRIM_M


@configclass
class Task000100MimicObservationsCfg(EpisodicShowroomObservationsCfg):
    @configclass
    class SubtaskCfg(ObsGroup):
        block_lifted = None
        released_in_basket = None

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class Task000100MimicTerminationsCfg:
    time_out = DoneTerm(func=isaaclab_mdp.time_out, time_out=True)
    success = None


@configclass
class Task000100MimicSeedEnvCfg(Task000100EnvCfg, MimicEnvCfg):
    """Shared deterministic seed adapter; concrete registrations select one arm."""

    env_name: str = ""
    mimic_side: str = ""
    observations: Task000100MimicObservationsCfg = Task000100MimicObservationsCfg()
    events: WhiteRoomResetEventsCfg = WhiteRoomResetEventsCfg()
    terminations: Task000100MimicTerminationsCfg = Task000100MimicTerminationsCfg()

    def __post_init__(self):
        requested_env_name = self.env_name
        requested_side = self.mimic_side
        super().__post_init__()
        self.env_name = requested_env_name
        self.mimic_side = requested_side
        if self.mimic_side not in ("left", "right"):
            raise ValueError("Task000100 Mimic requires mimic_side='left' or 'right'")

        signal_params = {"side": self.mimic_side}
        self.observations.subtask_terms.block_lifted = ObsTerm(
            func=mimic_terms.block_lifted, params=signal_params
        )
        self.observations.subtask_terms.released_in_basket = ObsTerm(
            func=mimic_terms.released_in_basket, params=signal_params
        )
        self.terminations.success = DoneTerm(
            func=mimic_terms.task_success, params=signal_params
        )
        configure_contact_failure_sensors(self)

        self.datagen_config.name = (
            f"task000100_GENERATED_{self.mimic_side}_white_room_v1"
        )
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 200
        self.datagen_config.generation_select_src_per_subtask = False
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = False
        self.datagen_config.max_num_failures = 100
        self.datagen_config.seed = 20260928

        self.subtask_configs.clear()
        arm_name = f"{self.mimic_side}_arm"
        self.subtask_configs[arm_name] = [
            SubTaskConfig(
                object_ref="navy_block",
                subtask_term_signal="block_lifted",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 1},
                action_noise=0.0,
                num_interpolation_steps=10,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Approach and lift the navy cosmetic block",
                next_subtask_description="Carry the block to the white basket and release",
            ),
            SubTaskConfig(
                object_ref="white_basket",
                subtask_term_signal="released_in_basket",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 1},
                action_noise=0.0,
                num_interpolation_steps=10,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Carry the block to the white basket and release",
                next_subtask_description="Return the active arm to its initial pose",
            ),
            SubTaskConfig(
                object_ref=None,
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="random",
                selection_strategy_kwargs={},
                action_noise=0.0,
                num_interpolation_steps=10,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Return the active arm to its initial pose",
                next_subtask_description="Task complete",
            ),
        ]

    def init_action_cfg(self, mode: str) -> None:
        if mode == "mimic_ik":
            passive_side = "r" if self.mimic_side == "left" else "l"
            configure_ffw_sg2_mimic_ik_actions(
                self.actions,
                joint_hold_sides=(passive_side,),
                eef_offset_z_m=TASK000100_SOFT_GRIPPER_TCP_Z_M,
            )
            return
        super().init_action_cfg(mode)


@configclass
class Task000100LeftMimicSeedEnvCfg(Task000100MimicSeedEnvCfg):
    env_name: str = "Cyclo-Real-WhiteRoom-Task000100-Left-Mimic-Seed-FFW-SG2-v0"
    mimic_side: str = "left"
    seed_side: str = "left"


@configclass
class Task000100RightMimicSeedEnvCfg(Task000100MimicSeedEnvCfg):
    env_name: str = "Cyclo-Real-WhiteRoom-Task000100-Right-Mimic-Seed-FFW-SG2-v0"
    mimic_side: str = "right"
    seed_side: str = "right"


@configclass
class Task000100LeftMimicGenerateEnvCfg(Task000100LeftMimicSeedEnvCfg):
    env_name: str = "Cyclo-Real-WhiteRoom-Task000100-Left-Mimic-Generate-FFW-SG2-v0"
    randomization: ShowroomRandomizationCfg = WHITE_ROOM_RANDOMIZATION

    def __post_init__(self):
        requested_env_name = self.env_name
        super().__post_init__()
        self.env_name = requested_env_name
        self.events.randomize_navy_block_pose = EventTerm(
            func=randomize_navy_block_pose,
            mode="reset",
            params={"side": self.mimic_side},
        )


@configclass
class Task000100RightMimicGenerateEnvCfg(Task000100RightMimicSeedEnvCfg):
    env_name: str = "Cyclo-Real-WhiteRoom-Task000100-Right-Mimic-Generate-FFW-SG2-v0"
    randomization: ShowroomRandomizationCfg = WHITE_ROOM_RANDOMIZATION

    def __post_init__(self):
        requested_env_name = self.env_name
        super().__post_init__()
        self.env_name = requested_env_name
        self.events.randomize_navy_block_pose = EventTerm(
            func=randomize_navy_block_pose,
            mode="reset",
            params={"side": self.mimic_side, "right_outer_trim_m": RIGHT_MIMIC_OUTER_TRIM_M},
        )
