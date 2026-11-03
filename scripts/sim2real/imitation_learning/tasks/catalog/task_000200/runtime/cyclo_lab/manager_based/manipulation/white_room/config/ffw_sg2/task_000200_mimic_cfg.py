"""Right-arm seed and independent-object random generation for Task000200."""

from isaaclab.envs import mdp as isaaclab_mdp
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from cyclo_lab.manager_based.manipulation.common import configure_ffw_sg2_mimic_ik_actions
from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.common import (
    EpisodicShowroomObservationsCfg,
)

from . import task_000200_mimic_terms as terms
from .task_000200 import SOFT_GRIPPER_TCP_Z_M, Task000200EnvCfg
from .task_000200_events import randomize_task000200_objects


@configclass
class Task000200MimicObservationsCfg(EpisodicShowroomObservationsCfg):
    @configclass
    class SubtaskCfg(ObsGroup):
        tape_lifted = None
        tape_placed_on_plate = None

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class Task000200MimicTerminationsCfg:
    time_out = DoneTerm(func=isaaclab_mdp.time_out, time_out=True)
    success = None


@configclass
class Task000200RightMimicSeedEnvCfg(Task000200EnvCfg, MimicEnvCfg):
    env_name: str = "Cyclo-Real-WhiteRoom-Task000200-Right-Mimic-Seed-FFW-SG2-v0"
    observations: Task000200MimicObservationsCfg = Task000200MimicObservationsCfg()
    terminations: Task000200MimicTerminationsCfg = Task000200MimicTerminationsCfg()

    def __post_init__(self):
        requested_name = self.env_name
        super().__post_init__()
        self.env_name = requested_name
        self.observations.subtask_terms.tape_lifted = ObsTerm(func=terms.tape_lifted)
        self.observations.subtask_terms.tape_placed_on_plate = ObsTerm(
            func=terms.tape_placed_on_plate
        )
        self.terminations.success = DoneTerm(func=terms.task_success)

        self.datagen_config.name = "task000200_GENERATED_right_white_room_v1"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 200
        self.datagen_config.generation_select_src_per_subtask = False
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = False
        self.datagen_config.max_num_failures = 100
        self.datagen_config.seed = 20261001

        self.subtask_configs.clear()
        self.subtask_configs["right_arm"] = [
            SubTaskConfig(
                object_ref="blue_tape",
                subtask_term_signal="tape_lifted",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 1},
                action_noise=0.0,
                num_interpolation_steps=10,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Grasp and lift the blue tape with the right gripper",
                next_subtask_description="Place the tape on the yellow plate",
            ),
            SubTaskConfig(
                object_ref="yellow_plate",
                subtask_term_signal="tape_placed_on_plate",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 1},
                action_noise=0.0,
                num_interpolation_steps=10,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Place the tape on the yellow plate",
                next_subtask_description="Return the right arm to home",
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
                description="Return the right arm to home",
                next_subtask_description="Task complete",
            ),
        ]

    def init_action_cfg(self, mode: str) -> None:
        if mode == "mimic_ik":
            configure_ffw_sg2_mimic_ik_actions(
                self.actions,
                joint_hold_sides=("l",),
                eef_offset_z_m=SOFT_GRIPPER_TCP_Z_M,
            )
            return
        super().init_action_cfg(mode)


@configclass
class Task000200RightMimicGenerateEnvCfg(Task000200RightMimicSeedEnvCfg):
    env_name: str = "Cyclo-Real-WhiteRoom-Task000200-Right-Mimic-Generate-FFW-SG2-v0"

    def __post_init__(self):
        requested_name = self.env_name
        super().__post_init__()
        self.env_name = requested_name
        self.events.set_task000200_seed_objects = None
        self.events.randomize_task000200_objects = EventTerm(
            func=randomize_task000200_objects,
            mode="reset",
        )
