"""MimicGen environment adapter for WhiteRoom Task000200."""

from collections.abc import Sequence

import isaaclab.utils.math as PoseUtils
import torch

from cyclo_lab.manager_based.manipulation.common import FFWSG2MimicEnv

from . import task_000200_mimic_terms as terms


class Task000200MimicEnv(FFWSG2MimicEnv):
    def get_generation_success(self, env_id: int, success_term) -> bool:
        return bool(terms.task_success(self)[env_id].item())

    def update_generation_success_tracking(self, env_id: int, success_term) -> bool:
        return bool(terms.task_success(self)[env_id].item())

    def reset(self, seed=None, env_ids=None, options=None):
        _, extras = super().reset(seed=seed, env_ids=env_ids, options=options)
        terms.reset_metric_state(self, env_ids)
        self.obs_buf = self.observation_manager.compute(update_history=True)
        return self.obs_buf, extras

    def reset_to(self, state, env_ids, seed=None, is_relative=False):
        _, extras = super().reset_to(state, env_ids, seed=seed, is_relative=is_relative)
        terms.reset_metric_state(self, env_ids)
        self.obs_buf = self.observation_manager.compute(update_history=True)
        return self.obs_buf, extras

    def get_object_poses(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        scene_state = self.scene.get_state(is_relative=True)
        robot_root = scene_state["articulation"]["robot"]["root_pose"]
        root_pos = robot_root[env_ids, :3]
        root_quat = robot_root[env_ids, 3:7]
        poses = {}
        for name, object_state in scene_state["rigid_object"].items():
            position, quaternion = PoseUtils.subtract_frame_transforms(
                root_pos, root_quat,
                object_state["root_pose"][env_ids, :3],
                object_state["root_pose"][env_ids, 3:7],
            )
            poses[name] = PoseUtils.make_pose(position, PoseUtils.matrix_from_quat(quaternion))
        return poses
