"""Isaac Lab Mimic adapter for WhiteRoom Task000100."""

from __future__ import annotations

import os
from collections.abc import Sequence

import isaaclab.utils.math as PoseUtils
import torch

from cyclo_lab.manager_based.manipulation.common import FFWSG2MimicEnv

from . import mimic_terms, task_000300_mimic_terms
from .contact_failure import contact_failures, reset_contact_failures


class Task000100MimicEnv(FFWSG2MimicEnv):
    """Expose WhiteRoom rigid objects in the randomized robot-root frame."""

    def get_generation_success(self, env_id: int, success_term) -> bool:
        """Use final containment and latched forbidden contacts for the label."""
        success = bool(mimic_terms.task_success(self, self.cfg.mimic_side)[env_id].item())
        if os.environ.get("CYCLO_TASK000100_LOG_OUTCOME") == "1":
            table, basket = contact_failures(self)
            delta = self.scene["navy_block"].data.root_pos_w[env_id] - self.scene["white_basket"].data.root_pos_w[env_id]
            print(
                f"TASK000100_OUTCOME env={env_id} "
                f"status={'ACCEPTED' if success else 'REJECTED'}"
                f" success={int(success)} table_contact={int(table[env_id].item())}"
                f" gripper_basket_contact={int(basket[env_id].item())}"
                f" block_basket_delta_m={delta.tolist()}"
                f" table_peak_n={self._task000100_table_peak_force_n[env_id].item():.6g}"
                f" basket_peak_n={self._task000100_basket_peak_force_n[env_id].item():.6g}",
                flush=True,
            )
        return success

    def update_generation_success_tracking(self, env_id: int, success_term) -> bool:
        """Sample contact history every action so early collisions remain failures."""
        return bool(mimic_terms.task_success(self, self.cfg.mimic_side)[env_id].item())

    def _finish_episode_reset(self, env_ids: Sequence[int] | None) -> None:
        reset_contact_failures(self, env_ids)
        mimic_terms.reset_metric_state(self, env_ids)
        self.obs_buf = self.observation_manager.compute(update_history=True)

    def reset(self, seed=None, env_ids=None, options=None):
        _, extras = super().reset(seed=seed, env_ids=env_ids, options=options)
        self._finish_episode_reset(env_ids)
        return self.obs_buf, extras

    def reset_to(self, state, env_ids, seed=None, is_relative=False):
        _, extras = super().reset_to(
            state, env_ids, seed=seed, is_relative=is_relative
        )
        self._finish_episode_reset(env_ids)
        return self.obs_buf, extras

    def get_object_poses(
        self, env_ids: Sequence[int] | None = None
    ) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        scene_state = self.scene.get_state(is_relative=True)
        robot_root = scene_state["articulation"]["robot"]["root_pose"]
        root_pos = robot_root[env_ids, :3]
        root_quat = robot_root[env_ids, 3:7]
        object_poses = {}
        for name, object_state in scene_state["rigid_object"].items():
            position, quaternion = PoseUtils.subtract_frame_transforms(
                root_pos,
                root_quat,
                object_state["root_pose"][env_ids, :3],
                object_state["root_pose"][env_ids, 3:7],
            )
            object_poses[name] = PoseUtils.make_pose(
                position, PoseUtils.matrix_from_quat(quaternion)
            )
        return object_poses


class Task000300MimicEnv(Task000100MimicEnv):
    """Use the upright bottle and shallow tray outcome for Task000300."""

    def _finish_episode_reset(self, env_ids: Sequence[int] | None) -> None:
        super()._finish_episode_reset(env_ids)
        task_000300_mimic_terms.reset_metric_state(self, env_ids)
        self._task000300_contact_logged = False

    def get_generation_success(self, env_id: int, success_term) -> bool:
        success = bool(task_000300_mimic_terms.task_success(self, self.cfg.mimic_side)[env_id].item())
        if os.environ.get("CYCLO_TASK000300_LOG_OUTCOME") == "1":
            table, tray = contact_failures(self)
            delta = self.scene["navy_block"].data.root_pos_w[env_id] - self.scene["white_basket"].data.root_pos_w[env_id]
            print(
                f"TASK000300_OUTCOME env={env_id} "
                f"status={'ACCEPTED' if success else 'REJECTED'}"
                f" table_contact={int(table[env_id].item())}"
                f" gripper_tray_contact={int(tray[env_id].item())}"
                f" bottle_tray_delta_m={delta.tolist()}"
                f" table_peak_n={self._task000100_table_peak_force_n[env_id].item():.6g}"
                f" tray_peak_n={self._task000100_basket_peak_force_n[env_id].item():.6g}",
                flush=True,
            )
        return success

    def update_generation_success_tracking(self, env_id: int, success_term) -> bool:
        success = bool(task_000300_mimic_terms.task_success(self, self.cfg.mimic_side)[env_id].item())
        if os.environ.get("CYCLO_TASK000300_LOG_OUTCOME") == "1":
            _, tray_contact = contact_failures(self)
            if bool(tray_contact[env_id].item()) and not getattr(self, "_task000300_contact_logged", False):
                self._task000300_contact_logged = True
                bottle = self.scene["navy_block"].data.root_pos_w[env_id]
                tray = self.scene["white_basket"].data.root_pos_w[env_id]
                print(
                    f"TASK000300_FIRST_TRAY_CONTACT sim_step={self._sim_step_counter} "
                    f"bottle_tray_delta_m={(bottle - tray).tolist()} "
                    f"peak_n={self._task000100_basket_peak_force_n[env_id].item():.6g}",
                    flush=True,
                )
        return success
