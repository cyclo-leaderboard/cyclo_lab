"""Both hands participate in every Mimic action (unlike the shared SG2 adapter)."""

import torch

from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.utils import math as mu


class Task901BimanualMimicEnv(ManagerBasedRLMimicEnv):
    EEF_SLICES = {"left_arm": slice(0, 7), "right_arm": slice(8, 15)}
    GRIP_INDICES = {"left_arm": 7, "right_arm": 15}

    def get_robot_eef_pose(self, eef_name, env_ids=None):
        if eef_name not in self.EEF_SLICES:
            raise ValueError(f"Unknown Task901 end effector: {eef_name}")
        ids = slice(None) if env_ids is None else env_ids
        pose = self.obs_buf["policy"][f"{eef_name.split('_')[0]}_eef_pose"][ids]
        return mu.make_pose(pose[..., :3], mu.matrix_from_quat(pose[..., 3:7]))

    def target_eef_pose_to_action(self, target_eef_pose_dict, gripper_action_dict,
                                  action_noise_dict=None, env_id=0):
        if set(target_eef_pose_dict) != set(self.EEF_SLICES):
            raise ValueError("Task901 requires BOTH left_arm and right_arm pose targets")
        if set(gripper_action_dict) != set(self.EEF_SLICES):
            raise ValueError("Task901 requires BOTH gripper commands")
        if action_noise_dict and any(float(v) != 0 for v in action_noise_dict.values()):
            raise ValueError("Contact-preserving pilot requires action_noise=0")
        parts = []
        for name in self.EEF_SLICES:
            pos, rot = mu.unmake_pose(target_eef_pose_dict[name])
            parts.extend((pos.reshape(3), mu.quat_from_matrix(rot).reshape(4),
                          gripper_action_dict[name].reshape(1)))
        # Source head/lift are effectively stationary; retain their reset targets.
        parts.append(self.obs_buf["policy"]["joint_pos_target"][env_id, 16:19])
        action = torch.cat(parts).unsqueeze(0)
        if action.shape != (1, 19) or not bool(torch.isfinite(action).all()):
            raise ValueError("Invalid Task901 EEF19 action")
        return action

    def action_to_target_eef_pose(self, action):
        return {name: mu.make_pose(action[..., span][..., :3],
                                  mu.matrix_from_quat(action[..., span][..., 3:7]))
                for name, span in self.EEF_SLICES.items()}

    def actions_to_gripper_actions(self, actions):
        return {name: actions[..., index:index + 1]
                for name, index in self.GRIP_INDICES.items()}

    def get_object_poses(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        robot = self.scene["robot"]
        out = {}
        for name in ("plate_blue", "plate_yellow", "plate_pink"):
            obj = self.scene[name]
            pos, quat = mu.subtract_frame_transforms(
                robot.data.root_pos_w[ids], robot.data.root_quat_w[ids],
                obj.data.root_pos_w[ids], obj.data.root_quat_w[ids])
            out[name] = mu.make_pose(pos, mu.matrix_from_quat(quat))
        # Holder is static AssetBase, not a RigidObject, so use its authored pose.
        holder = self.cfg.scene.plate_holder.init_state
        pos = torch.tensor(holder.pos, device=self.device).expand(self.num_envs, -1)
        quat = torch.tensor(holder.rot, device=self.device).expand(self.num_envs, -1)
        pos, quat = mu.subtract_frame_transforms(
            robot.data.root_pos_w[ids], robot.data.root_quat_w[ids],
            (pos + self.scene.env_origins)[ids], quat[ids])
        out["plate_holder"] = mu.make_pose(pos, mu.matrix_from_quat(quat))
        return out

    def get_subtask_term_signals(self, env_ids=None):
        raise NotImplementedError("Use the reviewed episode6 phase annotations in the prepared seed")

    def _reset_idx(self, env_ids):
        if hasattr(self, "_rack_stable_steps"):
            self._rack_stable_steps[env_ids] = 0
        self._rack_signal_tick = None
        super()._reset_idx(env_ids)

    def update_generation_success_tracking(self, env_id, success_term):
        return bool(success_term.func(self, **success_term.params)[env_id])

    def get_generation_success(self, env_id, success_term):
        # Final-state success, not a latch that survives a later dropped plate.
        return bool(success_term.func(self, **success_term.params)[env_id])
