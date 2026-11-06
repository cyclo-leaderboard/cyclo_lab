"""Task-local compatibility and final-state verdict for the Mimic core."""
import torch
import isaaclab.utils.math as PoseUtils
from isaaclab_mimic.datagen.data_generator import DataGenerator


def _get_delta_object_pose(current_object_pose, source_object_pose):
    """Return the rigid transform that maps source-world poses to current-world poses.

    The vendored Mimic generator still calls ``get_delta_object_pose``, while
    the vendored Isaac Lab math module no longer exports that helper.  Keep the
    compatibility shim local to the Task901 process instead of changing either
    shared package.
    """
    return current_object_pose @ PoseUtils.pose_inv(source_object_pose)


if not hasattr(PoseUtils, "get_delta_object_pose"):
    PoseUtils.get_delta_object_pose = _get_delta_object_pose


class Task901DataGenerator(DataGenerator):
    async def generate(self, env_id, success_term, env_reset_queue=None,
                       env_action_queue=None, pause_subtask=False,
                       export_demo=True, motion_planner=None):
        result = await super().generate(
            env_id=env_id, success_term=success_term,
            env_reset_queue=env_reset_queue, env_action_queue=env_action_queue,
            pause_subtask=pause_subtask, export_demo=False, motion_planner=motion_planner)
        # Upstream OR-latches success over time. Do not accept an episode that
        # briefly succeeded and then dropped a plate during the terminal hold.
        result['success'] = bool(success_term.func(self.env, **success_term.params)[env_id])
        ids = torch.tensor([env_id], dtype=torch.long, device=self.env.device)
        self.env.recorder_manager.set_success_to_episodes(
            ids, torch.tensor([[result['success']]], dtype=torch.bool, device=self.env.device))
        if export_demo:
            self.env.recorder_manager.export_episodes(ids)
        return result
