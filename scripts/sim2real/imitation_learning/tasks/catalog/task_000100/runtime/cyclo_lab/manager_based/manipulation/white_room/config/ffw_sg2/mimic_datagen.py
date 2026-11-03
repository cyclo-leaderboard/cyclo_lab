"""Task000100 trajectory stitching for a fixed home pose and randomized objects."""

import torch

from isaaclab_mimic.datagen.data_generator import DataGenerator


def anchor_segment_to_current_pose(trajectory, current_pose, subtask_index: int):
    """Remove a phase-start jump, then blend back to the retargeted endpoint.

    Generic MimicGen transforms every waypoint in an object-referenced phase.
    Its first waypoint therefore moves with the randomized block or basket,
    even though this task's robot always starts from the same joint pose. At a
    carry transition, the basket transform can relocate the first carry
    waypoint away from the grasp pose. Keep the first target at the measured
    current EEF pose and fade that correction out before the phase endpoint.
    """
    count = len(trajectory)
    if count < 2:
        raise ValueError("Task000100 phase needs at least two EEF waypoints")
    positions = torch.stack([trajectory[index].pose[:3, 3] for index in range(count)])
    correction = current_pose[:3, 3] - positions[0]
    if not bool(torch.isfinite(correction).all().item()):
        raise ValueError("Task000100 EEF anchor correction is not finite")
    if float(torch.linalg.vector_norm(correction).item()) > 0.35:
        raise ValueError("Task000100 phase-start EEF jump exceeds 35 cm")

    # Preserve the seed's initial stationary frames. The first phase's source
    # arm does not move until the human starts the reach; randomization should
    # not turn those frames into an artificial backward approach.
    if subtask_index == 0:
        moved = torch.nonzero(torch.linalg.vector_norm(positions - positions[0], dim=1) > 0.005)
        fade_start = min(int(moved[0].item()), count - 2) if len(moved) else 0
    else:
        fade_start = 0
    fade_end = count - 1
    for index in range(count):
        if index <= fade_start:
            weight = 1.0
        else:
            fraction = (index - fade_start) / (fade_end - fade_start)
            weight = 1.0 - (3.0 * fraction**2 - 2.0 * fraction**3)
        waypoint = trajectory[index]
        waypoint.pose = waypoint.pose.clone()
        waypoint.pose[:3, 3] += correction * weight
    return trajectory


class Task000100DataGenerator(DataGenerator):
    """Use the source motion while making each transformed phase continuous."""

    def generate_eef_subtask_trajectory(
        self, env_id, eef_name, subtask_ind, all_randomized_subtask_boundaries,
        runtime_subtask_constraints_dict, selected_src_demo_inds,
    ):
        trajectory = super().generate_eef_subtask_trajectory(
            env_id=env_id,
            eef_name=eef_name,
            subtask_ind=subtask_ind,
            all_randomized_subtask_boundaries=all_randomized_subtask_boundaries,
            runtime_subtask_constraints_dict=runtime_subtask_constraints_dict,
            selected_src_demo_inds=selected_src_demo_inds,
        )
        current_pose = self.env.get_robot_eef_pose(eef_name=eef_name, env_ids=[env_id])[0]
        return anchor_segment_to_current_pose(trajectory, current_pose, subtask_ind)


class Task000300DataGenerator(Task000100DataGenerator):
    """Keep the grasped bottle above the 2 cm tray rim through release."""

    def generate_eef_subtask_trajectory(
        self, env_id, eef_name, subtask_ind, all_randomized_subtask_boundaries,
        runtime_subtask_constraints_dict, selected_src_demo_inds,
    ):
        trajectory = super().generate_eef_subtask_trajectory(
            env_id=env_id,
            eef_name=eef_name,
            subtask_ind=subtask_ind,
            all_randomized_subtask_boundaries=all_randomized_subtask_boundaries,
            runtime_subtask_constraints_dict=runtime_subtask_constraints_dict,
            selected_src_demo_inds=selected_src_demo_inds,
        )
        if subtask_ind == 1:
            count = len(trajectory)
            for index in range(count):
                fraction = index / max(1, count - 1)
                lift_m = 0.012 * (3.0 * fraction**2 - 2.0 * fraction**3)
                waypoint = trajectory[index]
                waypoint.pose = waypoint.pose.clone()
                waypoint.pose[2, 3] += lift_m
        return trajectory
