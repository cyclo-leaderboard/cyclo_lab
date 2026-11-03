"""Head-camera, dual-arm deployment contract for Task000668."""

from cyclo_lab.robot_specs.ffw.sg2 import FFW_SG2_ACTION_JOINT_NAMES


_ARM_AND_GRIPPER_NAMES = FFW_SG2_ACTION_JOINT_NAMES[:16]

POLICY_CONTRACT = {
    "task": "task_000668",
    "task_name": "Move_PeanutMix_Showroom_SimtoReal",
    "robot": "ffw_sg2_rev1",
    "policy_hz": 15,
    "state_components": ("arm_left", "arm_right", "lift"),
    "action_components": ("arm_left", "arm_right"),
    "state_names": (*_ARM_AND_GRIPPER_NAMES, "lift_joint"),
    "action_names": _ARM_AND_GRIPPER_NAMES,
    "inactive_actions": {
        "head": "hold_current",
        "lift": "hold_current",
        "mobile": "hold_current",
    },
    "cameras": {"cam_left_head": {"width": 672, "height": 376}},
    "simulation": {
        "environment": "Cyclo-Real-Showroom-Task000668-FFW-SG2-v0",
        "randomized_environment": (
            "Cyclo-Real-Showroom-Task000668-Random-FFW-SG2-v0"
        ),
        "default_reset": "random_left_or_right_target_fixed_lift_-0.10m",
        "randomized_reset": "random_left_or_right_target_uniform_lift_[-0.15,-0.05]m",
    },
}
