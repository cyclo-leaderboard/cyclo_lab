"""UI environment classification stays independent of task numbers."""

import importlib.util
from pathlib import Path
import sys
import unittest


MODULE = Path(__file__).resolve().parents[2] / "source/cyclo_lab/cyclo_lab/runtime/bridges/capabilities.py"
SPEC = importlib.util.spec_from_file_location("bridge_capabilities_test", MODULE)
capabilities = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = capabilities
SPEC.loader.exec_module(capabilities)


class BridgeCapabilitiesTest(unittest.TestCase):
    def test_regular_sg2_and_bg2_joint_tasks_accept_policy_control(self):
        terms = ("arm_l_action", "gripper_l_action", "arm_r_action", "gripper_r_action", "head_action", "lift_action")
        for gym_id in (
            "Cyclo-Real-Showroom-Task000668-FFW-SG2-v0",
            "Cyclo-Real-WhiteRoom-Task000853-FFW-BG2-v0",
        ):
            self.assertTrue(capabilities.supports_policy_control(gym_id, terms, 19))

    def test_generation_and_non_joint_actions_are_preview_only(self):
        terms = ("arm_l_action", "gripper_l_action", "arm_r_action", "gripper_r_action")
        for marker in ("Mimic-Seed", "Trajectory-Generation", "Visual-Replay", "Locomanipulation-SDG"):
            gym_id = f"Cyclo-Real-WhiteRoom-Task000853-{marker}-FFW-BG2-v0"
            self.assertFalse(capabilities.supports_policy_control(gym_id, terms, 19))
        self.assertFalse(capabilities.supports_policy_control(
            "Cyclo-Real-WhiteRoom-Task000999-FFW-BG2-v0", ("right_arm_ik",), 19,
        ))


if __name__ == "__main__":
    unittest.main()
