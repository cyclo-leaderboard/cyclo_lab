"""Identity and platform contract for showroom Task000668."""

from cyclo_lab.assets.environments.robotis_showroom import ROBOTIS_SHOWROOM_BACKGROUND_USD_PATH
from cyclo_lab.robot_specs.ffw.sg2 import DEFAULT_SG2_CAMERA_PROFILE

from ..common import ShowroomTaskSpec
from .contract import TASK_INSTRUCTION


TASK_000668_NAME = "Move_PeanutMix_Showroom_SimtoReal"
TASK_000668_INSTRUCTION = TASK_INSTRUCTION

TASK_000668_SPEC = ShowroomTaskSpec(
    task_id="000668",
    env_name="Cyclo-Real-Showroom-Task000668-FFW-SG2-v0",
    instruction=TASK_000668_INSTRUCTION,
    target_object="peanut_mix_bag",
    target_side="left_or_right",
    environment_usd_path=ROBOTIS_SHOWROOM_BACKGROUND_USD_PATH,
    policy_cameras=("cam_head", "cam_wrist_left", "cam_wrist_right"),
    robot_profile=DEFAULT_SG2_CAMERA_PROFILE,
)
