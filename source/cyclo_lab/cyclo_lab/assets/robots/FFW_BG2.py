# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Taehyeong Kim

import math
from copy import deepcopy

from isaacsim.core.utils.stage import get_current_stage
from pxr import Usd, UsdPhysics

from isaaclab.sim import UsdFileCfg, RigidBodyPropertiesCfg, ArticulationRootPropertiesCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim.spawners.from_files import from_files
from isaaclab.sim.utils import clone, make_uninstanceable

from cyclo_lab.assets.robots import CYCLO_LAB_ASSETS_DATA_DIR


_BG2_HEAD_JOINT1_UPPER_LIMIT_RAD = 1.0


@clone
def spawn_bg2_with_deployed_head_limit(
    prim_path, cfg, translation=None, orientation=None, **kwargs
):
    """Spawn BG2 with the deployed 1.0 rad head joint range."""
    prim = from_files.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    stage = get_current_stage()
    make_uninstanceable(prim_path, stage)
    matches = []
    for child_prim in Usd.PrimRange(stage.GetPrimAtPath(prim_path)):
        if child_prim.GetName() != "head_joint1" or not child_prim.IsA(
            UsdPhysics.RevoluteJoint
        ):
            continue
        joint = UsdPhysics.RevoluteJoint(child_prim)
        joint.GetUpperLimitAttr().Set(math.degrees(_BG2_HEAD_JOINT1_UPPER_LIMIT_RAD))
        matches.append(str(child_prim.GetPath()))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one BG2 head_joint1 below {prim_path}, found {matches}."
        )
    print(
        "[BG2 head limits] set head_joint1 upper limit to "
        f"{_BG2_HEAD_JOINT1_UPPER_LIMIT_RAD:.3f} rad."
    )
    return prim

FFW_BG2_CFG = ArticulationCfg(
    spawn=UsdFileCfg(
        usd_path=f"{CYCLO_LAB_ASSETS_DATA_DIR}/robots/FFW/FFW_BG2_soft_tips.usda",
        rigid_props=RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # Lift joint
            "lift_joint": 0.0,

            # Left arm joints
            **{f"arm_l_joint{i + 1}": 0.0 for i in range(7)},
            # Right arm joints
            **{f"arm_r_joint{i + 1}": 0.0 for i in range(7)},

            # Left and right gripper joints
            **{f"gripper_l_joint{i + 1}": 0.0 for i in range(4)},
            **{f"gripper_r_joint{i + 1}": 0.0 for i in range(4)},

            # Head joints
            **{f"head_joint{i + 1}": 0.0 for i in range(2)},
        },
    ),
    actuators={
        # Actuator for vertical lift joint
        "lift": ImplicitActuatorCfg(
            joint_names_expr=["lift_joint"],
            velocity_limit_sim=0.2,
            effort_limit_sim=1000000.0,
            stiffness=10000.0,
            damping=100.0,
        ),

        # Actuators for both arms
        "DY_80": ImplicitActuatorCfg(
            joint_names_expr=[
                "arm_l_joint[1-2]",
                "arm_r_joint[1-2]",
            ],
            velocity_limit_sim=15.0,
            effort_limit_sim=61.4,
            stiffness=600.0,
            damping=30.0,
        ),
        "DY_70": ImplicitActuatorCfg(
            joint_names_expr=[
                "arm_l_joint[3-6]",
                "arm_r_joint[3-6]",
            ],
            velocity_limit_sim=15.0,
            effort_limit_sim=31.7,
            stiffness=600.0,
            damping=20.0,
        ),
        "DP-42" : ImplicitActuatorCfg(
            joint_names_expr=[
                "arm_l_joint7",
                "arm_r_joint7",
            ],
            velocity_limit_sim=6.0,
            effort_limit_sim=5.1,
            stiffness=200.0,
            damping=3.0,
        ),

        # Actuators for grippers
        "gripper_master": ImplicitActuatorCfg(
            joint_names_expr=["gripper_l_joint1", "gripper_r_joint1"],
            velocity_limit_sim=2.2,
            effort_limit_sim=30.0,
            stiffness=100.0,
            damping=4.0,
        ),
        "gripper_slave": ImplicitActuatorCfg(
            joint_names_expr=["gripper_l_joint[2-4]", "gripper_r_joint[2-4]"],
            effort_limit_sim=20.0,
            stiffness=2.0,
            damping=0.5,
        ),

        # Actuators for head joints
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_joint1", "head_joint2"],
            velocity_limit_sim=2.0,
            effort_limit_sim=30.0,
            stiffness=150.0,
            damping=3.0,
        ),
    }
)

FFW_BG2_CFG.spawn.func = spawn_bg2_with_deployed_head_limit
FFW_BG2_WITHOUT_MIMIC_CFG = deepcopy(FFW_BG2_CFG)
FFW_BG2_WITHOUT_MIMIC_CFG.spawn.usd_path = f'{CYCLO_LAB_ASSETS_DATA_DIR}/robots/FFW/FFW_BG2_without_mimic_soft_tips.usda'
