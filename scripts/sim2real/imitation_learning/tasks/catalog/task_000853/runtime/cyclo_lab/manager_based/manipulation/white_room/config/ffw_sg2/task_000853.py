"""Task000400 room/BG2 asset with Task000853 placement and current bringup pose."""
import json,math
from pathlib import Path
from dataclasses import replace
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg,RigidObjectCfg
from isaaclab.managers import EventTermCfg
from isaaclab.utils import configclass
from .task_000400 import Task000400EnvCfg

ROOT=Path(__file__).resolve().parents[7]
ROOM_USD_PATH=str(ROOT/'assets/room.usda')
LAYOUT=json.loads((ROOT/'layout.json').read_text())
ROBOT_SETUP=json.loads((ROOT/'robot_setup.json').read_text())
SOFT_GRIPPER_TCP_Z_M=float(ROBOT_SETUP['gripper']['tcp_z_m'])
ENV_NAME='Cyclo-Real-WallRoom-Task000853-FFW-BG2-v0'
# Task853-only visual extrinsic fit to the 1063 head frame (2026-10-27).
# Board silhouette and hook center were matched at head_joint1=0.67 rad.
# Keep the shared Task000400 camera intrinsics and ROS optical convention.
HEAD_CAMERA_OFFSET_POS_M=(0.05167023,0.04028590,0.00124562)
HEAD_CAMERA_OFFSET_ROT_WXYZ=(0.47909312,-0.50988651,0.52015945,-0.48981596)

def reset_cup(env,env_ids):
    for name in ('blue_cup','pegboard','pegboard_hook'):
        obj=env.scene[name];state=obj.data.default_root_state[env_ids].clone();state[:,:3]+=env.scene.env_origins[env_ids]
        obj.write_root_pose_to_sim(state[:,:7],env_ids=env_ids)
        obj.write_root_velocity_to_sim(torch.zeros_like(state[:,7:13]),env_ids=env_ids)

@configclass
class Task000853EnvCfg(Task000400EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Shared by waypoint seed, MimicGen and camera rerendering.
        self.scene.cam_head.offset.pos=HEAD_CAMERA_OFFSET_POS_M
        self.scene.cam_head.offset.rot=HEAD_CAMERA_OFFSET_ROT_WXYZ
        self.scene.environment.spawn.usd_path=ROOM_USD_PATH
        from cyclo_lab.robot_specs.ffw.sg2 import FFW_SG2_ACTION_JOINT_NAMES
        joints=dict(ROBOT_SETUP['joint_positions'])
        # Task 853 simulation pose: lower the lift 1 cm from the captured -0.13 m setup.
        joints['lift_joint']=-0.14
        self.scene.robot.init_state.pos=tuple(ROBOT_SETUP['root_position_m'])
        self.scene.robot.init_state.rot=tuple(ROBOT_SETUP['root_quaternion_wxyz'])
        self.scene.robot.init_state.joint_pos.update(joints)
        self.events.set_robot_joint_pose.params['joint_positions']=dict(joints)
        self.events.randomize_robot_root_pose=None
        if hasattr(self,'idle_action'):
            idle=list(self.idle_action)
            for i,name in enumerate(FFW_SG2_ACTION_JOINT_NAMES):idle[i]=joints[name]
            self.idle_action=tuple(idle)
        self.robot_setup_id=f"{ROBOT_SETUP['id']}_sim_lift_minus014"
        from cyclo_lab.assets.robots.FFW_SG2 import spawn_sg2_with_softgripper_physics
        # Shared FFW spawn hook only processes gripper meshes, material and contact proxies.
        self.scene.robot.spawn.usd_path=str(ROOT/ROBOT_SETUP['gripper']['usd'])
        self.scene.robot.spawn.func=spawn_sg2_with_softgripper_physics
        for frame in (self.scene.left_eef,self.scene.right_eef):
            frame.target_frames[0].offset.pos=(0.,0.,SOFT_GRIPPER_TCP_Z_M)
        self.task_spec=replace(self.task_spec,task_id='000853',env_name=ENV_NAME,
            instruction='Manipulate the blue handled cup with the pegboard hook.',target_object='blue_cup',
            environment_usd_path=ROOM_USD_PATH)
        self.task_id='000853';self.env_name=ENV_NAME;self.task_instruction=self.task_spec.instruction;self.target_object='blue_cup'
        self.operator_camera_title='BG2 Task000853'
        self.scene.navy_block=None;self.scene.white_basket=None
        self.events.randomize_navy_block_pose=None
        self.events.reset_cup=EventTermCfg(func=reset_cup,mode='reset')
        pos=LAYOUT['board_pos_m'];yaw=LAYOUT['board_yaw_rad'];rot=(math.cos(yaw/2),0,0,math.sin(yaw/2))
        self.scene.pegboard=RigidObjectCfg(prim_path='{ENV_REGEX_NS}/Pegboard',spawn=sim_utils.UsdFileCfg(usd_path=str(ROOT/'assets/pegboard.usda')),init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(pos),rot=rot))
        self.scene.pegboard_hook=RigidObjectCfg(prim_path='{ENV_REGEX_NS}/PegboardHook',spawn=sim_utils.UsdFileCfg(usd_path=str(ROOT/'assets/hook.usda')),init_state=RigidObjectCfg.InitialStateCfg(pos=(pos[0],pos[1],pos[2]+.457),rot=rot))
        self.scene.blue_cup=RigidObjectCfg(prim_path='{ENV_REGEX_NS}/BlueCup',spawn=sim_utils.UsdFileCfg(
            usd_path=str(ROOT/'assets/blue_cup.usda'),rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False,solver_position_iteration_count=16,solver_velocity_iteration_count=4,max_depenetration_velocity=.5)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(LAYOUT['cup_pos_m'])))
