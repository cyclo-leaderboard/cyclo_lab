"""SG2 cuRobo approach planning plus continuity-regularized contact IK. No simulator required."""
import argparse,json,time
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.util_file import load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen,MotionGenConfig,MotionGenPlanConfig
from curobo.wrap.reacher.ik_solver import IKSolver,IKSolverConfig
from curobo.geom.sdf.world import CollisionCheckerType

p=argparse.ArgumentParser();p.add_argument('--yaws',nargs='+',type=int,default=[-60,-30,0,30,60]);p.add_argument('--output',default='clearance');p.add_argument('--hover',action='store_true');a=p.parse_args()
base=Path('/experiment');scene=json.loads((base/'task_000759/scene.json').read_text());src=np.load(base/'input/source_episode.npz');fk=np.load(base/'input/source_fk.npz')
output=base/'output'/a.output;output.mkdir(parents=True,exist_ok=True)
robot_cfg=load_yaml(str(base/'task_000759/sg2_right.yml'))['robot_cfg'];ta=TensorDeviceType()
world={'cuboid':{'table':{'dims':[*scene['table_size_xy'],.05],'pose':[*scene['table_center_xy'],scene['table_top_m']-.025,1,0,0,0]},'torso':{'dims':[.25,.30,.55],'pose':[-.045,0.,.88,1,0,0,0]}}}
# Passive left link origins represented as fixed boxes. Approximate, not a full mesh audit.
body=fk['body_pose'][0];names=list(fk['body_names'])
for i in range(2,8):
    b=body[names.index(f'arm_l_link{i}')];world['cuboid'][f'passive_left_{i}']={'dims':[.11,.11,.11],'pose':[*b[:3].tolist(),1,0,0,0]}
approach_world=deepcopy(world)
# Conservative approach-only clearance for the open soft fingertips; contact uses original table.
approach_world['cuboid']['table']['dims'][2]+=.025
approach_world['cuboid']['table']['pose'][2]+=.0125
approach_robot=deepcopy(robot_cfg)
approach_robot['kinematics']['collision_spheres']['tcp'].append({'center':[0.,0.,0.],'radius':.02})
mg=MotionGen(MotionGenConfig.load_from_robot_config(approach_robot,approach_world,tensor_args=ta,num_ik_seeds=32,num_trajopt_seeds=4,interpolation_dt=1/15,collision_checker_type=CollisionCheckerType.PRIMITIVE,use_cuda_graph=True,position_threshold=.002,rotation_threshold=.02,collision_activation_distance=.003,velocity_scale=.6,acceleration_scale=.5,jerk_scale=.5))
ik=IKSolver(IKSolverConfig.load_from_robot_config(robot_cfg,world,tensor_args=ta,num_seeds=32,regularization=True,collision_checker_type=CollisionCheckerType.PRIMITIVE,position_threshold=.002,rotation_threshold=.02,collision_activation_distance=.003))
state=mg.kinematics.get_state(torch.tensor(src['state'][:,8:15],device='cuda',dtype=torch.float32))
fk_error=float(np.linalg.norm(state.ee_position.cpu().numpy()-fk['eef'][:,:3,3],axis=1).max())
assert fk_error<.0001,fk_error
def pose(T):
    quat=Rotation.from_matrix(T[:3,:3]).as_quat()[[3,0,1,2]]
    return Pose(position=ta.to_device(T[:3,3])[None],quaternion=ta.to_device(quat)[None])
reports=[]
for yaw in a.yaws:
    began=time.monotonic();skill=np.load(base/'input'/f'skill_yaw_{yaw:+03d}.npz');poses=skill['poses'];grips=skill['gripper'].ravel()
    report={'yaw_delta_deg':yaw,'fk_max_position_error_m':fk_error,'source_episode':93,'planner':'cuRobo MotionGen + regularized sequential IK','physical_success':None,'scene_collision_approximate':True,'checks':['active arm self collision','table','approximate torso','static passive left-arm boxes'],'status':'started'}
    q0=ta.to_device(src['state'][0,8:15])[None]
    # Look ahead through the whole contact/lift skill before choosing the approach posture.
    # The demonstrated arm posture supplies the regularization prior, not just an EEF target.
    prev=ta.to_device(src['state'][scene['contact_start_frame'],8:15])[None];contact=[];max_step=0.
    for index,T in enumerate(poses):
        sol=ik.solve_single(pose(T),retract_config=prev,seed_config=prev[None],return_seeds=8)
        feasible=sol.success[0]
        if not bool(feasible.any()):report['status']='contact_ik_failed';report['failed_waypoint']=index;break
        qs=sol.solution[0][feasible];scores=((qs-prev)**2*ta.to_device([1,1,1,1,1,1.5,2])).sum(-1);q=qs[scores.argmin()][None]
        step=float(torch.max(torch.abs(q-prev))) if index else 0.;max_step=max(max_step,step)
        if step>.22:report['status']='joint_jump_rejected';report['failed_waypoint']=index;report['rejected_step_rad']=step;break
        prev=q;contact.append(q[0].cpu().numpy())
    if len(contact)==len(poses):
        contact=np.array(contact)
        approach_prefix=None;start_q=q0
        if a.hover:
            hover_pose=poses[0].copy();hover_pose[2,3]+=.12
            pre=ta.to_device(contact[0:1])
            hs=ik.solve_single(pose(hover_pose),retract_config=pre,seed_config=pre[None],return_seeds=8)
            feasible=hs.success[0]
            if not bool(feasible.any()):
                report['status']='hover_ik_failed';reports.append(report);print('PLAN_RESULT',report,flush=True);continue
            hq=hs.solution[0][feasible];start_q=hq[((hq-pre)**2).sum(-1).argmin()][None]
            first=mg.plan_single_js(JointState.from_position(q0,joint_names=mg.joint_names),JointState.from_position(start_q,joint_names=mg.joint_names),MotionGenPlanConfig(max_attempts=8,timeout=30.,enable_graph_attempt=2))
            if not bool(first.success.item()):
                report['status']='hover_approach_failed';reports.append(report);print('PLAN_RESULT',report,flush=True);continue
            approach_prefix=first.get_interpolated_plan().position.cpu().numpy()
        result=mg.plan_single_js(JointState.from_position(start_q,joint_names=mg.joint_names),JointState.from_position(ta.to_device(contact[0:1]),joint_names=mg.joint_names),MotionGenPlanConfig(max_attempts=8,timeout=30.,enable_graph_attempt=2))
        report['approach_status']=str(result.status)
        if not bool(result.success.item()):
            report['status']='approach_failed';reports.append(report);print('PLAN_RESULT',report,flush=True);continue
        approach=result.get_interpolated_plan().position.cpu().numpy()
        if approach_prefix is not None:approach=np.concatenate([approach_prefix,approach[1:]])
        report['hover_clearance_m']=.12 if a.hover else 0.
        arm=np.concatenate([approach,contact[1:]])
        cmd=np.repeat(src['state'][0:1,:19],len(arm),axis=0);cmd[:,8:15]=arm;cmd[:,15]=np.concatenate([np.full(len(approach),grips[0]),grips[1:]])
        dt=np.full(len(arm),1/15)
        # Slow down skill uniformly if required; preserve gripper-to-contact synchronization.
        speed=float(np.max(np.abs(np.diff(contact,axis=0)))*15);scale=max(1.,speed/.9)
        dt[len(approach):]*=scale
        report.update(status='planned',approach_frames=len(approach),contact_frames=len(contact),max_contact_joint_step_rad=max_step,contact_time_scale=scale,max_contact_speed_rad_s=speed/scale)
        folder=output/f'yaw_{yaw:+03d}';folder.mkdir(exist_ok=True)
        np.savez(folder/'trajectory.npz',joint_targets=cmd,dt=dt,skill_poses=poses,object_pose=skill['object_pose'])
    report['elapsed_s']=time.monotonic()-began;reports.append(report);print('PLAN_RESULT',json.dumps(report),flush=True)
    (output/'planning_report.json').write_text(json.dumps(reports,indent=2))
(output/'planning_report.json').write_text(json.dumps(reports,indent=2))
