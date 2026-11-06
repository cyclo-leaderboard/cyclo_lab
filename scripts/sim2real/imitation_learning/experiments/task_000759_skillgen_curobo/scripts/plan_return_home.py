"""Collision-checked C2 return to the original starting joints after placement.

No reset, joint teleport, grasp attachment, or original-demo end-pose shortcut.
Prefer a direct monotonic joint route; use cuRobo joint-space planning if blocked.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from curobo.geom.types import WorldConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.base import TensorDeviceType
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen,MotionGenConfig,MotionGenPlanConfig
from plan_goalset import BASE,check_route,joint_metrics
from conditional_common import static_world,robot_model,fit

p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--motion-speed',type=float,default=1.2);args=p.parse_args()
checkpoint=json.loads(Path(args.checkpoint).read_text());out=Path(args.output)
src=np.load(BASE/'input/source_episode.npz');fk=np.load(BASE/'input/source_fk.npz');ta=TensorDeviceType()
world=static_world(checkpoint['scene'],fk);obj=np.array(checkpoint['object_pose'])
world['placed_cucumber']=dict(dims=[.236,.042,.039],pose=obj.tolist())
mg=MotionGen(MotionGenConfig.load_from_robot_config(robot_model(),WorldConfig.from_dict({'cuboid':world}),tensor_args=ta,num_ik_seeds=32,num_trajopt_seeds=4,num_graph_seeds=4,trajopt_tsteps=48,interpolation_dt=1/60,collision_checker_type=CollisionCheckerType.PRIMITIVE,use_cuda_graph=True,collision_cache={'obb':24},minimize_jerk=True))
start=np.array(checkpoint['joint_target'])[8:15];goal=src['state'][0,8:15]
raw=np.linspace(start,goal,100);raw_dt=.04;method='direct_monotonic_joint_route'
if not check_route(mg,raw,False):
    def js(q):return JointState.from_position(ta.to_device(np.asarray(q,dtype=np.float32)[None]),joint_names=mg.joint_names)
    r=mg.plan_single_js(js(start),js(goal),MotionGenPlanConfig(max_attempts=8,timeout=20.,enable_graph_attempt=2,use_start_state_as_retract=True))
    if not bool(r.success.item()):raise RuntimeError('No collision-free return-home route')
    raw=r.get_interpolated_plan().position.cpu().numpy();raw_dt=float(r.interpolation_dt);method='curobo_joint_goal_route'
q,v,a,info=fit(raw,raw_dt,np.linspace(start,goal,100),'hybrid','home',args.motion_speed)
if not check_route(mg,q,False):raise RuntimeError('Return-home spline collision rejected')
if np.abs(q[-1]-goal).max()>1e-6:raise RuntimeError('Return endpoint is not source initial pose')
commands=np.repeat(np.array(checkpoint['joint_target'])[None],len(q)-1,axis=0);commands[:,8:15]=q[1:];commands[:,15]=0
vel=np.zeros_like(commands);vel[:,8:15]=v[1:];acc=np.zeros_like(commands);acc[:,8:15]=a[1:]
np.savez(out/'home.npz',joint_targets=commands,joint_velocity=vel,joint_acceleration=acc,dt=np.full(len(commands),1/60))
report=dict(status='planned',method=method,fit=info,metrics=joint_metrics(q,np.full(len(q),1/60)),initial_joint_pose=goal.tolist(),endpoint_error_rad=float(np.abs(q[-1]-goal).max()),collision_proxies_approximate=True)
(out/'home_planning_report.json').write_text(json.dumps(report,indent=2));print('HOME_PLAN',json.dumps(report),flush=True)
