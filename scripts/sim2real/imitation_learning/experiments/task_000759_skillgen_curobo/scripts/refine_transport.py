"""Preserve a physically tested reach; optimize lift/place jointly over place poses."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from curobo.geom.types import WorldConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.util_file import load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen,MotionGenConfig,MotionGenPlanConfig
from curobo.wrap.reacher.ik_solver import IKSolver,IKSolverConfig
from curobo.rollout.cost.pose_cost import PoseCostMetric
from plan_goalset import BASE,pose_array,check_route,WEIGHTS,joint_metrics
from plan_pickplace import prior_spline

p=argparse.ArgumentParser();p.add_argument('--input',default='pickplace_v3');p.add_argument('--yaw',type=int,default=90);p.add_argument('--output',default='pickplace_v8');p.add_argument('--guarded-place',action='store_true',help='Carry above a conservative object envelope; descend only over basket.');args=p.parse_args()
parent=BASE/'output'/args.input;out=BASE/'output'/args.output;out.mkdir(exist_ok=True);folder=out/f'yaw_{args.yaw:+03d}';folder.mkdir(exist_ok=True)
scene=json.loads((parent/'scene.json').read_text());(out/'scene.json').write_text(json.dumps(scene,indent=2));src=np.load(BASE/'input/source_episode.npz');fk=np.load(BASE/'input/source_fk.npz');old=np.load(parent/f'yaw_{args.yaw:+03d}/trajectory.npz');ta=TensorDeviceType();basket=scene['basket']
w={'table':dict(dims=[*scene['table_size_xy'],.05],pose=[*scene['table_center_xy'],scene['table_top_m']-.025,1,0,0,0]),'torso':dict(dims=[.25,.30,.55],pose=[-.045,0,.88,1,0,0,0])}
for i in range(2,8):w[f'passive_left_{i}']=dict(dims=[.11]*3,pose=[*fk['body_pose'][0,list(fk['body_names']).index(f'arm_l_link{i}'),:3].tolist(),1,0,0,0])
bx,by=basket['center_xy'];sx,sy=basket['inner_size_xy'];th=basket['wall_thickness_m'];height=basket['wall_top_m']-basket['floor_top_m']
w['basket_floor']=dict(dims=[sx+2*th,sy+2*th,.02],pose=[bx,by,basket['floor_top_m']-.01,1,0,0,0])
for axis,sign in [(0,-1),(0,1),(1,-1),(1,1)]:
    pos=[bx,by,basket['floor_top_m']+height/2];pos[axis]+=sign*([sx,sy][axis]/2+th/2);dims=[sx+2*th,sy+2*th,height];dims[axis]=th;w[f'wall_{axis}_{sign}']=dict(dims=dims,pose=[*pos,1,0,0,0])
world=WorldConfig.from_dict({'cuboid':w});robot=load_yaml(str(BASE/'task_000759/sg2_right.yml'))['robot_cfg']
robot['kinematics']['collision_spheres']['tcp']=[dict(center=[0,s,z],radius=.016) for s in [-.063,.063] for z in [.025,.055,.085]]+[dict(center=[0,0,.12],radius=.035)]
mg=MotionGen(MotionGenConfig.load_from_robot_config(robot,world,tensor_args=ta,num_ik_seeds=64,num_trajopt_seeds=8,num_graph_seeds=4,trajopt_tsteps=48,interpolation_dt=1/60,collision_checker_type=CollisionCheckerType.PRIMITIVE,position_threshold=.002,rotation_threshold=.025,collision_activation_distance=.005,minimize_jerk=True,use_cuda_graph=True,collision_cache={'obb':24}))
ik=IKSolver(IKSolverConfig.load_from_robot_config(robot,world,tensor_args=ta,num_seeds=64,regularization=True,use_cuda_graph=False,collision_checker_type=CollisionCheckerType.PRIMITIVE,position_threshold=.002,rotation_threshold=.025))
def js(q):return JointState.from_position(ta.to_device(np.array(q,dtype=np.float32).reshape(1,7)),joint_names=mg.joint_names)
def cfg():return MotionGenPlanConfig(max_attempts=6,timeout=12.,enable_graph_attempt=2,time_dilation_factor=.8,use_start_state_as_retract=True)
close=int(old['close_start_frame']);prefix_n=close+45;prefix=old['joint_targets'][:prefix_n];qgrasp=prefix[-1,8:15];grasp=old['grasp_pose']
kin=mg.kinematics.get_state(js(qgrasp).position);liftpose=Pose(position=kin.ee_position+ta.to_device([[0,0,.30 if args.guarded_place else .18]]),quaternion=kin.ee_quaternion.clone())
c=cfg();c.pose_cost_metric=PoseCostMetric(hold_partial_pose=True,hold_vec_weight=ta.to_device([1,1,1,1,1,0]),project_to_goal_frame=False)
mg.toggle_link_collision(['tcp'],False)
lift=mg.plan_single(js(qgrasp),liftpose,c)
assert bool(lift.success.item()),str(lift.status)
liftq=lift.get_interpolated_plan().position.cpu().numpy();lq,lv,la,li=prior_spline(liftq,float(lift.interpolation_dt),src['state'][40:66,8:15],.005)
assert check_route(mg,lq,True),'lift smoothing collision'
offset=grasp[:3,:3].T@(np.array(scene['cucumber_center'])-grasp[:3,3])
poses=[];metadata=[]
for yaw in [-35,-20,0,20,35]:
    for tilt in [0,25]:
        R=Rotation.from_euler('z',yaw,degrees=True).as_matrix()@fk['eef'][40,:3,:3]@Rotation.from_euler('y',tilt,degrees=True).as_matrix()
        T=np.eye(4);T[:3,:3]=R;T[:3,3]=np.array([bx,by,1.02 if args.guarded_place else .88])-R@offset;poses.append(T);metadata.append(dict(place_yaw_from_seed_deg=yaw,place_tilt_deg=tilt))
poses=np.array(poses);quat=Rotation.from_matrix(poses[:,:3,:3]).as_quat()[:,[3,0,1,2]]
gp=Pose(position=ta.to_device(np.ascontiguousarray(poses[:,:3,3])).contiguous(),quaternion=ta.to_device(np.ascontiguousarray(quat)).contiguous())
ref=ta.to_device(np.repeat(lq[-1:],len(poses),axis=0).astype(np.float32));result=ik.solve_batch(gp,retract_config=ref,seed_config=ref.unsqueeze(1),return_seeds=8)
ok=result.success.cpu().numpy();endq=result.solution.cpu().numpy();cost=np.sum((endq-lq[-1])**2*WEIGHTS,axis=-1);cost[~ok]=np.inf
ranking=[(int(i),int(j)) for i,j in np.array(np.unravel_index(np.argsort(cost.flatten()),cost.shape)).T if np.isfinite(cost[i,j])]
trials=[];solutions=[]
for i,j in ranking[:12]:
    trial=dict(placement=metadata[i],ik_joint_distance_cost=float(cost[i,j]));trials.append(trial)
    mg.toggle_link_collision(['tcp'],False)
    carry=mg.plan_single_js(js(lq[-1]),js(endq[i,j]),cfg())
    if not bool(carry.success.item()):trial['status']='carry_failed';continue
    cq,cv,ca,ci=prior_spline(carry.get_interpolated_plan().position.cpu().numpy(),float(carry.interpolation_dt),src['state'][65:81,8:15],.005)
    if not check_route(mg,cq,True):trial['status']='carry_spline_collision';continue
    k=mg.kinematics.get_state(ta.to_device(cq.astype(np.float32)));pos=k.ee_position.cpu().numpy();overshoot=float(max(0,pos[:,2].max()-max(pos[0,2],pos[-1,2])))
    if overshoot>.08:trial.update(status='excess_height_rejected',height_overshoot_m=overshoot);continue
    if args.guarded_place and pos[:,2].min()<1.00:trial.update(status='carried_object_clearance_rejected',min_tcp_z_m=float(pos[:,2].min()));continue
    if args.guarded_place:
        k=mg.kinematics.get_state(js(cq[-1]).position)
        downpose=Pose(position=k.ee_position-ta.to_device([[0,0,.14]]),quaternion=k.ee_quaternion.clone())
        downcfg=cfg();downcfg.pose_cost_metric=PoseCostMetric(hold_partial_pose=True,hold_vec_weight=ta.to_device([1,1,1,1,1,0]),project_to_goal_frame=False)
        mg.toggle_link_collision(['tcp'],False);down=mg.plan_single(js(cq[-1]),downpose,downcfg)
        if not bool(down.success.item()):trial['status']='guarded_descent_failed';continue
        dq,dv,da,di=prior_spline(down.get_interpolated_plan().position.cpu().numpy(),float(down.interpolation_dt),src['state'][75:83,8:15],0.)
        if not check_route(mg,dq,True):trial['status']='descent_spline_collision';continue
        cq=np.vstack([cq,dq[1:]]);cv=np.vstack([cv,dv[1:]]);ca=np.vstack([ca,da[1:]]);ci['guarded_descent']=di
    k=mg.kinematics.get_state(js(cq[-1]).position);withdrawpose=Pose(position=k.ee_position+ta.to_device([[0,0,.16]]),quaternion=k.ee_quaternion.clone())
    mg.toggle_link_collision(['tcp'],True);withdraw=mg.plan_single(js(cq[-1]),withdrawpose,cfg())
    if not bool(withdraw.success.item()):trial['status']='withdraw_failed';continue
    wq,wv,wa,wi=prior_spline(withdraw.get_interpolated_plan().position.cpu().numpy(),float(withdraw.interpolation_dt),src['state'][80:100,8:15],.005)
    if not check_route(mg,wq,False):trial['status']='withdraw_spline_collision';continue
    release=60
    arm=np.vstack([lq[1:],cq[1:],np.repeat(cq[-1:],release,axis=0),wq[1:]])
    v=np.vstack([lv[1:],cv[1:],np.zeros((release,7)),wv[1:]]);acc=np.vstack([la[1:],ca[1:],np.zeros((release,7)),wa[1:]])
    commands=np.repeat(src['state'][0:1,:19],len(arm),axis=0);commands[:,8:15]=arm;commands[:,15]=1.1;rel=len(lq)-1+len(cq)-1;commands[rel:rel+release,15]=np.linspace(1.1,0,release);commands[rel+release:,15]=0
    commands=np.vstack([prefix,commands]);vel=np.zeros_like(commands);vel[:prefix_n]=old['joint_velocity'][:prefix_n];vel[prefix_n:,8:15]=v;accel=np.zeros_like(commands);accel[:prefix_n]=old['joint_acceleration'][:prefix_n];accel[prefix_n:,8:15]=acc
    metrics=joint_metrics(commands[:,8:15],np.full(len(commands),1/60));score=metrics['weighted_joint_travel_rad']+2*metrics['excess_joint_travel_rad']+30*overshoot
    trial.update(status='valid',score=score,metrics=metrics,height_overshoot_m=overshoot,lift=li,carry=ci,withdraw=wi);solutions.append((score,commands,vel,accel,prefix_n+rel,trial));print('VALID_REFINEMENT',json.dumps(trial),flush=True)
    if len(solutions)>=3:break
report=dict(yaw_delta_deg=args.yaw,parent=args.input,unchanged_reach_and_closure_prefix_frames=prefix_n,vertical_lift_orientation_locked=True,place_pose_and_joint_branch_candidates=True,guarded_place=args.guarded_place,trials=trials,status='planned' if solutions else 'failed',physics_verified=False)
if solutions:
    score,commands,vel,accel,release,trial=min(solutions,key=lambda x:x[0]);np.savez(folder/'trajectory.npz',joint_targets=commands,dt=np.full(len(commands),1/60),joint_velocity=vel,joint_acceleration=accel,close_start_frame=close,release_start_frame=release,grasp_pose=grasp,yaw_delta_deg=args.yaw);report['selected']=trial
(out/'planning_report.json').write_text(json.dumps(report,indent=2));print('REFINE_RESULT',report['status'],flush=True)
