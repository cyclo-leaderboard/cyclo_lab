"""Plan from measured held-object transform; continuous IK and checked local fits.

Invoked at a stationary lifted checkpoint by the isolated physical executor. The
object is never attached/teleported: the transform is only a planning prediction.
"""
import argparse,ast,json,re,os
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from curobo.geom.types import WorldConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.base import TensorDeviceType
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen,MotionGenConfig
from plan_goalset import BASE,pose_array,check_route,joint_metrics,WEIGHTS
from conditional_common import static_world,robot_model,fit
from plan_pickplace import prior_spline

p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--profile',choices=['reference','natural','hybrid'],default='hybrid');p.add_argument('--allow-axial-roll',action='store_true');p.add_argument('--motion-speed',type=float,default=1.);args=p.parse_args()
out=Path(args.output);out.mkdir(exist_ok=True);checkpoint=json.loads(Path(args.checkpoint).read_text());scene=checkpoint['scene'];fk=np.load(BASE/'input/source_fk.npz');src=np.load(BASE/'input/source_episode.npz');ta=TensorDeviceType()
mg=MotionGen(MotionGenConfig.load_from_robot_config(robot_model(),WorldConfig.from_dict({'cuboid':static_world(scene,fk)}),tensor_args=ta,num_ik_seeds=64,num_trajopt_seeds=4,trajopt_tsteps=32,interpolation_dt=1/60,collision_checker_type=CollisionCheckerType.PRIMITIVE,use_cuda_graph=True,position_threshold=.0015,rotation_threshold=.02,collision_activation_distance=.004,collision_cache={'obb':24}))
mg.toggle_link_collision(['tcp'],False)
Tactual=np.array(checkpoint['tcp_transform']);Oactual=np.array(checkpoint['object_transform']);relative=np.linalg.inv(Tactual)@Oactual
qstart=np.array(checkpoint['joint_target'])[8:15];start_fk=mg.kinematics.get_state(ta.to_device(qstart.astype(np.float32)[None]));Tstart=np.eye(4);Tstart[:3,:3]=Rotation.from_quat(start_fk.ee_quaternion.cpu().numpy()[0,[1,2,3,0]]).as_matrix();Tstart[:3,3]=start_fk.ee_position.cpu().numpy()[0]
Ostart=Tstart@relative;basket=scene['basket'];center=np.array(basket['center_xy']);floor=basket['floor_top_m'];rim=basket['wall_top_m'];half=np.array(basket['inner_size_xy'])/2
mesh=np.array(ast.literal_eval(re.search(r'point3f\[\] points = (\[.*\])',(BASE/'task_000759/cucumber.usda').read_text())[1]))

def interpolate(O0,O1,count):
    u=np.linspace(0,1,count);rots=Slerp([0,1],Rotation.from_matrix([O0[:3,:3],O1[:3,:3]]))(u).as_matrix();values=[]
    for i,t in enumerate(u):
        O=np.eye(4);O[:3,:3]=rots[i];O[:3,3]=(1-t)*O0[:3,3]+t*O1[:3,3];values.append(O@np.linalg.inv(relative))
    return np.array(values)

def continuous_ik(poses,q0):
    qs=[np.array(q0)];previous=ta.to_device(np.array(q0,dtype=np.float32)[None])
    for i,T in enumerate(poses[1:],1):
        result=mg.ik_solver.solve_single(pose_array(T,ta),retract_config=previous,seed_config=previous.unsqueeze(1),return_seeds=16)
        feasible=result.success.cpu().numpy()[0]
        if not feasible.any():return None,dict(reason='ik',waypoint=i)
        possible=result.solution.cpu().numpy()[0,feasible];cost=np.sum((possible-qs[-1])**2*WEIGHTS,axis=1);selected=possible[cost.argmin()]
        if np.abs(selected-qs[-1]).max()>.18:return None,dict(reason='branch_jump',waypoint=i,jump_rad=float(np.abs(selected-qs[-1]).max()))
        qs.append(selected);previous=ta.to_device(selected[None].copy())
    return np.array(qs),{}

def object_geometry_ok(q,allow_floor=False):
    k=mg.kinematics.get_state(ta.to_device(np.asarray(q,dtype=np.float32)));R=Rotation.from_quat(k.ee_quaternion.cpu().numpy()[:,[1,2,3,0]]).as_matrix();pos=k.ee_position.cpu().numpy()
    OR=R@relative[:3,:3];op=pos+np.einsum('nij,j->ni',R,relative[:3,3]);points=np.einsum('nij,vj->nvi',OR,mesh)+op[:,None,:]
    in_xy=(np.abs(points[:,:,:2]-center)<half-.004).all(axis=(1,2))
    above=points[:,:,2].min(axis=1)>rim+.012
    if not (in_xy|above).all():return False
    lower=floor-.004 if allow_floor else scene['table_top_m']+.002
    return bool((points[:,:,2].min(axis=1)>lower).all())

def best_stage(raw,dt,reference,stage,carried=True,allow_floor=False):
    options=[fit(raw,dt,reference,args.profile,stage,args.motion_speed)]
    if os.environ.get('TASK759_TIMING_PROFILE')=='rapid':
        from rapid_timing import retime,monotone
        sq,sv,sa,si=prior_spline(raw,dt,np.linspace(raw[0],raw[-1],len(raw)),0.,n_coeff=10,jerk_regularization=1e-6,travel_regularization=.002)
        sq,sv,sa,ti=retime(sq,sv,sa,stage);si.update(ti,smoothing='low-complexity joint spline');options.append((sq,sv,sa,si))
        mq,mv,ma,mi=monotone(raw[0],raw[-1],stage);mi['smoothing']='monotone minimum-jerk joints';options.append((mq,mv,ma,mi))
    feasible=[]
    for q,v,a,info in options:
        if not check_route(mg,q,carried):continue
        if carried and not object_geometry_ok(q,allow_floor):continue
        k=mg.kinematics.get_state(ta.to_device(q.astype(np.float32)));positions=k.ee_position.cpu().numpy();rotations=Rotation.from_quat(k.ee_quaternion.cpu().numpy()[:,[1,2,3,0]])
        arc=float(np.degrees((rotations[:-1].inv()*rotations[1:]).magnitude().sum()));direct=float(np.degrees((rotations[0].inv()*rotations[-1]).magnitude()))
        length=float(np.linalg.norm(np.diff(positions,axis=0),axis=1).sum());excess=length-float(np.linalg.norm(positions[-1]-positions[0]))
        if os.environ.get('TASK759_LOW_ROTATION')=='1' and arc>direct+8:continue
        if excess>.07 or positions[:,2].max()>max(positions[0,2],positions[-1,2])+.04:continue
        info.update(eef_rotation_arc_deg=arc,eef_path_excess_m=excess)
        score=info['duration_s']+.02*arc+5*excess+.02*np.abs(np.diff(q,axis=0)).sum()
        feasible.append((score,q,v,a,info))
    if not feasible:return None
    _,q,v,a,info=min(feasible,key=lambda value:value[0]);return q,v,a,info

trials=[];solutions=[]
# The object yaw, not just TCP yaw, is optimized for the follow-up placement.
initial_object_yaw=np.degrees(np.arctan2(Ostart[1,0],Ostart[0,0]))
goal_yaws=sorted([0,30,60,90,120,150,180],key=lambda angle:abs((angle-initial_object_yaw+180)%360-180))
initial_roll=float(Rotation.from_matrix(Ostart[:3,:3]).as_euler('xyz',degrees=True)[0])
goal_offsets=[(0.,0.),(-.03,-.04),(0.,-.04),(-.04,0.),(.04,-.04)] if args.allow_axial_roll else [(0.,0.)]
goal_pairs=[(y,r,offset) for offset in goal_offsets for y in goal_yaws for r in ([initial_roll,0.] if args.allow_axial_roll else [0.])]
low_rotation=os.environ.get('TASK759_LOW_ROTATION')=='1';rapid=os.environ.get('TASK759_TIMING_PROFILE')=='rapid'
if low_rotation:
    offsets=sorted([(x,y) for x in [-.06,-.04,-.02,0.,.02,.04,.06] for y in [-.03,-.015,0.,.015,.03]],key=lambda p:p[0]**2+p[1]**2)
    orientations=[(0,0,0),(0,-10,0),(0,10,0),(0,0,-10),(0,0,10),(-15,0,0),(15,0,0),(0,-20,0),(0,20,0),(-30,0,0),(30,0,0)]
    tilt_probe=float(os.environ.get('TASK759_PLACEMENT_TILT_DEG','0'))
    if tilt_probe:orientations=[(0,-tilt_probe,0),(0,tilt_probe,0),(0,0,-tilt_probe),(0,0,tilt_probe)]
    goal_pairs=[(initial_object_yaw+delta,initial_roll,offset,Rotation.from_euler('z',delta,degrees=True).as_matrix()@Ostart[:3,:3]@Rotation.from_euler('yx',[pitch,roll],degrees=True).as_matrix(),clearance,delta) for delta,pitch,roll in orientations for clearance in [.002,.020,.035] for offset in offsets]
else:goal_pairs=[(y,r,offset,None,.012 if rapid else .002,None) for y,r,offset in goal_pairs]
for object_yaw,object_roll,offset,goal_rotation,release_clearance,yaw_delta in goal_pairs:
    # A feasible centered solution takes priority. Expand interior placement
    # positions only if every centered orientation is infeasible.
    if not low_rotation and offset!=(0.,0.) and solutions:break
    if low_rotation and solutions:
        angle=float(np.degrees((Rotation.from_matrix(Ostart[:3,:3]).inv()*Rotation.from_matrix(goal_rotation)).magnitude()))
        if angle>solutions[0][-1]['orientation_change_deg']+1e-5:break
    trial=dict(place_object_yaw_deg=object_yaw,place_object_roll_deg=object_roll,place_xy_offset_m=list(offset));trials.append(trial)
    Oabove=np.eye(4);Oabove[:3,:3]=goal_rotation if goal_rotation is not None else Rotation.from_euler('z',object_yaw,degrees=True).as_matrix()@Rotation.from_euler('x',object_roll,degrees=True).as_matrix()
    rotated_mesh=mesh@Oabove[:3,:3].T
    transport_height=max(Ostart[2,3],rim-float(rotated_mesh[:,2].min())+.015) if low_rotation else max(.94,Ostart[2,3])
    Oabove[:3,3]=[*(center+np.array(offset)),transport_height]
    if low_rotation:
        trial.update(requested_yaw_delta_deg=yaw_delta,release_clearance_m=release_clearance,orientation_change_deg=float(np.degrees((Rotation.from_matrix(Ostart[:3,:3]).inv()*Rotation.from_matrix(Oabove[:3,:3])).magnitude())))
        if not (np.abs(rotated_mesh[:,:2]+np.array(offset))<half-.004).all():trial['status']='goal_footprint_outside_basket';continue
        test_floor=Oabove.copy();test_floor[2,3]=floor-float(rotated_mesh[:,2].min())+release_clearance
        endpoint=mg.ik_solver.solve_single(pose_array(test_floor@np.linalg.inv(relative),ta),retract_config=ta.to_device(qstart.astype(np.float32)[None]),seed_config=ta.to_device(qstart.astype(np.float32)[None,None]),return_seeds=16)
        if not endpoint.success.any().item():trial['status']='release_pose_unreachable';continue
    carry_poses=interpolate(Ostart,Oabove,65)
    raw,err=continuous_ik(carry_poses,qstart)
    if raw is None:trial.update(status='carry_rejected',**err);continue
    fitted=best_stage(raw,.06,src['state'][65:81,8:15],'carry')
    if fitted is None:trial['status']='carry_fit_collision';continue
    q,v,a,info=fitted
    Ofloor=Oabove.copy();Ofloor[2,3]=floor-float((mesh@Oabove[:3,:3].T)[:,2].min())+release_clearance if args.allow_axial_roll else floor+.022
    down_poses=interpolate(Oabove,Ofloor,50);rawdown,err=continuous_ik(down_poses,q[-1])
    if rawdown is None:trial.update(status='descent_rejected',**err);continue
    fitted=best_stage(rawdown,.05,src['state'][78:84,8:15],'descent',allow_floor=True)
    if fitted is None:trial['status']='descent_fit_collision';continue
    dq,dv,da,di=fitted
    # Release only after placing near the floor; do not drop the object from 15 cm.
    Trelease=Ofloor@np.linalg.inv(relative);Twithdraw=Trelease.copy();Twithdraw[2,3]+=.20
    object_for_retreat=np.eye(4);object_for_retreat[:3,:3]=Ofloor[:3,:3];object_for_retreat[:3,3]=Ofloor[:3,3]+[0,0,.14 if rapid else .20]
    wr,err=continuous_ik(interpolate(Ofloor,object_for_retreat,45),dq[-1])
    if wr is None:trial.update(status='withdraw_rejected',**err);continue
    fitted=best_stage(wr,.05,src['state'][84:108,8:15],'withdraw',carried=False)
    if fitted is None:trial['status']='withdraw_fit_collision';continue
    wq,wv,wa,wi=fitted
    settle=6 if rapid else 30;release=30 if rapid else 75
    arm=np.vstack([q[1:],dq[1:],np.repeat(dq[-1:],settle+release,axis=0),wq[1:]])
    vel=np.vstack([v[1:],dv[1:],np.zeros((settle+release,7)),wv[1:]]);acc=np.vstack([a[1:],da[1:],np.zeros((settle+release,7)),wa[1:]])
    commands=np.repeat(np.array(checkpoint['joint_target'])[None],len(arm),axis=0);commands[:,8:15]=arm;commands[:,15]=1.1
    release_start=len(q)-1+len(dq)-1+settle;commands[release_start:release_start+release,15]=np.linspace(1.1,0,release);commands[release_start+release:,15]=0
    velocities=np.zeros_like(commands);velocities[:,8:15]=vel;accelerations=np.zeros_like(commands);accelerations[:,8:15]=acc
    metrics=joint_metrics(arm,np.full(len(arm),1/60));score=metrics['weighted_joint_travel_rad']+3*metrics['excess_joint_travel_rad']
    wrist_travel=float(np.abs(np.diff(arm[:,4:7],axis=0)).sum())
    trial['wrist_travel_rad']=wrist_travel
    if low_rotation:score+=8*np.deg2rad(trial['orientation_change_deg'])+float(os.environ.get('TASK759_WRIST_PENALTY','.5'))*wrist_travel
    trial.update(status='valid',score=score,metrics=metrics,carry=info,descent=di,withdraw=wi)
    solutions.append((score,commands,velocities,accelerations,release_start,trial));print('PLACE_CANDIDATE',json.dumps(trial),flush=True)
    if args.allow_axial_roll and len(solutions)>=(4 if low_rotation else 5):break
report=dict(profile=args.profile,allow_axial_roll=args.allow_axial_roll,measured_held_transform=relative.tolist(),object_scripted=False,world_collision_approximate=True,method='continuous closest-previous IK + local spline optimization + carried-mesh clearance + floor-supported release',trials=trials,status='planned' if solutions else 'failed')
report.update(low_rotation=low_rotation,rapid_timing=rapid,release_method='near-floor support or <=35 mm controlled drop' if low_rotation else 'floor-near release')
if solutions:
    _,commands,v,a,release,trial=min(solutions,key=lambda x:x[0]);np.savez(out/'suffix.npz',joint_targets=commands,joint_velocity=v,joint_acceleration=a,dt=np.full(len(commands),1/60),release_start_frame=release)
    report['selected']=trial
(out/'planning_report.json').write_text(json.dumps(report,indent=2));print('CONDITIONED_PLACE',report['status'],flush=True)
if not solutions:raise RuntimeError('No complete carried-object-safe placement path')
