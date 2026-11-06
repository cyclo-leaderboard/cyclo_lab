"""Hybrid free-space reach plus an explicitly verified final approach corridor.

The final ten centimetres retain grasp orientation and a straight TCP approach;
global source-shape fitting is not allowed to cut this contact-sensitive corner.
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
from plan_goalset import BASE,WEIGHTS,pose_array,check_route,joint_metrics
from plan_pickplace import grasp_candidates
from conditional_common import static_world,robot_model,fit

p=argparse.ArgumentParser();p.add_argument('--yaws',type=int,nargs='+',default=[-75,-90]);p.add_argument('--candidate',type=int,default=229);p.add_argument('--height-offset',type=float,default=0.);p.add_argument('--approach-world-z',action='store_true');p.add_argument('--approach-distance',type=float,default=.10);p.add_argument('--output',default='hybrid11_corridor_profiles');p.add_argument('--scene',default='');args=p.parse_args()
out=BASE/'output'/args.output;out.mkdir(exist_ok=True)
scene=json.loads(Path(args.scene or BASE/'output/hybrid11_profiles/scene.json').read_text());(out/'scene.json').write_text(json.dumps(scene,indent=2))
src=np.load(BASE/'input/source_episode.npz');fk=np.load(BASE/'input/source_fk.npz');ta=TensorDeviceType()
world=static_world(scene,fk);world['cucumber']=dict(dims=[.236,.042,.039],pose=[*scene['cucumber_center'],1,0,0,0])
mg=MotionGen(MotionGenConfig.load_from_robot_config(robot_model(),WorldConfig.from_dict({'cuboid':world}),tensor_args=ta,num_ik_seeds=64,num_trajopt_seeds=8,num_graph_seeds=4,trajopt_tsteps=48,interpolation_dt=1/60,collision_checker_type=CollisionCheckerType.PRIMITIVE,position_threshold=.0015,rotation_threshold=.02,collision_activation_distance=.004,minimize_jerk=True,use_cuda_graph=True,collision_cache={'obb':24}))

def js(q):return JointState.from_position(ta.to_device(np.asarray(q,dtype=np.float32)[None]),joint_names=mg.joint_names)
def continuous(poses,q0):
    qs=[q0.copy()];previous=ta.to_device(q0.astype(np.float32)[None])
    mg.toggle_link_collision(['tcp'],False)
    for i,T in enumerate(poses[1:],1):
        result=mg.ik_solver.solve_single(pose_array(T,ta),retract_config=previous,seed_config=previous.unsqueeze(1),return_seeds=16)
        ok=result.success.cpu().numpy()[0]
        if not ok.any():raise RuntimeError(f'Continuous IK failed at {i}')
        candidates=result.solution.cpu().numpy()[0,ok];cost=np.sum((candidates-qs[-1])**2*WEIGHTS,axis=1);q=candidates[cost.argmin()]
        if np.abs(q-qs[-1]).max()>.18:raise RuntimeError('IK branch discontinuity')
        qs.append(q);previous=ta.to_device(q[None].copy())
    return np.array(qs)

def corridor_error(q,T0,T1):
    state=mg.kinematics.get_state(ta.to_device(q.astype(np.float32)));positions=state.ee_position.cpu().numpy();direction=T1[:3,3]-T0[:3,3]
    u=((positions-T0[:3,3])@direction)/(direction@direction)
    projected=T0[:3,3]+u[:,None]*direction
    error=float(np.linalg.norm(positions-projected,axis=1).max())
    orientations=Rotation.from_quat(state.ee_quaternion.cpu().numpy()[:,[1,2,3,0]])
    angular=float((orientations.inv()*Rotation.from_matrix(T0[:3,:3])).magnitude().max())
    if error>.0025 or angular>.025 or u.min()<-.02 or u.max()>1.02 or np.diff(u).min()<-.005:raise RuntimeError(f'Corridor fit rejected: {error} m, {angular} rad')
    return dict(max_lateral_error_m=error,max_rotation_error_rad=angular,monotone_progress_min_delta=float(np.diff(u).min()))

reports=[]
for yaw in args.yaws:
    report=dict(yaw_delta_deg=yaw,candidate=args.candidate,height_offset_m=args.height_offset,method='hybrid reach, separately verified straight final approach, continuous-IK lift',status='failed')
    try:
        values,meta=grasp_candidates(scene,fk,yaw);goal=values[args.candidate].copy();goal[2,3]+=args.height_offset
        pre=goal.copy();pre[:3,3]+=args.approach_distance*(np.array([0.,0.,1.]) if args.approach_world_z else goal[:3,2]);report['approach_axis']='world_z' if args.approach_world_z else 'tool_z';report['approach_distance_m']=args.approach_distance
        ori=Rotation.from_euler('z',scene['cucumber_yaw_rad']+np.deg2rad(yaw)).as_quat()[[3,0,1,2]]
        world['cucumber']['pose']=[*scene['cucumber_center'],*ori.tolist()];mg.update_world(WorldConfig.from_dict({'cuboid':world}));mg.reset(reset_seed=True);mg.toggle_link_collision(['tcp'],True)
        result=mg.plan_single(js(src['state'][0,8:15]),pose_array(pre,ta),MotionGenPlanConfig(max_attempts=8,timeout=18.,enable_graph_attempt=2,use_start_state_as_retract=True,time_dilation_factor=.8))
        if not bool(result.success.item()):raise RuntimeError('Pregrasp reach planning failed')
        raw=result.get_interpolated_plan().position.cpu().numpy()
        reach,rv,ra,ri=fit(raw,float(result.interpolation_dt),src['state'][:37,8:15],'hybrid','reach',1.2)
        if not check_route(mg,reach,False):raise RuntimeError('Pregrasp fit collision')
        # Begin exactly at FK of the executed endpoint; account for finite IK error.
        kin=mg.kinematics.get_state(ta.to_device(reach[-1:].astype(np.float32)));pre_actual=pre.copy();pre_actual[:3,3]=kin.ee_position.cpu().numpy()[0];pre_actual[:3,:3]=Rotation.from_quat(kin.ee_quaternion.cpu().numpy()[0,[1,2,3,0]]).as_matrix()
        poses=np.repeat(goal[None],65,axis=0);poses[:,:3,3]=np.linspace(pre_actual[:3,3],goal[:3,3],65);poses[0]=pre_actual
        raw=continuous(poses,reach[-1]);aq,av,aa,ai=fit(raw,.035,np.linspace(raw[0],raw[-1],65),'hybrid','approach',1.2)
        ai.update(corridor_error(aq,pre_actual,goal))
        if not check_route(mg,aq,False):raise RuntimeError('Approach open-gripper proxy collision')
        transport={k:v for k,v in world.items() if k!='cucumber'};mg.update_world(WorldConfig.from_dict({'cuboid':transport}))
        actual=mg.kinematics.get_state(ta.to_device(aq[-1:].astype(np.float32)));start=goal.copy();start[:3,3]=actual.ee_position.cpu().numpy()[0];start[:3,:3]=Rotation.from_quat(actual.ee_quaternion.cpu().numpy()[0,[1,2,3,0]]).as_matrix()
        end=start.copy();end[2,3]+=.20;poses=np.repeat(start[None],65,axis=0);poses[:,:3,3]=np.linspace(start[:3,3],end[:3,3],65)
        raw=continuous(poses,aq[-1]);lq,lv,la,li=fit(raw,.04,src['state'][40:66,8:15],'hybrid','lift',1.2)
        if not check_route(mg,lq,True):
            mg.toggle_link_collision(['tcp'],False)
            lift_plan=mg.plan_single(js(aq[-1]),pose_array(end,ta),MotionGenPlanConfig(max_attempts=8,timeout=18.,enable_graph_attempt=2,use_start_state_as_retract=True,time_dilation_factor=.8))
            if not bool(lift_plan.success.item()):raise RuntimeError('No collision-free lift fallback')
            raw=lift_plan.get_interpolated_plan().position.cpu().numpy();lq,lv,la,li=fit(raw,float(lift_plan.interpolation_dt),src['state'][40:66,8:15],'hybrid','lift',1.2)
            li['collision_aware_curobo_lift_fallback']=True
            if not check_route(mg,lq,True):raise RuntimeError('Lift fallback fit collision')
        hold=45;arm=np.vstack([reach,aq[1:],np.repeat(aq[-1:],hold,axis=0),lq[1:]]);close=len(reach)+len(aq)-1
        cmd=np.repeat(src['state'][0:1,:19],len(arm),axis=0);cmd[:,8:15]=arm;cmd[:,15]=0;cmd[close:close+hold,15]=np.linspace(0,1.1,hold);cmd[close+hold:,15]=1.1
        vel=np.zeros_like(cmd);vel[:,8:15]=np.vstack([rv,av[1:],np.zeros((hold,7)),lv[1:]]);acc=np.zeros_like(cmd);acc[:,8:15]=np.vstack([ra,aa[1:],np.zeros((hold,7)),la[1:]])
        dest=out/f'hybrid_{yaw:+03d}';dest.mkdir(exist_ok=True);np.savez(dest/'trajectory.npz',joint_targets=cmd,joint_velocity=vel,joint_acceleration=acc,dt=np.full(len(cmd),1/60),close_start_frame=close,release_start_frame=-1,grasp_pose=goal,pickup_only=True,yaw_delta_deg=yaw,approach_start_frame=len(reach))
        report.update(status='planned',candidate_metadata=meta[args.candidate],reach=ri,approach=ai,lift=li,metrics=joint_metrics(arm,np.full(len(arm),1/60)))
    except Exception as error:
        import traceback;traceback.print_exc();report['error']=str(error)
    reports.append(report);(out/'planning_report.json').write_text(json.dumps(reports,indent=2));print('CORRIDOR_PICKUP',json.dumps(report),flush=True)
