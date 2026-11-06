"""Preserve accepted reach, then close while clearing the table by 8 mm.

Reuses the first part of the old collision-checked lift, not a new IK branch.
All source outputs remain untouched. No collision disable or lift locking.
"""
import argparse,json,os
import numpy as np
from pathlib import Path
from curobo.geom.types import WorldConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.base import TensorDeviceType
from curobo.wrap.reacher.motion_gen import MotionGen,MotionGenConfig
from plan_goalset import BASE,check_route,joint_metrics
from conditional_common import static_world,robot_model,fit
from rapid_timing import retime,monotone

p=argparse.ArgumentParser();p.add_argument('--yaws',type=int,nargs='+',default=[-30,75,-90]);p.add_argument('--clearance',type=float,default=.008);p.add_argument('--output',default='rapid10_pickups');p.add_argument('--monotone-clearance',action='store_true');p.add_argument('--protect-approach',action='store_true');p.add_argument('--input-pickups',default='');p.add_argument('--input-profiles',default='');p.add_argument('--approach-slowdown',type=float,default=1.);p.add_argument('--pregrasp-settle',type=float,default=0.);p.add_argument('--scene',default='');args=p.parse_args()
os.environ['TASK759_TIMING_PROFILE']='rapid'
out=BASE/'output'/args.output;out.mkdir(exist_ok=True);selected={} if args.input_profiles or args.input_pickups else json.loads((BASE/'output/hybrid11_selection.json').read_text());src=np.load(BASE/'input/source_episode.npz');fk=np.load(BASE/'input/source_fk.npz');scene=json.loads(Path(args.scene or BASE/'output/hybrid11_profiles/scene.json').read_text());(out/'scene.json').write_text(json.dumps(scene,indent=2));ta=TensorDeviceType()
mg=MotionGen(MotionGenConfig.load_from_robot_config(robot_model(),WorldConfig.from_dict({'cuboid':static_world(scene,fk)}),tensor_args=ta,num_ik_seeds=16,num_trajopt_seeds=2,trajopt_tsteps=32,interpolation_dt=1/60,collision_checker_type=CollisionCheckerType.PRIMITIVE,use_cuda_graph=True,collision_cache={'obb':24}))
reports=[]
for yaw in args.yaws:
    path=str(BASE/'output'/args.input_profiles/f'hybrid_{yaw:+03d}'/'trajectory.npz') if args.input_profiles else str(BASE/'output'/args.input_pickups/f'yaw_{yaw:+03d}'/'trajectory.npz') if args.input_pickups else json.loads((BASE/'output'/selected[str(yaw)]/'scene_used.json').read_text())['arguments']['trajectory'];old=np.load(path);close=int(old['close_start_frame']);q=old['joint_targets'][:,8:15];v=old['joint_velocity'][:,8:15];a=old['joint_acceleration'][:,8:15]
    rq,rv,ra,ri=retime(q[:close],v[:close],a[:close],'reach')
    if args.protect_approach and 'approach_start_frame' in old:
        app=int(old['approach_start_frame']);fq,fv,fa,fi=retime(q[:app],v[:app],a[:app],'reach');aq,av,aa,ai=retime(q[app-1:close],v[app-1:close],a[app-1:close],'approach',slowdown=args.approach_slowdown)
        dwell=int(round(args.pregrasp_settle*60))
        rq=np.vstack([fq,np.repeat(fq[-1:],dwell,axis=0),aq[1:]]);rv=np.vstack([fv,np.zeros((dwell,7)),av[1:]]);ra=np.vstack([fa,np.zeros((dwell,7)),aa[1:]]);ri=dict(duration_s=(len(rq)-1)/60,free_reach=fi,protected_approach=ai,pregrasp_settle_s=dwell/60)
    lift=np.vstack([q[close-1],q[close+45:]])
    positions=mg.kinematics.get_state(ta.to_device(lift.astype(np.float32))).ee_position.cpu().numpy();dz=positions[:,2]-positions[0,2]
    indices=np.where(dz>=args.clearance)[0];k=int(indices[0]) if len(indices) else 0
    if k<2:raise RuntimeError('Cannot locate initial table-clearance segment')
    early=lift[:k+1];eq,ev,ea,ei=fit(early,1/60,np.linspace(early[0],early[-1],len(early)),'hybrid','close_lift')
    if args.monotone_clearance:eq,ev,ea,ei=monotone(early[0],early[-1],'close_lift')
    lq,lv,la,li=fit(lift[k:],1/60,src['state'][40:66,8:15],'hybrid','lift')
    if not check_route(mg,rq,False) or not check_route(mg,eq,True) or not check_route(mg,lq,True):raise RuntimeError('Rapid path collision rejection')
    hold=max(30,len(eq)+6);still=hold-len(eq)
    arm=np.vstack([rq,eq,np.repeat(eq[-1:],still,axis=0),lq[1:]])
    vel=np.vstack([rv,ev,np.zeros((still,7)),lv[1:]]);acc=np.vstack([ra,ea,np.zeros((still,7)),la[1:]])
    cmd=np.repeat(src['state'][0:1,:19],len(arm),axis=0);cmd[:,8:15]=arm;cmd[:,15]=0
    delay=3;start=len(rq)+delay;cmd[start:len(rq)+hold,15]=np.linspace(0,1.1,hold-delay);cmd[len(rq)+hold:,15]=1.1
    vv=np.zeros_like(cmd);vv[:,8:15]=vel;aa=np.zeros_like(cmd);aa[:,8:15]=acc
    dest=out/f'hybrid_{yaw:+03d}';dest.mkdir(exist_ok=True);np.savez(dest/'trajectory.npz',joint_targets=cmd,joint_velocity=vv,joint_acceleration=aa,dt=np.full(len(cmd),1/60),close_start_frame=start,release_start_frame=-1,pickup_only=True,yaw_delta_deg=yaw,grasp_pose=old['grasp_pose'])
    report=dict(yaw_deg=yaw,source_accepted_pickup=path,requested_initial_clearance_m=args.clearance,actual_initial_clearance_m=float(dz[k]),reach=ri,close_lift=ei,lift=li,closure_duration_s=(hold-delay)/60,metrics=joint_metrics(arm,np.full(len(arm),1/60)),status='planned_not_physics_verified')
    reports.append(report);(out/'planning_report.json').write_text(json.dumps(reports,indent=2));print('RAPID_PICKUP',json.dumps(report),flush=True)
