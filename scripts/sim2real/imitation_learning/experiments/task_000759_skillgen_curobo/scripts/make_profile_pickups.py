"""Controlled reference/natural/hybrid comparison from identical initialized paths."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from curobo.geom.types import WorldConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.base import TensorDeviceType
from curobo.wrap.reacher.motion_gen import MotionGen,MotionGenConfig
from plan_goalset import BASE,check_route,joint_metrics
from conditional_common import static_world,robot_model,fit

p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--yaws',type=int,nargs='+',required=True);p.add_argument('--profiles',nargs='+',default=['reference','natural','hybrid']);p.add_argument('--output',default='conditional_profiles');p.add_argument('--motion-speed',type=float,default=1.);args=p.parse_args()
parent=BASE/'output'/args.input;out=BASE/'output'/args.output;out.mkdir(exist_ok=True);scene=json.loads((parent/'scene.json').read_text());(out/'scene.json').write_text(json.dumps(scene,indent=2));src=np.load(BASE/'input/source_episode.npz');fk=np.load(BASE/'input/source_fk.npz');ta=TensorDeviceType()
w=static_world(scene,fk);w['cucumber']=dict(dims=[.236,.042,.039],pose=[*scene['cucumber_center'],1,0,0,0])
mg=MotionGen(MotionGenConfig.load_from_robot_config(robot_model(),WorldConfig.from_dict({'cuboid':w}),tensor_args=ta,num_ik_seeds=16,num_trajopt_seeds=4,trajopt_tsteps=32,interpolation_dt=1/60,collision_checker_type=CollisionCheckerType.PRIMITIVE,use_cuda_graph=True,collision_cache={'obb':24}))
reports=[]
for yaw in args.yaws:
    old=np.load(parent/f'yaw_{yaw:+03d}/trajectory.npz');close=int(old['close_start_frame']);raw_reach=old['raw_reach'] if 'raw_reach' in old else old['joint_targets'][:close,8:15];raw_lift=old['raw_lift'] if 'raw_lift' in old else old['joint_targets'][close+45:,8:15];reach_dt=float(old['raw_reach_dt']) if 'raw_reach_dt' in old else 1/60;lift_dt=float(old['raw_lift_dt']) if 'raw_lift_dt' in old else 1/60
    ori=Rotation.from_euler('z',scene['cucumber_yaw_rad']+np.deg2rad(yaw)).as_quat()[[3,0,1,2]];w['cucumber']['pose']=[*scene['cucumber_center'],*ori.tolist()]
    for profile in args.profiles:
        mg.update_world(WorldConfig.from_dict({'cuboid':w}))
        q,v,a,ri=fit(raw_reach,reach_dt,src['state'][:41,8:15],profile,'reach',args.motion_speed);valid=check_route(mg,q,False)
        mg.update_world(WorldConfig.from_dict({'cuboid':{k:v for k,v in w.items() if k!='cucumber'}}))
        lq,lv,la,li=fit(raw_lift,lift_dt,src['state'][40:66,8:15],profile,'lift',args.motion_speed);valid=valid and check_route(mg,lq,True)
        report=dict(profile=profile,yaw_delta_deg=yaw,initialization=str(parent),same_grasp_goal=True,same_initial_route_for_profiles=True,reach=ri,lift=li,status='planned' if valid else 'collision_rejected')
        if valid:
            hold=45;arm=np.vstack([q,np.repeat(q[-1:],hold,axis=0),lq[1:]]);cmd=np.repeat(src['state'][0:1,:19],len(arm),axis=0);cmd[:,8:15]=arm;cmd[:,15]=0;cmd[len(q):len(q)+hold,15]=np.linspace(0,1.1,hold);cmd[len(q)+hold:,15]=1.1
            vel=np.zeros_like(cmd);vel[:,8:15]=np.vstack([v,np.zeros((hold,7)),lv[1:]]);acc=np.zeros_like(cmd);acc[:,8:15]=np.vstack([a,np.zeros((hold,7)),la[1:]])
            folder=out/f'{profile}_{yaw:+03d}';folder.mkdir(exist_ok=True);np.savez(folder/'trajectory.npz',joint_targets=cmd,joint_velocity=vel,joint_acceleration=acc,dt=np.full(len(arm),1/60),close_start_frame=len(q),release_start_frame=-1,grasp_pose=old['grasp_pose'],yaw_delta_deg=yaw,pickup_only=True)
            report['metrics']=joint_metrics(arm,np.full(len(arm),1/60));report['trajectory']=str(folder/'trajectory.npz')
        reports.append(report);print('PROFILE_RESULT',json.dumps(report),flush=True)
(out/'profiles.json').write_text(json.dumps(reports,indent=2))
