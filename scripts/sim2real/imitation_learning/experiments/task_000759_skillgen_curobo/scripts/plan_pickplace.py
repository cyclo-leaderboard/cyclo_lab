"""Isolated, original-posture-biased cuRobo cucumber pick/place experiment.

Analytic grasps, not GraspGen. One measured source episode. No object attachment.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation
from curobo.geom.types import WorldConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.util_file import load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from plan_goalset import BASE, WEIGHTS, pose_array, check_route, joint_metrics


def grasp_candidates(scene, fk, yaw):
    ref=fk['eef'][40];center=np.array(scene['cucumber_center'])
    axis=Rotation.from_euler('z',scene['cucumber_yaw_rad']+np.deg2rad(yaw)).apply([1.,0.,0.])
    values=[];meta=[]
    # Oblique closing is geometrically possible on a long convex object; strict
    # rigid rotation of the seed is unnecessarily restrictive. Physics tests it.
    for oblique in [0,-20,20,-35,35,-50,50,180,160,200,145,215,130,230,-60,240]:
        for tilt in [0,25,50]:
            for symmetry in [0,180]:
                for longitudinal in [-.035,0.,.035]:
                    for height in [.006,.018]:
                        T=np.eye(4)
                        T[:3,:3]=Rotation.from_euler('z',yaw+oblique,degrees=True).as_matrix()@ref[:3,:3]@Rotation.from_euler('y',tilt,degrees=True).as_matrix()@Rotation.from_euler('z',symmetry,degrees=True).as_matrix()
                        T[:3,3]=center+longitudinal*axis;T[2,3]=ref[2,3]+height
                        values.append(T);meta.append(dict(oblique_deg=oblique,tilt_deg=tilt,local_gripper_symmetry_deg=symmetry,longitudinal_offset_m=longitudinal,height_offset_m=height))
    return np.array(values),meta


def prior_spline(q,raw_dt,reference,weight=.02,hz=60,n_coeff=20,jerk_regularization=1e-7,travel_regularization=0.):
    """Whole-phase C2 fit with explicit time-aligned measured-joint shape prior.

    Endpoint correction preserves the human motion's shape rather than asking
    a rotated grasp to end at the unrotated original joint configuration.
    Every output path is collision checked; a prior never overrides feasibility.
    """
    q=np.asarray(q,dtype=float);q=q[np.r_[True,np.max(np.abs(np.diff(q,axis=0)),axis=1)>1e-8]]
    u=np.linspace(0,1,len(q));ref=np.asarray(reference,dtype=float)
    original=np.column_stack([np.interp(u,np.linspace(0,1,len(ref)),ref[:,j]) for j in range(7)])
    corrected=original+(1-u[:,None])*(q[0]-original[0])+u[:,None]*(q[-1]-original[-1])
    n=n_coeff;k=5;knots=np.r_[np.zeros(k+1),np.linspace(0,1,n-k+1)[1:-1],np.ones(k+1)]
    basis=BSpline(knots,np.eye(n),k);A=basis(u);D=basis(np.linspace(0,1,250),3)
    fixed=np.r_[np.arange(3),np.arange(n-3,n)];free=np.arange(3,n-3)
    c=np.zeros((n,7));c[:3]=q[0];c[-3:]=q[-1]
    weights=np.broadcast_to(weight(u) if callable(weight) else weight,(len(q),));prior=np.sqrt(weights)[:,None]
    reg=np.sqrt(jerk_regularization*len(q)/len(D));D1=basis(np.linspace(0,1,250),1);travel=np.sqrt(travel_regularization*len(q)/len(D1))
    lhs=np.vstack([A[:,free],reg*D[:,free],prior*A[:,free],travel*D1[:,free]])
    rhs=np.vstack([q-A[:,fixed]@c[fixed],-reg*D[:,fixed]@c[fixed],prior*(corrected-A[:,fixed]@c[fixed]),-travel*D1[:,fixed]@c[fixed]])
    c[free]=np.linalg.lstsq(lhs,rhs,rcond=None)[0]
    total=max(raw_dt*(len(q)-1),.5);s=BSpline(knots*total,c,k)
    dense=np.linspace(0,total,max(500,len(q)*3));vmax=np.abs(s(dense,1)).max();amax=np.abs(s(dense,2)).max();jmax=np.abs(s(dense,3)).max()
    dilation=max(1,vmax/.8,np.sqrt(amax/2),np.cbrt(jmax/15))*1.02
    duration=np.ceil(total*dilation*hz)/hz;grid=np.linspace(0,total,int(round(duration*hz))+1);scale=total/duration
    return s(grid),s(grid,1)*scale,s(grid,2)*scale**2,dict(duration_s=duration,original_joint_prior_weight=float(weights.mean()),reference_weight_range=[float(weights.min()),float(weights.max())],travel_regularization=travel_regularization,reference_rmse_before_rad=float(np.sqrt(np.mean((q-corrected)**2))),reference_rmse_after_rad=float(np.sqrt(np.mean((s(u*total)-corrected)**2))),max_fit_deviation_rad=float(np.abs(s(u*total)-q).max()),analytic_peak_velocity_rad_s=float(vmax*scale),analytic_peak_acceleration_rad_s2=float(amax*scale**2),analytic_peak_jerk_rad_s3=float(jmax*scale**3))


def main():
    p=argparse.ArgumentParser();p.add_argument('--yaws',type=int,nargs='+',default=[0,30,60,90]);p.add_argument('--output',default='pickplace_v1');p.add_argument('--attempts',type=int,default=10);p.add_argument('--prior-weight',type=float,default=.02);p.add_argument('--candidate-indices',type=int,nargs='+',default=None,help='Optional explicit diagnostic candidates, recorded in report.');p.add_argument('--place-orientation',choices=['grasp','seed'],default='grasp');p.add_argument('--pickup-only',action='store_true');args=p.parse_args()
    import os
    candidate_map=json.loads(os.environ.get('TASK759_CANDIDATE_MAP','{}'))
    out=BASE/'output'/args.output;out.mkdir(parents=True,exist_ok=True)
    scene=json.loads((BASE/'task_000759/scene.json').read_text());src=np.load(BASE/'input/source_episode.npz');fk=np.load(BASE/'input/source_fk.npz');ta=TensorDeviceType()
    basket=dict(center_xy=[.51,.11],inner_size_xy=[.34,.32],floor_top_m=.724,wall_top_m=.814,wall_thickness_m=.012)
    scene['basket']=basket;(out/'scene.json').write_text(json.dumps(scene,indent=2))
    def boxes(yaw):
        ori=Rotation.from_euler('z',scene['cucumber_yaw_rad']+np.deg2rad(yaw)).as_quat()[[3,0,1,2]].tolist()
        w={'table':dict(dims=[*scene['table_size_xy'],.05],pose=[*scene['table_center_xy'],scene['table_top_m']-.025,1,0,0,0]),'torso':dict(dims=[.25,.30,.55],pose=[-.045,0,.88,1,0,0,0]),'cucumber':dict(dims=[.236,.042,.039],pose=[*scene['cucumber_center'],*ori])}
        names=list(fk['body_names'])
        for i in range(2,8):w[f'passive_left_{i}']=dict(dims=[.11]*3,pose=[*fk['body_pose'][0,names.index(f'arm_l_link{i}'),:3].tolist(),1,0,0,0])
        bx,by=basket['center_xy'];sx,sy=basket['inner_size_xy'];th=basket['wall_thickness_m'];height=basket['wall_top_m']-basket['floor_top_m'];z=basket['floor_top_m']+height/2
        w['basket_floor']=dict(dims=[sx+2*th,sy+2*th,.02],pose=[bx,by,basket['floor_top_m']-.01,1,0,0,0])
        for axis,sign in [(0,-1),(0,1),(1,-1),(1,1)]:
            pos=[bx,by,z];pos[axis]+=sign*([sx,sy][axis]/2+th/2)
            dims=[sx+2*th,sy+2*th,height];dims[axis]=th
            w[f'basket_wall_{axis}_{sign}']=dict(dims=dims,pose=[*pos,1,0,0,0])
        return w
    robot=load_yaml(str(BASE/'task_000759/sg2_right.yml'))['robot_cfg']
    robot['kinematics']['cspace']['retract_config']=src['state'][40,8:15].tolist()
    robot['kinematics']['collision_spheres']['tcp']=[dict(center=[0.,s,z],radius=.016) for s in [-.063,.063] for z in [.025,.055,.085]]+[dict(center=[0,0,.12],radius=.035)]
    config=MotionGenConfig.load_from_robot_config(robot,WorldConfig.from_dict({'cuboid':boxes(args.yaws[0])}),tensor_args=ta,num_ik_seeds=64,num_trajopt_seeds=8,num_graph_seeds=4,trajopt_tsteps=48,interpolation_dt=1/60,collision_checker_type=CollisionCheckerType.PRIMITIVE,position_threshold=.002,rotation_threshold=.025,collision_activation_distance=.005,minimize_jerk=True,use_cuda_graph=not bool(os.environ.get('TASK759_CLEARANCE_WAYPOINT')),collision_cache={'obb':24})
    mg=MotionGen(config)
    ik=IKSolver(IKSolverConfig.load_from_robot_config(robot,None,tensor_args=ta,num_seeds=32,regularization=True,self_collision_check=False,self_collision_opt=False,use_cuda_graph=False,position_threshold=.002,rotation_threshold=.025))
    q0=src['state'][0,8:15];refq=src['state'][40,8:15];reports=[]
    def js(q):return JointState.from_position(ta.to_device(np.asarray(q,dtype=np.float32).reshape(1,7)),joint_names=mg.joint_names)
    def planconfig():return MotionGenPlanConfig(max_attempts=6,timeout=12.,enable_graph_attempt=2,enable_finetune_trajopt=True,parallel_finetune=True,time_dilation_factor=.8,use_start_state_as_retract=False)
    for yaw in args.yaws:
        began=time.monotonic();world=boxes(yaw);mg.update_world(WorldConfig.from_dict({'cuboid':world}));mg.reset(reset_seed=True)
        values,meta=grasp_candidates(scene,fk,yaw)
        if os.environ.get('TASK759_REFINED_GRASP_GRID')=='1':
            values=[];meta=[];axis=Rotation.from_euler('z',scene['cucumber_yaw_rad']+np.deg2rad(yaw)).apply([1.,0.,0.]);ref=fk['eef'][40]
            for oblique in [20,35,50]:
                for tilt in [35,40]:
                    for symmetry in [0,180]:
                        for longitudinal in [-.035,0.,.035]:
                            for height in [.026,.032]:
                                T=np.eye(4);T[:3,:3]=Rotation.from_euler('z',yaw+oblique,degrees=True).as_matrix()@ref[:3,:3]@Rotation.from_euler('y',tilt,degrees=True).as_matrix()@Rotation.from_euler('z',symmetry,degrees=True).as_matrix();T[:3,3]=np.array(scene['cucumber_center'])+longitudinal*axis;T[2,3]=ref[2,3]+height
                                values.append(T);meta.append(dict(oblique_deg=oblique,tilt_deg=tilt,local_gripper_symmetry_deg=symmetry,longitudinal_offset_m=longitudinal,height_offset_m=height,refined_grasp_grid=True))
            values=np.array(values)
        extra_height=float(os.environ.get('TASK759_GRASP_Z_OFFSET','0'))
        if extra_height:
            values[:,2,3]+=extra_height
            for metadata in meta:metadata['extra_grasp_height_m']=extra_height
        quat=Rotation.from_matrix(values[:,:3,:3]).as_quat()[:,[3,0,1,2]]
        pose=Pose(position=ta.to_device(np.ascontiguousarray(values[:,:3,3])).contiguous(),quaternion=ta.to_device(np.ascontiguousarray(quat)).contiguous())
        ref=ta.to_device(np.repeat(refq[None],len(values),axis=0).astype(np.float32))
        # Pinned 0.7.7 implementation expects batch x seeds x dof despite its
        # docstring saying seeds x batch x dof; do not rely on that stale docstring.
        r=ik.solve_batch(pose,retract_config=ref,seed_config=ref.unsqueeze(1),return_seeds=8)
        success=r.success.cpu().numpy();sol=r.solution.cpu().numpy()
        costs=np.mean((sol-refq[None,None,:])**2*WEIGHTS,axis=-1);costs[~success]=np.inf
        ranked=np.argsort(costs.min(axis=1));ranked=[int(i) for i in ranked if np.isfinite(costs[i].min())]
        # Preserve baseline zero-obliquity trials, then explore actual original-near IK.
        # Ranking is an explicit original-q preference, not just a seed position.
        trial_indices=[]
        for idx in ranked:
            family=tuple(meta[idx][key] for key in ['oblique_deg','tilt_deg','local_gripper_symmetry_deg'])
            if family not in [tuple(meta[k][key] for key in ['oblique_deg','tilt_deg','local_gripper_symmetry_deg']) for k in trial_indices]:trial_indices.append(idx)
        if args.candidate_indices is not None:trial_indices=args.candidate_indices
        if str(yaw) in candidate_map:trial_indices=candidate_map[str(yaw)]
        trials=[];solutions=[]
        pre_path=None;plan_start=q0
        clearance_height=float(os.environ.get('TASK759_CLEARANCE_WAYPOINT','0'))
        if clearance_height:
            clearance=fk['eef'][0].copy();clearance[:3,3]=[.40,float(fk['eef'][0,1,3]),clearance_height]
            preconfig=planconfig();preconfig.use_start_state_as_retract=True
            prepose=pose_array(clearance,ta);prepose=Pose(position=prepose.position.unsqueeze(1),quaternion=prepose.quaternion.unsqueeze(1))
            preliminary=mg.plan_single(js(q0),prepose,preconfig)
            if not bool(preliminary.success.item()):raise RuntimeError('Initial orientation-preserving clearance waypoint unreachable')
            pre_path=preliminary.get_interpolated_plan().position.cpu().numpy();plan_start=pre_path[-1]
        print('IK_DIAGNOSTIC',yaw,'reachable',len(ranked),'/',len(values),'best',[(i,meta[i],float(costs[i].min())) for i in trial_indices[:args.attempts]],flush=True)
        for attempt,index in enumerate(trial_indices[:args.attempts]):
            trial=dict(candidate_index=index,metadata=meta[index],original_grasp_joint_mse=float(costs[index].min()))
            trials.append(trial)
            try:
                grasp_goal=pose_array(values[index],ta)
                grasp_goal=Pose(position=grasp_goal.position.unsqueeze(1),quaternion=grasp_goal.quaternion.unsqueeze(1))
                result=mg.plan_grasp(js(plan_start),grasp_goal,planconfig(),grasp_approach_offset=Pose.from_list([0,0,.10,1,0,0,0],ta),retract_offset=Pose.from_list([0,0,.20,1,0,0,0],ta),grasp_approach_path_constraint=[1.,1.,1.,1.,1.,0.],retract_path_constraint=[1.,1.,1.,1.,1.,0.],grasp_approach_constraint_in_goal_frame=True,retract_constraint_in_goal_frame=False,disable_collision_links=['tcp'])
                if isinstance(result,tuple) or result.success is None or not bool(result.success.item()):trial['status']=str(getattr(result,'status','stitch_failure'));continue
                grasp=result.grasp_interpolated_trajectory.position.cpu().numpy();lift=result.retract_interpolated_trajectory.position.cpu().numpy()
                if pre_path is not None:
                    grasp=np.vstack([pre_path,grasp[1:]]);trial['orientation_preserving_clearance_height_m']=clearance_height
                if args.pickup_only:
                    reach,rv,ra,ri=prior_spline(grasp,float(result.grasp_interpolation_dt),src['state'][:41,8:15],args.prior_weight)
                    if not check_route(mg,reach,False):trial['status']='reach_spline_collision';continue
                    lq,lv,la,li=prior_spline(lift,float(result.retract_interpolation_dt),src['state'][40:66,8:15],min(args.prior_weight,.005))
                    transport_world={k:v for k,v in world.items() if k!='cucumber'}
                    mg.update_world(WorldConfig.from_dict({'cuboid':transport_world}))
                    if not check_route(mg,lq,True):trial['status']='lift_spline_collision';continue
                    hold=45;arm=np.vstack([reach,np.repeat(reach[-1:],hold,axis=0),lq[1:]])
                    cmd=np.repeat(src['state'][0:1,:19],len(arm),axis=0);cmd[:,8:15]=arm;close=len(reach);cmd[:,15]=0;cmd[close:close+hold,15]=np.linspace(0,1.1,hold);cmd[close+hold:,15]=1.1
                    vel=np.zeros_like(cmd);vel[:,8:15]=np.vstack([rv,np.zeros((hold,7)),lv[1:]]);accel=np.zeros_like(cmd);accel[:,8:15]=np.vstack([ra,np.zeros((hold,7)),la[1:]])
                    metrics=joint_metrics(arm,np.full(len(arm),1/60));score=metrics['weighted_joint_travel_rad']+4*joint_metrics(reach,np.full(len(reach),1/60))['excess_joint_travel_rad']
                    trial.update(status='valid_pickup_only',score=score,metrics=metrics,reach=ri,lift=li)
                    dest=out/f'yaw_{yaw:+03d}';dest.mkdir(exist_ok=True)
                    np.savez(dest/'trajectory.npz',joint_targets=cmd,dt=np.full(len(cmd),1/60),joint_velocity=vel,joint_acceleration=accel,close_start_frame=close,release_start_frame=-1,grasp_pose=values[index],yaw_delta_deg=yaw,pickup_only=True,raw_reach=grasp,raw_lift=lift,raw_reach_dt=float(result.grasp_interpolation_dt),raw_lift_dt=float(result.retract_interpolation_dt))
                    solutions.append((score,index,None,None,None,None,close,-1,trial))
                    print('PICKUP_PLAN',yaw,index,json.dumps(trial),flush=True);break
                # Once lifted, initial object obstacle no longer exists at the table.
                # Object is NOT attached in physics. Conservative carried-object swept
                # volume is checked analytically below and actual PhysX decides success.
                transport_world={k:v for k,v in world.items() if k!='cucumber'}
                mg.update_world(WorldConfig.from_dict({'cuboid':transport_world}))
                drop=values[index].copy()
                if args.place_orientation=='seed':
                    # A yaw-only change keeps the held object's long axis horizontal
                    # and avoids forcing an awkward pickup wrist orientation at place.
                    drop[:3,:3]=Rotation.from_euler('z',-(yaw+meta[index]['oblique_deg']),degrees=True).as_matrix()@values[index,:3,:3]
                object_offset_local=values[index,:3,:3].T@(np.array(scene['cucumber_center'])-values[index,:3,3])
                drop[:3,3]=np.array([*basket['center_xy'],.88])-drop[:3,:3]@object_offset_local
                carry_cfg=planconfig();carry_cfg.use_start_state_as_retract=True
                carry=mg.plan_single(js(lift[-1]),pose_array(drop,ta),carry_cfg)
                if not bool(carry.success.item()):trial['status']='basket_transport_failed';continue
                carryq=carry.get_interpolated_plan().position.cpu().numpy()
                retreat=drop.copy();retreat[2,3]+=.16
                withdraw=mg.plan_single(js(carryq[-1]),pose_array(retreat,ta),carry_cfg)
                if not bool(withdraw.success.item()):trial['status']='withdraw_failed';continue
                wq=withdraw.get_interpolated_plan().position.cpu().numpy()
                reach,rv,ra,ri=prior_spline(grasp,float(result.grasp_interpolation_dt),src['state'][:41,8:15],args.prior_weight)
                transport,tv,taa,ti=prior_spline(np.vstack([lift,carryq[1:]]),float(result.retract_interpolation_dt),src['state'][40:81,8:15],args.prior_weight)
                retreatq,wv,wa,wi=prior_spline(wq,float(withdraw.interpolation_dt),src['state'][80:100,8:15],args.prior_weight)
                mg.update_world(WorldConfig.from_dict({'cuboid':world}))
                if not check_route(mg,reach,False):
                    trial['raw_open_reach_collision_free']=check_route(mg,grasp,False)
                    recovered=False
                    for n_coeff,prior_weight,jerk_weight in [(32,.005,1e-8),(48,0.,1e-9)]:
                        reach,rv,ra,ri=prior_spline(grasp,float(result.grasp_interpolation_dt),src['state'][:41,8:15],prior_weight,n_coeff=n_coeff,jerk_regularization=jerk_weight)
                        if check_route(mg,reach,False):
                            ri.update(spline_control_points=n_coeff,jerk_regularization=jerk_weight,collision_aware_prior_relaxation=True);recovered=True;break
                    if not recovered:trial['status']='reach_spline_collision';continue
                mg.update_world(WorldConfig.from_dict({'cuboid':transport_world}))
                if not check_route(mg,transport,True) or not check_route(mg,retreatq,False):trial['status']='transport_spline_collision';continue
                if max(np.abs(reach[-1]-transport[0]).max(),np.abs(transport[-1]-retreatq[0]).max())>.003:trial['status']='joint_seam';continue
                hold=45;release=60
                arm=np.vstack([reach,np.repeat(reach[-1:],hold,axis=0),transport[1:],np.repeat(transport[-1:],release,axis=0),retreatq[1:]])
                vel=np.vstack([rv,np.zeros((hold,7)),tv[1:],np.zeros((release,7)),wv[1:]])
                acc=np.vstack([ra,np.zeros((hold,7)),taa[1:],np.zeros((release,7)),wa[1:]])
                close_start=len(reach);release_start=close_start+hold+len(transport)-1
                cmd=np.repeat(src['state'][0:1,:19],len(arm),axis=0);cmd[:,8:15]=arm;cmd[:,15]=0
                cmd[close_start:close_start+hold,15]=np.linspace(0,1.1,hold);cmd[close_start+hold:release_start,15]=1.1;cmd[release_start:release_start+release,15]=np.linspace(1.1,0,release)
                dt=np.full(len(arm),1/60);metrics=joint_metrics(arm,dt)
                posture=float(np.mean((reach[-1]-refq)**2*WEIGHTS))
                score=metrics['weighted_joint_travel_rad']+2*metrics['excess_joint_travel_rad']+3*posture+.02*metrics['duration_s']
                reach_metrics=joint_metrics(reach,np.full(len(reach),1/60))
                eef_metrics={}
                for phase,path in [('reach',reach),('transport',transport)]:
                    positions=mg.kinematics.get_state(ta.to_device(path.astype(np.float32))).ee_position.cpu().numpy()
                    length=float(np.linalg.norm(np.diff(positions,axis=0),axis=1).sum())
                    excess=length-float(np.linalg.norm(positions[-1]-positions[0]))
                    overshoot=max(0.,float(positions[:,2].max()-max(positions[0,2],positions[-1,2]))-.05)
                    eef_metrics[phase]=dict(length_m=length,excess_length_m=excess,unnecessary_height_overshoot_m=overshoot)
                    score+=15*excess+30*overshoot
                score+=4*reach_metrics['excess_joint_travel_rad']
                trial.update(status='valid',score=score,metrics=metrics,reach=ri,transport=ti,withdraw=wi,selected_grasp_original_joint_mse=posture)
                trial.update(reach_metrics=reach_metrics,eef_metrics=eef_metrics)
                v=np.zeros_like(cmd);v[:,8:15]=vel;a=np.zeros_like(cmd);a[:,8:15]=acc
                solutions.append((score,index,cmd,dt,v,a,close_start,release_start,trial))
                print('VALID_PICKPLACE',yaw,attempt,json.dumps(trial),flush=True)
                if len(solutions)>=3:break
            except Exception as e:
                import traceback;traceback.print_exc();trial.update(status='exception',error=repr(e))
            finally:mg.update_world(WorldConfig.from_dict({'cuboid':world}))
        report=dict(yaw_delta_deg=yaw,kinematic_reachable_candidates=len(ranked),candidate_count=len(values),explicit_diagnostic_candidates=args.candidate_indices,place_orientation=args.place_orientation,transport_ik_regularized_to_previous_state=True,ik_reference_frame=40,ik_reference_regularization=True,original_trajectory_prior_weight=args.prior_weight,original_grasp_posture_score_weight=3.,trials=trials,status='planned' if solutions else 'failed',elapsed_s=time.monotonic()-began,physics_verified=False,geometry_is_approximate=True,continuous_all_angle_guarantee=False,global_optimality_guarantee=False)
        if solutions:
            score,index,cmd,dt,v,a,close_start,release_start,trial=min(solutions,key=lambda s:s[0]);folder=out/f'yaw_{yaw:+03d}';folder.mkdir(exist_ok=True)
            if not args.pickup_only:np.savez(folder/'trajectory.npz',joint_targets=cmd,dt=dt,joint_velocity=v,joint_acceleration=a,close_start_frame=close_start,release_start_frame=release_start,grasp_pose=values[index],yaw_delta_deg=yaw)
            report.update(selected=meta[index],selected_candidate=index,score=score,metrics=trial['metrics'],close_start_frame=close_start,release_start_frame=release_start,reach=trial['reach'],transport=trial.get('transport',trial.get('lift')),withdraw=trial.get('withdraw'),pickup_only=args.pickup_only)
        reports.append(report);(out/'planning_report.json').write_text(json.dumps(reports,indent=2));print('PICKPLACE_RESULT',yaw,report['status'],report.get('selected'),flush=True)


if __name__=='__main__':main()
