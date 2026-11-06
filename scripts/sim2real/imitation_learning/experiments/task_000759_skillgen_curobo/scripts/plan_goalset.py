"""Object-aware cuRobo goal-set grasp planning, carry lookahead and C2 joint execution.

No learned grasp generator, no per-frame IK execution, no edits to shared Cyclo Lab.
The analytic cucumber symmetries are a task-specific candidate generator.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation
import torch

from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.util_file import load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

BASE = Path('/experiment')
WEIGHTS = np.array([1., 1., 1., 1., 1.5, 1.5, 2.])


def pose_array(matrices, ta):
    matrices = np.asarray(matrices)
    if matrices.ndim == 2:
        matrices = matrices[None]
    quat = Rotation.from_matrix(matrices[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]
    if len(matrices) == 1:
        return Pose(position=ta.to_device(np.ascontiguousarray(matrices[:, :3, 3])).contiguous(), quaternion=ta.to_device(np.ascontiguousarray(quat)).contiguous())
    return Pose(position=ta.to_device(np.ascontiguousarray(matrices[None, :, :3, 3])).contiguous(), quaternion=ta.to_device(np.ascontiguousarray(quat[None])).contiguous())


def joint_metrics(q, dt):
    q = np.asarray(q, dtype=float)
    dt = np.broadcast_to(np.asarray(dt, dtype=float), (len(q),))
    velocity = np.diff(q, axis=0) / dt[1:, None]
    acc = np.diff(velocity, axis=0) / ((dt[1:-1]+dt[2:])[:, None]/2)
    jerk = np.diff(acc, axis=0) / dt[3:, None]
    travel = np.sum(np.abs(np.diff(q, axis=0)), axis=0)
    return {'joint_travel_rad': travel.tolist(), 'weighted_joint_travel_rad': float(travel @ WEIGHTS),
            'excess_joint_travel_rad': float(np.sum((travel-np.abs(q[-1]-q[0]))*WEIGHTS)),
            'peak_velocity_rad_s': float(np.max(np.abs(velocity))),
            'peak_acceleration_rad_s2': float(np.max(np.abs(acc))),
            'peak_jerk_rad_s3': float(np.max(np.abs(jerk))),
            'duration_s': float(dt.sum())}


def smooth_phase(q, raw_dt, hz=60):
    """One quintic joint spline over the whole route; zero end velocity/acceleration.

    This is checked again against cuRobo collisions after interpolation; it is not assumed
    that smoothing preserves collision clearance. Slow down using analytic derivative bounds.
    """
    q = np.asarray(q, dtype=float)
    # Remove repeated boundary samples inserted by plan_grasp's stitching.
    keep = np.r_[True, np.max(np.abs(np.diff(q, axis=0)), axis=1) > 1e-8]
    q = q[keep]
    if len(q) < 3:
        raise ValueError('Degenerate route')
    # Optimize all spline control points together. Hard endpoint C2 constraints;
    # least-squares route fidelity + integrated third derivative regularization.
    # This removes interpolation wiggles rather than merely slowing them down.
    total_time=max(raw_dt*(len(q)-1),.5)
    degree=5;n_coeff=20
    knots=np.r_[np.zeros(degree+1),np.linspace(0.,1.,n_coeff-degree+1)[1:-1],np.ones(degree+1)]
    u=np.linspace(0.,1.,len(q))
    basis=BSpline(knots,np.eye(n_coeff),degree)
    A=basis(u);D=basis(np.linspace(0.,1.,250),3)
    fixed=np.r_[np.arange(3),np.arange(n_coeff-3,n_coeff)]
    free=np.arange(3,n_coeff-3)
    coefficients=np.zeros((n_coeff,7));coefficients[:3]=q[0];coefficients[-3:]=q[-1]
    reg=np.sqrt(1e-7*len(q)/len(D))
    lhs=np.vstack([A[:,free],reg*D[:,free]])
    rhs=np.vstack([q-A[:,fixed]@coefficients[fixed],-reg*D[:,fixed]@coefficients[fixed]])
    coefficients[free]=np.linalg.lstsq(lhs,rhs,rcond=None)[0]
    spline=BSpline(knots*total_time,coefficients,degree)
    dense = np.linspace(0., total_time, max(500, len(q)*3))
    vmax = float(np.max(np.abs(spline(dense, 1))))
    amax = float(np.max(np.abs(spline(dense, 2))))
    jmax = float(np.max(np.abs(spline(dense, 3))))
    dilation = max(1., vmax/.8, np.sqrt(amax/2.), np.cbrt(jmax/15.)) * 1.02
    duration = np.ceil(total_time*dilation*hz)/hz
    n = int(round(duration*hz))
    grid = np.linspace(0., total_time, n+1)
    scale = total_time/duration
    return spline(grid), spline(grid, 1)*scale, spline(grid, 2)*scale**2, {
        'duration_s': duration, 'time_scale': duration/total_time,
        'spline_control_points':n_coeff,'global_smoothness_regularization':1e-7,
        'max_joint_fit_deviation_rad':float(np.max(np.abs(spline(u*total_time)-q))),
        'analytic_peak_velocity_rad_s': vmax*scale,
        'analytic_peak_acceleration_rad_s2': amax*scale**2,
        'analytic_peak_jerk_rad_s3': jmax*scale**3,
        'endpoint_velocity_max': float(np.max(np.abs(spline([0,total_time],1)*scale))),
        'endpoint_acceleration_max': float(np.max(np.abs(spline([0,total_time],2)*scale**2))),
    }


def candidates(scene, fk, yaw):
    """Cucumber axial symmetry + tilt freedom; not a claim of stable grasps."""
    reference = fk['eef'][40]
    center = np.array(scene['cucumber_center'])
    original_yaw = scene['cucumber_yaw_rad']
    values, meta = [], []
    for flip in [0, 180]:
        delta = Rotation.from_euler('z', yaw+flip, degrees=True).as_matrix()
        axis = np.array([np.cos(original_yaw+np.deg2rad(yaw+flip)),
                         np.sin(original_yaw+np.deg2rad(yaw+flip)), 0.])
        for tilt in [0, 25, 50]:
            for longitudinal in [-.035, 0.]:
                for height in [.006, .018]:
                    T = np.eye(4)
                    T[:3,:3] = delta @ reference[:3,:3] @ Rotation.from_euler('y',tilt,degrees=True).as_matrix()
                    T[:3,3] = center + longitudinal*axis
                    T[2,3] = reference[2,3]+height
                    values.append(T)
                    meta.append({'flip_deg':flip,'tilt_from_seed_deg':tilt,
                                 'longitudinal_offset_m':longitudinal,'height_offset_m':height})
    return np.array(values), meta


def check_route(mg, q, ignore_fingers=False):
    # Dense 60 Hz samples, plus sweep checks in the cuRobo optimization itself.
    mg.toggle_link_collision(['tcp'], not ignore_fingers)
    try:
        valid=[]
        for chunk in np.array_split(q, max(1,int(np.ceil(len(q)/256)))):
            s=JointState.from_position(mg.tensor_args.to_device(chunk.astype(np.float32)),joint_names=mg.joint_names)
            valid.extend(mg.check_constraints(s).feasible.flatten().cpu().tolist())
        return bool(all(valid))
    finally:
        mg.toggle_link_collision(['tcp'], True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--yaws',type=int,nargs='+',default=[-60,-30,0,30,60])
    p.add_argument('--output',default='goalset_v1')
    p.add_argument('--alternatives',type=int,default=6)
    p.add_argument('--approach-mode',choices=['tool','world'],default='tool')
    args=p.parse_args()
    out=BASE/'output'/args.output;out.mkdir(parents=True,exist_ok=True)
    scene=json.loads((BASE/'task_000759/scene.json').read_text())
    src=np.load(BASE/'input/source_episode.npz');fk=np.load(BASE/'input/source_fk.npz')
    ta=TensorDeviceType()
    robot=load_yaml(str(BASE/'task_000759/sg2_right.yml'))['robot_cfg']
    # Gripper-open proxy: two fingertip rails and a palm proxy. Full soft mesh still
    # runs in PhysX; proxy collision checks are deliberately labelled approximate.
    robot['kinematics']['collision_spheres']['tcp']=[
        {'center':[0.,s,z],'radius':.016} for s in [-.063,.063] for z in [.025,.055,.085]]
    robot['kinematics']['collision_spheres']['tcp'].append({'center':[0.,0.,.12],'radius':.035})
    def world(yaw):
        ori=Rotation.from_euler('z',scene['cucumber_yaw_rad']+np.deg2rad(yaw)).as_quat()[[3,0,1,2]].tolist()
        w={'cuboid':{'table':{'dims':[*scene['table_size_xy'],.05],
            'pose':[*scene['table_center_xy'],scene['table_top_m']-.025,1,0,0,0]},
            'torso':{'dims':[.25,.30,.55],'pose':[-.045,0.,.88,1,0,0,0]},
            'cucumber':{'dims':[.236,.042,.039], 'pose':[*scene['cucumber_center'],*ori]}}}
        names=list(fk['body_names'])
        for i in range(2,8):
            pos=fk['body_pose'][0,names.index(f'arm_l_link{i}'),:3].tolist()
            w['cuboid'][f'passive_left_{i}']={'dims':[.11,.11,.11],'pose':[*pos,1,0,0,0]}
        return WorldConfig.from_dict(w)
    mg=MotionGen(MotionGenConfig.load_from_robot_config(robot,world(args.yaws[0]),tensor_args=ta,
        num_ik_seeds=64,num_trajopt_seeds=8,num_graph_seeds=4,trajopt_tsteps=48,
        interpolation_dt=1/60,collision_checker_type=CollisionCheckerType.PRIMITIVE,
        position_threshold=.002,rotation_threshold=.025,collision_activation_distance=.005,
        minimize_jerk=True,use_cuda_graph=True,collision_cache={'obb':16}))
    q0=src['state'][0,8:15]
    all_reports=[]
    for yaw in args.yaws:
        began=time.monotonic();mg.update_world(world(yaw));mg.reset(reset_seed=True)
        values,meta=candidates(scene,fk,yaw);remaining=list(range(len(values)));trials=[];solutions=[]
        for attempt in range(args.alternatives):
            if not remaining:break
            # Try distinct tilt / 180-degree symmetry families, not just successive
            # nearest poses from one family. cuRobo still chooses within each goal set.
            family=[0,3,1,4,2,5][attempt%6]
            pool=[i for i in remaining if i//4==family]
            if not pool:pool=remaining.copy()
            start=JointState.from_position(ta.to_device(q0[None]),joint_names=mg.joint_names)
            plan_cfg=MotionGenPlanConfig(max_attempts=6,timeout=12.,enable_graph_attempt=2,
                enable_finetune_trajopt=True,parallel_finetune=True,time_dilation_factor=.8)
            try:
                result=mg.plan_grasp(start,pose_array(values[pool],ta),plan_cfg,
                    grasp_approach_offset=Pose.from_list([0,0,.10,1,0,0,0],ta),
                    retract_offset=Pose.from_list([0,0,.22,1,0,0,0],ta),
                    grasp_approach_path_constraint=[1.,1.,1.,1.,1.,0.],
                    retract_path_constraint=[1.,1.,1.,1.,1.,0.],
                    grasp_approach_constraint_in_goal_frame=args.approach_mode=='tool',
                    retract_constraint_in_goal_frame=False,disable_collision_links=['tcp'])
            except Exception as e:
                import traceback
                traceback.print_exc()
                trials.append({'status':'planner_exception','error':repr(e)});break
            if isinstance(result,tuple):trials.append({'status':'planner_stitch_failure'});break
            index=int(result.goalset_index.item()) if result.goalset_index is not None else None
            candidate=pool[index] if index is not None else None
            if candidate is not None:remaining.remove(candidate)
            trial={'attempt':attempt,'candidate_index':candidate,'status':str(result.status),
                'metadata':meta[candidate] if candidate is not None else None}
            trials.append(trial)
            if result.success is None or not bool(result.success.item()):
                continue
            grasp=result.grasp_interpolated_trajectory.position.cpu().numpy()
            lift=result.retract_interpolated_trajectory.position.cpu().numpy()
            # Look ahead to the real seed's first post-lift carry waypoint, preserving
            # the selected grasp orientation rather than imposing the seed's wrist turn.
            carry_pose=values[candidate].copy();carry_pose[:3,3]=fk['eef'][80,:3,3]
            carry=mg.plan_single(result.retract_trajectory[-1].unsqueeze(0),pose_array(carry_pose,ta),
                MotionGenPlanConfig(max_attempts=5,timeout=10.,enable_graph_attempt=2,time_dilation_factor=.8))
            if not bool(carry.success.item()):trial['status']='carry_lookahead_failed';continue
            carry_q=carry.get_interpolated_plan().position.cpu().numpy()
            try:
                reach,reach_v,reach_a,reach_info=smooth_phase(grasp,float(result.grasp_interpolation_dt))
                transport,transport_v,transport_a,transport_info=smooth_phase(np.vstack([lift,carry_q[1:]]),float(result.retract_interpolation_dt))
            except Exception as e:trial['status']='spline_failed';trial['error']=repr(e);continue
            # Open fingers must avoid object/table before final entry. Contact disables
            # ONLY the tcp finger proxy; all arm links and obstacles remain checked.
            if not check_route(mg,reach,False) or not check_route(mg,transport,True):
                trial['status']='smoothed_collision_rejected';continue
            if np.max(np.abs(reach[-1]-transport[0]))>.003:
                trial['status']='joint_seam_rejected';continue
            hold=45  # 0.75 s closure dwell, while all arm joints stay still.
            arm=np.vstack([reach,np.repeat(reach[-1:],hold,axis=0),transport[1:]])
            vel=np.vstack([reach_v,np.zeros((hold,7)),transport_v[1:]])
            acc=np.vstack([reach_a,np.zeros((hold,7)),transport_a[1:]])
            commands=np.repeat(src['state'][0:1,:19],len(arm),axis=0);commands[:,8:15]=arm
            close_start=len(reach)
            commands[:,15]=np.r_[np.zeros(close_start),np.linspace(0.,1.1,hold),np.full(len(transport)-1,1.1)]
            dt=np.full(len(arm),1/60)
            metrics=joint_metrics(arm,dt)
            reach_metrics=joint_metrics(reach,np.full(len(reach),1/60))
            kin=mg.kinematics.get_state(ta.to_device(reach.astype(np.float32)))
            positions=kin.ee_position.cpu().numpy()
            eef_length=float(np.linalg.norm(np.diff(positions,axis=0),axis=1).sum())
            eef_excess=eef_length-float(np.linalg.norm(positions[-1]-positions[0]))
            score=metrics['weighted_joint_travel_rad']+1.5*metrics['excess_joint_travel_rad']
            score+=4.*reach_metrics['excess_joint_travel_rad']+15.*eef_excess
            score+=.02*metrics['duration_s']+.05*metrics['peak_acceleration_rad_s2']
            trial.update(status='valid',score=score,metrics=metrics,reach=reach_info,transport=transport_info,
                reach_metrics=reach_metrics,reach_eef_length_m=eef_length,reach_eef_excess_m=eef_excess,
                joint_seam_max_rad=float(np.max(np.abs(reach[-1]-transport[0]))))
            velocities=np.zeros_like(commands);velocities[:,8:15]=vel
            accelerations=np.zeros_like(commands);accelerations[:,8:15]=acc
            solutions.append((score,candidate,commands,dt,velocities,accelerations,close_start,trial))
            print('VALID_CANDIDATE',yaw,attempt,json.dumps(trial),flush=True)
        report={'yaw_delta_deg':yaw,'candidate_count':len(values),'trials':trials,
            'planner':'cuRobo MotionGen.plan_grasp goalset + carry lookahead + checked C2 quintic joint splines',
            'cucumber_in_collision_world':True,'physics_grasp_verified':False,
            'approach_mode':args.approach_mode,
            'collision_geometry_is_approximate':True,'elapsed_s':time.monotonic()-began,
            'status':'planned' if solutions else 'failed'}
        if solutions:
            selected=min(solutions,key=lambda x:x[0]);score,index,cmd,dt,v,a,close_start,trial=selected
            folder=out/f'yaw_{yaw:+03d}';folder.mkdir(exist_ok=True)
            np.savez(folder/'trajectory.npz',joint_targets=cmd,dt=dt,joint_velocity=v,joint_acceleration=a,
                close_start_frame=close_start,grasp_pose=values[index],yaw_delta_deg=yaw)
            report.update(selected_candidate=index,selected=meta[index],score=score,metrics=trial['metrics'],
                close_start_frame=close_start,reach=trial['reach'],transport=trial['transport'])
            old=BASE/'output/clearance_v2'/f'yaw_{yaw:+03d}/trajectory.npz'
            if old.exists():
                baseline=np.load(old);report['previous_command_metrics']=joint_metrics(baseline['joint_targets'][:,8:15],baseline['dt'])
        all_reports.append(report);(out/'planning_report.json').write_text(json.dumps(all_reports,indent=2))
        print('GOALSET_RESULT',yaw,report['status'],report.get('selected'),flush=True)
    return all_reports


if __name__=='__main__':
    main()
