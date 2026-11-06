"""Isolated Task000759: physical joint-target replay; no attachment or object scripting."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--trajectory', default='')
p.add_argument('--name', default='baseline')
p.add_argument('--yaw', type=float, default=0.)
p.add_argument('--video', action='store_true')
p.add_argument('--interpolate-targets',action='store_true')
p.add_argument('--velocity-feedforward',action='store_true')
p.add_argument('--object-offset', type=float, nargs=3, default=[0.,0.,0.])
p.add_argument('--scene',default='',help='Experiment-local scene JSON, optionally with a basket.')
p.add_argument('--plan-place',action='store_true',help='Plan full placement from a measured, stationary lifted checkpoint.')
p.add_argument('--profile',choices=['reference','natural','hybrid'],default='hybrid')
p.add_argument('--allow-axial-roll',action='store_true')
p.add_argument('--return-home',action='store_true')
p.add_argument('--motion-speed',type=float,default=1.,help='Already applied to pickup file; forwarded to placement/home moving phases only.')
p.add_argument('--audit-contacts',action='store_true')
p.add_argument('--diagnostic-no-table-contact',action='store_true',help='Counterfactual diagnosis only; NOT an accepted task condition.')
p.add_argument('--max-time',type=float,default=0.,help='Diagnostic early stop only.')
p.add_argument('--lift-damping',type=float,default=0.,help='Experiment-local finite lift hold damping; zero preserves source config. Not a lock.')
AppLauncher.add_app_launcher_args(p);args=p.parse_args()
Path('/experiment/output').mkdir(parents=True,exist_ok=True)
if args.return_home:
    # Process-level leases survive host exec disconnections and cap subsequent
    # task-owned Kit instances. Locks are automatically released on process exit.
    import fcntl,time
    slot_lease=None
    while slot_lease is None:
        for slot in range(2):
            candidate=Path(f'/experiment/output/.task759_kit_slot_{slot}.lock').open('a')
            try:fcntl.flock(candidate,fcntl.LOCK_EX|fcntl.LOCK_NB);slot_lease=candidate;break
            except BlockingIOError:candidate.close()
        if slot_lease is None:time.sleep(2)
if args.video:args.enable_cameras=True
app=AppLauncher(args).app
import json,os
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sensors import Camera, CameraCfg
from cyclo_lab.assets.robots.FFW_SG2 import FFW_SG2_CFG

base=Path('/experiment');cfg=json.loads(Path(args.scene or base/'task_000759/scene.json').read_text())
src=np.load(base/'input/source_episode.npz');out=base/'output'/args.name;out.mkdir(parents=True,exist_ok=True)
writer=None
rapid=os.environ.get('TASK759_TIMING_PROFILE')=='rapid'
try:
    if not (base/'task_000759/cucumber.usda').exists():
        import runpy
        runpy.run_path('/experiment/scripts/make_cucumber.py')
    sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/cfg['physics_hz'],device=args.device))
    sim_utils.GroundPlaneCfg().func('/World/Ground',sim_utils.GroundPlaneCfg())
    light=sim_utils.DomeLightCfg(intensity=1700.);light.func('/World/Light',light)
    table=sim_utils.CuboidCfg(size=(*cfg['table_size_xy'],.05),collision_props=sim_utils.CollisionPropertiesCfg(),visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.58,.43,.28)),physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.6,dynamic_friction=.5))
    table.func('/World/Table',table,translation=(*cfg['table_center_xy'],cfg['table_top_m']-.025))
    if 'basket' in cfg:
        basket=cfg['basket'];bx,by=basket['center_xy'];sx,sy=basket['inner_size_xy'];th=basket['wall_thickness_m'];height=basket['wall_top_m']-basket['floor_top_m']
        basket_parts=[('Floor',(sx+2*th,sy+2*th,.02),(bx,by,basket['floor_top_m']-.01))]
        for axis,sign in [(0,-1),(0,1),(1,-1),(1,1)]:
            position=[bx,by,basket['floor_top_m']+height/2];position[axis]+=sign*([sx,sy][axis]/2+th/2)
            dimensions=[sx+2*th,sy+2*th,height];dimensions[axis]=th
            basket_parts.append((f'Wall{axis}{"Plus" if sign>0 else "Minus"}',tuple(dimensions),tuple(position)))
        for name,dimensions,position in basket_parts:
            part=sim_utils.CuboidCfg(size=dimensions,collision_props=sim_utils.CollisionPropertiesCfg(),visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.94,.72,.06)),physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.8,dynamic_friction=.6,restitution=0.))
            part.func('/World/Basket'+name,part,translation=position)
    rc=FFW_SG2_CFG.copy();rc.prim_path='/World/Robot';rc.spawn.articulation_props.fix_root_link=True
    if args.lift_damping:rc.actuators['lift'].damping=args.lift_damping
    if args.audit_contacts:rc.spawn.activate_contact_sensors=True
    rc.init_state.pos=(0.,0.,0.);rc.init_state.rot=(1.,0.,0.,0.)
    rc.init_state.joint_pos.update({str(n):float(q) for n,q in zip(src['joint_names'][:19],src['state'][0,:19])})
    robot=Articulation(rc)
    from pxr import UsdPhysics,Sdf
    rigid_paths={prim.GetName():prim.GetPath().pathString for prim in sim.stage.Traverse() if prim.GetPath().pathString.startswith('/World/Robot/') and prim.HasAPI(UsdPhysics.RigidBodyAPI)}
    if args.diagnostic_no_table_contact:
        for body_path in rigid_paths.values():
            prim=sim.stage.GetPrimAtPath(body_path)
            if prim.IsValid():UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel().AddTarget(Sdf.Path('/World/Table'))
    contact_sensors=[];contact_body_names=['arm_r_link7','gripper_r_rh_p12_rn_r1','gripper_r_rh_p12_rn_l1','gripper_r_rh_p12_rn_r2','gripper_r_rh_p12_rn_l2']
    if args.audit_contacts:
        from isaaclab.sensors import ContactSensor,ContactSensorCfg
        sim_utils.activate_contact_sensors('/World/Robot',stage=sim.stage)
        (out/'rigid_body_paths.json').write_text(json.dumps(rigid_paths,indent=2))
        for body_name in contact_body_names:
            contact_sensors.append(ContactSensor(ContactSensorCfg(prim_path=rigid_paths[body_name],update_period=0.,history_length=0,filter_prim_paths_expr=['/World/Table','/World/Cucumber'],max_contact_data_count_per_prim=64)))
    contact_events=[];audit_clock={'frame':-1,'time_s':0.}
    if args.audit_contacts:
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools
        def on_contact_report(headers,data):
            for header in headers:
                actors=[str(PhysicsSchemaTools.intToSdfPath(header.actor0)),str(PhysicsSchemaTools.intToSdfPath(header.actor1))]
                colliders=[str(PhysicsSchemaTools.intToSdfPath(header.collider0)),str(PhysicsSchemaTools.intToSdfPath(header.collider1))]
                if not (any(v.startswith('/World/Table') for v in actors+colliders) and any(v.startswith('/World/Robot') for v in actors+colliders)):continue
                indices=range(header.contact_data_offset,header.contact_data_offset+header.num_contact_data)
                points=[data[j] for j in indices]
                if not points:continue
                impulse=float(sum(np.linalg.norm(np.array(point.impulse)) for point in points))
                contact_events.append(dict(**audit_clock,actors=actors,colliders=colliders,impulse_magnitude_sum_Ns=impulse,force_equivalent_N=impulse*cfg['physics_hz'],minimum_separation_m=float(min(point.separation for point in points)),positions=[list(point.position) for point in points]))
        contact_report_subscription=get_physx_simulation_interface().subscribe_contact_report_events(on_contact_report)
    center=np.array(cfg['cucumber_center'])+args.object_offset;yaw=cfg['cucumber_yaw_rad']+np.deg2rad(args.yaw)
    cucumber=RigidObject(RigidObjectCfg(prim_path='/World/Cucumber',spawn=sim_utils.UsdFileCfg(usd_path='/experiment/task_000759/cucumber.usda',rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=16,solver_velocity_iteration_count=4,max_depenetration_velocity=1.,linear_damping=.1,angular_damping=.1),mass_props=sim_utils.MassPropertiesCfg(mass=cfg['cucumber_mass_kg']),collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.001,rest_offset=0.)),init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(center),rot=(float(np.cos(yaw/2)),0.,0.,float(np.sin(yaw/2))))))
    material=sim_utils.RigidBodyMaterialCfg(static_friction=cfg['cucumber_static_friction'],dynamic_friction=cfg['cucumber_dynamic_friction'],restitution=0.)
    material.func('/World/CucumberPhysics',material)
    sim_utils.bind_physics_material('/World/Cucumber/geometry','/World/CucumberPhysics')
    camera=None
    if args.video:
        camera=Camera(CameraCfg(prim_path='/World/Observer',update_period=0.,height=480,width=640,data_types=['rgb'],spawn=sim_utils.PinholeCameraCfg(focal_length=24.,horizontal_aperture=26.,clipping_range=(.05,20.))))
    sim.reset();robot.update(1/240);cucumber.update(1/240)
    ids=[robot.joint_names.index(str(n)) for n in src['joint_names'][:19]]
    target=robot.data.default_joint_pos.clone();target[0,ids]=torch.as_tensor(src['state'][0,:19],device=sim.device)
    robot.write_joint_state_to_sim(target,torch.zeros_like(target))
    if camera:
        camera.set_world_poses_from_view(torch.tensor([[1.65,-1.65,1.65]],device=sim.device),torch.tensor([[.43,-.05,.82]],device=sim.device))
        import imageio.v2 as imageio
        writer=imageio.get_writer(out/'observer.mp4',fps=cfg['control_hz'],codec='libx264',quality=7,macro_block_size=16)
    if args.trajectory:
        planned=np.load(args.trajectory);commands=planned['joint_targets'];durations=planned['dt']
        velocity_commands=planned['joint_velocity'] if 'joint_velocity' in planned else np.zeros_like(commands)
        close_start_frame=int(planned['close_start_frame']) if 'close_start_frame' in planned else None
        release_start_frame=int(planned['release_start_frame']) if 'release_start_frame' in planned else None
    else:
        count=cfg['skill_end_exclusive'];commands=np.repeat(src['state'][0:1,:19],count,axis=0)
        commands[:,8:15]=src['state'][:count,8:15];commands[:,15]=src['action'][:count,15]
        durations=np.full(count,1/cfg['control_hz'])
        velocity_commands=np.zeros_like(commands);close_start_frame=None;release_start_frame=None
    initial_hold=(2 if rapid else 6) if args.return_home else 30
    commands=np.concatenate([commands,np.repeat(commands[-1:],initial_hold,axis=0)])
    durations=np.concatenate([durations,np.full(initial_hold,1/cfg['control_hz'])])
    velocity_commands=np.concatenate([velocity_commands,np.zeros((initial_hold,19))])
    log_q=[];log_obj=[];log_tcp=[];log_tcp_quat=[];log_error=[];times=[];t=0.;physics_valid=True;video_frames=0;place_planned=False;place_error=None
    home_planned=False;home_error=None;home_start_frame=None;home_start_time=None
    log_contacts=[];log_body_positions=[]
    (out/'scene_used.json').write_text(json.dumps({'scene':cfg,'arguments':vars(args)},indent=2,default=str))
    for _ in range(120):
        robot.set_joint_position_target(target);robot.write_data_to_sim();sim.step(render=False);robot.update(1/240);cucumber.update(1/240)
    z_initial=float(cucumber.data.root_pos_w[0,2]);initial_obj=cucumber.data.root_state_w[0,:7].cpu().numpy().copy()
    if camera:
        for _ in range(6):sim.render()
    previous_command=src['state'][0,:19].copy();previous_velocity=np.zeros(19)
    i=0
    while i<len(commands):
        audit_clock.update(frame=i,time_s=t)
        command=commands[i];duration=durations[i]
        steps=max(1,round(float(duration)*cfg['physics_hz']))
        # Mimic joints are driven by the USD mimic constraints, never teleported.
        for substep in range(steps):
            alpha=(substep+1)/steps if args.interpolate_targets else 1.
            target[0,ids]=torch.as_tensor(previous_command+alpha*(command-previous_command),device=sim.device,dtype=target.dtype)
            if args.velocity_feedforward:
                velocity_target=torch.zeros_like(target)
                velocity_target[0,ids]=torch.as_tensor(previous_velocity+alpha*(velocity_commands[i]-previous_velocity),device=sim.device,dtype=target.dtype)
                robot.set_joint_velocity_target(velocity_target)
            robot.set_joint_position_target(target);robot.write_data_to_sim();sim.step(render=False);robot.update(1/cfg['physics_hz']);cucumber.update(1/cfg['physics_hz']);t+=1/cfg['physics_hz']
        previous_command=command.copy();previous_velocity=velocity_commands[i].copy()
        q=robot.data.joint_pos[0,ids].cpu().numpy().copy();obj=cucumber.data.root_state_w[0,:7].cpu().numpy().copy()
        if not bool(torch.isfinite(robot.data.body_pos_w).all()) or float(robot.data.body_pos_w.abs().max())>10.:
            physics_valid=False;print('PHYSICS_INVALID',i,flush=True);break
        from isaaclab.utils.math import quat_apply
        bi=robot.body_names.index('arm_r_link7');tcp=robot.data.body_pos_w[0,bi]+quat_apply(robot.data.body_quat_w[0,bi].unsqueeze(0),torch.tensor([[0.,0.,-.2]],device=sim.device))[0]
        tcp_quat=robot.data.body_quat_w[0,bi].cpu().numpy().copy()
        log_q.append(q);log_obj.append(obj);log_tcp.append(tcp.cpu().numpy());log_tcp_quat.append(tcp_quat);log_error.append(np.max(np.abs(q[8:15]-command[8:15])));times.append(t)
        if args.audit_contacts:
            forces=[]
            for sensor in contact_sensors:
                sensor.update(float(duration),force_recompute=True);forces.append(sensor.data.force_matrix_w[0,0,0].cpu().numpy().copy())
            log_contacts.append(forces);log_body_positions.append(robot.data.body_pos_w[0].cpu().numpy().copy())
            if i%15==0 and np.linalg.norm(forces,axis=1).max()>1.:print('TABLE_CONTACT_N',i,np.round(np.linalg.norm(forces,axis=1),2),'LIFT_MM',float(q[18]-command[18])*1000,flush=True)
        if camera and video_frames<round(t*cfg['control_hz']):
            sim.render();camera.update(float(duration),force_recompute=True)
            rgb=camera.data.output['rgb'][0,:,:,:3].cpu().numpy()
            while video_frames<round(t*cfg['control_hz']):
                writer.append_data(rgb);video_frames+=1
        if i%30==0:print('SIM_FRAME',i,'OBJECT',np.round(obj[:3],4),'TRACK_ERR',log_error[-1],flush=True)
        if args.max_time and t>=args.max_time:break
        if args.return_home and place_planned and place_error is None and not home_planned and i==len(commands)-1:
            home_planned=True
            import subprocess,os
            checkpoint={'scene':cfg,'object_pose':obj.tolist(),'joint_actual':q.tolist(),'joint_target':command.tolist(),'physical_time_s':t}
            (out/'home_checkpoint.json').write_text(json.dumps(checkpoint,indent=2))
            with (out/'home_planner.log').open('w') as planner_log:
                env=os.environ.copy();env['CUDA_HOME']='/usr/local/cuda-12.8'
                completed=subprocess.run(['/isaac-sim/python.sh','-u','/experiment/scripts/plan_return_home.py','--checkpoint',str(out/'home_checkpoint.json'),'--output',str(out),'--motion-speed',str(args.motion_speed)],stdout=planner_log,stderr=subprocess.STDOUT,env=env,timeout=180)
            if completed.returncode!=0 or not (out/'home.npz').exists():home_error='return_home_planner_failed';print('HOME_REJECTED',home_error,flush=True)
            else:
                home=np.load(out/'home.npz');follow=home['joint_targets'];home_start_frame=len(commands);home_start_time=t
                home_hold=6 if rapid else 15
                commands=np.vstack([commands,follow,np.repeat(follow[-1:],home_hold,axis=0)]);durations=np.r_[durations,home['dt'],np.full(home_hold,1/cfg['control_hz'])];velocity_commands=np.vstack([velocity_commands,home['joint_velocity'],np.zeros((home_hold,19))])
                print('HOME_APPENDED',len(follow),'START',home_start_frame,flush=True)
        if args.plan_place and not place_planned and i==len(commands)-1:
            place_planned=True
            if obj[2]<z_initial+.10 or np.linalg.norm(obj[:3]-tcp.cpu().numpy())>.20:
                place_error='pickup_did_not_hold_object';print('PLACE_REJECTED',place_error,flush=True)
            else:
                from scipy.spatial.transform import Rotation
                import subprocess,os
                T=np.eye(4);T[:3,:3]=Rotation.from_quat(tcp_quat[[1,2,3,0]]).as_matrix();T[:3,3]=tcp.cpu().numpy()
                O=np.eye(4);O[:3,:3]=Rotation.from_quat(obj[[4,5,6,3]]).as_matrix();O[:3,3]=obj[:3]
                checkpoint={'scene':cfg,'profile':args.profile,'yaw_delta_deg':args.yaw,'tcp_transform':T.tolist(),'object_transform':O.tolist(),'joint_actual':q.tolist(),'joint_target':command.tolist(),'prefix_frames':len(commands),'physical_time_s':t}
                (out/'lift_checkpoint.json').write_text(json.dumps(checkpoint,indent=2))
                with (out/'place_planner.log').open('w') as planner_log:
                    env=os.environ.copy();env['CUDA_HOME']='/usr/local/cuda-12.8'
                    planner_args=['/isaac-sim/python.sh','-u','/experiment/scripts/plan_conditioned_place.py','--checkpoint',str(out/'lift_checkpoint.json'),'--output',str(out),'--profile',args.profile,'--motion-speed',str(args.motion_speed)]
                    if args.allow_axial_roll or args.return_home:planner_args.append('--allow-axial-roll')
                    completed=subprocess.run(planner_args,stdout=planner_log,stderr=subprocess.STDOUT,env=env,timeout=240)
                if completed.returncode!=0 or not (out/'suffix.npz').exists():place_error='placement_planner_failed';print('PLACE_REJECTED',place_error,flush=True)
                else:
                    suffix=np.load(out/'suffix.npz');release_start_frame=len(commands)+int(suffix['release_start_frame']);follow=suffix['joint_targets'];follow_dt=suffix['dt'];follow_v=suffix['joint_velocity']
                    final_hold=(2 if rapid else 6) if args.return_home else 30
                    commands=np.vstack([commands,follow,np.repeat(follow[-1:],final_hold,axis=0)]);durations=np.r_[durations,follow_dt,np.full(final_hold,1/cfg['control_hz'])];velocity_commands=np.vstack([velocity_commands,follow_v,np.zeros((final_hold,19))])
                    print('PLACE_APPENDED',len(follow),'RELEASE',release_start_frame,flush=True)
        i+=1
    obj=np.array(log_obj);tcp=np.array(log_tcp);q=np.array(log_q)
    held=(obj[-15:,2]>z_initial+.10)&(np.linalg.norm(obj[-15:,:3]-tcp[-15:],axis=1)<.20)
    report={'name':args.name,'yaw_delta_deg':args.yaw,'physical_success':bool(physics_valid and len(obj)==len(commands) and held.all()),'physics_valid':physics_valid,'criterion':'last 15 control frames object > initial settled z + 0.10 m and within 0.20 m of TCP; complete finite simulation','initial_object_pose':initial_obj.tolist(),'final_object_pose':obj[-1].tolist(),'max_lift_m':float(obj[:,2].max()-z_initial),'final_lift_m':float(obj[-1,2]-z_initial),'max_joint_tracking_error_rad':float(max(log_error)),'num_frames':len(obj),'object_scripted_after_reset':False,'geometry_is_approximate':True}
    report.update(profile=args.profile,online_placement_requested=args.plan_place,placement_planner_error=place_error,timing_profile='rapid' if rapid else 'conservative')
    if 'basket' in cfg:
        basket=cfg['basket'];xy=np.array(basket['center_xy']);half=np.array(basket['inner_size_xy'])/2
        from scipy.spatial.transform import Rotation
        # Test the actual collider, not fictitious corners of its bounding box.
        import ast,re
        vertices=np.array(ast.literal_eval(re.search(r'point3f\[\] points = (\[.*\])',(base/'task_000759/cucumber.usda').read_text())[1]))
        fits=[]
        for pose in obj[-15:]:
            points=Rotation.from_quat(pose[[4,5,6,3]]).apply(vertices)+pose[:3]
            fits.append(bool((np.abs(points[:,:2]-xy)<half+.003).all() and (points[:,2]<basket['wall_top_m']+.003).all() and (points[:,2]>basket['floor_top_m']-.005).all()))
        stable=bool(np.linalg.norm(obj[-1,:3]-obj[-15,:3])<.01)
        opened=bool((q[-15:,15]<.15).all());clear=bool((np.linalg.norm(tcp[-15:]-obj[-15:,:3],axis=1)>.12).all())
        lifted=bool(obj[:,2].max()>z_initial+.10)
        report.update(physical_success=bool(physics_valid and len(obj)==len(commands) and all(fits) and stable and opened and clear and lifted),criterion='complete finite simulation; lifted > 10 cm; actual 159 collider vertices inside basket for final 15 samples (xy/rim 3 mm, floor 5 mm tolerance), stable < 1 cm, gripper open < .15 rad and TCP > 12 cm away',basket=basket,whole_cucumber_in_basket_final=bool(all(fits)),object_stable_final=stable,gripper_open_final=opened,tcp_clear_final=clear,pick_lift_success=lifted,release_start_frame=release_start_frame)
    if close_start_frame is not None:
        before=obj[:min(close_start_frame,len(obj)),:3]
        drift=np.linalg.norm(before-initial_obj[:3],axis=1)
        report['preclose_max_object_displacement_m']=float(drift.max())
        report['preclose_object_displacement_at_close_m']=float(drift[-1])
        report['no_preclose_push_within_5mm']=bool(drift.max()<.005)
        report['close_start_frame']=close_start_frame
    if args.return_home:
        error=np.abs(q[-15:,8:15]-src['state'][0,8:15])
        returned=bool(home_start_frame is not None and home_error is None and (error<.03).all())
        report.update(return_home_requested=True,return_home_success=returned,return_home_error=home_error,home_start_frame=home_start_frame,home_start_time_s=home_start_time,final_home_max_joint_error_rad=float(error[-1].max()),physical_time_s=t,motion_speed=args.motion_speed)
        report['pick_place_success_before_home_gate']=report['physical_success']
        report['physical_success']=bool(report['physical_success'] and returned and report.get('no_preclose_push_within_5mm',True))
        report['criterion']+='; returned to source initial right-arm joints within .03 rad for final 15 samples; no preclose push > 5 mm'
    report.update(lift_command_range_m=float(np.ptp(commands[:,18])),lift_actual_range_m=float(np.ptp(q[:,18])),lift_max_tracking_error_m=float(np.abs(q[:,18]-commands[:len(q),18]).max()),lift_controller_damping=float(rc.actuators['lift'].damping),lift_controller_stiffness=float(rc.actuators['lift'].stiffness),diagnostic_table_collision_disabled=args.diagnostic_no_table_contact,diagnostic_early_stop=bool(args.max_time))
    if args.audit_contacts:
        contact_array=np.array(log_contacts);peaks=np.linalg.norm(contact_array,axis=-1).max(axis=0)
        report.update(table_contact_peak_N=dict(zip(contact_body_names,peaks.tolist())),max_robot_table_contact_N=float(peaks.max()),static_table_force_filter_validated=False)
        np.savez(out/'contact_audit.npz',table_forces=contact_array,contact_body_names=contact_body_names,body_positions=np.array(log_body_positions),body_names=robot.body_names,time=times)
        (out/'table_contact_events.json').write_text(json.dumps(contact_events,indent=2))
        report.update(direct_table_contact_event_count=len(contact_events),direct_table_contact_peak_force_equivalent_N=max((event['force_equivalent_N'] for event in contact_events),default=0.))
    if args.diagnostic_no_table_contact or args.max_time:report['accepted_task_test']=False
    (out/'result.json').write_text(json.dumps(report,indent=2))
    np.savez(out/'execution.npz',joint_actual=q,joint_targets=commands,object_pose=obj,tcp=tcp,tcp_quaternion=np.array(log_tcp_quat),time=times)
    np.savez(out/'executed_trajectory.npz',joint_targets=commands,dt=durations,joint_velocity=velocity_commands,close_start_frame=close_start_frame if close_start_frame is not None else -1,release_start_frame=release_start_frame if release_start_frame is not None else -1,home_start_frame=home_start_frame if home_start_frame is not None else -1,yaw_delta_deg=args.yaw)
    print('RESULT',json.dumps(report),flush=True)
except Exception:
    import traceback
    traceback.print_exc()
    raise
finally:
    if writer is not None:writer.close()
    app.close(wait_for_replicator=False,skip_cleanup=True)
