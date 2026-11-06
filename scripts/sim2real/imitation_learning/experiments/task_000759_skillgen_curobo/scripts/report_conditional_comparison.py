"""Twelve physical runs, honest metrics, and four streamed A/B/C comparison MP4s."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import cv2
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial.transform import Rotation
from plan_goalset import BASE,WEIGHTS,joint_metrics

p=argparse.ArgumentParser();p.add_argument('--output',default='conditional_comparison_final');p.add_argument('--angles',type=int,nargs='+',default=[15,45,75,90]);p.add_argument('--rollfree-75',action='store_true');args=p.parse_args();out=BASE/'output'/args.output;out.mkdir(exist_ok=True);source=np.load(BASE/'input/source_episode.npz');reports=[]
profiles=['reference','natural','hybrid'];labels={'reference':'A: Reference .25','natural':'B: Natural, reference 0','hybrid':'C: Hybrid, contact-weighted'}
for yaw in args.angles:
    for profile in profiles:
        folder=BASE/'output'/(f'conditional_{profile}_{yaw}'+('_rollfree' if args.rollfree_75 and yaw==75 else ''));r=json.loads((folder/'result.json').read_text());trace=np.load(folder/'execution.npz');cmd=trace['joint_targets'][:len(trace['joint_actual']),8:15];actual=trace['joint_actual'];times=trace['time'];dt=np.diff(np.r_[0,times]);close=r['close_start_frame'];release=r['release_start_frame'];obj=trace['object_pose'];tcp=trace['tcp']
        prefix=np.load(BASE/'output'/('comparison90_profiles' if yaw==90 else ('conditional_extra_profiles' if profile=='hybrid' else 'conditional_extra_ablation'))/f'{profile}_{yaw:+03d}/trajectory.npz')
        normalized=(times[:close]-times[0])/(times[close-1]-times[0]);ref=np.column_stack([np.interp(normalized,np.linspace(0,1,41),source['state'][:41,j]) for j in range(8,15)])
        corrected=ref+(1-normalized[:,None])*(cmd[0]-ref[0])+normalized[:,None]*(cmd[close-1]-ref[-1])
        cutoff=release if release is not None and release>0 else len(obj);held=(obj[:cutoff,2]>r['initial_object_pose'][2]+.10)&(np.linalg.norm(tcp[:cutoff]-obj[:cutoff,:3],axis=1)<.15)&(actual[:cutoff,15]>.5);held_duration=float(dt[:cutoff][held].sum())
        r['usable_full_task_success']=bool(r['physical_success'] and r['no_preclose_push_within_5mm'] and held_duration>=.3)
        r['held_above_10cm_s']=held_duration;r['arm_tracking_within_0p05_rad']=bool(r['max_joint_tracking_error_rad']<.05)
        r['command_metrics']=joint_metrics(cmd,dt);r['approach_command_metrics']=joint_metrics(cmd[:close],dt[:close]);r['source_raw_approach_rmse_rad']=float(np.sqrt(np.mean((cmd[:close]-ref)**2)));r['source_endpoint_adjusted_shape_rmse_rad']=float(np.sqrt(np.mean((cmd[:close]-corrected)**2)))
        v=np.gradient(actual[:,8:15],times,axis=0);acc=np.gradient(v,times,axis=0);jerk=np.gradient(acc,times,axis=0)
        r['actual_peak_velocity_rad_s']=float(np.abs(v).max());r['actual_peak_acceleration_rad_s2']=float(np.abs(acc).max());r['actual_peak_jerk_rad_s3']=float(np.abs(jerk).max());r['actual_derivative_note']='Finite differences of physical feedback; command jerk bound does not bound these contact/PD derivatives.'
        T=prefix['grasp_pose'];quats=trace['tcp_quaternion'];k=close-1;r['grasp_tcp_position_error_m']=float(np.linalg.norm(tcp[k]-T[:3,3]));r['grasp_tcp_orientation_error_deg']=float(np.degrees((Rotation.from_matrix(T[:3,:3]).inv()*Rotation.from_quat(quats[k,[1,2,3,0]])).magnitude()))
        steps=np.linalg.norm(np.diff(tcp[:close],axis=0),axis=1);r['approach_eef_path_length_m']=float(steps.sum());r['approach_eef_straight_distance_m']=float(np.linalg.norm(tcp[close-1]-tcp[0]));r['video']=str(folder/'observer.mp4')
        cap=cv2.VideoCapture(r['video']);expected=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);count=0
        while True:
            ok,_=cap.read()
            if not ok:break
            count+=1
        cap.release();assert expected==count and count>0;r['video_frames']=count;r['video_fps']=fps;r['video_duration_s']=count/fps;r['full_video_decode_pass']=True
        if not r['usable_full_task_success']:
            r['failure_stage']='preclose_push' if not r['no_preclose_push_within_5mm'] else 'grasp_or_lift' if held_duration<.3 else 'placement_planner' if r.get('placement_planner_error') else 'placement_or_release'
        (folder/'comparison_evaluation.json').write_text(json.dumps(r,indent=2));reports.append(r)
try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
except OSError:font=ImageFont.load_default()
for yaw in args.angles:
    rows=[next(r for r in reports if r['yaw_delta_deg']==yaw and r['profile']==profile) for profile in profiles];caps=[cv2.VideoCapture(r['video']) for r in rows];last=[None]*3;maxframes=max(r['video_frames'] for r in rows);writer=imageio.get_writer(out/f'compare_{yaw:03d}.mp4',fps=15,codec='libx264',quality=7,macro_block_size=16)
    for i in range(maxframes):
        frame=Image.new('RGB',(1920,512));draw=ImageDraw.Draw(frame)
        for n,(r,cap) in enumerate(zip(rows,caps)):
            ok,bgr=cap.read()
            if ok:last[n]=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            assert last[n] is not None
            frame.paste(Image.fromarray(last[n]),(n*640,32));status='PASS' if r['usable_full_task_success'] else 'FAIL'
            draw.text((n*640+8,5),f'{labels[r["profile"]]} | {yaw} deg | {status}',font=font,fill='white')
        writer.append_data(np.asarray(frame))
    writer.close()
    for c in caps:c.release()
    # Decode the complete composed MP4 as well.
    cap=cv2.VideoCapture(str(out/f'compare_{yaw:03d}.mp4'));count=0
    while True:
        ok,_=cap.read()
        if not ok:break
        count+=1
    cap.release();assert count==maxframes
summary=dict(source_episode=93,yaw_definition='Relative to original 26.43 degree yaw, positions and physical assets unchanged.',profiles={'reference':dict(reference_weight=.25,travel_regularization=0.),'natural':dict(reference_weight=0.,travel_regularization=.001),'hybrid':dict(free_space_reference_weight=.005,contact_reference_weight_max=.08,travel_regularization=.0005)},method_note='Same grasp goals and initial approach paths per angle. Reference weighting and joint-travel regularization vary; all use hard joint endpoint/C2 constraints, retiming, dense collision rejection, and actual-held-object placement replanning. Not a pure SkillGen baseline or cuRobo V2 integrated custom cost.',num_runs=len(reports),num_usable_successes=sum(r['usable_full_task_success'] for r in reports),successes_by_profile={p:sum(r['usable_full_task_success'] for r in reports if r['profile']==p) for p in profiles},scope='One selected physical run per condition, after development. Not a population success-rate estimate or human-naturalness guarantee.',comparison_time_note='Physical time at 15 fps; shorter/failed clips hold their final frame. Planner compute time while simulation is paused is not included.',geometry_is_approximate=True,results=reports)
(out/'summary.json').write_text(json.dumps(summary,indent=2))
summary['goal_freedom_75']='All three 75 degree profiles use the same extra axial-roll and interior-XY placement goal set.' if args.rollfree_75 else 'Fixed flat roll and fixed basket-centre goal, same for all profiles.'
(out/'summary.json').write_text(json.dumps(summary,indent=2))
fields=['angle','profile','usable_success','raw_task_success','preclose_push_mm','duration_s','approach_joint_travel_weighted_rad','approach_excess_joint_travel_rad','joint_travel_weighted_rad','excess_joint_travel_rad','source_raw_rmse_rad','source_adjusted_rmse_rad','command_peak_velocity','command_peak_acceleration','command_peak_jerk','actual_peak_jerk','tracking_error_rad','failure_stage']
with (out/'metrics.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
    for r in reports:
        m=r['command_metrics'];am=r['approach_command_metrics'];writer.writerow(dict(angle=r['yaw_delta_deg'],profile=r['profile'],usable_success=r['usable_full_task_success'],raw_task_success=r['physical_success'],preclose_push_mm=r['preclose_max_object_displacement_m']*1000,duration_s=r['video_duration_s'],approach_joint_travel_weighted_rad=am['weighted_joint_travel_rad'],approach_excess_joint_travel_rad=am['excess_joint_travel_rad'],joint_travel_weighted_rad=m['weighted_joint_travel_rad'],excess_joint_travel_rad=m['excess_joint_travel_rad'],source_raw_rmse_rad=r['source_raw_approach_rmse_rad'],source_adjusted_rmse_rad=r['source_endpoint_adjusted_shape_rmse_rad'],command_peak_velocity=m['peak_velocity_rad_s'],command_peak_acceleration=m['peak_acceleration_rad_s2'],command_peak_jerk=m['peak_jerk_rad_s3'],actual_peak_jerk=r['actual_peak_jerk_rad_s3'],tracking_error_rad=r['max_joint_tracking_error_rad'],failure_stage=r.get('failure_stage','')))
print('COMPARISON_RESULT',summary['successes_by_profile'],flush=True)
