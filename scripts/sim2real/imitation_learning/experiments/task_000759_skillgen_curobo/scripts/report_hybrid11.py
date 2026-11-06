"""Audit actual pick/place/home executions and fully decode all rendered MP4s."""
import argparse,csv,json,hashlib,shutil
from pathlib import Path
import numpy as np
import cv2,imageio.v2 as imageio
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw,ImageFont
from plan_goalset import BASE

p=argparse.ArgumentParser();p.add_argument('--runs',default='hybrid11');p.add_argument('--output',default='hybrid11_report');p.add_argument('--mapping',default='');args=p.parse_args()
out=BASE/'output'/args.output;out.mkdir(exist_ok=True)
angles=[0,30,-30,45,-45,60,-60,75,-75,90,-90]
mapping=json.loads(Path(args.mapping).read_text()) if args.mapping else {}
reports=[];captures=[];lengths=[];reviews=[]
font=ImageFont.load_default(size=19)
small=ImageFont.load_default(size=16)
for yaw in angles:
    folder=BASE/'output'/mapping.get(str(yaw),f'{args.runs}_{yaw:+03d}')
    if not (folder/'result.json').exists():raise RuntimeError(f'Missing actual simulation report: {folder}')
    r=json.loads((folder/'result.json').read_text());ex=np.load(folder/'execution.npz');cmd=ex['joint_targets'];t=ex['time'];actual=ex['joint_actual']
    quality=bool(r['physical_success'] and r['max_joint_tracking_error_rad']<.05)
    velocity=np.diff(cmd[:,8:15],axis=0)/np.diff(t)[:,None]
    acceleration=np.diff(velocity,axis=0)/np.diff(t)[1:,None]
    jerk=np.diff(acceleration,axis=0)/np.diff(t)[2:,None]
    actual_velocity=np.diff(actual[:,8:15],axis=0)/np.diff(t)[:,None]
    actual_acceleration=np.diff(actual_velocity,axis=0)/np.diff(t)[1:,None]
    actual_jerk=np.diff(actual_acceleration,axis=0)/np.diff(t)[2:,None]
    home=r.get('home_start_frame');home_exc=None
    if home is not None:
        route=cmd[home-1:,8:15];home_exc=float((np.abs(np.diff(route,axis=0)).sum(axis=0)-np.abs(route[-1]-route[0])).sum())
    cap=cv2.VideoCapture(str(folder/'observer.mp4'));n=0;snapshots=[]
    expected=int(round(float(t[-1])*15));selected=set(np.linspace(0,max(0,expected-1),12,dtype=int))
    while True:
        ok,frame=cap.read()
        if not ok:break
        if n in selected:snapshots.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        n+=1
    cap.release()
    if n!=expected:raise RuntimeError(f'Video frame count mismatch: {folder}: {n}!={expected}')
    exported=out/f'yaw_{yaw:+03d}.mp4';shutil.copy2(folder/'observer.mp4',exported)
    assert hashlib.sha256(exported.read_bytes()).digest()==hashlib.sha256((folder/'observer.mp4').read_bytes()).digest()
    payload=dict(np.load(folder/'executed_trajectory.npz'));payload.update(home_start_frame=r.get('home_start_frame',-1),motion_speed=r.get('motion_speed',1.))
    np.savez(out/f'trajectory_{yaw:+03d}.npz',**payload)
    board=Image.new('RGB',(1280,960),(20,24,32))
    for index,frame in enumerate(snapshots):board.paste(Image.fromarray(frame).resize((320,240)),((index%4)*320,(index//4)*240))
    draw=ImageDraw.Draw(board);draw.text((12,740),f'{yaw:+d} deg | {"PASS" if quality else "FAIL"} | pick, place, return home',font=font,fill='white')
    draw.text((12,780),f'pre-close push: {r.get("preclose_max_object_displacement_m",0)*1000:.2f} mm | final home error: {r.get("final_home_max_joint_error_rad",999):.5f} rad',font=small,fill='white')
    board.save(out/f'review_{yaw:+03d}.jpg')
    entry=dict(yaw_deg=yaw,success=r['physical_success'],pick_place_success=r.get('pick_place_success_before_home_gate',False),return_home_success=r.get('return_home_success',False),no_preclose_push=r.get('no_preclose_push_within_5mm',False),preclose_push_mm=r.get('preclose_max_object_displacement_m',0)*1000,home_error_rad=r.get('final_home_max_joint_error_rad'),physical_duration_s=float(t[-1]),peak_command_velocity_rad_s=float(np.abs(velocity).max()),peak_command_acceleration_rad_s2=float(np.abs(acceleration).max()),peak_command_jerk_rad_s3=float(np.abs(jerk).max()),home_excess_joint_travel_rad=home_exc,max_tracking_error_rad=r['max_joint_tracking_error_rad'],video_frames=n,video_fully_decoded=True,video=str(folder/'observer.mp4'),result=str(folder/'result.json'),place_error=r.get('placement_planner_error'),home_error=r.get('return_home_error'))
    quality=bool(r['physical_success'] and r['max_joint_tracking_error_rad']<.05)
    entry.update(physical_task_success=r['physical_success'],motion_quality_check=quality,success=quality,quality_check_definition='Task success plus max right-arm tracking error < .05 rad; heuristic screen, not proof of human-like motion.',peak_actual_velocity_rad_s=float(np.abs(actual_velocity).max()),peak_actual_acceleration_rad_s2=float(np.abs(actual_acceleration).max()),peak_actual_jerk_rad_s3=float(np.abs(actual_jerk).max()),actual_joint_jerk_bound_claimed=False)
    entry.update(exported_video=str(exported),exported_full_trajectory=str(out/f'trajectory_{yaw:+03d}.npz'))
    checkpoint=json.loads((folder/'lift_checkpoint.json').read_text())
    begin=int(np.searchsorted(t,checkpoint['physical_time_s']));end=int(r['release_start_frame'])
    rotations=Rotation.from_quat(ex['tcp_quaternion'][begin:end+1][:,[1,2,3,0]])
    wrist_travel=float(np.abs(np.diff(actual[begin:end+1,12:15],axis=0)).sum())
    entry.update(lift_command_range_mm=r.get('lift_command_range_m',0)*1000,lift_actual_range_mm=r.get('lift_actual_range_m',0)*1000,lift_max_tracking_error_mm=r.get('lift_max_tracking_error_m',0)*1000,lift_controller_damping=r.get('lift_controller_damping'),lift_locked=False,table_collisions_enabled=not r.get('diagnostic_table_collision_disabled',False),carry_to_release_eef_net_rotation_deg=float(np.degrees((rotations[0].inv()*rotations[-1]).magnitude())),carry_to_release_eef_rotation_arc_deg=float(np.degrees((rotations[:-1].inv()*rotations[1:]).magnitude().sum())),carry_to_release_wrist_travel_rad=wrist_travel,timing_profile=r.get('timing_profile','conservative'))
    reports.append(entry);captures.append(cv2.VideoCapture(str(folder/'observer.mp4')));lengths.append(n)
canvas_size=(1440,1536);last=[None]*len(reports)
with imageio.get_writer(out/'all_11_angles.mp4',fps=15,codec='libx264',quality=7,macro_block_size=16) as writer:
    for index in range(max(lengths)):
        canvas=Image.new('RGB',canvas_size,(18,23,31));draw=ImageDraw.Draw(canvas)
        for j,(r,cap) in enumerate(zip(reports,captures)):
            ok,frame=cap.read()
            if ok:last[j]=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            x=(j%3)*480;y=(j//3)*384
            if last[j] is not None:canvas.paste(Image.fromarray(last[j]).resize((480,360)),(x,y+24))
            draw.text((x+8,y+2),f'{r["yaw_deg"]:+d} deg  {"PASS" if r["success"] else "FAIL"}  {r["physical_duration_s"]:.1f}s',font=small,fill=(116,244,157) if r['success'] else (255,128,128))
        draw.text((975,1190),'HYBRID: pick / place / home',font=small,fill='white');draw.text((975,1220),'1x physical simulation time',font=small,fill='white');draw.text((975,1250),'15 fps actual simulation',font=small,fill='white');draw.text((975,1280),'Finished clips hold last frame',font=small,fill='white')
        writer.append_data(np.asarray(canvas))
for cap in captures:cap.release()
cap=cv2.VideoCapture(str(out/'all_11_angles.mp4'));decoded=0
while cap.read()[0]:decoded+=1
cap.release()
assert decoded==max(lengths)
summary=dict(angles=angles,total=11,successful=sum(r['success'] for r in reports),runs=reports,comparison_video=str(out/'all_11_angles.mp4'),comparison_frames_decoded=decoded,one_source_episode=True,approximate_scene_and_physics=True,all_angles_relative_to_source=True,no_continuous_angle_or_real_robot_guarantee=True,failed_attempts_preserved=True)
attempts=[]
for path in sorted((BASE/'output').glob(f'{args.runs}_*/result.json')):
    value=json.loads(path.read_text());attempts.append(dict(name=value['name'],yaw_deg=value['yaw_delta_deg'],success=value['physical_success'],quality_accepted=bool(value['physical_success'] and value['max_joint_tracking_error_rad']<.05),max_tracking_error_rad=value['max_joint_tracking_error_rad'],preclose_push_mm=value.get('preclose_max_object_displacement_m',0)*1000,return_home_success=value.get('return_home_success'),result=str(path)))
summary.update(all_physical_attempts=attempts,number_of_physical_attempts=len(attempts),number_of_failed_attempts=sum(not a['success'] for a in attempts),interpretation='Tuned final examples, not an unbiased one-shot success rate; every earlier failed physical attempt is retained.',source_provenance=json.loads((BASE/'input/provenance.json').read_text()))
(out/'summary.json').write_text(json.dumps(summary,indent=2))
with (out/'metrics.csv').open('w') as stream:
    csv_writer=csv.DictWriter(stream,fieldnames=reports[0]);csv_writer.writeheader();csv_writer.writerows(reports)
print(json.dumps(summary,indent=2))
