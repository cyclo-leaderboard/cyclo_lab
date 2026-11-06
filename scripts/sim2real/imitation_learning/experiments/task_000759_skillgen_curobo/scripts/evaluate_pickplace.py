"""Summarize physical four-angle checks and compose an honest same-time 2x2 MP4."""
import argparse
import ast
import json
import re
from pathlib import Path
import numpy as np
import cv2
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from plan_goalset import joint_metrics
from scipy.spatial.transform import Rotation

p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',default='pickplace_four_angles');a=p.parse_args()
base=Path('/experiment');out=base/'output'/a.output;out.mkdir(exist_ok=True)
mesh=np.array(ast.literal_eval(re.search(r'point3f\[\] points = (\[.*\])',(base/'task_000759/cucumber.usda').read_text())[1]))
reports=[];clips=[]
for name in a.runs:
    folder=base/'output'/name;r=json.loads((folder/'result.json').read_text());trace=np.load(folder/'execution.npz');time=trace['time'];obj=trace['object_pose'];tcp=trace['tcp'];actual=trace['joint_actual'];q=trace['joint_targets'][:len(actual)]
    cutoff=r['release_start_frame'];close=r['close_start_frame'];z0=r['initial_object_pose'][2]
    held=(obj[:cutoff,2]>z0+.10)&(np.linalg.norm(tcp[:cutoff]-obj[:cutoff,:3],axis=1)<.15)&(actual[:cutoff,15]>.5)
    durations=np.diff(np.r_[0,time])
    held_duration=float(durations[:cutoff][held].sum())
    basket=r['basket'];xy=np.array(basket['center_xy']);half=np.array(basket['inner_size_xy'])/2
    mesh_fits=[];world_vertices=[]
    for pose in obj[-15:]:
        points=Rotation.from_quat(pose[[4,5,6,3]]).apply(mesh)+pose[:3];world_vertices.append(points)
        mesh_fits.append(bool((np.abs(points[:,:2]-xy)<half+.003).all() and (points[:,2]>basket['floor_top_m']-.005).all() and (points[:,2]<basket['wall_top_m']+.003).all()))
    r['raw_simulator_success']=r['physical_success']
    r['raw_simulator_criterion']=r['criterion']
    legacy='actual 159 collider vertices' not in r['criterion']
    r['legacy_conservative_box_success']=r['physical_success'] if legacy else None
    r['legacy_conservative_box_containment']=r['whole_cucumber_in_basket_final'] if legacy else None
    r['whole_cucumber_in_basket_final']=bool(all(mesh_fits))
    r['containment_method']='All 159 vertices of the actual convex collider, not the eight nonexistent corners of its bounding box; xy 3mm / floor 5mm / rim 3mm tolerances.'
    r['final_mesh_bounds_m']=[world_vertices[-1].min(axis=0).tolist(),world_vertices[-1].max(axis=0).tolist()]
    r['physical_success']=bool(r['physics_valid'] and len(actual)==len(trace['joint_targets']) and all(mesh_fits) and r['object_stable_final'] and r['gripper_open_final'] and r['tcp_clear_final'] and r['pick_lift_success'])
    strict=bool(r['physical_success'] and r['no_preclose_push_within_5mm'] and held_duration>=.3)
    r['criterion']='Complete physics; actual collider vertices inside basket after release; gripper open and withdrawn; >=0.3s held >10cm; <=5mm preclose displacement.'
    r['arm_tracking_within_0p05_rad']=bool(r['max_joint_tracking_error_rad']<=.05)
    r['naturalness_is_not_certified_by_task_success']=True
    vel=np.gradient(actual[:,8:15],time,axis=0);acc=np.gradient(vel,time,axis=0);jerk=np.gradient(acc,time,axis=0)
    velocity=np.diff(q[:,8:15],axis=0)/durations[1:,None]
    m=joint_metrics(q[:,8:15],durations)
    reach_metrics=joint_metrics(q[:close,8:15],durations[:close])
    frames=[];cap=cv2.VideoCapture(str(folder/'observer.mp4'));expected=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
    while True:
        ok,bgr=cap.read()
        if not ok:break
        frames.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
    cap.release();assert len(frames)==expected and fps==15
    # This is a discrete engineering smoke test, not a population success rate.
    r.update(strict_full_task_success=strict,held_above_10cm_duration_s=held_duration,command_metrics=m,reach_command_metrics=reach_metrics,actual_peak_velocity_rad_s=float(np.abs(vel).max()),actual_peak_acceleration_rad_s2=float(np.abs(acc).max()),actual_peak_jerk_rad_s3=float(np.abs(jerk).max()),actual_derivative_note='Finite-difference physical joint measurements; not analytically bounded by command retiming.',video=str(folder/'observer.mp4'),video_frames_decoded=len(frames),video_fps=fps,video_duration_s=len(frames)/fps)
    if not strict:
        r['failure_phase']='pregrasp_push' if not r['no_preclose_push_within_5mm'] else 'grasp_or_lift' if held_duration<.3 else 'transport_or_place'
    (folder/'evaluation.json').write_text(json.dumps(r,indent=2));reports.append(r);clips.append(frames)
summary=dict(num_tests=len(reports),num_successes=sum(x['strict_full_task_success'] for x in reports),success_fraction=sum(x['strict_full_task_success'] for x in reports)/len(reports),angle_definition='Object yaw relative to the original seed orientation of 26.43 degrees; positions unchanged.',scope='One trial per selected plan after development. Four discrete angles, not an all-angle guarantee or statistical success-rate estimate.',success_criterion='Finite complete physics; held above 10 cm for >=0.3s; <=5mm preclose displacement; whole cucumber stable inside basket after release, gripper open and TCP withdrawn.',original_episode=93,object_attachment=False,geometry_is_approximate=True,results=reports)
(out/'summary.json').write_text(json.dumps(summary,indent=2))
assert len(clips)==4,'Comparison layout is for four angles.'
try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
except OSError:font=ImageFont.load_default()
writer=imageio.get_writer(out/'four_angles.mp4',fps=15,codec='libx264',quality=7,macro_block_size=16)
for i in range(max(len(c) for c in clips)):
    canvas=Image.new('RGB',(1280,1024))
    for n,clip in enumerate(clips):
        x=(n%2)*640;y=(n//2)*512;frame=clip[min(i,len(clip)-1)];canvas.paste(Image.fromarray(frame),(x,y+32))
        r=reports[n];status='PASS' if r['strict_full_task_success'] else 'FAIL'
        ImageDraw.Draw(canvas).text((x+12,y+4),f'Seed yaw {r["yaw_delta_deg"]:+.0f} deg | {status} | t={min(i,len(clip)-1)/15:.1f}s',font=font,fill='white')
    writer.append_data(np.array(canvas))
writer.close()
print(json.dumps({k:v for k,v in summary.items() if k!='results'},indent=2))
