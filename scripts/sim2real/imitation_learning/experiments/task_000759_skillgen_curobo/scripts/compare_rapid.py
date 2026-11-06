"""Measured old/new timings, lift tracking and placement rotations; no guarantees."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from plan_goalset import BASE

root=BASE/'output';out=root/'rapid10_report'
current=json.loads((out/'summary.json').read_text())
previous=json.loads((root/'hybrid11_selection.json').read_text())

def measure(folder):
    r=json.loads((folder/'result.json').read_text());e=np.load(folder/'execution.npz')
    checkpoint=json.loads((folder/'lift_checkpoint.json').read_text())
    start=int(np.searchsorted(e['time'],checkpoint['physical_time_s']));end=r['release_start_frame']
    rotations=Rotation.from_quat(e['tcp_quaternion'][start:end+1][:,[1,2,3,0]])
    close=r['close_start_frame'];window=slice(max(0,close-30),min(len(e['time']),close+60))
    return dict(duration_s=r['physical_time_s'],physical_success=r['physical_success'],
        carry_to_release_eef_net_rotation_deg=float(np.degrees((rotations[0].inv()*rotations[-1]).magnitude())),
        carry_to_release_eef_rotation_arc_deg=float(np.degrees((rotations[:-1].inv()*rotations[1:]).magnitude().sum())),
        carry_to_release_wrist_travel_rad=float(np.abs(np.diff(e['joint_actual'][start:end+1,12:15],axis=0)).sum()),
        lift_command_range_mm=float(np.ptp(e['joint_targets'][:,18])*1000),
        lift_actual_range_mm=float(np.ptp(e['joint_actual'][:,18])*1000),
        lift_max_tracking_error_mm=float(np.abs(e['joint_actual'][:,18]-e['joint_targets'][:,18]).max()*1000),
        grasp_window_lift_range_mm=float(np.ptp(e['joint_actual'][window,18])*1000),
        preclose_push_mm=r['preclose_max_object_displacement_m']*1000,result=str(folder/'result.json'))

comparisons=[]
for run in current['runs']:
    yaw=run['yaw_deg'];old=measure(root/previous[str(yaw)]);new=measure(Path(run['result']).parent)
    comparisons.append(dict(yaw_deg=yaw,old=old,new=new,duration_ratio=new['duration_s']/old['duration_s']))

diagnostics=[]
for name in ['lift_contact_baseline_v2','lift_contact_counterfactual_v2','lift_contact_events_rapid']:
    folder=root/name;r=json.loads((folder/'result.json').read_text())
    diagnostics.append(dict(name=name,table_collisions_enabled=not r['diagnostic_table_collision_disabled'],
        lift_command_range_mm=r['lift_command_range_m']*1000,
        lift_actual_range_mm=r['lift_actual_range_m']*1000,
        lift_max_tracking_error_mm=r['lift_max_tracking_error_m']*1000,
        diagnostic_only=True,result=str(folder/'result.json')))
report=dict(comparisons=comparisons,contact_diagnostics=diagnostics,
    interpretation='Table collision counterfactual supports contact as a major cause. Zero static-table sensor readings were not validated and are not evidence of no contact.',
    corrected_local_lift_damping=600,original_local_lift_damping=100,lift_stiffness_unchanged=10000,lift_not_locked=True,
    retimed_60hz_commands=True,physics_hz=240,video_fps=15,playback_speed=1,
    source_episode=93,approximate_scene=True,simulation_only_limits=True,
    global_optimality_or_continuous_angle_success_claimed=False,
    final_examples_are_tuned_selection_not_one_shot_success_rate=True)
(out/'old_vs_rapid.json').write_text(json.dumps(report,indent=2))
for pair in comparisons:
    print(pair['yaw_deg'],*[f"{pair[k]['duration_s']:.2f}s / lift {pair[k]['lift_actual_range_mm']:.2f}mm / EEF {pair[k]['carry_to_release_eef_rotation_arc_deg']:.2f}deg / wrist {pair[k]['carry_to_release_wrist_travel_rad']:.2f}rad" for k in ['old','new']])
