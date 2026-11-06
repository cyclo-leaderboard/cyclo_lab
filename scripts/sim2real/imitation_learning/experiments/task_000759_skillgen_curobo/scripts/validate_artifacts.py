"""Evidence gates. Structural checks never substitute for a physical-success result."""
import json
from pathlib import Path
import numpy as np
import h5py
base=Path('/experiment');checks={}
src=np.load(base/'input/source_episode.npz');assert src['state'].shape==(115,22)
with h5py.File(base/'input/seed.hdf5') as h:
    assert list(h['data'])==['demo_0'];d=h['data/demo_0'];assert len(d['actions'])==66
    source_obj=d['obs/datagen_info/object_pose/cucumber'][0]
    source=np.concatenate([d['obs/datagen_info/eef_pose/right_arm'][30:31],d['obs/datagen_info/target_eef_pose/right_arm'][30:66]])
checks['single_source_115_frames_one_annotated_demo']=True
for yaw in [-60,-30,0,30,60]:
    s=np.load(base/'input'/f'skill_yaw_{yaw:+03d}.npz')
    expected=s['object_pose']@np.linalg.inv(source_obj)@source
    assert np.max(abs(s['poses']-expected))<1e-5
    checks[f'official_skill_transform_{yaw}']=True
reports=json.loads((base/'output/clearance_v2/planning_report.json').read_text())
for report in reports:
    if report['status']!='planned':continue
    yaw=report['yaw_delta_deg'];t=np.load(base/'output/clearance_v2'/f'yaw_{yaw:+03d}/trajectory.npz');q=t['joint_targets'];dt=t['dt']
    assert q.shape[1]==19 and np.isfinite(q).all() and (dt>0).all()
    assert np.max(abs(np.diff(q[:,8:15],axis=0)))<.22
    assert np.allclose(q[:,:8],src['state'][0,:8]) and np.allclose(q[:,16:19],src['state'][0,16:19])
    checks[f'finite_continuous_fixed_inactive_joints_{yaw}']=True
physical={}
for folder in ['baseline_final','final_yaw_-60','final_yaw_-30','final_yaw_+00','hover_yaw_-60']:
    result=base/'output'/folder/'result.json'
    if result.exists():physical[folder]=json.loads(result.read_text())
report={'artifact_checks':checks,'physical_results':physical,'warning':'Angle smoke tests after tuning, not an unbiased success-rate estimate. See each MP4 for visual quality.'}
(base/'output/validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
