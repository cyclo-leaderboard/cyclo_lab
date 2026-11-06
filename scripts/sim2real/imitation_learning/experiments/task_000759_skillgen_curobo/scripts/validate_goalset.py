"""Check exported C2 joint commands and summarize independent physics evidence."""
import argparse,json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--output',default='goalset_v6');args=p.parse_args()
base=Path('/experiment');folder=base/'output'/args.output
reports=json.loads((folder/'planning_report.json').read_text());source=np.load(base/'input/source_episode.npz')
checks={};physical={}
for report in reports:
    yaw=report['yaw_delta_deg']
    if report['status']!='planned':continue
    d=np.load(folder/f'yaw_{yaw:+03d}/trajectory.npz');q=d['joint_targets'];v=d['joint_velocity'];a=d['joint_acceleration'];close=int(d['close_start_frame'])
    assert np.isfinite(q).all() and np.isfinite(v).all() and np.isfinite(a).all()
    assert q.shape[1]==19 and q.shape==v.shape==a.shape
    assert np.allclose(d['dt'],1/60)
    assert np.max(abs(q[0]-source['state'][0,:19]))<1e-6
    assert np.allclose(q[:,:8],source['state'][0,:8]) and np.allclose(q[:,16:],source['state'][0,16:19])
    assert np.max(abs(v[:,8:15]))<=.801 and np.max(abs(a[:,8:15]))<=2.001
    assert np.max(abs(v[[0,close-1,close+44,-1],8:15]))<1e-6
    assert np.max(abs(a[[0,close-1,close+44,-1],8:15]))<1e-6
    assert np.max(abs(q[close:close+45,8:15]-q[close-1,8:15]))<1e-6
    assert report['cucumber_in_collision_world']
    assert report['reach']['analytic_peak_jerk_rad_s3']<=15.01
    assert report['transport']['analytic_peak_jerk_rad_s3']<=15.01
    checks[str(yaw)]={'finite_commands':True,'fixed_passive_joints':True,'60hz':True,
        'position_velocity_acceleration_continuous_at_closure':True,'derivative_bounds':True,
        'cucumber_in_planner_world':True}
for path in sorted((base/'output').glob('goalset_v*/result.json')):
    physical[path.parent.name]=json.loads(path.read_text())
result={'trajectory_checks':checks,'physics_runs':physical,
    'note':'Physical grasp, no pre-closure pushing, and subjective naturalness are separate criteria. Unexecuted paths are not physical successes.'}
(folder/'validation.json').write_text(json.dumps(result,indent=2))
print('VALIDATED',len(checks),'planned orientations')
for name,r in physical.items():print(name,'grasp=',r['physical_success'],'preclose_drift_m=',r.get('preclose_max_object_displacement_m'))
