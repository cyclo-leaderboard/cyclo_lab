"""Retiming assertions and offline return-home planning probe, not task success."""
import json,subprocess,os
import numpy as np
from conditional_common import fit
from plan_goalset import BASE
src=np.load(BASE/'input/source_episode.npz');q=src['state'][:41,8:15]
out=BASE/'output/hybrid11_math_checks';out.mkdir(exist_ok=True)
p,v,a,info=fit(q,1/15,q,'hybrid','reach',1.2)
assert np.max(np.abs(p[[0,-1]]-q[[0,-1]]))<1e-8
assert np.abs(v[[0,-1]]).max()<1e-8 and np.abs(a[[0,-1]]).max()<1e-8
assert info['effective_motion_speed']>1.19 and info['effective_motion_speed']<=1.2+1e-8
old=BASE/'output/conditional_hybrid_90';ex=np.load(old/'execution.npz')
scene=json.loads((old/'scene_used.json').read_text())['scene']
checkpoint=dict(scene=scene,object_pose=ex['object_pose'][-1].tolist(),joint_target=ex['joint_targets'][-1].tolist())
(out/'home_checkpoint.json').write_text(json.dumps(checkpoint,indent=2))
env=os.environ.copy();env['CUDA_HOME']='/usr/local/cuda-12.8'
subprocess.run(['/isaac-sim/python.sh','/experiment/scripts/plan_return_home.py','--checkpoint',str(out/'home_checkpoint.json'),'--output',str(out),'--motion-speed','1.2'],check=True,env=env)
home=np.load(out/'home.npz');assert np.abs(home['joint_targets'][-1,8:15]-src['state'][0,8:15]).max()<1e-6
(out/'retiming_checks.json').write_text(json.dumps(info,indent=2));print('RETIMING C2/ENDPOINT/LIMITS AND COLLISION-CHECKED RETURN PROBE PASSED')
