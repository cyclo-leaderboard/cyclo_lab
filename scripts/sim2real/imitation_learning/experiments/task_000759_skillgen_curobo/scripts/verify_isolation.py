"""Host-side read-only audit of experiment boundaries."""
import json,subprocess
from pathlib import Path
base=Path(__file__).resolve().parent.parent
def run(*args):return subprocess.check_output(args,text=True).strip()
info=json.loads(run('docker','inspect','task000759_skillgen_20261105'))[0]
mounts=[{k:m[k] for k in ['Source','Destination','RW']} for m in info['Mounts']]
assert all(not m['RW'] for m in mounts if m['Destination'] in ['/dataset','/workspace/cyclo_lab','/robot_description'])
assert info['HostConfig']['NetworkMode']=='bridge'
assert not info['HostConfig']['Privileged']
original=run('docker','exec','cyclo_lab','/isaac-sim/python.sh','-c','import importlib.util,torch; print("curobo:",importlib.util.find_spec("curobo")); print("torch:",torch.__version__)')
assert 'curobo: None' in original
status=run('git','-C','/home/robotis-ai/cyclo_lab','status','--short')
report={'container':info['Name'],'base_image':info['Image'],'container_status':info['State']['Status'],'mounts':mounts,'network_mode':info['HostConfig']['NetworkMode'],'privileged':info['HostConfig']['Privileged'],'original_container_package_probe':original,'original_worktree_status':status,'experiment_only_install':True,'note':'Original worktree was already dirty and preserved. GPU/CPU are shared hardware, not resource-isolated machines. No existing processes were stopped.'}
(base/'output/isolation_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
