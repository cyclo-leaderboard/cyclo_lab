"""Read-only, finite-sample IK diagnosis of the old grasp-candidate restrictions."""
import json
import numpy as np
from scipy.spatial.transform import Rotation
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.util_file import load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver,IKSolverConfig
from plan_goalset import BASE,candidates
ta=TensorDeviceType();scene=json.loads((BASE/'task_000759/scene.json').read_text());fk=np.load(BASE/'input/source_fk.npz');src=np.load(BASE/'input/source_episode.npz')
robot=load_yaml(str(BASE/'task_000759/sg2_right.yml'))['robot_cfg']
solver=IKSolver(IKSolverConfig.load_from_robot_config(robot,None,tensor_args=ta,num_seeds=128,regularization=True,self_collision_check=False,self_collision_opt=False,use_cuda_graph=False,position_threshold=.002,rotation_threshold=.025))
reports=[]
for yaw in [0,30,60,90]:
    values,meta=candidates(scene,fk,yaw);quat=Rotation.from_matrix(values[:,:3,:3]).as_quat()[:,[3,0,1,2]]
    pose=Pose(position=ta.to_device(np.ascontiguousarray(values[:,:3,3])).contiguous(),quaternion=ta.to_device(np.ascontiguousarray(quat)).contiguous())
    ref=ta.to_device(np.repeat(src['state'][40:41,8:15],len(values),axis=0).copy())
    r=solver.solve_batch(pose,retract_config=ref,seed_config=ref.unsqueeze(1),return_seeds=16)
    success=r.success.cpu().numpy();valid=np.where(success.any(axis=1))[0]
    report=dict(yaw_delta_deg=yaw,num_candidates=len(values),num_seeds_per_candidate=128,world_and_self_collision_disabled=True,reachable_candidate_count=len(valid),reachable_candidate_metadata=[meta[i] for i in valid],interpretation='Finite randomized IK search only; no mathematical infeasibility proof.')
    reports.append(report);print(json.dumps(report),flush=True)
(BASE/'output/pickplace_old_candidate_ik_diagnosis.json').write_text(json.dumps(reports,indent=2))
