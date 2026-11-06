"""Run the installed Isaac Lab SkillGen segmentation and object-frame adaptation core."""
import argparse
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--output',default='/experiment/output/skillgen_poses');AppLauncher.add_app_launcher_args(p);args=p.parse_args()
app=AppLauncher(args).app
try:
    import json
    from pathlib import Path
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    from isaaclab.envs.mimic_env_cfg import MimicEnvCfg,SubTaskConfig
    from isaaclab_mimic.datagen.data_generator import DataGenerator
    base=Path('/experiment');out=Path(args.output);out.mkdir(parents=True,exist_ok=False);scene=json.loads((base/'task_000759/scene.json').read_text());fk=np.load(base/'input/source_fk.npz')
    class Adapter:
        device='cuda:0'
        cfg=MimicEnvCfg()
        cfg.datagen_config.use_skillgen=True
        cfg.subtask_configs={'right_arm':[SubTaskConfig(object_ref='cucumber',subtask_term_signal='grasp',selection_strategy='random',action_noise=0.,num_interpolation_steps=0)]}
        object_pose=torch.eye(4,device=device)
        def get_object_poses(self,env_ids=None):return {'cucumber':self.object_pose[None]}
        def get_robot_eef_pose(self,env_ids=None,eef_name=None):return torch.tensor(fk['eef'][0:1],device=self.device,dtype=torch.float32)
        def actions_to_gripper_actions(self,actions):return {'right_arm':actions[:,-1:]}
    adapter=Adapter();generator=DataGenerator(adapter,dataset_path='/experiment/input/seed.hdf5')
    boundaries=generator.randomize_subtask_boundaries()
    for yaw in scene['yaw_deltas_deg']:
        adapter.object_pose[:3,:3]=torch.tensor(Rotation.from_euler('z',scene['cucumber_yaw_rad']+np.deg2rad(yaw)).as_matrix(),device=adapter.device,dtype=torch.float32)
        adapter.object_pose[:3,3]=torch.tensor(scene['cucumber_center'],device=adapter.device)
        traj=generator.generate_eef_subtask_trajectory(0,'right_arm',0,boundaries,{}, {'right_arm':None})
        seq=traj.get_full_sequence().sequence
        np.savez(out/f'skill_yaw_{yaw:+03d}.npz',poses=np.array([w.pose.cpu().numpy() for w in seq]),gripper=np.array([w.gripper_action.cpu().numpy() for w in seq]),object_pose=adapter.object_pose.cpu().numpy())
    result={'implementation':'installed isaaclab_mimic.datagen.DataGenerator.generate_eef_subtask_trajectory','use_skillgen':True,'source_demo_count':len(generator.src_demo_datagen_info_pool.datagen_infos),'boundaries':{k:np.asarray(v).tolist() for k,v in boundaries.items()},'yaw_deltas_deg':scene['yaw_deltas_deg'],'waypoints_per_skill':len(seq),'note':'Custom SG2 joint-space execution adapter replaces stock EEF-only execution; no physical-success claim from pose generation.'}
    (out/'skillgen_pose_report.json').write_text(json.dumps(result,indent=2));print('SKILLGEN_POSES',result,flush=True)
finally:app.close(wait_for_replicator=False,skip_cleanup=True)
