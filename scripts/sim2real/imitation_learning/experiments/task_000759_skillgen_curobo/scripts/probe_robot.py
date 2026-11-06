"""Read measured source joints into isolated Isaac to derive robot FK (no success claim)."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();AppLauncher.add_app_launcher_args(p);a=p.parse_args()
app=AppLauncher(a).app
import json
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from cyclo_lab.assets.robots.FFW_SG2 import FFW_SG2_CFG
from pxr import Usd,UsdPhysics,UsdGeom

src=np.load('/experiment/input/source_episode.npz')
cfg=FFW_SG2_CFG.copy();cfg.prim_path='/World/Robot'
cfg.spawn.articulation_props.fix_root_link=True
cfg.init_state.pos=(0.,0.,0.)
cfg.init_state.rot=(1.,0.,0.,0.)
cfg.init_state.joint_pos.update({str(n):float(q) for n,q in zip(src['joint_names'][:19],src['state'][0,:19])})
sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/240,device=a.device))
robot=Articulation(cfg)
sim.reset()
poses=[];body_poses=[]
ids=[robot.joint_names.index(str(n)) for n in src['joint_names'][:19]]
try:
    for frame in range(len(src['state'])):
        q=robot.data.default_joint_pos.clone()
        q[0,ids]=torch.tensor(src['state'][frame,:19],device=sim.device)
        for s,g in [('l',7),('r',15)]:
            for i in range(2,5):
                name=f'gripper_{s}_joint{i}'
                if name in robot.joint_names:q[0,robot.joint_names.index(name)]=float(src['state'][frame,g])
        robot.write_joint_state_to_sim(q,torch.zeros_like(q));sim.forward();robot.update(1/15)
        body=torch.cat([robot.data.body_pos_w[0],robot.data.body_quat_w[0]],dim=-1).cpu().numpy()
        body_poses.append(body.copy())
        b=body[robot.body_names.index('arm_r_link7')]
        T=np.eye(4);T[:3,:3]=Rotation.from_quat(b[[4,5,6,3]]).as_matrix();T[:3,3]=b[:3]
        T[:3,3]+=T[:3,:3]@np.array([0.,0.,-.2]);poses.append(T)
    np.savez('/experiment/input/source_fk.npz',eef=np.asarray(poses),body_pose=np.asarray(body_poses),body_names=np.array(robot.body_names),joint_names=np.array(robot.joint_names))
    joints=[]
    for prim in Usd.PrimRange(sim.stage.GetPrimAtPath('/World/Robot')):
        if prim.IsA(UsdPhysics.Joint):
            joint=UsdPhysics.Joint(prim)
            def v(attr):
                if not attr:return None
                x=attr.Get()
                if hasattr(x,'GetReal'):return [float(x.GetReal()),*map(float,x.GetImaginary())]
                if isinstance(x,(str,int,float,bool)) or x is None:return x
                try:return list(x)
                except TypeError:pass
                return x
            joints.append({'name':prim.GetName(),'type':prim.GetTypeName(),'parent':[str(x) for x in joint.GetBody0Rel().GetTargets()],'child':[str(x) for x in joint.GetBody1Rel().GetTargets()], 'localPos0':v(joint.GetLocalPos0Attr()),'localPos1':v(joint.GetLocalPos1Attr()),'localRot0':v(joint.GetLocalRot0Attr()),'localRot1':v(joint.GetLocalRot1Attr()),'axis':v(prim.GetAttribute('physics:axis')),'lower':v(prim.GetAttribute('physics:lowerLimit')),'upper':v(prim.GetAttribute('physics:upperLimit'))})
    report={'source_frames':len(poses),'body_names':robot.body_names,'joint_names':robot.joint_names,'joints':joints,'sample_tcp':{str(i):poses[i].tolist() for i in [0,30,35,40,45,55,65,80]}}
    Path('/experiment/input/robot_probe.json').write_text(json.dumps(report,indent=2))
    print('PROBE_TCP', {i:np.round(poses[i][:3,3],4).tolist() for i in [0,30,35,40,45,55,65,80]},flush=True)
except Exception:
    import traceback
    traceback.print_exc()
    raise
finally:app.close(wait_for_replicator=False, skip_cleanup=True)
