"""Build isolated scene settings, one annotated seed, and exact USD-derived arm kinematics."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import yaml
import h5py

base=Path('/experiment'); task=base/'task_000759';src=np.load(base/'input/source_episode.npz'); fk=np.load(base/'input/source_fk.npz')
probe=json.loads((base/'input/robot_probe.json').read_text())
eef=fk['eef'];frame=40
axis=eef[frame,:2,0];axis=axis/np.linalg.norm(axis)
yaw=float(np.arctan2(axis[1],axis[0]));center=eef[frame,:3,3].copy();center[:2]+=.035*axis
radius=.018;center[2]=.722
scene={'task_id':'Task000759-Cucumber-SkillGen-SG2-v0','source_episode':93,'source_frames':115,'control_hz':15,'physics_hz':240,'table_top_m':.704,'table_center_xy':[.78,0.],'table_size_xy':[1.,1.2],'cucumber_center':center.tolist(),'cucumber_yaw_rad':yaw,'cucumber_radius_m':radius,'cucumber_cylinder_length_m':.20,'cucumber_mass_kg':.12,'cucumber_static_friction':1.2,'cucumber_dynamic_friction':1.,'tcp_offset':[0,0,-.2],'contact_start_frame':30,'grasp_complete_frame':52,'skill_end_exclusive':66,'yaw_deltas_deg':[-60,-30,0,30,60],'geometry_note':'Approximate rigid capsule and table for rotation feasibility, not measured geometry or friction.'}
scene['geometry_note']='Approximate flattened convex cucumber mesh (236 x 42 x 39 mm) and table; not measured geometry or friction.'
scene['cucumber_asset']='/experiment/task_000759/cucumber.usda'
scene['cucumber_linear_damping']=.1
scene['cucumber_angular_damping']=.1
(task/'scene.json').write_text(json.dumps(scene,indent=2))
# Derive a minimal exact kinematic chain from the SAME simulated USD joint frames.
robot=ET.Element('robot',{'name':'sg2_right_skillgen'})
ET.SubElement(robot,'link',{'name':'world'})
chain=['lift_joint']+[f'arm_r_joint{i}' for i in range(1,8)]
for name in chain:
    j=next(v for v in probe['joints'] if v['name']==name)
    parent=j['parent'][0].split('/')[-1];child=j['child'][0].split('/')[-1]
    assert np.allclose(j['localPos1'],0) and np.allclose(j['localRot1'],[1,0,0,0]),j
    ET.SubElement(robot,'link',{'name':child})
    typ='fixed' if name=='lift_joint' else 'revolute'
    joint=ET.SubElement(robot,'joint',{'name':name,'type':typ})
    ET.SubElement(joint,'parent',{'link':parent});ET.SubElement(joint,'child',{'link':child})
    xyz=np.array(j['localPos0'],float)
    if name=='lift_joint':xyz[2]+=float(src['state'][0,18])
    quat=np.array(j['localRot0'])
    rpy=Rotation.from_quat(quat[[1,2,3,0]]).as_euler('xyz')
    ET.SubElement(joint,'origin',{'xyz':' '.join(map(str,xyz)),'rpy':' '.join(map(str,rpy))})
    if typ=='revolute':
        vector=np.eye(3)['XYZ'.index(j['axis'])]
        ET.SubElement(joint,'axis',{'xyz':' '.join(map(str,vector))})
        ET.SubElement(joint,'limit',{'lower':str(np.deg2rad(j['lower'])),'upper':str(np.deg2rad(j['upper'])),'effort':'60','velocity':'1.5'})
ET.SubElement(robot,'link',{'name':'tcp'})
j=ET.SubElement(robot,'joint',{'name':'tcp_fixed','type':'fixed'})
ET.SubElement(j,'parent',{'link':'arm_r_link7'});ET.SubElement(j,'child',{'link':'tcp'})
ET.SubElement(j,'origin',{'xyz':'0 0 -0.2','rpy':'0 0 0'})
ET.indent(robot);ET.ElementTree(robot).write(task/'sg2_right.urdf',encoding='utf-8',xml_declaration=True)
spheres={}
for i in range(1,8):
    name=f'arm_r_link{i}'
    if i<7:
        nj=next(v for v in probe['joints'] if v['name']==f'arm_r_joint{i+1}')
        endpoint=np.array(nj['localPos0'])
    else:endpoint=np.array([0,0,-.12])
    radius_i=.045 if i<5 else .03
    spheres[name]=[{'center':(endpoint*t).tolist(),'radius':radius_i} for t in np.linspace(0,1,max(2,int(np.linalg.norm(endpoint)/.04)+1))]
spheres['tcp']=[{'center':[0.,s,.055],'radius':.014} for s in [-.065,.065]]
links=list(spheres)
ignore={name:links[i+1:i+3] for i,name in enumerate(links)}
cfg={'robot_cfg':{'kinematics':{'use_usd_kinematics':False,'urdf_path':str(task/'sg2_right.urdf'),'asset_root_path':str(task),'base_link':'world','ee_link':'tcp','link_names':links,'collision_link_names':links,'collision_spheres':spheres,'collision_sphere_buffer':.003,'self_collision_ignore':ignore,'self_collision_buffer':{n:0. for n in links},'use_global_cumul':True,'mesh_link_names':[],'cspace':{'joint_names':chain[1:],'retract_config':src['state'][0,8:15].tolist(),'null_space_weight':[1.]*7,'cspace_distance_weight':[1.,1.,1.,1.,1.,1.5,2.],'max_acceleration':4.,'max_jerk':40.}}}}
(task/'sg2_right.yml').write_text(yaml.safe_dump(cfg,sort_keys=False))
# Input is a measured-joint-derived skill, not claimed as physics-validated yet.
object_pose=np.eye(4);object_pose[:3,:3]=Rotation.from_euler('z',yaw).as_matrix();object_pose[:3,3]=center
count=scene['skill_end_exclusive']
with h5py.File(base/'input/seed.hdf5','w') as h:
    data=h.create_group('data');data.attrs['env_args']=json.dumps({'env_name':scene['task_id'],'type':2,'env_kwargs':{}});data.attrs['total']=count
    demo=data.create_group('demo_0');demo.attrs['num_samples']=count
    demo.attrs['source']='episode 93, measured joint FK; real grasp visually verified; simulated source validation separate'
    demo.create_dataset('source_joint_states',data=src['state'][:count,:19]);demo.create_dataset('source_joint_actions',data=src['action'][:count,:19])
    poses=eef[:count].astype(np.float32);target=np.concatenate([poses[1:],poses[-1:]])
    action=np.concatenate([target[:,:3,3],Rotation.from_matrix(target[:,:3,:3]).as_quat()[:,[3,0,1,2]],src['action'][:count,15:16]],axis=1).astype(np.float32)
    demo.create_dataset('actions',data=action)
    info=demo.create_group('obs/datagen_info')
    info.create_dataset('eef_pose/right_arm',data=poses);info.create_dataset('target_eef_pose/right_arm',data=target)
    info.create_dataset('object_pose/cucumber',data=np.repeat(object_pose[None],count,axis=0).astype(np.float32))
    info.create_dataset('subtask_start_signals/grasp',data=(np.arange(count)>=scene['contact_start_frame']).astype(np.int32))
    info.create_dataset('subtask_term_signals/grasp',data=(np.arange(count)>=count-1).astype(np.int32))
print(json.dumps(scene,indent=2))
