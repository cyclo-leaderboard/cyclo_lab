"""Render the actual registered Task901 plate-holder variant without robot commands."""
from pathlib import Path
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True,enable_cameras=True,kit_args='--enable omni.usd.schema.omni_lens_distortion').app
failed=False;env=None
try:
 import sys,json,os,cv2,numpy as np,torch,gymnasium as gym
 import isaaclab.sim as sim_utils
 from isaaclab.sensors import Camera,CameraCfg
 from isaaclab_tasks.utils import parse_env_cfg
 from pxr import UsdGeom,Gf,Usd
 p=Path('/tmp/task901_calibration');out=p/'renders';out.mkdir(parents=True,exist_ok=True)
 root=Path('/workspace/cyclo_lab/scripts/sim2real/imitation_learning');sys.path.insert(0,str(root))
 from local_plugins import load_for_task
 name='Cyclo-Real-WhiteRoom-Task000901-PlateHolder-FFW-SG2-v0'
 bundle=load_for_task(name);bundle.register_environments()
 assert name in gym.registry
 cfg=parse_env_cfg(name,device='cuda:0',num_envs=1)
 print('CFG_REFERENCE_JOINTS',json.dumps({k:v for k,v in cfg.scene.robot.init_state.joint_pos.items() if k in ('lift_joint','arm_l_joint1','arm_r_joint1','head_joint1','head_joint2')}),flush=True)
 if os.getenv('TASK901_USE_FACTORY_CAMERA_OFFSETS'):
  cfg.scene.cam_head.offset=CameraCfg.OffsetCfg(pos=(0.0,0.03,0.0),rot=(0.5,0.5,-0.5,-0.5),convention='isaac')
  for side in ('left','right'):
   getattr(cfg.scene,f'cam_wrist_{side}').offset=CameraCfg.OffsetCfg(pos=(0.0,0.0,0.0),rot=(0.7071068,0.0,0.7071068,0.0),convention='ros')
 if os.getenv('TASK901_TEST_ROOT_IDENTITY'):
  cfg.scene.robot.init_state.rot=(1.0,0.0,0.0,0.0)
 flip=os.getenv('TASK901_TEST_WRIST_FLIP')
 if flip:
  q={'x':(0.0,0.7071068,0.0,-0.7071068),'y':(-0.7071068,0.0,0.7071068,0.0)}[flip]
  cfg.scene.cam_wrist_left.offset.rot=q
  cfg.scene.cam_wrist_right.offset.rot=q
 env=gym.make(name,cfg=cfg).unwrapped;env.reset()
 sim=env.sim;stage=sim.stage
 preview=Camera(CameraCfg(prim_path='/World/PlateHolderPreviewCamera',height=900,width=1400,data_types=['rgb'],spawn=sim_utils.PinholeCameraCfg(focal_length=32,horizontal_aperture=36,clipping_range=(.02,100))))
 fitted_wrist={
  'left':{'eye':(-.147,-.352,.961),'forward':(-.905,.017,-.426),'intrinsics':(392.2402,392.8599,242.0067,317.3694)},
  'right':{'eye':(-.15623,.12186,.93794),'forward':(-.90560,.04689,-.42153),'intrinsics':(391.9070,392.4101,240.0282,323.3480)},
 }
 fitted_cameras={}
 for side,pose in fitted_wrist.items():
  fitted_cameras[side]=Camera(CameraCfg(prim_path=f'/World/Task901FittedWrist{side.title()}',height=640,width=480,data_types=['rgb'],spawn=sim_utils.PinholeCameraCfg(focal_length=17.12,horizontal_aperture=20.955,vertical_aperture=27.94,clipping_range=(.01,8))))
 probe=None
 if os.getenv('TASK901_TEST_DIRECT_CHILD'):
  probe=Camera(CameraCfg(prim_path='/World/envs/env_0/Robot/ffw_sg2_follower/arm_l_link7/Task901CameraProbe',height=120,width=160,data_types=['rgb'],spawn=sim_utils.PinholeCameraCfg(focal_length=17.12,horizontal_aperture=20.955,clipping_range=(.01,8)),offset=CameraCfg.OffsetCfg(pos=(0.,0.,0.),rot=(1.,0.,0.,0.),convention='opengl')))
 sim.reset()
 env.reset()
 robot=env.scene['robot']
 if os.getenv('TASK901_FORCE_ZERO_JOINTS'):
  target=torch.zeros_like(robot.data.joint_pos)
  robot.write_joint_state_to_sim(target,torch.zeros_like(target))
  robot.set_joint_position_target(target)
 elif os.getenv('TASK901_APPLY_REFERENCE_JOINTS'):
  target=robot.data.joint_pos.clone()
  for joint_name,value in cfg.scene.robot.init_state.joint_pos.items():
   if joint_name in robot.joint_names:target[:,robot.joint_names.index(joint_name)]=float(value)
  robot.write_joint_state_to_sim(target,torch.zeros_like(target))
  robot.set_joint_position_target(target)
 else:
  robot.set_joint_position_target(robot.data.joint_pos.clone())
 env.scene.write_data_to_sim()
 sim.step(render=True)
 env.scene.update(sim.get_physics_dt())
 joints=robot.data.joint_pos.clone()
 joint_state={name:float(joints[0,i]) for i,name in enumerate(robot.joint_names)}
 if not os.getenv('TASK901_FORCE_ZERO_JOINTS'):
  for name in ('lift_joint','arm_l_joint1','arm_l_joint4','arm_r_joint1','arm_r_joint4','head_joint1'):
   assert abs(joint_state[name]-cfg.scene.robot.init_state.joint_pos[name])<1e-4,(name,joint_state[name])
 print('SIM_JOINT_POSITIONS',json.dumps({k:v for k,v in joint_state.items() if k in ('lift_joint','arm_l_joint1','arm_r_joint1','head_joint1','head_joint2')}),flush=True)
 body_names=('arm_base_link','arm_l_link7','arm_r_link7','head_link2','camera_l_link','camera_r_link','gripper_l_rh_p12_rn_l2','gripper_l_rh_p12_rn_r2','gripper_r_rh_p12_rn_l2','gripper_r_rh_p12_rn_r2')
 body_report={name:{'pos_w':robot.data.body_pos_w[0,robot.body_names.index(name)].tolist(),'quat_w':robot.data.body_quat_w[0,robot.body_names.index(name)].tolist()} for name in body_names if name in robot.body_names}
 print('SIM_BODY_POSES',json.dumps(body_report),flush=True)
 head=env.scene['cam_head'];head_prim=stage.GetPrimAtPath(head.cfg.prim_path.replace('env_.*','env_0'))
 assert head_prim.IsValid()
 head_prim.ApplyAPI('OmniLensDistortionOpenCvPinholeAPI');head_prim.GetAttribute('omni:lensdistortion:model').Set('opencvPinhole')
 head_focal_px=head.cfg.spawn.focal_length*head.cfg.width/head.cfg.spawn.horizontal_aperture
 for key,val in {'fx':head_focal_px,'fy':head_focal_px,'cx':626.33935546875,'cy':360.0394592285156,'imageSize':Gf.Vec2i(1280,720)}.items():head_prim.GetAttribute('omni:lensdistortion:opencvPinhole:'+key).Set(val)
 for key in ['k1','k2','k3','k4','k5','k6','p1','p2','s1','s2','s3','s4']:
  a=head_prim.GetAttribute('omni:lensdistortion:opencvPinhole:'+key)
  if a:a.Set(0.)
 for side,pose in fitted_wrist.items():
  fx,fy,cx,cy=pose['intrinsics']
  wrist_path=env.scene[f'cam_wrist_{side}'].cfg.prim_path.replace('env_.*','env_0')
  for path in (fitted_cameras[side].cfg.prim_path,wrist_path):
   p=stage.GetPrimAtPath(path)
   assert p.IsValid(),path
   p.ApplyAPI('OmniLensDistortionOpenCvPinholeAPI');p.GetAttribute('omni:lensdistortion:model').Set('opencvPinhole')
   for key,val in {'fx':fx,'fy':fy,'cx':cx,'cy':cy,'imageSize':Gf.Vec2i(480,640)}.items():p.GetAttribute('omni:lensdistortion:opencvPinhole:'+key).Set(val)
   for key in ['k1','k2','k3','k4','k5','k6','p1','p2','s1','s2','s3','s4']:
    a=p.GetAttribute('omni:lensdistortion:opencvPinhole:'+key)
    if a:a.Set(0.)
 bbox=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render'])
 table=bbox.ComputeWorldBound(stage.GetPrimAtPath('/World/envs/env_0/WhiteRoom/WhiteTable/Top')).ComputeAlignedBox()
 tmin=np.array(table.GetMin());tmax=np.array(table.GetMax());assert abs(tmax[2]-.70)<1e-5
 report={'env_id':'Cyclo-Real-WhiteRoom-Task000901-PlateHolder-FFW-SG2-v0','table_top_z_m':float(tmax[2]),'objects':{},'no_hardware_commands':True,'sim_body_poses':body_report,'head_focal_px':head_focal_px,'head_offset_pos':list(head.cfg.offset.pos),'head_offset_rot':list(head.cfg.offset.rot)}
 report['reference_joints_verified']={key:joint_state[key] for key in ('lift_joint','arm_l_joint1','arm_l_joint4','arm_r_joint1','arm_r_joint4','head_joint1')}
 panel=stage.GetPrimAtPath('/World/envs/env_0/WhiteRoom/WhiteRoomPartitionWall/Panel_04')
 assert panel.IsValid()
 panel_box=bbox.ComputeWorldBound(panel).ComputeAlignedBox()
 panel_width_m=float(panel_box.GetMax()[1]-panel_box.GetMin()[1])
 assert abs(panel_width_m-0.5)<1e-5,panel_width_m
 report['partition_panel_width_m']=panel_width_m
 for obj in ['PlatePink','PlateBlue','PlateYellow','PlateHolder']:
  prim=stage.GetPrimAtPath('/World/envs/env_0/'+obj);assert prim.IsValid(),obj
  b=bbox.ComputeWorldBound(prim).ComputeAlignedBox();lo=np.array(b.GetMin());hi=np.array(b.GetMax())
  assert lo[2]>=tmax[2]-.001 and np.all(lo[:2]>tmin[:2]) and np.all(hi[:2]<tmax[:2]),(obj,lo,hi)
  report['objects'][obj]={'min_m':lo.tolist(),'max_m':hi.tolist(),'size_m':(hi-lo).tolist()}
 preview.set_world_poses_from_view(torch.tensor([[-.07,-.25,1.28]],device=sim.device),torch.tensor([[-.55862,-.10,.73]],device=sim.device))
 for side,pose in fitted_wrist.items():
  eye=np.array(pose['eye']);forward=np.array(pose['forward']);forward=forward/np.linalg.norm(forward)
  fitted_cameras[side].set_world_poses_from_view(torch.tensor([eye],device=sim.device,dtype=torch.float32),torch.tensor([eye+forward],device=sim.device,dtype=torch.float32))
 for _ in range(55):sim.render()
 for camera_name in ('cam_head','cam_wrist_left','cam_wrist_right'):
  sensor=env.scene[camera_name]
  p,q=sensor._view.get_world_poses()
  print('CAMERA_VIEW_POSE',camera_name,p[0].tolist(),q[0].tolist(),flush=True)
 if probe is not None:
  p,q=probe._view.get_world_poses()
  print('DIRECT_CHILD_PROBE_POSE',p[0].tolist(),q[0].tolist(),flush=True)
 preview.update(1/60,force_recompute=True)
 hero=preview.data.output['rgb'][0,:,:,:3].cpu().numpy();assert hero.shape==(900,1400,3) and hero.std()>5
 cv2.imwrite(str(out/'sim_hero.png'),cv2.cvtColor(hero,cv2.COLOR_RGB2BGR))
 head.update(1/60,force_recompute=True)
 rgb=head.data.output['rgb'][0,:,:,:3].cpu().numpy();assert rgb.shape==(720,1280,3) and rgb.std()>5
 cv2.imwrite(str(out/'sim_head.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR));report['sim_head_rgb_std']=float(rgb.std());report['sim_hero_rgb_std']=float(hero.std())
 print('HEAD_RGB_STATS',float(rgb.std()),rgb.reshape(-1,3).mean(axis=0).tolist(),flush=True)
 head_matrix=UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(head_prim)
 head_parent_matrix=UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(head_prim.GetParent())
 report['sim_head_usd_matrix']=[[float(head_matrix[i][j]) for j in range(4)] for i in range(4)]
 report['sim_head_parent_usd_matrix']=[[float(head_parent_matrix[i][j]) for j in range(4)] for i in range(4)]
 for side in ('left','right'):
  wrist=env.scene[f'cam_wrist_{side}']
  wrist.update(1/60,force_recompute=True)
  raw=wrist.data.output['rgb'][0,:,:,:3].cpu().numpy()
  assert raw.shape==(640,480,3),(side,raw.shape)
  native=cv2.rotate(raw,cv2.ROTATE_90_CLOCKWISE)
  assert native.shape==(480,640,3)
  cv2.imwrite(str(out/f'sim_wrist_{side}_raw.png'),cv2.cvtColor(raw,cv2.COLOR_RGB2BGR))
  cv2.imwrite(str(out/f'sim_wrist_{side}_native.png'),cv2.cvtColor(native,cv2.COLOR_RGB2BGR))
  report[f'sim_wrist_{side}_rgb_std']=float(native.std())
  wrist_path=wrist.cfg.prim_path.replace('env_.*','env_0')
  wrist_prim=stage.GetPrimAtPath(wrist_path)
  assert wrist_prim.IsValid(),wrist_path
  matrix=UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(wrist_prim)
  report[f'sim_wrist_{side}_usd_matrix']=[[float(matrix[i][j]) for j in range(4)] for i in range(4)]
  parent_matrix=UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(wrist_prim.GetParent())
  report[f'sim_wrist_{side}_parent_usd_matrix']=[[float(parent_matrix[i][j]) for j in range(4)] for i in range(4)]
  fitted=fitted_cameras[side];fitted.update(1/60,force_recompute=True)
  aligned=fitted.data.output['rgb'][0,:,:,:3].cpu().numpy()
  assert aligned.shape==(640,480,3) and aligned.std()>5
  cv2.imwrite(str(out/f'sim_wrist_{side}_fitted.png'),cv2.cvtColor(aligned,cv2.COLOR_RGB2BGR))
  report[f'sim_wrist_{side}_fitted_pose']={'eye_m':pose['eye'],'forward':pose['forward']}
  report[f'sim_wrist_{side}_mounted_vs_fitted_mean_abs_rgb']=float(np.abs(raw.astype(np.float32)-aligned.astype(np.float32)).mean())
 # Visual visibility is checked on the rendered images. USD rest transforms
 # do not describe the articulated PhysX pose used by RTX in these frames.
 report['robot_link_positions_m']={}
 report['sim_body_usd_matrices']={}
 names={'arm_base_link','arm_l_link7','arm_r_link7','head_link1','head_link2','zed','camera_l_link','camera_r_link',
        'gripper_l_rh_p12_rn_l2','gripper_l_rh_p12_rn_r2',
        'gripper_r_rh_p12_rn_l2','gripper_r_rh_p12_rn_r2'}
 xforms=UsdGeom.XformCache(Usd.TimeCode.Default())
 for prim in stage.Traverse():
  if prim.GetName() in names and '/Robot/' in str(prim.GetPath()):
   matrix=xforms.GetLocalToWorldTransform(prim)
   report['robot_link_positions_m'][prim.GetName()]=list(map(float,matrix.ExtractTranslation()))
   if prim.GetName() in ('head_link2','arm_l_link7','arm_r_link7'):
    report['sim_body_usd_matrices'][prim.GetName()]=[[float(matrix[i][j]) for j in range(4)] for i in range(4)]
 print('ROBOT_LINK_POSITIONS',json.dumps(report['robot_link_positions_m']),flush=True)
 report['robot_joints_unchanged']=bool(torch.equal(joints,robot.data.joint_pos));assert report['robot_joints_unchanged']
 (out/'validation.json').write_text(json.dumps(report,indent=2));print('RENDER_VALIDATION',json.dumps(report),flush=True)
except BaseException:
 import traceback;failed=True;traceback.print_exc()
finally:
 if env is not None:env.close()
 app.close(wait_for_replicator=False,skip_cleanup=True)
if failed:raise SystemExit(1)
