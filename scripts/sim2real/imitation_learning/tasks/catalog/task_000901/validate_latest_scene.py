"""Finite headless profile smoke test: no ROS, hardware or replay commands."""
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)

from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
env = None
try:
    import importlib.util
    import json
    import math
    import sys
    import unittest
    import numpy as np
    import torch
    import gymnasium as gym
    from pxr import Usd, UsdGeom, UsdPhysics
    from isaaclab_tasks.utils import parse_env_cfg

    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root.parents[2]))
    from local_plugins import load_for_task
    name = 'Cyclo-Real-WhiteRoom-Task000901-PlateHolder-FFW-SG2-v0'
    load_for_task(name).register_environments()
    from _cyclo_task000901_scene import (
        Task000901SceneCfg, PROFILE, TABLE_TOP_Z_M, make_task000901_sg2_cfg,
    )
    spec = importlib.util.spec_from_file_location('test_latest_profile', root/'test_latest_profile.py')
    tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tests)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
    if not result.wasSuccessful():
        raise AssertionError('CPU USD profile tests failed')
    visual = Task000901SceneCfg()
    for prop in ('cucumber', 'onion', 'pot', 'lid'):
        assert abs(getattr(visual, prop).init_state.pos[2]-(TABLE_TOP_Z_M+.0002)) < 1e-8
    legacy = make_task000901_sg2_cfg(latest=False)
    assert legacy.spawn.func.__name__ == 'spawn_sg2_with_softgripper_physics'
    cfg = parse_env_cfg(name, device='cuda:0', num_envs=1)
    cfg.seed = 901
    for sensor in ('cam_head', 'cam_wrist_left', 'cam_wrist_right'):
        setattr(cfg.scene, sensor, None)
    assert cfg.scene.robot.spawn.articulation_props.enabled_self_collisions
    assert cfg.scene.robot.spawn.func.__name__ == 'spawn_latest_robot'
    env = gym.make(name, cfg=cfg).unwrapped
    env.reset()
    stage = env.sim.stage
    robot = env.scene['robot']
    rp = stage.GetPrimAtPath('/World/envs/env_0/Robot')
    assert rp.GetCustomDataByKey('task901:physicsProfile') == PROFILE['id']
    articulation = stage.GetPrimAtPath('/World/envs/env_0/Robot/ffw_sg2_follower')
    assert articulation.GetAttribute('physxArticulation:enabledSelfCollisions').Get() is True
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render']).ComputeWorldBound(
        stage.GetPrimAtPath('/World/envs/env_0/WhiteRoom/WhiteTable/Top')).ComputeAlignedBox()
    assert abs(bounds.GetMax()[2]-TABLE_TOP_Z_M) < 1e-6
    pieces = [p for p in Usd.PrimRange(rp) if '_cyclo_softgripper_proxy' in str(p.GetPath())
              and p.HasAPI(UsdPhysics.CollisionAPI)
              and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False]
    filters = sum(len(UsdPhysics.FilteredPairsAPI(p).GetFilteredPairsRel().GetTargets())
                  for p in pieces if p.HasAPI(UsdPhysics.FilteredPairsAPI))
    assert (len(pieces), filters) == (20, 50)
    materials = robot.root_physx_view.get_material_properties().cpu().numpy().reshape(-1, 3)
    assert np.any(np.all(np.isclose(materials[:, :2], [4., 3.5]), axis=1))
    joints = {}
    for p in Usd.PrimRange(rp):
        if p.GetName() in {f'gripper_{s}_joint{j}' for s in 'lr' for j in (2, 4)}:
            assert abs(p.GetAttribute('physics:upperLimit').Get()-math.degrees(1.1)) < 1e-4
            joints[p.GetName()] = {
                'frequency': p.GetAttribute('physxMimicJoint:rotX:naturalFrequency').Get(),
                'damping': p.GetAttribute('physxMimicJoint:rotX:dampingRatio').Get(),
            }
    assert len(joints) == 4 and all(x == {'frequency': 1500., 'damping': 5.} for x in joints.values())
    plates = {}
    for n in ('plate_blue', 'plate_pink', 'plate_yellow'):
        obj = env.scene[n]
        assert abs(obj.cfg.init_state.pos[2]-(TABLE_TOP_Z_M+.0005)) < 1e-8
        assert abs(float(obj.root_physx_view.get_masses()[0, 0])-.015) < 1e-6
        prim = stage.GetPrimAtPath(obj.cfg.prim_path.replace('env_.*', 'env_0'))
        colliders = [p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.CollisionAPI)
                     and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False]
        assert len(colliders) == 384
        plates[n] = {'initial_z_m': obj.cfg.init_state.pos[2], 'colliders': len(colliders)}
    target = robot.data.joint_pos.clone()
    for _ in range(24):
        robot.set_joint_position_target(target)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(cfg.sim.dt)
    assert bool(torch.isfinite(robot.data.joint_pos).all())
    assert all(bool(torch.isfinite(env.scene[n].data.root_state_w).all()) for n in plates)
    report = {'profile': PROFILE['id'], 'table_top_z_m': float(bounds.GetMax()[2]),
              'active_tip_pieces': len(pieces), 'opposing_tip_filters': filters,
              'joint24': joints, 'plates': plates, 'physics_steps': 24,
              'cpu_usd_tests': result.testsRun, 'rendered_video': False,
              'grasp_or_insertion_success_tested': False, 'no_real_robot_commands': True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(report, f, indent=2)
    print('TASK901_LATEST_SMOKE', json.dumps(report), flush=True)
finally:
    if env is not None:
        env.close()
    app.close(wait_for_replicator=False, skip_cleanup=True)
