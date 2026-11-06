"""Finite Task901-local diagnostic runner using Isaac Lab's existing Mimic core.

One environment only; never connects to ROS/Zenoh or accesses a real robot.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--mode', choices=('joint-reference', 'mimic'), default='mimic')
parser.add_argument('--trials', type=int, default=1)
parser.add_argument('--xy-range', type=float, default=0.)
parser.add_argument('--seed', type=int, default=901)
parser.add_argument('--output', type=Path, required=True)
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.trials < 1 or not 0 <= args.xy_range <= .03:
    parser.error('Require trials>=1 and 0<=xy-range<=.03 m for the first pilot')
if args.output.exists():
    parser.error(f'Refusing to overwrite {args.output}')
args.output.mkdir(parents=True)
app = AppLauncher(args).app
env = None
failed = False
try:
    import gymnasium as gym
    import h5py
    import numpy as np
    import torch
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
    from isaaclab.managers import DatasetExportMode
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from local_plugins import load_for_task
    task = 'Cyclo-Real-WhiteRoom-Task000901-Bimanual-Mimic-Seed-FFW-SG2-v0'
    load_for_task(task).register_environments()
    from _cyclo_task000901_mimic.env_cfg import REFERENCE, rack_success
    cfg = parse_env_cfg(task, device=args.device, num_envs=1)
    cfg.seed = args.seed
    cfg.plate_xy_half_range_m = args.xy_range
    cfg.datagen_config.seed = args.seed
    success_term = cfg.terminations.success
    cfg.terminations = None
    cfg.recorders = ActionStateRecorderManagerCfg()
    cfg.recorders.dataset_export_dir_path = str(args.output)
    cfg.recorders.dataset_filename = 'episodes'
    cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES
    if args.mode == 'mimic':
        cfg.init_action_cfg('mimic_ik')
    expected = json.loads((REFERENCE/'provenance.json').read_text())
    robot_sha = hashlib.sha256(Path(cfg.scene.robot.spawn.usd_path).read_bytes()).hexdigest()
    if robot_sha != expected['source_robot_sha256']:
        raise ValueError('SG2 USD changed since reference; refusing silent robot substitution')
    env = gym.make(task, cfg=cfg).unwrapped
    env.reset()
    print('PILOT_ENV_READY', flush=True)
    robot = env.scene['robot']
    # Exercise the actual two-hand API, including both grippers and pose inverse.
    target_poses = {name: env.get_robot_eef_pose(name)[0] for name in ('left_arm','right_arm')}
    test_grips = {name: torch.tensor([v], device=env.device)
                  for name, v in [('left_arm', .23), ('right_arm', .71)]}
    test_action = env.target_eef_pose_to_action(target_poses, test_grips)
    restored = env.action_to_target_eef_pose(test_action)
    for name in target_poses:
        assert torch.allclose(restored[name][0], target_poses[name], atol=1e-5)
    assert abs(float(test_action[0,7])-.23)<1e-5
    assert abs(float(test_action[0,15])-.71)<1e-5
    print('BIMANUAL_API_ROUNDTRIP_OK', flush=True)
    rows = []
    if args.mode == 'joint-reference':
        # Precisely reproduce the old 240 Hz interpolated measured-position PD
        # targets. Do not confuse a held 15 Hz action with the original replay.
        data = np.load(REFERENCE/'input/aligned_replay.npz')
        lookup = {n: i for i, n in enumerate(robot.joint_names)}
        ids = [lookup[n] for n in data['joint_names'].tolist()]
        derived = json.loads((REFERENCE/'input/input_manifest.json').read_text())['mimic_from_recorded_urdf']
        t_end = float(data['time_sec'][-1])
        states = []
        for step in range(round((t_end+3)/cfg.sim.dt)+1):
            t = min(step*cfg.sim.dt, t_end)
            q = robot.data.default_joint_pos.clone()
            values = [np.interp(t, data['time_sec'], data['reference_position'][:, j]) for j in range(len(ids))]
            q[0, ids] = torch.tensor(values, device=env.device, dtype=q.dtype)
            for name, relation in derived.items():
                q[0, lookup[name]] = q[0, lookup[relation['joint']]]*float(relation.get('multiplier',1))
            robot.set_joint_position_target(q)
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(cfg.sim.dt)
            if step % cfg.decimation == 0:
                env.common_step_counter += 1
                env.observation_manager.compute()
                rack_success(env)
                states.append([env.scene[n].data.root_state_w[0].cpu().numpy() for n in expected['expected_slot_xyz']])
                if step % (cfg.decimation*75) == 0:
                    print('REFERENCE_SECONDS', round(step*cfg.sim.dt,2), flush=True)
        np.savez_compressed(args.output/'reference_states.npz', plate_names=list(expected['expected_slot_xyz']), object_states=states)
        rows.append(dict(trial=0, success=bool(rack_success(env)[0]),
                         final_xyz={n:env.scene[n].data.root_pos_w[0].cpu().tolist() for n in expected['expected_slot_xyz']}))
    else:
        from _cyclo_task000901_mimic.generator import Task901DataGenerator
        from isaaclab_mimic.datagen.datagen_info_pool import DataGenInfoPool
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        reset_queue, action_queue = asyncio.Queue(), asyncio.Queue()
        pool = DataGenInfoPool(env, cfg, env.device, asyncio_lock=asyncio.Lock())
        pool.load_from_dataset_file(str(REFERENCE/'seed.hdf5'))
        print('MIMIC_SOURCE_BOUNDARIES', pool.subtask_boundaries, flush=True)
        generator = Task901DataGenerator(env=env, src_demo_datagen_info_pool=pool)
        trace = []
        for trial in range(args.trials):
            task_future = loop.create_task(generator.generate(
                env_id=0, success_term=success_term,
                env_reset_queue=reset_queue, env_action_queue=action_queue))
            steps = 0
            while not task_future.done():
                loop.run_until_complete(asyncio.sleep(0))
                while not reset_queue.empty():
                    reset_queue.get_nowait()
                    env.reset(env_ids=torch.tensor([0], device=env.device))
                    reset_queue.task_done()
                if not action_queue.empty():
                    env_id, action = action_queue.get_nowait()
                    with torch.inference_mode():
                        # MultiWaypoint queues one unbatched action per env;
                        # this single-env runner must restore the batch axis.
                        env.step(action.unsqueeze(0))
                    action_queue.task_done()
                    trace.append([env.scene[n].data.root_state_w[0].cpu().numpy() for n in expected['expected_slot_xyz']])
                    steps += 1
                    if steps % 100 == 0:
                        print('MIMIC_PROGRESS', trial, steps, flush=True)
                    if steps > 2000:
                        raise RuntimeError('Mimic coordination exceeded bounded pilot step budget')
            result = task_future.result()
            row = dict(trial=trial, success=bool(result['success']), steps=steps,
                       offsets=env.last_plate_spawn_offsets,
                       final_xyz={n:env.scene[n].data.root_pos_w[0].cpu().tolist() for n in expected['expected_slot_xyz']})
            rows.append(row)
            (args.output/'trials.json').write_text(json.dumps(rows, indent=2))
            print('MIMIC_TRIAL_DONE', json.dumps(row), flush=True)
        np.savez_compressed(args.output/'pilot_object_states.npz', object_states=trace, plate_names=list(expected['expected_slot_xyz']))
        loop.close()
    summary = dict(mode=args.mode, trials=rows, success_count=sum(x['success'] for x in rows),
                   source='episode6 left/original_joint_limits', robot_usd_sha256=robot_sha,
                   xy_half_range_m=args.xy_range, seed=args.seed,
                   original_teleop_commands_available=False,
                   object_pose_writes_after_reset=0, real_robot_access=False)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    print('PILOT_DONE', json.dumps(summary), flush=True)
except BaseException:
    failed = True
    traceback.print_exc()
finally:
    if env is not None:
        env.close()
    app.close(wait_for_replicator=False, skip_cleanup=True)
sys.stdout.flush()
sys.stderr.flush()
os._exit(1 if failed else 0)
