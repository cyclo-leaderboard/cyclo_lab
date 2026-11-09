"""Convert the LEFT episode6 replay archive into an explicitly derived Mimic seed.

Runs without Isaac Sim. This is a measured-state-derived seed, NOT recovered
teleoperation actions. Never edits the source experiment or overwrites an output.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import h5py
import numpy as np

HOST_REFERENCE = Path('/home/robotis-ai/agents/experiments/2026-11-03_09-56-43_task000901_real_sim_replay/insert_plate_real2sim/tip_self_contact_joint24_comparison/episode_006')
SEED_ID = 'Cyclo-Real-WhiteRoom-Task000901-Bimanual-Mimic-Seed-FFW-SG2-v0'
ROOT_POS = np.array([-.0200190219, -.1099593887, 0.])
ROOT_ROT = np.diag([-1., -1., 1.])
HOLDER_POSE = [-.5464056504055486, .15655533467208427, .7002,
               .7006398248933365, 0., 0., .7135151265204086]
TCP_Z = -.2
# Seconds in the *0.8x simulation* clock. In each cycle: lift, coordinated
# handover, then holder-relative insertion/release/retreat. Both hands share cuts.
PHASES = [
    ('blue_lift', 'plate_blue', 6.5, 'TRANSFORM'),
    ('blue_handover', None, 12., 'REPLAY'),
    ('blue_place', 'plate_holder', 19.2, 'TRANSFORM'),
    ('yellow_lift', 'plate_yellow', 25.5, 'TRANSFORM'),
    ('yellow_handover', None, 31.2, 'REPLAY'),
    ('yellow_place', 'plate_holder', 38., 'TRANSFORM'),
    ('pink_lift', 'plate_pink', 44.5, 'TRANSFORM'),
    ('pink_handover', None, 49.2, 'REPLAY'),
    ('pink_place', 'plate_holder', None, 'TRANSFORM'),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quat_matrix(q):
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), axis=-1).reshape(q.shape[:-1]+(3, 3))


def world_to_root_pose(pose):
    out = np.broadcast_to(np.eye(4), (len(pose), 4, 4)).copy()
    out[:, :3, :3] = ROOT_ROT.T @ quat_matrix(pose[:, 3:7])
    out[:, :3, 3] = (pose[:, :3] - ROOT_POS) @ ROOT_ROT
    return out.astype(np.float32)


def prepare(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    inp = source/'input'
    render = source/'original_joint_limits/render'
    required = [inp/'aligned_replay.npz', inp/'input_manifest.json',
                inp/'scene_snapshot.py', render/'physics_states.npz',
                render/'gripper_kinematics.npz', render/'physics_result.json',
                render/'all_plates_hold_3s.npy']
    hashes = {str(p.relative_to(source)): sha(p) for p in required}
    manifest = json.loads((inp/'input_manifest.json').read_text())
    report = json.loads((render/'physics_result.json').read_text())
    if manifest['episode_index'] != 6 or report.get('object_pose_writes_after_initialization') != 0:
        raise ValueError('Not the requested freely simulated episode6 reference')
    data = np.load(inp/'aligned_replay.npz')
    physics = np.load(render/'physics_states.npz')
    kin = np.load(render/'gripper_kinematics.npz')
    count = len(data['time_sec'])
    if len(kin['time_sec']) != count or not np.allclose(kin['time_sec'], data['time_sec']):
        raise ValueError('Reference clocks do not match')
    # Preserve original 15 Hz archive; do not invent higher-rate contact poses.
    extra = 45  # terminal three-second settling segment
    def extend(x):
        return np.concatenate((x, np.repeat(x[-1:], extra, axis=0)))
    eef = {}
    eef_pose7 = {}
    for side in ('left', 'right'):
        index = kin['body_names'].tolist().index(f'arm_{side[0]}_link7')
        world = kin['body_pose_xyzw'][:, index].copy()
        world[:, 3:7] = world[:, [6, 3, 4, 5]]  # PhysX XYZW -> WXYZ
        world[:, :3] += quat_matrix(world[:, 3:7]) @ np.array([0., 0., TCP_Z])
        eef[side+'_arm'] = extend(world_to_root_pose(world))
        root_quat = world[:, [6, 5, 4, 3]] * np.array([-1., -1., 1., 1.])
        eef_pose7[side] = extend(np.concatenate(((world[:, :3]-ROOT_POS) @ ROOT_ROT, root_quat), axis=1))
    target = {k: np.concatenate((v[1:], v[-1:])) for k, v in eef.items()}
    names = data['joint_names'].tolist()
    ordered = ([f'arm_l_joint{i}' for i in range(1, 8)] + ['gripper_l_joint1'] +
               [f'arm_r_joint{i}' for i in range(1, 8)] +
               ['gripper_r_joint1', 'head_joint1', 'head_joint2', 'lift_joint'])
    joint_targets = extend(data['reference_position'][:, [names.index(n) for n in ordered]]).astype(np.float32)
    # Pose actions are annotated below from matrices; the core reads grippers.
    actions = joint_targets.copy()
    for side, span in [('left', slice(0,7)), ('right', slice(8,15))]:
        poses = eef_pose7[side]
        actions[:, span] = np.concatenate((poses[1:], poses[-1:]))
    object_poses = {str(n): extend(world_to_root_pose(physics['object_states'][:, j, :7]))
                    for j, n in enumerate(physics['plate_names'])}
    object_poses['plate_holder'] = world_to_root_pose(np.repeat([HOLDER_POSE], count+extra, axis=0))
    cuts = [int(np.searchsorted(data['time_sec'], p[2])) if p[2] is not None else count+extra
            for p in PHASES]
    output.mkdir(parents=True)
    shutil.copytree(inp, output/'input')
    for p in required:
        if p.parent == render:
            (output/'render').mkdir(exist_ok=True)
            shutil.copy2(p, output/'render'/p.name)
    with h5py.File(output/'seed.hdf5', 'x') as f:
        group = f.create_group('data')
        group.attrs['env_args'] = json.dumps({'env_name': SEED_ID, 'type': 2, 'env_kwargs': {}})
        group.attrs['total'] = count+extra
        group.attrs['action_representation'] = 'eef19_wxyz'
        group.attrs['source_action_provenance'] = 'derived from measured-position PD replay; original teleop actions unavailable'
        demo = group.create_group('demo_0')
        demo.attrs['num_samples'] = count+extra
        demo.attrs['success'] = True  # user-approved source, not a new runtime result
        demo.create_dataset('actions', data=actions)
        demo.create_dataset('source_joint_targets', data=joint_targets)
        info = demo.create_group('obs/datagen_info')
        for name in eef:
            info.create_dataset('eef_pose/'+name, data=eef[name])
            info.create_dataset('target_eef_pose/'+name, data=target[name])
        for name, poses in object_poses.items():
            info.create_dataset('object_pose/'+name, data=poses)
        for phase, cut in zip(PHASES[:-1], cuts[:-1]):
            # Pool treats transition+1 as end-exclusive.
            info.create_dataset('subtask_term_signals/'+phase[0],
                                data=(np.arange(count+extra) >= cut-1).astype(np.int32))
    provenance = dict(source_host=str(HOST_REFERENCE), source_read_from=str(source),
                      variant='original_joint_limits', video_side='left',
                      source_bag=manifest['source_bag'], mcap_sha256=manifest['mcap_sha256'],
                      source_hashes=hashes, source_robot_sha256=report['robot_usd_sha256'],
                      original_teleop_actions_available=False, tcp_z_m=TCP_Z,
                      root_pos=ROOT_POS.tolist(), root_quat_wxyz=[0.,0.,0.,1.],
                      control_hz=15, physics_hz=240,
                      source_frames=count, seed_frames=count+extra,
                      phases=[dict(name=p[0], object_ref=p[1], end_exclusive=c,
                                   coordination=p[3]) for p, c in zip(PHASES, cuts)],
                      expected_slot_xyz={str(n): physics['object_states'][-1, j, :3].tolist()
                                         for j, n in enumerate(physics['plate_names'])})
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=HOST_REFERENCE)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'output/reference')
    args = parser.parse_args()
    prepare(args.source, args.output)
