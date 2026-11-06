"""Fresh-output orchestration; source samples/settings are shipped, old runs aren't.

Run through ../run.sh, not a workstation-specific experiment path. No ROS, object
attachment, original-checkout writes, or hidden success-only retries are used.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / 'output'
SETTINGS = json.loads((ROOT / 'final_settings.json').read_text())
LOCK = json.loads((ROOT / 'runtime.lock.json').read_text())
PYTHON = '/isaac-sim/python.sh'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def busy_guard(allow_shared):
    if allow_shared:
        return
    listing = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=process_name', '--format=csv,noheader'
    ], text=True)
    if any('isaac-sim' in p or 'kit/python' in p for p in listing.splitlines()):
        raise RuntimeError('An Isaac process is already using the GPU. Leave it running; '
                           'retry later, or explicitly opt in with --allow-shared-gpu.')


def doctor(static=False):
    expected = json.loads((ROOT / 'input/manifest.json').read_text())
    for name, sha in expected.items():
        assert digest(ROOT / name) == sha, f'Missing, LFS-pointer, or changed file: {name}'
    for path in (ROOT / 'scripts').glob('*.py'):
        ast.parse(path.read_bytes(), filename=str(path))
    report = {'input_hashes_valid': True, 'python_sources_parse': True,
              'package': str(ROOT), 'static_only': static, 'runtime_lock': LOCK}
    if not static:
        import numpy as np
        import h5py
        import torch
        import curobo
        seed = np.load(ROOT / 'input/source_episode.npz')
        fk = np.load(ROOT / 'input/source_fk.npz')
        assert seed['state'].shape == (115, 22)
        assert seed['action'].shape == (115, 22)
        assert len(seed['joint_names']) >= 19
        assert fk['eef'].shape == (115, 4, 4)
        assert np.isfinite(seed['state']).all() and np.isfinite(fk['eef']).all()
        with h5py.File(ROOT / 'input/seed.hdf5') as h:
            assert len(h['data']) == 1
        assert torch.__version__ == LOCK['torch'], torch.__version__
        assert torch.version.cuda == LOCK['cuda_toolkit'], torch.version.cuda
        assert torch.cuda.is_available(), 'CUDA GPU is required'
        curobo_root = next((p for p in Path(curobo.__file__).parents if (p / '.git').exists()), None)
        assert curobo_root is not None, 'Cannot verify cuRobo source revision'
        revision = subprocess.check_output(['git', '-C', str(curobo_root), 'rev-parse', 'HEAD'], text=True).strip()
        assert revision == LOCK['curobo_commit'], revision
        robot = Path('/workspace/cyclo_lab/source/cyclo_lab/data/robots/FFW/FFW_SG2_softgripper_soft_tips.usda')
        assert robot.is_file() and not robot.read_bytes().startswith(b'version https://git-lfs'), 'Run git lfs pull'
        versions = {}
        for line in (ROOT / 'docker/requirements.txt').read_text().splitlines():
            if '==' not in line or line.startswith('#'):
                continue
            name, wanted = line.split('==')
            actual = importlib.metadata.version(name)
            assert actual == wanted, f'{name}: {actual} != {wanted}'
            versions[name] = actual
        report.update(versions=versions, torch=torch.__version__, cuda=torch.version.cuda,
                      gpu=torch.cuda.get_device_name(0), curobo_commit=revision,
                      source_state_shape=list(seed['state'].shape), robot_asset=str(robot))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    save(OUTPUT / 'doctor.json', report)
    print(json.dumps(report, indent=2))


def run(script, arguments, environment, log):
    command = [PYTHON, '-u', str(ROOT / 'scripts' / script), *map(str, arguments)]
    with Path(log).open('w') as stream:
        stream.write(json.dumps({'argv': command, 'task759_environment': {
            k: v for k, v in environment.items() if k.startswith('TASK759_')
        }}) + '\n')
        stream.flush()
        subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)


def environment():
    env = {k: v for k, v in os.environ.items() if not k.startswith('TASK759_')}
    env['CUDA_HOME'] = '/usr/local/cuda-12.8'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    return env


def round_path(name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,80}', name):
        raise ValueError('Use a simple run name without slashes')
    return OUTPUT / name


def selected_angles(yaws):
    values = SETTINGS['angles'] if not yaws else yaws
    if len(set(values)) != len(values) or any(v not in SETTINGS['angles'] for v in values):
        raise ValueError(f'Choose distinct supported delta angles: {SETTINGS["angles"]}')
    return values


def plan(name, yaws):
    folder = round_path(name)
    folder.mkdir(parents=True, exist_ok=False)
    scene = json.loads((ROOT / 'task_000759/scene.json').read_text())
    scene['basket'] = SETTINGS['basket']
    save(folder / 'scene.json', scene)
    report = {'angles': yaws, 'settings': SETTINGS, 'planned': {}, 'errors': {},
              'historical_output_required': False, 'physical_success_claimed': False}
    for yaw in yaws:
        signed = f'{yaw:+03d}'
        print('PLAN', signed, flush=True)
        try:
            env = environment()
            initial = f'{name}/initial_{signed}'
            if yaw in SETTINGS['protected_tool_axis_approach']:
                run('plan_corridor_pickups.py', [
                    '--yaws', yaw, '--candidate', SETTINGS['candidate_indices'][str(yaw)],
                    '--scene', folder / 'scene.json', '--output', initial
                ], env, folder / f'initial_{signed}.log')
                source_args = ['--input-profiles', initial, '--protect-approach',
                               '--approach-slowdown', SETTINGS['approach_slowdown'][str(yaw)],
                               '--pregrasp-settle', SETTINGS['pregrasp_settle_s'][str(yaw)]]
            else:
                env['TASK759_CANDIDATE_MAP'] = json.dumps({str(yaw): [SETTINGS['candidate_indices'][str(yaw)]]})
                if str(yaw) in SETTINGS['initial_clearance_waypoint_m']:
                    env['TASK759_CLEARANCE_WAYPOINT'] = str(SETTINGS['initial_clearance_waypoint_m'][str(yaw)])
                if yaw in SETTINGS['refined_grasp_grid']:
                    env['TASK759_REFINED_GRASP_GRID'] = '1'
                run('plan_pickplace.py', ['--yaws', yaw, '--output', initial,
                    '--pickup-only', '--prior-weight', '.005', '--attempts', '1'], env,
                    folder / f'initial_{signed}.log')
                if yaw in SETTINGS['direct_pickup_retiming']:
                    source_args = ['--input-pickups', initial]
                else:
                    profile = f'{name}/profile_{signed}'
                    run('make_profile_pickups.py', ['--input', initial, '--yaws', yaw,
                        '--profiles', 'hybrid', '--motion-speed', '1.2', '--output', profile],
                        env, folder / f'profile_{signed}.log')
                    source_args = ['--input-profiles', profile]
            # Profiles fit with the original conservative timing; retime only now.
            rapid = f'{name}/pickup_{signed}'
            run('make_rapid_pickups.py', ['--yaws', yaw, *source_args,
                '--clearance', SETTINGS['closure_rise_m'].get(str(yaw), SETTINGS['closure_rise_m']['default']),
                '--monotone-clearance', '--scene', folder / 'scene.json', '--output', rapid],
                environment(), folder / f'rapid_{signed}.log')
            trajectory = f'{rapid}/hybrid_{signed}/trajectory.npz'
            if not (OUTPUT / trajectory).is_file():
                raise RuntimeError('Planner did not produce an accepted pickup trajectory')
            report['planned'][str(yaw)] = trajectory
        except Exception as error:
            report['errors'][str(yaw)] = str(error)
            print('PLAN_FAILED', signed, str(error), flush=True)
        save(folder / 'plan_manifest.json', report)
    return folder, report


def simulate(folder, plan_report):
    report = {'angles': plan_report['angles'], 'runs': [], 'planning_errors': plan_report['errors'],
              'tuned_settings_not_unbiased_success_rate': True, 'physics_hz': 240,
              'normal_playback': True, 'source_episode_count': 1,
              'requested': len(plan_report['angles']), 'successful': 0}
    mapping = {}
    for yaw in plan_report['angles']:
        if str(yaw) not in plan_report['planned']:
            continue
        signed = f'{yaw:+03d}'
        name = f'{folder.name}/run_{signed}'
        if (OUTPUT / name).exists():
            raise FileExistsError(f'Refusing to overwrite an earlier physical attempt: {name}')
        env = environment()
        env['TASK759_TIMING_PROFILE'] = 'rapid'
        env['TASK759_LOW_ROTATION'] = '1' if yaw in SETTINGS['low_rotation_placement'] else '0'
        if str(yaw) in SETTINGS['wrist_travel_penalty']:
            env['TASK759_WRIST_PENALTY'] = str(SETTINGS['wrist_travel_penalty'][str(yaw)])
        if str(yaw) in SETTINGS['placement_tilt_probe_deg']:
            env['TASK759_PLACEMENT_TILT_DEG'] = str(SETTINGS['placement_tilt_probe_deg'][str(yaw)])
        print('SIMULATE', signed, flush=True)
        try:
            run('simulate.py', ['--headless', '--video', '--interpolate-targets',
                '--velocity-feedforward', '--plan-place', '--return-home',
                '--lift-damping', SETTINGS['lift_damping'], '--profile', 'hybrid',
                '--yaw', yaw, '--name', name, '--scene', folder / 'scene.json',
                '--trajectory', OUTPUT / plan_report['planned'][str(yaw)]], env,
                folder / f'simulation_{signed}.log')
            result = json.loads((OUTPUT / name / 'result.json').read_text())
            success = bool(result['physical_success'] and result['max_joint_tracking_error_rad'] < .05)
            entry = {'yaw_deg': yaw, 'success': success, 'result': str(OUTPUT / name / 'result.json'),
                     'video': str(OUTPUT / name / 'observer.mp4'), 'physical_duration_s': result['physical_time_s']}
            if success:
                mapping[str(yaw)] = name
        except Exception as error:
            entry = {'yaw_deg': yaw, 'success': False, 'error': str(error)}
        report['runs'].append(entry)
        report['successful'] = sum(r['success'] for r in report['runs'])
        report['requested'] = len(plan_report['angles'])
        save(folder / 'generation_report.json', report)
    save(folder / 'selection.json', mapping)
    save(folder / 'generation_report.json', report)
    if len(mapping) == 11:
        destination = f'{folder.name}/report'
        run('report_hybrid11.py', ['--runs', f'{folder.name}/run', '--mapping',
            folder / 'selection.json', '--output', destination], environment(), folder / 'report.log')
        run('concat_hybrid11.py', ['--folder', OUTPUT / destination, '--speed', '1'],
            environment(), folder / 'concat.log')
    print(json.dumps(report, indent=2))
    return len(mapping) == len(plan_report['angles'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    check = sub.add_parser('doctor');check.add_argument('--static', action='store_true')
    for name in ('plan', 'generate', 'simulate'):
        command = sub.add_parser(name)
        command.add_argument('--name', required=True)
        command.add_argument('--yaws', nargs='+', type=int)
        command.add_argument('--allow-shared-gpu', action='store_true',
                             help='Explicitly permit GPU sharing; never interrupts existing processes')
    historical = sub.add_parser('skillgen-poses')
    historical.add_argument('--name', default='skillgen_poses')
    args = p.parse_args()
    if args.command == 'doctor':
        doctor(args.static);return
    if args.command == 'skillgen-poses':
        busy_guard(False)
        target = round_path(args.name)
        if target.exists():
            raise FileExistsError(f'Refusing to overwrite {target}')
        OUTPUT.mkdir(parents=True, exist_ok=True)
        run('skillgen_poses.py', ['--headless', '--output', target],
            environment(), OUTPUT / f'{args.name}.log')
        print(target / 'skillgen_pose_report.json')
        return
    busy_guard(args.allow_shared_gpu)
    if args.command == 'simulate':
        folder = round_path(args.name)
        report = json.loads((folder / 'plan_manifest.json').read_text())
        if args.yaws and args.yaws != report['angles']:
            raise ValueError('Simulation angles must match this plan manifest')
    else:
        folder, report = plan(args.name, selected_angles(args.yaws))
    if args.command == 'plan':
        print(json.dumps(report, indent=2))
        raise SystemExit(1 if report['errors'] else 0)
    raise SystemExit(0 if simulate(folder, report) else 1)


if __name__ == '__main__':
    main()
