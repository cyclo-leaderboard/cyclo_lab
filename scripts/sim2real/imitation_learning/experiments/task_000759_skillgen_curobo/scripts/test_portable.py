"""CPU-only tests for routing and safety; these do not claim physical success."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('task759_portable', Path(__file__).with_name('portable.py'))
portable = importlib.util.module_from_spec(spec)
spec.loader.exec_module(portable)


class PortableTests(unittest.TestCase):
    def test_names_and_angles(self):
        for bad in ('../escape', '/absolute', '.', '..', 'a/b'):
            with self.assertRaises(ValueError):
                portable.round_path(bad)
        for angles in ([0, 0], [15], [75, 180]):
            with self.assertRaises(ValueError):
                portable.selected_angles(angles)
        self.assertEqual(len(portable.selected_angles(None)), 11)

    def test_busy_guard_never_kills(self):
        with patch.object(portable.subprocess, 'check_output', return_value='/isaac-sim/kit/python/bin/python3\n'):
            with self.assertRaises(RuntimeError):
                portable.busy_guard(False)
        with patch.object(portable.subprocess, 'check_output') as query:
            portable.busy_guard(True)
            query.assert_not_called()

    def test_fresh_output_all_angle_routing(self):
        calls = []
        with tempfile.TemporaryDirectory(prefix='task759-check-') as temporary:
            with patch.object(portable, 'OUTPUT', Path(temporary)):
                def fake(script, args, env, log):
                    calls.append((script, args, dict(env)))
                    if script == 'make_rapid_pickups.py':
                        yaw = args[args.index('--yaws') + 1]
                        out = args[args.index('--output') + 1]
                        path = portable.OUTPUT / out / f'hybrid_{yaw:+03d}' / 'trajectory.npz'
                        path.parent.mkdir(parents=True)
                        path.touch()
                with patch.object(portable, 'run', side_effect=fake):
                    folder, report = portable.plan('test11', portable.selected_angles(None))
                self.assertEqual(len(report['planned']), 11)
                self.assertFalse(report['errors'])
                with self.assertRaises(FileExistsError):
                    portable.plan('test11', [0])
                self.assertTrue((folder / 'scene.json').exists())
        corridor = [c for c in calls if c[0] == 'plan_corridor_pickups.py']
        self.assertEqual(len(corridor), 1)
        self.assertIn(-75, corridor[0][1])
        refined = [c for c in calls if c[2].get('TASK759_REFINED_GRASP_GRID') == '1']
        self.assertTrue(refined)
        self.assertTrue(all(c[2].get('TASK759_TIMING_PROFILE') is None for c in calls))
        zero = next(c for c in calls if c[0] == 'make_rapid_pickups.py' and c[1][1] == 0)
        self.assertEqual(zero[1][zero[1].index('--clearance') + 1], .020)
        self.assertTrue(all('--scene' in c[1] for c in calls if c[0] == 'make_rapid_pickups.py'))

    def test_special_placement_flags_and_failure_accounting(self):
        calls = []
        with tempfile.TemporaryDirectory(prefix='task759-check-') as temporary:
            with patch.object(portable, 'OUTPUT', Path(temporary)):
                folder = portable.OUTPUT / 'small';folder.mkdir()
                manifest = {'angles': [75, -90], 'planned': {'75': 'p75.npz', '-90': 'pm90.npz'}, 'errors': {}}
                def fake(script, args, env, log):
                    calls.append((script, args, dict(env)))
                    name = args[args.index('--name') + 1]
                    out = portable.OUTPUT / name;out.mkdir(parents=True)
                    portable.save(out / 'result.json', {'physical_success': True,
                        'max_joint_tracking_error_rad': .01, 'physical_time_s': 10.})
                with patch.object(portable, 'run', side_effect=fake):
                    self.assertTrue(portable.simulate(folder, manifest))
                p75, n90 = calls
                self.assertEqual(p75[2]['TASK759_WRIST_PENALTY'], '5')
                self.assertEqual(p75[2]['TASK759_PLACEMENT_TILT_DEG'], '20')
                self.assertEqual(n90[2]['TASK759_LOW_ROTATION'], '1')
                self.assertNotIn('TASK759_PLACEMENT_TILT_DEG', n90[2])
                for call in calls:
                    self.assertNotIn('--diagnostic-no-table-contact', call[1])
                    self.assertIn('--return-home', call[1])
                result = json.loads((folder / 'generation_report.json').read_text())
                self.assertEqual(result['successful'], 2)


if __name__ == '__main__':
    unittest.main()
