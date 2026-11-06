"""Offline seed integrity tests; requires only numpy and h5py, no simulator."""
import json
from pathlib import Path
import unittest

import h5py
import numpy as np

from prepare_reference import PHASES, quat_matrix, sha

ROOT = Path(__file__).parent/'output/reference'


@unittest.skipUnless((ROOT/'seed.hdf5').exists(), 'prepare the reference first')
class ReferenceTests(unittest.TestCase):
    def test_source_identity(self):
        p = json.loads((ROOT/'provenance.json').read_text())
        self.assertEqual(p['video_side'], 'left')
        self.assertEqual(p['variant'], 'original_joint_limits')
        self.assertFalse(p['original_teleop_actions_available'])
        # Compare the source archive to its preserved provenance, not filenames.
        source = Path(p['source_read_from'])
        for rel, digest in p['source_hashes'].items():
            self.assertEqual(sha(source/rel), digest)

    def test_both_pose_actions_match_matrices(self):
        with h5py.File(ROOT/'seed.hdf5') as f:
            demo = f['data/demo_0']
            actions = demo['actions'][:]
            self.assertEqual(actions.shape, (910,19))
            self.assertTrue(np.isfinite(actions).all())
            for name, span in [('left_arm',slice(0,7)), ('right_arm',slice(8,15))]:
                pose = actions[:,span]
                matrix = demo['obs/datagen_info/target_eef_pose/'+name][:]
                np.testing.assert_allclose(pose[:,:3],matrix[:,:3,3],atol=1e-6)
                np.testing.assert_allclose(quat_matrix(pose[:,3:7]),matrix[:,:3,:3],atol=1e-6)

    def test_both_grippers_are_preserved(self):
        source = np.load(ROOT/'input/aligned_replay.npz')
        with h5py.File(ROOT/'seed.hdf5') as f:
            actions = f['data/demo_0/actions'][:]
            for side,index in [('l',7),('r',15)]:
                ref = source['reference_position'][:,source['joint_names'].tolist().index('gripper_'+side+'_joint1')]
                np.testing.assert_allclose(actions[:len(ref),index],ref,atol=1e-7)

    def test_phase_boundaries(self):
        with h5py.File(ROOT/'seed.hdf5') as f:
            signals = f['data/demo_0/obs/datagen_info/subtask_term_signals']
            cuts = []
            for phase in PHASES[:-1]:
                transition = np.flatnonzero(np.diff(signals[phase[0]][:]))
                self.assertEqual(len(transition),1)
                cuts.append(int(transition[0])+2)
            self.assertTrue(all(a<b for a,b in zip([0]+cuts,cuts+[910])))


if __name__ == '__main__':
    unittest.main()
