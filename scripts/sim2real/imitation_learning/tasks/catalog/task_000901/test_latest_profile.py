"""CPU-only USD composition/authoring tests; does not launch Kit or simulate.

Run with an Isaac Python environment exposing pxr/PhysxSchema, numpy and trimesh.
The AST loader isolates real collision authoring from Isaac's GPU app imports.
This is not a physical grasp/insertion success test.
"""
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import unittest

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[5]
PROFILE = json.loads((ROOT/'physics_profile.json').read_text())


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def authoring_namespace(path, injected=None):
    """Keep all real functions/constants, omit GPU-dependent package imports."""
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and
        (node.module or '').startswith(('isaaclab', 'cyclo_lab', '_cyclo_task')))]
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.decorator_list = []
    namespace = {'__file__': str(path), **(injected or {})}
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace


def box(stage, path):
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render']).ComputeWorldBound(
        stage.GetPrimAtPath(path)).ComputeAlignedBox()
    return np.array(bounds.GetMin()), np.array(bounds.GetMax())


class LatestProfileTests(unittest.TestCase):
    def test_exact_current_profile_and_candidate(self):
        self.assertEqual(PROFILE['table_top_z_m'], .695)
        self.assertEqual(PROFILE['table_height_offset_m'], -.005)
        self.assertEqual(PROFILE['contact_mode'], 'recess5mm')
        self.assertEqual(PROFILE['opposing_tip_filtered_pairs_retained'], 50)
        self.assertEqual(PROFILE['plate_friction'], [.6, .45])
        self.assertEqual(hashlib.sha256((ROOT/'contact_candidate.py').read_bytes()).hexdigest(),
                         PROFILE['source_sha256']['contact_candidate.py'])
        plugin = load(ROOT/'plugin.py')
        self.assertEqual(plugin.provenance()['table_top_z_m'], .695)

    def test_room_collision_and_tags_move_together(self):
        for old, new in [('white_room_task000901_visual_v2.usda', 'white_room_latest.usda'),
                         ('white_room_plate_holder.usda', 'white_room_plate_holder_latest.usda')]:
            before = Usd.Stage.Open(str(ROOT/'runtime/data'/old))
            after = Usd.Stage.Open(str(ROOT/'runtime/data'/new))
            self.assertTrue(before and after)
            lo, hi = box(after, '/Root/WhiteTable/Top')
            self.assertAlmostEqual(hi[2], PROFILE['table_top_z_m'], places=6)
            old_lo, old_hi = box(before, '/Root/WhiteTable/Top')
            np.testing.assert_allclose(hi-old_hi, [0, 0, -.005], atol=1e-7)
            np.testing.assert_allclose(lo-old_lo, [0, 0, -.005], atol=1e-7)
            colliders = [p for p in Usd.PrimRange(after.GetPrimAtPath('/Root/WhiteTable'))
                         if p.HasAPI(UsdPhysics.CollisionAPI)]
            self.assertTrue(colliders)
            for i in range(5):
                path = f'/Root/WhiteTable/Tag{i}Paper'
                old_tag = box(before, path)
                new_tag = box(after, path)
                np.testing.assert_allclose(new_tag[0]-old_tag[0], [0, 0, -.005], atol=1e-7)

    def test_latest_plate_geometry(self):
        root = ROOT/'runtime/data/objects/plates_latest'
        meta = json.loads((root/'plate_profile.json').read_text())
        self.assertEqual(meta['revision'], PROFILE['plate_profile'])
        self.assertEqual(meta['flat_eating_surface_diameter_m'], .16)
        self.assertEqual(meta['outer_diameter_m'], .185)
        self.assertEqual(meta['convex_collision_sectors'], 384)
        for color in ('light_blue', 'pastel_pink', 'lime_yellow'):
            stage = Usd.Stage.Open(str(root/f'{color}.usda'))
            self.assertTrue(stage)
            colliders = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)
                         and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False]
            self.assertEqual(len(colliders), 384)

    def test_real_robot_collision_authoring(self):
        robots = REPO/'source/cyclo_lab/cyclo_lab/assets/robots'
        geometry = load(robots/'ffw_softgripper_geometry.py')
        fc = authoring_namespace(robots/'ffw_softgripper_collision.py',
                                {'softgripper_source_points': geometry.softgripper_source_points})
        contact = load(ROOT/'contact_candidate.py')
        latest = authoring_namespace(ROOT/'latest_physics.py', {'apply_candidate': contact.apply_candidate})
        stage = Usd.Stage.CreateInMemory()
        robot = stage.DefinePrim('/Robot', 'Xform')
        robot.GetReferences().AddReference(str(REPO/'source/cyclo_lab/data/robots/FFW/FFW_SG2_softgripper_soft_tips.usda'))
        while True:
            instances = [p for p in stage.Traverse() if p.IsInstance()]
            if not instances:
                break
            for p in instances:
                p.SetInstanceable(False)
        UsdPhysics.Scene.Define(stage, '/PhysicsScene')
        UsdPhysics.MaterialAPI.Apply(stage.DefinePrim('/Robot/distalGripperPhysicsMaterial', 'Material'))
        fc['apply_ffw_softgripper_collision_defaults'](stage, '/Robot')
        visuals = {str(p.GetPath()): np.array(UsdGeom.Mesh(p).GetPointsAttr().Get()).copy()
                   for p in Usd.PrimRange(robot) if p.IsA(UsdGeom.Mesh) and '/visuals/' in str(p.GetPath())}
        joints = [p for p in Usd.PrimRange(robot) if p.GetName() in
                  {f'gripper_{s}_joint{j}' for s in 'lr' for j in (2, 4)}]
        mimic_before = [(p.GetAttribute('physxMimicJoint:rotX:naturalFrequency').Get(),
                         p.GetAttribute('physxMimicJoint:rotX:dampingRatio').Get()) for p in joints]
        audit = latest['apply_latest_robot_physics'](robot)
        self.assertEqual(len(audit['pieces']), 20)
        shift = max(p['max_world_shift_m'] for p in audit['pieces'])
        self.assertGreater(shift, .004)
        self.assertLessEqual(shift, .00500001)
        self.assertEqual(robot.GetCustomDataByKey('task901:physicsProfile'), PROFILE['id'])
        self.assertEqual(len(joints), 4)
        for p, original in zip(joints, mimic_before):
            self.assertAlmostEqual(p.GetAttribute('physics:upperLimit').Get(), math.degrees(1.1), places=4)
            self.assertEqual((p.GetAttribute('physxMimicJoint:rotX:naturalFrequency').Get(),
                              p.GetAttribute('physxMimicJoint:rotX:dampingRatio').Get()), original)
        for path, original in visuals.items():
            np.testing.assert_array_equal(UsdGeom.Mesh(stage.GetPrimAtPath(path)).GetPointsAttr().Get(), original)
        print(json.dumps({'cpu_usd_only': True, 'tip_pieces': len(audit['pieces']),
                          'max_recess_shift_m': shift, 'visual_meshes_unchanged': len(visuals)}))


if __name__ == '__main__':
    unittest.main(verbosity=2)
