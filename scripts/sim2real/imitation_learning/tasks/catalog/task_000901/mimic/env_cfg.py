"""Isolated 901 bimanual variants for canonical or selected Task901 seeds."""
import json
import os
from pathlib import Path
import importlib.util

import torch
from pxr import PhysxSchema, Usd, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.envs import MimicEnvCfg, SubTaskConfig
from isaaclab.envs.mimic_env_cfg import (
    SubTaskConstraintConfig, SubTaskConstraintCoordinationScheme, SubTaskConstraintType,
)
from isaaclab.managers import EventTermCfg, TerminationTermCfg
from isaaclab.sim.utils import bind_physics_material
from isaaclab.utils import configclass, math as mu
from cyclo_lab.manager_based.actions.ffw_sg2 import FFWSG2JointPositionActionsCfg
from cyclo_lab.manager_based.manipulation.common import configure_ffw_sg2_mimic_ik_actions
from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.common import (
    EpisodicShowroomObservationsCfg, make_eef_frame,
)
from _cyclo_task000901_scene import Task000901PlateHolderEnvCfg, Task000901PlateHolderSceneCfg

from .prepare_reference import PHASES, TCP_Z

REFERENCE = Path(os.environ.get(
    'TASK901_MIMIC_REFERENCE', Path(__file__).parent/'output/reference')).resolve()

# Table-frame centers of the four printed 40 mm tags.  Order 0 -> 1 -> 3 -> 2
# is counter-clockwise around the requested spawn region.
APRILTAG_0123_XY = (
    (-0.7339083330653123, -0.5828106085655894),
    (-0.4253956648567431, -0.5898924214782611),
    (-0.3651127450816598, -0.0741429687618186),
    (-0.7319294830597425, -0.0612887391870866),
)
PLATE_OUTER_RADIUS_M = 0.185 / 2
PLATE_CENTER_EDGE_CLEARANCE_M = 0.02
PLATE_PAIR_CLEARANCE_M = 0.005


def reference_provenance():
    return json.loads((REFERENCE/'provenance.json').read_text())


def spawn_reference_robot(path, cfg, *args, **kwargs):
    from cyclo_lab.assets.robots.FFW_SG2 import spawn_sg2_with_softgripper_physics
    result = spawn_sg2_with_softgripper_physics(path, cfg, *args, **kwargs)
    stage = result.GetStage()
    material = stage.GetPrimAtPath(str(result.GetPath())+'/distalGripperPhysicsMaterial')
    if not material or not material.HasAPI(UsdPhysics.MaterialAPI):
        raise ValueError('Expected original SG2 real-tip physics material')
    provenance = reference_provenance()
    profile = provenance.get('physics_profile')
    api = UsdPhysics.MaterialAPI(material)
    api.CreateStaticFrictionAttr().Set(float(profile['tip_static_friction']) if profile else 4.)
    api.CreateDynamicFrictionAttr().Set(float(profile['tip_dynamic_friction']) if profile else 3.5)
    physx = PhysxSchema.PhysxMaterialAPI.Apply(material)
    physx.CreateCompliantContactStiffnessAttr().Set(
        float(profile['contact_stiffness_N_per_m']) if profile else 0.)
    physx.CreateCompliantContactDampingAttr().Set(
        float(profile['contact_damping_Ns_per_m']) if profile else 0.)
    physx.CreateCompliantContactAccelerationSpringAttr().Set(False)
    filtered = 0
    for child in Usd.PrimRange(result):
        if '_cyclo_softgripper_proxy' in str(child.GetPath()) and child.HasAPI(UsdPhysics.FilteredPairsAPI):
            relation = UsdPhysics.FilteredPairsAPI(child).GetFilteredPairsRel()
            targets = list(relation.GetTargets())
            if not all('_cyclo_softgripper_proxy' in str(t) for t in targets):
                raise ValueError('Unexpected non-tip collision exclusion')
            filtered += len(targets)
            if not profile:
                relation.SetTargets([])
        if child.GetName() in {f'gripper_{s}_joint{j}' for s in 'lr' for j in (2,4)}:
            import math
            tip = profile['tip_contact'] if profile else {}
            child.GetAttribute('physics:upperLimit').Set(math.degrees(float(tip.get('joint24_upper_limit_rad', 1.1))))
            child.GetAttribute('physxMimicJoint:rotX:naturalFrequency').Set(float(tip.get('joint24_mimic_natural_frequency', 1500.)))
            child.GetAttribute('physxMimicJoint:rotX:dampingRatio').Set(float(tip.get('joint24_mimic_damping_ratio', 5.)))
    if filtered != 50:
        raise ValueError(f'Expected 50 opposing-tip exclusions, got {filtered}')
    if profile:
        expected = int(profile['tip_contact']['opposing_tip_filtered_pairs_retained'])
        if expected != 50:
            raise ValueError(f'Selected Task901 seed expects unsupported retained-pair count {expected}')
        contact_path = Path(__file__).resolve().parent.parent/'test4_contact.py'
        spec = importlib.util.spec_from_file_location('_task901_selected_contact', contact_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        audit = module.apply_candidate(stage, str(result.GetPath()), profile['contact_mode'])
        if not abs(float(audit['recess_max_m'])-float(profile['tip_recess_m'])) < 1e-9:
            raise ValueError('Selected seed collision recess was not reproduced')
    return result


def spawn_reference_plate(path, cfg, *args, **kwargs):
    from _cyclo_task000901_scene import spawn_task000901_plate
    result = spawn_task000901_plate(path, cfg, *args, **kwargs)
    material = result.GetStage().GetPrimAtPath(str(result.GetPath())+'/PlatePhysicsMaterial')
    api = UsdPhysics.MaterialAPI(material)
    profile = reference_provenance().get('physics_profile')
    api.CreateStaticFrictionAttr().Set(float(profile['plate_static_friction']) if profile else .45)
    api.CreateDynamicFrictionAttr().Set(float(profile['plate_dynamic_friction']) if profile else .30)
    PhysxSchema.PhysxMaterialAPI.Apply(material).CreateFrictionCombineModeAttr().Set('max' if profile else 'min')
    return result


def spawn_reference_holder(path, cfg, *args, **kwargs):
    from isaaclab.sim.spawners.from_files import from_files
    result = from_files.spawn_from_usd(path, cfg, *args, **kwargs)
    if reference_provenance().get('physics_profile', {}).get('use_source_scene_holder_material'):
        return result
    material_path = str(result.GetPath())+'/HolderPhysicsMaterial'
    material = sim_utils.RigidBodyMaterialCfg(static_friction=.25, dynamic_friction=.20,
        restitution=0., friction_combine_mode='min', restitution_combine_mode='min')
    material.func(material_path, material)
    bind_physics_material(str(result.GetPath()), material_path)
    return result


def reset_reference(env, env_ids):
    """Only initial-state writes; all later plate movement is physical."""
    from isaaclab.envs.mdp.events import reset_scene_to_default
    reset_scene_to_default(env, env_ids, reset_joint_targets=True)
    env._rack_signal_tick = None
    if hasattr(env, '_rack_stable_steps'):
        env._rack_stable_steps[env_ids] = 0
    names = ('plate_blue', 'plate_yellow', 'plate_pink')
    base_xy = torch.stack([env.scene[n].data.default_root_state[env_ids, :2] for n in names], dim=1)
    all_xy = base_xy.clone()
    pending = torch.ones(len(env_ids), dtype=torch.bool, device=env.device)
    if env.cfg.plate_spawn_region == 'apriltag_0_1_2_3':
        polygon = torch.tensor(APRILTAG_0123_XY, device=env.device, dtype=base_xy.dtype)
        edge = torch.roll(polygon, -1, 0) - polygon
        edge_length = torch.linalg.vector_norm(edge, dim=-1)
        # The printed tag centers define the requested center-spawn polygon.
        # A small inset keeps centers off the paper tags.  Requiring the full
        # 185 mm disc inside this polygon would reject the recorded layouts.
        margin = PLATE_CENTER_EDGE_CLEARANCE_M
        low, high = polygon.amin(dim=0), polygon.amax(dim=0)
        for _ in range(10000):
            count = int(pending.sum())
            candidate = low + torch.rand((count, 3, 2), device=env.device) * (high-low)
            relative = candidate[:, :, None, :] - polygon[None, None, :, :]
            cross = edge[None, None, :, 0]*relative[..., 1] - edge[None, None, :, 1]*relative[..., 0]
            valid = (cross >= margin*edge_length[None, None, :]).all(dim=(1,2))
            for i, j in ((0,1),(0,2),(1,2)):
                valid &= torch.linalg.vector_norm(candidate[:,i]-candidate[:,j], dim=-1) >= (
                    2*PLATE_OUTER_RADIUS_M + PLATE_PAIR_CLEARANCE_M)
            # Internal object names carry spatial roles for Mimic transforms:
            # blue=right-bottom, yellow=left-bottom, pink=top.  Colors are not
            # task semantics and may be permuted visually in a later pass.
            top = candidate[:, :, 0].argmin(dim=1)
            remaining = torch.ones((count, 3), dtype=torch.bool, device=env.device)
            remaining.scatter_(1, top[:, None], False)
            masked_y = candidate[:, :, 1].masked_fill(~remaining, float('-inf'))
            right_bottom = masked_y.argmax(dim=1)
            masked_y = candidate[:, :, 1].masked_fill(~remaining, float('inf'))
            left_bottom = masked_y.argmin(dim=1)
            row = torch.arange(count, device=env.device)
            candidate = torch.stack((
                candidate[row, right_bottom], candidate[row, left_bottom], candidate[row, top]), dim=1)
            pending_ids = pending.nonzero(as_tuple=False).squeeze(-1)
            accepted = pending_ids[valid]
            all_xy[accepted] = candidate[valid]
            pending[accepted] = False
            if not bool(pending.any()):
                break
    else:
        bounds = float(env.cfg.plate_xy_half_range_m)
        all_offsets = torch.zeros_like(base_xy)
        for _ in range(1000):
            all_offsets[pending] = (torch.rand((int(pending.sum()), 3, 2), device=env.device)*2-1)*bounds
            candidate = base_xy + all_offsets
            valid = torch.ones(len(env_ids), dtype=torch.bool, device=env.device)
            for i, j in ((0,1),(0,2),(1,2)):
                valid &= torch.linalg.vector_norm(candidate[:,i]-candidate[:,j], dim=-1) >= (
                    2*PLATE_OUTER_RADIUS_M + PLATE_PAIR_CLEARANCE_M)
            all_xy[~pending] = candidate[~pending]
            accepted = pending & valid
            all_xy[accepted] = candidate[accepted]
            pending[accepted] = False
            if not bool(pending.any()):
                break
    if bool(pending.any()):
        raise ValueError('Could not sample non-overlapping plate positions')
    offsets = {}
    for index, name in enumerate(names):
        obj = env.scene[name]
        state = obj.data.default_root_state[env_ids].clone()
        state[:, :3] += env.scene.env_origins[env_ids]
        xy = all_xy[:, index]
        offset = xy - base_xy[:, index]
        state[:, :2] = xy + env.scene.env_origins[env_ids, :2]
        state[:, 7:] = 0
        obj.write_root_state_to_sim(state, env_ids=env_ids)
        offsets[name] = offset.cpu().tolist()
    env.last_plate_spawn_offsets = offsets
    env.last_plate_spawn_xy = {name: all_xy[:, index].cpu().tolist() for index, name in enumerate(names)}
    env.last_plate_spatial_roles = {
        'right_bottom': 'plate_blue', 'left_bottom': 'plate_yellow', 'top': 'plate_pink'}
    # Same pre-motion settling as reference replay (0.75 s, PD-held robot).
    # Do not advance episode_length_buf or record fictitious pre-motion actions.
    robot = env.scene['robot']
    for _ in range(180):
        robot.set_joint_position_target(robot.data.default_joint_pos)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(env.cfg.sim.dt)


def rack_success(env):
    if getattr(env, '_rack_signal_tick', None) == env.common_step_counter:
        return env._rack_success
    if not hasattr(env, '_rack_stable_steps'):
        env._rack_stable_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    good = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    names = tuple(env.cfg.expected_slot_xyz)
    positions = torch.stack([env.scene[name].data.root_pos_w for name in names], dim=1)
    if env.cfg.expected_slot_centers_xyz:
        slots = torch.tensor(env.cfg.expected_slot_centers_xyz, device=env.device)
        slots = slots[None] + env.scene.env_origins[:, None]
        distance = torch.linalg.vector_norm(positions[:, :, None]-slots[:, None, :], dim=-1)
        assignments = torch.tensor(
            ((0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)), device=env.device)
        rows = torch.arange(3, device=env.device)
        matched = torch.stack([distance[:, rows, assignment].amax(dim=1) for assignment in assignments], dim=1)
        good &= matched.amin(dim=1) < .035
    for name in names:
        obj = env.scene[name]
        if not env.cfg.expected_slot_centers_xyz:
            target = torch.tensor(env.cfg.expected_slot_xyz[name], device=env.device) + env.scene.env_origins
            good &= torch.linalg.vector_norm(obj.data.root_pos_w-target, dim=-1) < .035
        normal = mu.matrix_from_quat(obj.data.root_quat_w)[:, :, 2]
        good &= normal[:, 2].abs() < .55
        good &= torch.linalg.vector_norm(obj.data.root_lin_vel_w, dim=-1) < .03
        good &= torch.linalg.vector_norm(obj.data.root_ang_vel_w, dim=-1) < .5
        for side in ('left', 'right'):
            tcp = env.scene[side+'_eef'].data.target_pos_w[:, 0]
            good &= torch.linalg.vector_norm(tcp-obj.data.root_pos_w, dim=-1) > .13
    env._rack_stable_steps = torch.where(good, env._rack_stable_steps+1, 0)
    env._rack_success = env._rack_stable_steps >= round(.5/env.step_dt)
    env._rack_signal_tick = env.common_step_counter
    return env._rack_success


@configclass
class BimanualScene(Task000901PlateHolderSceneCfg):
    left_eef = make_eef_frame('left')
    right_eef = make_eef_frame('right')


@configclass
class ReferenceEvents:
    reset_reference = EventTermCfg(func=reset_reference, mode='reset')


@configclass
class ReferenceTerminations:
    success = TerminationTermCfg(func=rack_success)


@configclass
class Task901BimanualSeedEnvCfg(Task000901PlateHolderEnvCfg, MimicEnvCfg):
    scene = BimanualScene(num_envs=1, env_spacing=5.)
    actions = FFWSG2JointPositionActionsCfg()
    observations = EpisodicShowroomObservationsCfg()
    events = ReferenceEvents()
    terminations = ReferenceTerminations()
    plate_xy_half_range_m: float = 0.
    plate_spawn_region: str = 'reference_offsets'
    expected_slot_xyz: dict = {}
    expected_slot_centers_xyz: list = []
    env_name: str = 'Cyclo-Real-WhiteRoom-Task000901-Bimanual-Mimic-Seed-FFW-SG2-v0'

    def __post_init__(self):
        super().__post_init__()
        path = REFERENCE/'provenance.json'
        if not path.exists():
            raise FileNotFoundError('First run task_000901/mimic/prepare_reference.py')
        provenance = json.loads(path.read_text())
        manifest = json.loads((REFERENCE/'input/input_manifest.json').read_text())
        import numpy as np
        data = np.load(REFERENCE/'input/aligned_replay.npz')
        self.expected_slot_xyz = provenance['expected_slot_xyz']
        self.expected_slot_centers_xyz = provenance.get('expected_slot_centers_xyz', [])
        self.scene.robot.init_state.joint_pos.update(
            {name: float(value) for name, value in zip(data['joint_names'].tolist(), data['reference_position'][0])})
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True
        profile = provenance.get('physics_profile')
        if profile:
            self.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = int(
                profile['tip_contact']['robot_solver_velocity_iterations'])
        self.scene.robot.spawn.func = spawn_reference_robot
        table_offset = float(provenance.get('table_height_offset_m', 0.0))
        if table_offset and (REFERENCE/'input/white_room_table_offset.usda').is_file():
            self.scene.environment.spawn.usd_path = str(REFERENCE/'input/white_room_table_offset.usda')
            holder = self.scene.plate_holder.init_state
            holder.pos = (*holder.pos[:2], holder.pos[2] + table_offset)
        for name, color in [('plate_blue', 'light_blue'), ('plate_pink', 'pastel_pink'), ('plate_yellow', 'lime_yellow')]:
            plate = getattr(self.scene, name)
            plate.init_state.pos = (*manifest['plate_initial_xy_m'][name], .7005 + table_offset)
            plate.spawn.usd_path = str(REFERENCE/f'input/plate_assets/{color}.usda')
            plate.spawn.func = spawn_reference_plate
        self.scene.plate_holder.spawn.func = spawn_reference_holder
        self.sim.dt = 1/240
        self.decimation = 16  # 15 Hz source archive, not invented 60 Hz contact data
        self.sim.render_interval = self.decimation
        self.episode_length_s = 100.
        # Pilot is physics/state only. Video rendering is a separate replay step.
        for name in ('cam_head', 'cam_wrist_left', 'cam_wrist_right'):
            setattr(self.scene, name, None)
            setattr(self.observations.policy, name, None)
        c = self.datagen_config
        c.name = f"task901_bimanual_{provenance.get('source_variant', 'episode6_tip_contact_left')}"
        c.generation_num_trials = 10
        c.generation_guarantee = False
        c.generation_keep_failed = True
        c.generation_select_src_per_subtask = False
        c.generation_select_src_per_arm = False
        c.generation_transform_first_robot_pose = False
        c.generation_interpolate_from_last_target_pose = True
        c.seed = 901
        self.subtask_configs = {
            side+'_arm': [SubTaskConfig(
                # The vendored core dereferences an object tensor before it
                # branches on REPLAY.  The fixed holder is only a device-bearing
                # placeholder for handover phases; REPLAY still returns identity
                # and applies no holder-relative transform.
                object_ref=ref if ref is not None else 'plate_holder',
                subtask_term_signal=name if i<len(PHASES)-1 else None,
                subtask_term_offset_range=(0,0), selection_strategy='random',
                action_noise=0., num_interpolation_steps=8 if i else 0,
                num_fixed_steps=0, apply_noise_during_interpolation=False)
                for i, (name, ref, _, _) in enumerate(PHASES)]
            for side in ('left', 'right')}
        self.task_constraint_configs = [SubTaskConstraintConfig(
            eef_subtask_constraint_tuple=[('left_arm', i), ('right_arm', i)],
            constraint_type=SubTaskConstraintType.COORDINATION,
            coordination_scheme=getattr(SubTaskConstraintCoordinationScheme, scheme),
            coordination_synchronize_start=True)
            for i, (_, _, _, scheme) in enumerate(PHASES)]

    def init_action_cfg(self, mode):
        if mode == 'mimic_ik':
            configure_ffw_sg2_mimic_ik_actions(self.actions, eef_offset_z_m=TCP_Z)
        elif mode != 'joint':
            raise ValueError(mode)


@configclass
class Task901BimanualGenerateEnvCfg(Task901BimanualSeedEnvCfg):
    # Deliberately small first pilot: independently randomize each plate ±1 cm.
    plate_xy_half_range_m: float = .01
    plate_spawn_region: str = 'apriltag_0_1_2_3'
    env_name: str = 'Cyclo-Real-WhiteRoom-Task000901-Bimanual-Mimic-Generate-FFW-SG2-v0'
