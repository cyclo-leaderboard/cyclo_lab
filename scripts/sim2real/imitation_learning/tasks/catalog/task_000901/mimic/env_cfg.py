"""Isolated 901 bimanual variants matching episode6 LEFT tip-contact physics."""
import json
from pathlib import Path

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

REFERENCE = Path(__file__).parent/'output/reference'


def spawn_reference_robot(path, cfg, *args, **kwargs):
    from cyclo_lab.assets.robots.FFW_SG2 import spawn_sg2_with_softgripper_physics
    result = spawn_sg2_with_softgripper_physics(path, cfg, *args, **kwargs)
    stage = result.GetStage()
    material = stage.GetPrimAtPath(str(result.GetPath())+'/distalGripperPhysicsMaterial')
    if not material or not material.HasAPI(UsdPhysics.MaterialAPI):
        raise ValueError('Expected original SG2 real-tip physics material')
    api = UsdPhysics.MaterialAPI(material)
    api.CreateStaticFrictionAttr().Set(4.)
    api.CreateDynamicFrictionAttr().Set(3.5)
    physx = PhysxSchema.PhysxMaterialAPI.Apply(material)
    physx.CreateCompliantContactStiffnessAttr().Set(0.)
    physx.CreateCompliantContactDampingAttr().Set(0.)
    physx.CreateCompliantContactAccelerationSpringAttr().Set(False)
    removed = 0
    for child in Usd.PrimRange(result):
        if '_cyclo_softgripper_proxy' in str(child.GetPath()) and child.HasAPI(UsdPhysics.FilteredPairsAPI):
            relation = UsdPhysics.FilteredPairsAPI(child).GetFilteredPairsRel()
            targets = list(relation.GetTargets())
            if not all('_cyclo_softgripper_proxy' in str(t) for t in targets):
                raise ValueError('Unexpected non-tip collision exclusion')
            removed += len(targets)
            relation.SetTargets([])
        if child.GetName() in {f'gripper_{s}_joint{j}' for s in 'lr' for j in (2,4)}:
            # Explicitly retain LEFT variant; no compliant-joint24 substitution.
            import math
            child.GetAttribute('physics:upperLimit').Set(math.degrees(1.1))
            child.GetAttribute('physxMimicJoint:rotX:naturalFrequency').Set(1500.)
            child.GetAttribute('physxMimicJoint:rotX:dampingRatio').Set(5.)
    if removed != 50:
        raise ValueError(f'Expected 50 opposing-tip exclusions, got {removed}')
    return result


def spawn_reference_plate(path, cfg, *args, **kwargs):
    from _cyclo_task000901_scene import spawn_task000901_plate
    result = spawn_task000901_plate(path, cfg, *args, **kwargs)
    material = result.GetStage().GetPrimAtPath(str(result.GetPath())+'/PlatePhysicsMaterial')
    api = UsdPhysics.MaterialAPI(material)
    api.CreateStaticFrictionAttr().Set(.45)
    api.CreateDynamicFrictionAttr().Set(.30)
    PhysxSchema.PhysxMaterialAPI.Apply(material).CreateFrictionCombineModeAttr().Set('min')
    return result


def spawn_reference_holder(path, cfg, *args, **kwargs):
    from isaaclab.sim.spawners.from_files import from_files
    result = from_files.spawn_from_usd(path, cfg, *args, **kwargs)
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
    bounds = float(env.cfg.plate_xy_half_range_m)
    names = ('plate_blue', 'plate_yellow', 'plate_pink')
    base_xy = torch.stack([env.scene[n].data.default_root_state[env_ids, :2] for n in names], dim=1)
    # Reference plates are 22 cm across. Rejection sampling prevents random
    # placements from starting with overlapping rims; Z/yaw remain unchanged.
    all_offsets = torch.zeros_like(base_xy)
    pending = torch.ones(len(env_ids), dtype=torch.bool, device=env.device)
    for _ in range(1000):
        all_offsets[pending] = (torch.rand((int(pending.sum()), 3, 2), device=env.device)*2-1)*bounds
        xy = base_xy + all_offsets
        valid = torch.ones(len(env_ids), dtype=torch.bool, device=env.device)
        for i, j in ((0,1),(0,2),(1,2)):
            valid &= torch.linalg.vector_norm(xy[:,i]-xy[:,j], dim=-1) >= .221
        pending &= ~valid
        if not bool(pending.any()):
            break
    if bool(pending.any()):
        raise ValueError('Could not sample non-overlapping plate positions')
    offsets = {}
    for index, name in enumerate(names):
        obj = env.scene[name]
        state = obj.data.default_root_state[env_ids].clone()
        state[:, :3] += env.scene.env_origins[env_ids]
        offset = all_offsets[:, index]
        state[:, :2] += offset
        state[:, 7:] = 0
        obj.write_root_state_to_sim(state, env_ids=env_ids)
        offsets[name] = offset.cpu().tolist()
    env.last_plate_spawn_offsets = offsets
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
    for name, xyz in env.cfg.expected_slot_xyz.items():
        obj = env.scene[name]
        target = torch.tensor(xyz, device=env.device) + env.scene.env_origins
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
    expected_slot_xyz: dict = {}
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
        self.scene.robot.init_state.joint_pos.update(
            {name: float(value) for name, value in zip(data['joint_names'].tolist(), data['reference_position'][0])})
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True
        self.scene.robot.spawn.func = spawn_reference_robot
        for name, color in [('plate_blue', 'light_blue'), ('plate_pink', 'pastel_pink'), ('plate_yellow', 'lime_yellow')]:
            plate = getattr(self.scene, name)
            plate.init_state.pos = (*manifest['plate_initial_xy_m'][name], .7005)
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
        c.name = 'task901_bimanual_episode6_tip_contact_left'
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
    env_name: str = 'Cyclo-Real-WhiteRoom-Task000901-Bimanual-Mimic-Generate-FFW-SG2-v0'
