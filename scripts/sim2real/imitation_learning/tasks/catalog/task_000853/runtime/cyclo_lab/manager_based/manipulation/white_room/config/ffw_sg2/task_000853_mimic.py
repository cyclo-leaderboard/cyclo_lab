"""BG2 cup-to-hook Mimic: cup approach, hook placement, and withdrawal."""
import importlib.util
import math
import numpy as np
import torch
from isaaclab.envs import mdp
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, TerminationTermCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as mu
from cyclo_lab.manager_based.manipulation.common import FFWSG2MimicEnv, configure_ffw_sg2_mimic_ik_actions
from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.common import EpisodicShowroomObservationsCfg
from .task_000853 import Task000853EnvCfg, ROOT, reset_cup, SOFT_GRIPPER_TCP_Z_M

_spec=importlib.util.spec_from_file_location('task853_geometry', ROOT/'mimic_geometry.py')
geometry=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(geometry)


def randomize_cup(env, env_ids):
    reset_cup(env,env_ids)
    if not env.cfg.randomize_cup_spawn:
        return
    obj=env.scene['blue_cup'];state=obj.data.default_root_state[env_ids].clone()
    state[:,:3]+=env.scene.env_origins[env_ids]
    n=len(state);bounds=env.cfg.cup_spawn_xy_half_range_m
    state[:,:2]+=(torch.rand((n,2),device=env.device)*2-1)*torch.tensor(bounds,device=env.device)
    yaw=math.radians(env.cfg.cup_spawn_yaw_base_deg)+(torch.rand(n,device=env.device)*2-1)*math.radians(env.cfg.cup_spawn_yaw_deg)
    zero=torch.zeros_like(yaw);state[:,3:7]=mu.quat_from_euler_xyz(zero,zero,yaw)
    obj.write_root_state_to_sim(state,env_ids=env_ids)


def signals(env):
    # Observation and termination managers may call this repeatedly in one step.
    tick=env.common_step_counter
    if getattr(env,'_cup_signal_tick',None)==tick:
        return env._cup_signals
    if not hasattr(env,'_cup_lifted'):
        env._cup_lifted=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
        env._cup_released=env._cup_lifted.clone();env._cup_stable=torch.zeros(env.num_envs,dtype=torch.long,device=env.device)
    reset=env.episode_length_buf==0
    env._cup_lifted[reset]=False;env._cup_released[reset]=False;env._cup_stable[reset]=0
    cup=env.scene['blue_cup'];hook=env.scene['pegboard_hook'];robot=env.scene['robot']
    grip=robot.data.joint_pos[:,robot.joint_names.index(f'gripper_{env.cfg.mimic_side[0]}_joint1')]
    lifted=(cup.data.root_pos_w[:,2]-env.scene.env_origins[:,2]>.775)&(grip>=.5)
    env._cup_lifted|=lifted
    threaded=torch.as_tensor(geometry.threaded_on_hook(cup.data.root_state_w[:,:7].detach().cpu().numpy(),hook.data.root_state_w[:,:7].detach().cpu().numpy()),device=env.device)
    released=threaded&(grip<=.2)&env._cup_lifted
    env._cup_released|=released
    # Both EEFs must have withdrawn; a gripper must not be supporting the cup.
    clear=torch.ones(env.num_envs,dtype=torch.bool,device=env.device)
    for side in ('left','right'):
        pos=env.scene[f'{side}_eef'].data.target_pos_w[:,0]
        clear&=torch.linalg.vector_norm(pos-cup.data.root_pos_w,dim=-1)>.12
    # A freely hanging cup may swing; persistent threading proves support.
    stable=released&clear
    env._cup_stable=torch.where(stable,env._cup_stable+1,0)
    success=env._cup_stable>=math.ceil(.5/env.step_dt)
    env._cup_signals={'cup_lifted':env._cup_lifted.clone(),'released_on_hook':env._cup_released.clone(),'success':success}
    env._cup_signal_tick=tick
    return env._cup_signals


def cup_lifted(env):return signals(env)['cup_lifted']
def released_on_hook(env):return signals(env)['released_on_hook']
def task_success(env):return signals(env)['success']


@configclass
class CupObservations(EpisodicShowroomObservationsCfg):
    @configclass
    class Subtasks(ObservationGroupCfg):
        cup_lifted=ObservationTermCfg(func=cup_lifted)
        released_on_hook=ObservationTermCfg(func=released_on_hook)
        def __post_init__(self):
            self.enable_corruption=False;self.concatenate_terms=False
    subtask_terms:Subtasks=Subtasks()


@configclass
class CupTerminations:
    time_out=TerminationTermCfg(func=mdp.time_out,time_out=True)
    success=TerminationTermCfg(func=task_success)


@configclass
class Task000853MimicSeedEnvCfg(Task000853EnvCfg,MimicEnvCfg):
    mimic_side:str='right'
    randomize_cup_spawn:bool=False
    cup_spawn_xy_half_range_m:tuple=(.075,.175)
    cup_spawn_yaw_deg:float=0.
    cup_spawn_yaw_base_deg:float=0.
    observations:CupObservations=CupObservations()
    terminations:CupTerminations=CupTerminations()
    def __post_init__(self):
        super().__post_init__()
        self.env_name=f'Cyclo-Real-WallRoom-Task000853-{self.mimic_side.title()}-Mimic-{"Generate" if self.randomize_cup_spawn else "Seed"}-FFW-BG2-v0'
        self.control_hz=60.;self.decimation=round(1/self.sim.dt/60.)
        self.scene.replicate_physics=True
        if self.randomize_cup_spawn:
            # Keep the board-side edge; add 5 cm toward the robot and 5 cm on each side.
            # World X [-0.64212, -0.49212], Y [-0.21, 0.14] at the current board pose.
            board=self.scene.pegboard.init_state.pos
            self.scene.blue_cup.init_state.pos=(board[0]+.225,board[1]+.025,.735)
        self.events.reset_cup.func=randomize_cup
        self.task_instruction='Pick up the blue cup and hang its handle on the pegboard hook, release and withdraw.'
        c=self.datagen_config;c.name='task000853_BG2_cup_hook'
        c.generation_guarantee=False;c.generation_keep_failed=True;c.generation_num_trials=10
        c.generation_select_src_per_subtask=False;c.generation_transform_first_robot_pose=False
        c.generation_interpolate_from_last_target_pose=False;c.seed=853
        self.subtask_configs={f'{self.mimic_side}_arm':[
            SubTaskConfig(object_ref=obj,subtask_term_signal=signal,subtask_term_offset_range=offset,
                selection_strategy='nearest_neighbor_object' if obj else 'random',selection_strategy_kwargs={'nn_k':1} if obj else {},
                action_noise=0.,num_interpolation_steps=interpolation_steps,num_fixed_steps=0,apply_noise_during_interpolation=False)
            for obj,signal,offset,interpolation_steps in [
                # The lift marker occurs halfway through the source lift. Finish
                # wrist tilting before switching from cup-relative to hook-relative poses.
                ('blue_cup','cup_lifted',(150,150),10),
                # A spawned cup may be 10+ cm from the source cup. Move that
                # handoff over 1.5 s instead of a 0.17 s lateral wrist jump.
                ('pegboard_hook','released_on_hook',(0,0),90),
                (None,None,(0,0),10),
            ]]}
    def init_action_cfg(self,mode):
        if mode=='mimic_ik':
            configure_ffw_sg2_mimic_ik_actions(self.actions,joint_hold_sides=('l' if self.mimic_side=='right' else 'r',),eef_offset_z_m=SOFT_GRIPPER_TCP_Z_M)
        else:super().init_action_cfg(mode)


@configclass
class Task000853RightMimicGenerateEnvCfg(Task000853MimicSeedEnvCfg):
    randomize_cup_spawn:bool=True

@configclass
class Task000853LeftMimicSeedEnvCfg(Task000853MimicSeedEnvCfg):
    mimic_side:str='left'

@configclass
class Task000853LeftMimicGenerateEnvCfg(Task000853LeftMimicSeedEnvCfg):
    randomize_cup_spawn:bool=True
    cup_spawn_xy_half_range_m:tuple=(.075,.075)
    cup_spawn_yaw_base_deg:float=180.
    def __post_init__(self):
        super().__post_init__()
        # Left side of the fixed pegboard centre Y=-0.060 m: [-0.210,-0.060].
        # The mirrored successful seed has the handle reversed (yaw 180 deg).
        board=self.scene.pegboard.init_state.pos
        self.scene.blue_cup.init_state.pos=(board[0]+.225,board[1]-.075,.735)


class Task000853MimicEnv(FFWSG2MimicEnv):
    def get_object_poses(self,env_ids=None):
        ids=slice(None) if env_ids is None else env_ids;r=self.scene['robot'];out={}
        for name in ('blue_cup','pegboard_hook','pegboard'):
            obj=self.scene[name]
            pos,quat=mu.subtract_frame_transforms(r.data.root_pos_w[ids],r.data.root_quat_w[ids],obj.data.root_pos_w[ids],obj.data.root_quat_w[ids])
            out[name]=mu.make_pose(pos,mu.matrix_from_quat(quat))
        return out
    def _reset_idx(self,env_ids):
        if hasattr(self,'_cup_lifted'):
            self._cup_lifted[env_ids]=False;self._cup_released[env_ids]=False;self._cup_stable[env_ids]=0
        self._cup_signal_tick=None
        super()._reset_idx(env_ids)
    def get_generation_success(self,env_id,success_term):return bool(task_success(self)[env_id])
    def update_generation_success_tracking(self,env_id,success_term):return self.get_generation_success(env_id,success_term)
