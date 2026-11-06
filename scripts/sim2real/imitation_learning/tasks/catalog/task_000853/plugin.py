"""Task000853 BG2: the Task000400 room with pegboard, hook and blue cup."""
import importlib
from pathlib import Path
TASK_ID='000853'
ENVIRONMENT='wall_room'
_ROOT=Path(__file__).resolve().parent

def register_environments():
    from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.catalog_runtime import mount_task_package, register_scene_dependency
    register_scene_dependency('white_room')
    assets='cyclo_lab.assets.environments'
    package='cyclo_lab.manager_based.manipulation.white_room.config.ffw_sg2'
    for task_id in ('000300','000400',TASK_ID):
        for module_name,relative in ((assets,'cyclo_lab/assets/environments'),(package,'cyclo_lab/manager_based/manipulation/white_room/config/ffw_sg2')):
            mount_task_package(task_id,module_name,relative,required=module_name!=assets or task_id!='000853')
    import gymnasium as gym
    gym.register(
        id='Cyclo-Real-WallRoom-Task000853-FFW-BG2-v0',
        entry_point='isaaclab.envs:ManagerBasedRLEnv',
        kwargs={'env_cfg_entry_point':f'{package}.task_000853:Task000853EnvCfg'},
        disable_env_checker=True,
    )
    gym.register(
        id='Cyclo-Real-WhiteRoom-Task000853-FFW-BG2-v0',
        entry_point='isaaclab.envs:ManagerBasedRLEnv',
        kwargs={'env_cfg_entry_point':f'{package}.task_000853:Task000853EnvCfg'},
        disable_env_checker=True,
    )

    for side in ('Left','Right'):
        for phase in ('Seed','Generate'):
            cls = 'Task000853MimicSeedEnvCfg' if (side,phase)==('Right','Seed') else f'Task000853{side}Mimic{phase}EnvCfg'
            gym.register(id=f'Cyclo-Real-WallRoom-Task000853-{side}-Mimic-{phase}-FFW-BG2-v0',
                entry_point=f'{package}.task_000853_mimic:Task000853MimicEnv',
                kwargs={'env_cfg_entry_point':f'{package}.task_000853_mimic:{cls}'},disable_env_checker=True)
            gym.register(id=f'Cyclo-Real-WhiteRoom-Task000853-{side}-Mimic-{phase}-FFW-BG2-v0',
                entry_point=f'{package}.task_000853_mimic:Task000853MimicEnv',
                kwargs={'env_cfg_entry_point':f'{package}.task_000853_mimic:{cls}'},disable_env_checker=True)

def mimic_data_generator_class():
    package='cyclo_lab.manager_based.manipulation.white_room.config.ffw_sg2'
    return importlib.import_module(f'{package}.task_000853_datagen').Task000853DataGenerator

def waypoint_robot_spec(environment):
    if environment!='Cyclo-Real-WallRoom-Task000853-FFW-BG2-v0':raise ValueError('Use Task000853 BG2 environment.')
    return {'urdf_path':str(_ROOT.parent/'task_000400/reference/robot1063_20261022/live1063.urdf'),'robot_contract_id':'ffw_bg2_1063_softgripper_fixed_base_19d_v1','robot_model':'FFW-BG2','gripper_type':'soft','eef_tcp_z_m':-.242}

def provenance():
    return {'id':TASK_ID,'environment':ENVIRONMENT,'source_task':'000400','robot_model':'FFW-BG2','objects':'measured pegboard, 100 mm hook, Daiso blue cup 10303251','reference':'1063 live cameras 2026-10-23','layout':'measured centered board; see layout.json; cup camera fit','success_criterion':'Mimic cup-to-hook: lifted, released, shaft through handle, withdrawn and supported for 0.5 s'}
