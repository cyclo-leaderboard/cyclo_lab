"""Task000901 SG2 scene registration while manipulation is being configured."""
from pathlib import Path
import importlib.util,sys
TASK_ID='000901'
ENVIRONMENT='white_room'
_ROOT=Path(__file__).resolve().parent

def register_environments():
    import gymnasium as gym
    contact_name='_cyclo_task000901_test4_contact'
    if contact_name not in sys.modules:
        spec=importlib.util.spec_from_file_location(contact_name,_ROOT/'test4_contact.py')
        module=importlib.util.module_from_spec(spec);sys.modules[contact_name]=module;spec.loader.exec_module(module)
    name='_cyclo_task000901_scene'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,_ROOT/'scene.py')
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    if 'Cyclo-Real-WhiteRoom-Task000901-Environment-v0' not in gym.registry:
        gym.register(id='Cyclo-Real-WhiteRoom-Task000901-Environment-v0',entry_point='isaaclab.envs:ManagerBasedRLEnv',kwargs={'env_cfg_entry_point':f'{name}:Task000901EnvCfg'},disable_env_checker=True)
    plate_holder_id='Cyclo-Real-WhiteRoom-Task000901-PlateHolder-FFW-SG2-v0'
    if plate_holder_id not in gym.registry:
        gym.register(
            id=plate_holder_id,
            entry_point='isaaclab.envs:ManagerBasedRLEnv',
            kwargs={'env_cfg_entry_point':f'{name}:Task000901PlateHolderEnvCfg'},
            disable_env_checker=True,
        )
    # Independent bimanual variants. Existing scene registrations stay untouched.
    package = '_cyclo_task000901_mimic'
    if package not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            package, _ROOT/'mimic/__init__.py',
            submodule_search_locations=[str(_ROOT/'mimic')])
        module = importlib.util.module_from_spec(spec)
        sys.modules[package] = module
        spec.loader.exec_module(module)
    for phase in ('Seed', 'Generate'):
        env_id = f'Cyclo-Real-WhiteRoom-Task000901-Bimanual-Mimic-{phase}-FFW-SG2-v0'
        if env_id not in gym.registry:
            gym.register(id=env_id, entry_point=f'{package}.adapter:Task901BimanualMimicEnv',
                kwargs={'env_cfg_entry_point': f'{package}.env_cfg:Task901Bimanual{phase}EnvCfg'},
                disable_env_checker=True)

def mimic_data_generator_class():
    import importlib
    return importlib.import_module('_cyclo_task000901_mimic.generator').Task901DataGenerator

def provenance():
    return {'id':TASK_ID,'environment':ENVIRONMENT,'source_task':'000400','table_top_z_m':0.69,'robot':'FFW-SG2','reference_robot':'1050','tag_family':'tag36h11','tag_detection_square_m':0.04,'tag_positions':'camera_estimated','status':'setup'}
