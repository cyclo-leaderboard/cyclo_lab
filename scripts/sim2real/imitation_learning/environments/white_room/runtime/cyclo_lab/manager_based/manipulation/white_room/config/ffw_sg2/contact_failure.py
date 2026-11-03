"""Latch robot/table and gripper/basket contact during Task000100 MimicGen."""

from functools import lru_cache

import torch


TABLE_COLLIDERS = ("Top", "LegFrontLeft", "LegBackLeft", "LegFrontRight", "LegBackRight")


@lru_cache(maxsize=4)
def _monitored_body_paths(usd_path: str) -> tuple[tuple[str, bool], ...]:
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"Cannot inspect Task000100 robot USD: {usd_path}")
    root = stage.GetDefaultPrim().GetPath()
    bodies = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        path = str(prim.GetPath().MakeRelativePath(root))
        is_gripper = "/left_gripper/" in path or "/right_gripper/" in path
        if "arm_l_link" in path or "arm_r_link" in path or is_gripper:
            bodies.append((path, is_gripper))
    if not bodies or not any(gripper for _, gripper in bodies):
        raise RuntimeError(f"Task000100 robot arm/gripper bodies missing: {usd_path}")
    return tuple(bodies)


def configure_contact_failure_sensors(cfg) -> None:
    """Filter table contact for both arms and basket contact for both grippers."""
    from isaaclab.sensors import ContactSensorCfg

    cfg.scene.robot.spawn.activate_contact_sensors = True
    cfg.scene.lazy_sensor_update = False
    table_filters = [
        f"{{ENV_REGEX_NS}}/WhiteRoom/WhiteTable/{name}"
        for name in TABLE_COLLIDERS
    ]
    basket_filter = "{ENV_REGEX_NS}/WhiteBasket"
    names = []
    gripper_names = []
    for body_path, is_gripper in _monitored_body_paths(cfg.scene.robot.spawn.usd_path):
        name = "task000100_contact_" + body_path.rsplit("/", 1)[-1]
        filters = table_filters + ([basket_filter] if is_gripper else [])
        setattr(cfg.scene, name, ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{body_path}",
            update_period=0.0,
            history_length=cfg.decimation + 1,
            filter_prim_paths_expr=filters,
        ))
        names.append(name)
        if is_gripper:
            gripper_names.append(name)
    cfg.task000100_contact_sensor_names = tuple(names)
    cfg.task000100_gripper_contact_sensor_names = tuple(gripper_names)


def reset_contact_failures(env, env_ids=None) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long).reshape(-1)
    if not hasattr(env, "_task000100_table_collision"):
        env._task000100_table_collision = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._task000100_gripper_basket_collision = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        env._task000100_table_peak_force_n = torch.zeros(env.num_envs, device=env.device)
        env._task000100_basket_peak_force_n = torch.zeros(env.num_envs, device=env.device)
        env._task000100_contact_reset_step = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
    env._task000100_table_collision[ids] = False
    env._task000100_gripper_basket_collision[ids] = False
    env._task000100_table_peak_force_n[ids] = 0.0
    env._task000100_basket_peak_force_n[ids] = 0.0
    env._task000100_contact_reset_step[ids] = env._sim_step_counter
    env._task000100_last_contact_sample_step = -1


def contact_failures(env, force_threshold_n: float = 1.0e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Return latched table and gripper/basket contacts from fresh physics steps."""
    if not hasattr(env, "_task000100_table_collision"):
        reset_contact_failures(env)
    if env._task000100_last_contact_sample_step == env._sim_step_counter:
        return env._task000100_table_collision.clone(), env._task000100_gripper_basket_collision.clone()
    names = env.cfg.task000100_contact_sensor_names
    if not names:
        raise RuntimeError("Task000100 contact sensors are not configured")
    gripper_names = set(env.cfg.task000100_gripper_contact_sensor_names)
    for name in names:
        forces = env.scene[name].data.force_matrix_w_history
        expected_filters = len(TABLE_COLLIDERS) + int(name in gripper_names)
        if forces is None or forces.shape[-2] != expected_filters:
            raise RuntimeError(f"Task000100 contact filters unresolved: {name}")
        if not bool(torch.isfinite(forces).all().item()):
            raise RuntimeError(f"Non-finite Task000100 contact forces: {name}")
        fresh_steps = env._sim_step_counter - env._task000100_contact_reset_step
        valid = torch.arange(forces.shape[1], device=env.device)[None, :] < fresh_steps[:, None]
        norms = torch.linalg.vector_norm(forces, dim=-1) * valid[:, :, None, None]
        table_peak = norms[..., :len(TABLE_COLLIDERS)].reshape(env.num_envs, -1).amax(dim=1)
        env._task000100_table_peak_force_n = torch.maximum(
            env._task000100_table_peak_force_n, table_peak
        )
        env._task000100_table_collision |= table_peak > force_threshold_n
        if name in gripper_names:
            basket_peak = norms[..., len(TABLE_COLLIDERS):].reshape(env.num_envs, -1).amax(dim=1)
            env._task000100_basket_peak_force_n = torch.maximum(
                env._task000100_basket_peak_force_n, basket_peak
            )
            env._task000100_gripper_basket_collision |= basket_peak > force_threshold_n
    env._task000100_last_contact_sample_step = env._sim_step_counter
    return env._task000100_table_collision.clone(), env._task000100_gripper_basket_collision.clone()


def invalid_contact(env) -> torch.Tensor:
    table, basket = contact_failures(env)
    return table | basket
