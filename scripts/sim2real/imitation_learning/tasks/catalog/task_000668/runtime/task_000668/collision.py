"""Robot/cabinet contact failure for Task000668."""

from functools import lru_cache

import torch


@lru_cache(maxsize=4)
def _robot_body_paths(usd_path: str) -> tuple[str, ...]:
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"Cannot inspect contact bodies: {usd_path}")
    root = stage.GetDefaultPrim().GetPath()
    paths = tuple(str(prim.GetPath().MakeRelativePath(root)) for prim in stage.Traverse()
                  if prim.HasAPI(UsdPhysics.RigidBodyAPI))
    if not paths:
        raise RuntimeError(f"No rigid bodies for cabinet contact monitoring: {usd_path}")
    return paths


def configure_cabinet_contact_sensors(cfg) -> None:
    """Use one sensor per body because filtered contacts are one-to-many."""
    from isaaclab.sensors import ContactSensorCfg

    cfg.scene.robot.spawn.activate_contact_sensors = True
    cfg.scene.lazy_sensor_update = False
    filters = [
        f"{{ENV_REGEX_NS}}/RobotisShowroom/robotis_showroom/{cabinet}"
        for cabinet in ("kolbjorn_cabinet_1", "kolbjorn_cabinet_02")
    ]
    names = []
    for path in _robot_body_paths(cfg.scene.robot.spawn.usd_path):
        name = "cabinet_contact_" + path.rsplit("/", 1)[-1]
        setattr(cfg.scene, name, ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{path}", update_period=0.0,
            history_length=cfg.decimation + 1, filter_prim_paths_expr=filters,
        ))
        names.append(name)
    cfg.cabinet_contact_sensor_names = tuple(names)


def reset_cabinet_collision(env, env_ids: torch.Tensor) -> None:
    if not hasattr(env, "_task000668_cabinet_collision"):
        env._task000668_cabinet_collision = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._task000668_cabinet_peak_force_n = torch.zeros(env.num_envs, device=env.device)
        env._task000668_contact_reset_step = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._task000668_cabinet_collision[env_ids] = False
    env._task000668_cabinet_peak_force_n[env_ids] = 0.0
    env._task000668_contact_reset_step[env_ids] = env._sim_step_counter


def cabinet_collision(env, force_threshold_n: float = 1.0e-6) -> torch.Tensor:
    """Latch any cabinet contact, including contact between control frames."""
    if not hasattr(env, "_task000668_cabinet_collision"):
        reset_cabinet_collision(env, torch.arange(env.num_envs, device=env.device))
    names = env.cfg.cabinet_contact_sensor_names
    if not names:
        raise RuntimeError("Task000668 cabinet contact monitoring is missing")
    for name in names:
        sensor = env.scene[name]
        forces = sensor.data.force_matrix_w_history
        if forces is None or forces.shape[-2] == 0:
            raise RuntimeError(f"Cabinet contact filters unresolved: {name}")
        if not bool(torch.isfinite(forces).all()):
            raise RuntimeError(f"Non-finite cabinet contacts: {name}")
        # reset()/forward() can expose the previous backend force snapshot.
        # Include only history samples produced by physics since this reset.
        fresh_steps = env._sim_step_counter - env._task000668_contact_reset_step
        valid = torch.arange(forces.shape[1], device=env.device)[None, :] < fresh_steps[:, None]
        norms = torch.linalg.vector_norm(forces, dim=-1)
        norms = norms * valid[:, :, None, None]
        peak = norms.reshape(env.num_envs, -1).amax(dim=1)
        env._task000668_cabinet_peak_force_n = torch.maximum(env._task000668_cabinet_peak_force_n, peak)
        env._task000668_cabinet_collision |= peak > force_threshold_n
    return env._task000668_cabinet_collision.clone()
