"""Fixed-grid WhiteRoom seed controller shared by Task000100/000300 copies."""

from __future__ import annotations

class Factory:
    def __init__(self, args, task_id):
        self.args, self.task_id = args, task_id
        raw = getattr(args, f"task{task_id}_seed_ids", None)
        self.requested = tuple(value.strip().upper() for value in raw.split(",")) if raw else ()
        if len(set(self.requested)) != len(self.requested) or any(not value for value in self.requested):
            raise ValueError(f"--task{task_id}_seed_ids requires unique nonempty IDs.")
        expected = len(self.requested) if self.requested else args.num_demos
        if self.requested and args.num_demos != expected:
            raise ValueError(f"--task{task_id}_seed_ids requires --num_demos {expected}.")

    def configure_env_cfg(self, env_cfg):
        from cyclo_lab.manager_based.manipulation.white_room.config.ffw_sg2.contact_failure import configure_contact_failure_sensors
        if not getattr(env_cfg, "task000100_contact_sensor_names", ()):
            configure_contact_failure_sensors(env_cfg)

    def create(self, env, teleop, **_kwargs): return Controller(self, env, teleop)

def build_recording_factory(args, task_id): return Factory(args, task_id)

class Controller:
    def __init__(self, factory, env, teleop):
        self.factory, self.env, self.teleop = factory, env, teleop
        from cyclo_lab.assets.environments.white_room_layout import task000100_seed_positions
        self.positions = task000100_seed_positions(side=env.cfg.seed_side)
        by_label = {seed.label: index for index, seed in enumerate(self.positions)}
        unknown = set(factory.requested) - by_label.keys()
        if unknown: raise ValueError(f"Unknown Task{factory.task_id} seed IDs: {sorted(unknown)}")
        self.schedule = tuple(by_label[label] for label in factory.requested) if factory.requested else tuple(range(len(self.positions)))
        if factory.args.num_demos != len(self.schedule):
            raise ValueError(f"Task{factory.task_id} seed capture requires --num_demos {len(self.schedule)}.")
        self.cursor = 0
        env._task000100_seed_index = self.schedule[0]

    def register_callbacks(self, **_kwargs): pass
    def after_initial_reset(self): self._reset_contacts()
    def on_recording_started(self): self._reset_contacts()
    def on_environment_reset(self): self._reset_contacts()
    def before_shutdown(self): pass
    def apply_actions(self, actions, **_kwargs): return actions
    def record_step_metadata(self, *_args):
        from cyclo_lab.manager_based.manipulation.white_room.config.ffw_sg2.contact_failure import contact_failures
        contact_failures(self.env)
    def _reset_contacts(self):
        import torch
        from cyclo_lab.manager_based.manipulation.white_room.config.ffw_sg2.contact_failure import reset_contact_failures
        reset_contact_failures(self.env, torch.arange(self.env.num_envs, device=self.env.device))
    def validate_save(self):
        from cyclo_lab.manager_based.manipulation.white_room.config.ffw_sg2.contact_failure import contact_failures
        table, destination = contact_failures(self.env)
        if bool(table.any().item()) or bool(destination.any().item()):
            print(f"[Task{self.factory.task_id}] Save refused by contact gate.")
            return False
        return True
    def advance_seed(self):
        if self.cursor + 1 < len(self.schedule):
            self.cursor += 1; self.env._task000100_seed_index = self.schedule[self.cursor]
            print(f"[Task{self.factory.task_id}] Next seed: {self.positions[self.schedule[self.cursor]].label}")
    def dataset_metadata(self):
        return {"seed_grid_id": f"white_room_3x3_{self.env.cfg.seed_side}_v1", "seed_grid_position_count": len(self.positions), "seed_capture_position_count": len(self.schedule), "seed_capture_ids": ",".join(self.positions[index].label for index in self.schedule), "target_side": self.env.cfg.seed_side}
    def episode_metadata(self):
        index = self.schedule[self.cursor]; seed = self.positions[index]
        return {"seed_position_index": index, "seed_position_id": seed.label, "object_world_x_m": seed.x_m, "object_world_y_m": seed.y_m}
