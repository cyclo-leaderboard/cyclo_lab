"""Task000200 seed schedule controller."""

from __future__ import annotations


class Factory:
    def __init__(self, args):
        self.args = args
        if args.robot_type != "FFW_SG2":
            raise ValueError("Task000200 recording requires --robot_type FFW_SG2.")
        if args.task000200_center_seed and args.num_demos != 1:
            raise ValueError("--task000200_center_seed requires --num_demos 1.")

    def configure_env_cfg(self, env_cfg):
        if not self.args.task000200_center_seed:
            return
        import torch
        from isaaclab.managers import EventTermCfg
        from cyclo_lab.assets.environments.white_room_task000200_layout import PLATE_BOX, TAPE_BOX

        def reset_center(env, env_ids):
            for name, box in (("yellow_plate", PLATE_BOX), ("blue_tape", TAPE_BOX)):
                obj = env.scene[name]
                pose = obj.data.default_root_state[env_ids, :7].clone()
                pose[:, :3] += env.scene.env_origins[env_ids]
                x_m, y_m = box.nominal_center()
                pose[:, 0] = env.scene.env_origins[env_ids, 0] + x_m
                pose[:, 1] = env.scene.env_origins[env_ids, 1] + y_m
                obj.write_root_pose_to_sim(pose, env_ids=env_ids)
                obj.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=obj.device, dtype=pose.dtype), env_ids=env_ids)

        env_cfg.events.set_task000200_seed_objects = EventTermCfg(func=reset_center, mode="reset")

    def create(self, env, teleop, **_kwargs):
        return Controller(self, env, teleop)


def build_recording_factory(args):
    return Factory(args)


class Controller:
    def __init__(self, factory, env, teleop):
        self.factory, self.env, self.teleop = factory, env, teleop
        from cyclo_lab.assets.environments.white_room_task000200_layout import task000200_tape_seed_positions
        self.seeds = () if factory.args.task000200_center_seed else task000200_tape_seed_positions()
        if self.seeds and factory.args.num_demos != len(self.seeds):
            raise ValueError(f"Task000200 seed capture requires --num_demos {len(self.seeds)}.")
        env._task000200_seed_index = 0

    def register_callbacks(self, **_kwargs): pass
    def validate_save(self): return True
    def after_initial_reset(self): pass
    def on_recording_started(self): pass
    def on_environment_reset(self): pass
    def record_step_metadata(self, *_args): pass
    def before_shutdown(self): pass
    def apply_actions(self, actions, **_kwargs): return actions

    def advance_seed(self):
        if self.seeds and self.env._task000200_seed_index + 1 < len(self.seeds):
            self.env._task000200_seed_index += 1
            print(f"[Task000200] Next seed: {self.seeds[self.env._task000200_seed_index].label}")

    def dataset_metadata(self):
        from cyclo_lab.assets.environments.white_room_task000200_layout import PLATE_BOX, TAPE_BOX
        center = self.factory.args.task000200_center_seed
        return {
            "seed_grid_id": "task000200_tape_center_plate_center_v1" if center else "task000200_tape_safe_corners_plate_center_v1",
            "seed_grid_position_count": 1 if center else len(self.seeds),
            "seed_capture_position_count": 1 if center else len(self.seeds),
            "seed_capture_ids": "CENTER" if center else ",".join(seed.label for seed in self.seeds),
            "target_side": "right",
            "plate_world_x_m": PLATE_BOX.nominal_center()[0],
            "plate_world_y_m": PLATE_BOX.nominal_center()[1],
            "plate_box_size_m": PLATE_BOX.x_max - PLATE_BOX.x_min,
            "tape_box_size_m": TAPE_BOX.x_max - TAPE_BOX.x_min,
        }

    def episode_metadata(self):
        if self.factory.args.task000200_center_seed:
            return {"seed_position_id": "CENTER", "seed_position_index": 0}
        seed = self.seeds[self.env._task000200_seed_index]
        return {"seed_position_id": seed.label, "seed_position_index": self.env._task000200_seed_index, "tape_world_x_m": seed.x_m, "tape_world_y_m": seed.y_m}
