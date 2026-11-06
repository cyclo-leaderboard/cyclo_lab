# Task000525

Task000525 moves the orange coffee can from `kolbjorn_cabinet_02` to the
fixed ivory mat on `central_dining_set`. Regions A/B use the left arm and
regions C/D use the right arm for the complete episode.

## Runtime contract

- Environment: `Cyclo-Real-Showroom-Task000525-FFW-SG2-v0`
- Randomized environment: `Cyclo-Real-Showroom-Task000525-Random-FFW-SG2-v0`
- Robot: `FFW_SG2_softgripper.usd`
- Target: `coffee_can_orange`
- Policy frequency: 15 Hz
- Policy state and action: joint position 19 + body-frame
  `(linear_x, linear_y, angular_z)` = 22 values
- Default reset: deterministic

The three base values are part of the canonical policy contract. A dataset
that contains only lateral velocity is not a Task000525 policy dataset.

## Code map

| Responsibility | File |
| --- | --- |
| identity, target, cameras, and rates | [spec.py](runtime/cyclo_lab/manager_based/manipulation/showroom/config/ffw_sg2/tasks/task_000525/spec.py) |
| region-to-object and region-to-arm rules | [arrangement.py](runtime/cyclo_lab/manager_based/manipulation/showroom/config/ffw_sg2/tasks/task_000525/arrangement.py) |
| measured shelf geometry | [layout.py](runtime/cyclo_lab/manager_based/manipulation/showroom/config/ffw_sg2/tasks/task_000525/layout.py) |
| randomization profiles | [profiles.py](runtime/cyclo_lab/manager_based/manipulation/showroom/config/ffw_sg2/tasks/task_000525/profiles.py) |
| environment and reset wiring | [env_cfg.py](runtime/cyclo_lab/manager_based/manipulation/showroom/config/ffw_sg2/tasks/task_000525/env_cfg.py) |
| learned-policy schema | [policy.py](runtime/cyclo_lab/manager_based/manipulation/showroom/config/ffw_sg2/tasks/task_000525/policy.py) |
| task-owned recording lifecycle | [recording.py](runtime/cyclo_lab/manager_based/manipulation/showroom/config/ffw_sg2/tasks/task_000525/recording.py) |
| live route controller | [online_dijkstra.py](runtime/cyclo_lab/manager_based/manipulation/showroom/config/ffw_sg2/tasks/task_000525/online_dijkstra.py) |
| SDG adapter geometry | [locomanipulation_sdg_contract.py](runtime/cyclo_lab/manager_based/manipulation/showroom/config/ffw_sg2/tasks/task_000525/locomanipulation_sdg_contract.py) |
| SDG environment adapter | [locomanipulation_sdg_env.py](runtime/cyclo_lab/manager_based/manipulation/showroom/config/ffw_sg2/tasks/task_000525/locomanipulation_sdg_env.py) |
| appearance-only randomization | [appearance_events.py](runtime/cyclo_lab/manager_based/manipulation/showroom/config/ffw_sg2/tasks/task_000525/appearance_events.py) |
| fixed destination mat | [destination_mat.py](runtime/cyclo_lab/manager_based/manipulation/showroom/config/ffw_sg2/tasks/task_000525/destination_mat.py) |
| Task525-only arm hold tuning | [robot_stability.py](runtime/cyclo_lab/manager_based/manipulation/showroom/config/ffw_sg2/tasks/task_000525/robot_stability.py) |

The generic recorder discovers Task525 through the shared task recorder
registry. Task-specific CLI validation, metadata, callbacks, holds, and
shutdown behavior stay in `recording.py`; the generic recorder does not branch
on the Task525 ID.

## Recording

The primary launcher uses the online Dijkstra controller after the operator
marks the completed pick/carry transition with `G`:

```bash
./scripts/sim2real/imitation_learning/record_task000525_mobile_demo.sh \
  --task525_seed_regions A,B --num_demos 2 \
  --dataset_file /workspace/cyclo_lab/datasets/task_000525_orange_seed_ab.hdf5 \
  --camera_view operator --render_episode_cameras
```

Use `C,D` for a right-arm session. One session must not mix left- and
right-arm regions.

The manual-base launcher is the safety fallback. It keeps the same 22D
dataset contract while accepting base commands from the foot pad or keyboard:

```bash
./scripts/sim2real/imitation_learning/record_task000525_manual_base_demo.sh \
  --task525_seed_regions A,B --num_demos 2 \
  --dataset_file /workspace/cyclo_lab/datasets/task_000525_manual_ab.hdf5
```

The controller refuses to save incomplete marker sequences. Automatic mode
also refuses to save until navigation has completed. Manual mode refuses the
place marker while the base is moving.

## Offline generation and policy staging

The retained trajectory generator is task-local and uses the registered
`Cyclo-Real-Showroom-Task000525-Trajectory-Generation-FFW-SG2-v0`
environment:

```bash
./scripts/sim2real/imitation_learning/run_task000525_trajectory_generation.sh \
  --dataset /workspace/cyclo_lab/datasets/task_000525_orange_seed_abcd.hdf5 \
  --output_file /workspace/cyclo_lab/datasets/task_000525_generated.hdf5 \
  --acceptance_scope all --num_runs 200 --max_attempts 1000 \
  --device cpu --headless --enable_cameras
```

Task-local export, visual replay, and staging audit tools live in
[`scripts/`](scripts/).
The repository-wide LeRobot v3 audit is
[`tasks/common/audit_lerobot_v30.py`](../../common/audit_lerobot_v30.py).

## Validation

Manual Isaac checks are grouped under
[`validation/task_000525`](../../../validation/task_000525):

- `validate_mobile.py`: record/replay all three body velocity directions;
- `validate_online_dijkstra.py`: record and inspect the live route segment;
- `validate_dijkstra_path.py`: inspect the planned path and docking endpoint;
- `validate_sdg_adapter.py`: smoke-test the retained SDG adapter.

These are explicit integration programs, not default unit tests. Unit tests
cover the pure layout, 22D policy schema, camera shapes, arm stability tuning,
carry-distance rule, policy staging, and recording-controller state changes.

## Safety invariants

- Appearance augmentation may change only existing visual transforms or
  material bindings; it must not write rigid pose, velocity, collision, or
  mass state.
- The orange target is never included in distractor label permutation.
- Automatic base motion holds the manipulation action while navigating and
  fails closed on planning or timeout errors.
- Task525 arm tuning changes only its arm actuators; lift, base, shared SG2,
  and other tasks keep their existing gains.
