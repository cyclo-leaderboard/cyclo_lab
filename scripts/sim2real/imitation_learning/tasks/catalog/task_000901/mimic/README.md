# Task000901 bimanual Mimic pilot

This is an additive Task000901 variant. Neither `scene.py`, shared SG2 code,
the shared robot USD, nor `task_901_haeun` is modified by this integration.

## Reference identity

The user-selected LEFT video is episode 6 / `original_joint_limits` in:

`/home/robotis-ai/agents/experiments/2026-11-03_09-56-43_task000901_real_sim_replay/insert_plate_real2sim/tip_self_contact_joint24_comparison`

Its inputs are `episode_006/input/aligned_replay.npz` and
`episode_006/original_joint_limits/render/{physics_states,gripper_kinematics}.npz`.
The original bag is `Task_000901_Insert_Plate_Real2Sim_MCAP/6`, MCAP SHA256
`0b237dec8f432b6716b1a0acb6f1596cdadad81efa6572772912da7ab5b5dd67`.
Original teleoperation actions were unavailable in that experiment. The video
uses **measured joint positions as PD targets**. Our seed derives its Cartesian
poses from the resulting simulated motion, preserving both measured gripper
targets. It must not be described as recovered original action data.

The source user approved has all three plates in the holder. This approval is
separate from the success checks for *newly generated* episodes.

## Contact and control contract

- SG2 real-tip USD fingerprint checked against reference before running.
- Self collision enabled; 50 opposing soft-tip proxy exclusions cleared.
- Distal tip friction 4.0 / 3.5, rigid contact, no recess/compliance.
- Joint 2/4: upper limit 1.1 rad, mimic natural frequency/damping 1500/5.
  This is **not** the video's RIGHT compliant-joint24 variant.
- Source-local plates: 15 g, 1152 collision pieces, friction 0.45 / 0.30.
- Holder friction 0.25 / 0.20. Plate/holder combine mode `min`.
- Both observations and IK use TCP `(0, 0, -0.2)` relative to each `arm_*_link7`.
  This is a consistent mathematical frame, not a claim of measured fingertip TCP.
- Robot-root frame, WXYZ quaternions; EEF19 and joint19 have different meanings.
- 240 Hz physics; source-derived Cartesian seed is 15 Hz. Head/lift held at
  reference initial targets (their source variation is <0.002 rad / 0.00001 m).

## Generation

Uses the existing `isaaclab_mimic.datagen.data_generator.DataGenerator` unchanged.
The local adapter handles **both** TCPs/grippers. Nine object-centric phases use
paired coordination constraints; handovers use shared-source REPLAY and other
segments use shared TRANSFORM. Phase cuts are provisional, inferred from the
source gripper/lift timelines, not an automatic proof of correct contact phases.
Every source segment retains its two-hand timing. Connections add equal-length
interpolation to both hands. The first pilot randomizes each plate XY by ±1 cm,
with no orientation/visual/friction randomization. Wider distributions require
overlap, reachability and quality preflight checks before use.

New success requires all three plates near their source rack slots, upright,
slow, hands withdrawn, for 0.5 s. Final success is not latched through later drops.
No plate poses are written after reset, and no grasp attachments are used.

## Commands

### Manually approved MCAP episode (default: Test4)

After reviewing a completed Test4 real/sim replay and confirming that the task
succeeded, build a Mimic seed with:

```bash
mcap2hdf5901 \
  /home/robotis-ai/cyclo_intelligence/docker/workspace/rosbag2/Task_000901_Insert_Plate_Real2Sim_MCAP/EPISODE \
  --approved-success
```

The default is the Test4 rigid-contact profile.  The seed directory contains
`seed.hdf5`, `camera_real_sim.mp4`, the exact contact candidate, source replay evidence, and
`provenance.json`.  The latter makes the selected Test4 physics profile apply
again when `TASK901_MIMIC_REFERENCE` points to this seed directory.  HDF5 itself
does not encode Isaac Sim materials.  Use `--dry-run` first to resolve the source
without creating or approving a seed.  `--variant test5` remains an explicit
comparison override only.

Phase cuts are duration-scaled from the canonical episode and must be reviewed
before augmentation; pass corrected boundaries with `--phase-cuts` when needed.

### Canonical episode-6 reference

Prepare once on the host (numpy+h5py interpreter), refuses overwrite:

```bash
/home/robotis-ai/.venvs/robotis-r2/bin/python \
  scripts/sim2real/imitation_learning/tasks/catalog/task_000901/mimic/prepare_reference.py
```

Inside `cyclo_lab` container, from `/workspace/cyclo_lab`:

```bash
/isaac-sim/python.sh scripts/sim2real/imitation_learning/tasks/catalog/task_000901/mimic/run_pilot.py \
  --headless --mode joint-reference \
  --output scripts/sim2real/imitation_learning/tasks/catalog/task_000901/mimic/output/new_nominal_joint

/isaac-sim/python.sh scripts/sim2real/imitation_learning/tasks/catalog/task_000901/mimic/run_pilot.py \
  --headless --mode mimic --trials 1 --xy-range 0 \
  --output scripts/sim2real/imitation_learning/tasks/catalog/task_000901/mimic/output/new_nominal_mimic

/isaac-sim/python.sh scripts/sim2real/imitation_learning/tasks/catalog/task_000901/mimic/run_pilot.py \
  --headless --mode mimic --trials 10 --xy-range 0.01 --seed 901 \
  --output scripts/sim2real/imitation_learning/tasks/catalog/task_000901/mimic/output/new_random10
```

Do not proceed to random generation until the nominal Mimic replay is acceptable.
10 means attempts, not 10 guaranteed successes. All scripts are finite and close
their own simulator. No ROS/Zenoh bridge, real robot connection, or source writes.
State/action HDF5 excludes cameras; generated data is not visually certified
until a separate saved-state rendering is reviewed.
