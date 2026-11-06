# Task000759: source-guided cuRobo cucumber pick/place/return

[한국어](README.ko.md)

Standalone experiment, not a registered production task or an unmodified SkillGen
generator. Early trials used Isaac Lab SkillGen's real `DataGenerator` with
`use_skillgen=True`; the final generator uses analytic grasp candidates, cuRobo,
source-joint priors, continuous IK, joint splines/minimum-jerk candidates, per-phase
retiming, and placement/home planned from the physically measured held state.
`scripts/skillgen_poses.py` preserves the initial official-core experiment.

The one numerical seed is episode 93 (source episode 95), 115 samples at 15 Hz.
`input/manifest.json` verifies the included numerical seed, FK, annotation, kinematic
model and approximate scene. No source camera video, complete source dataset,
historical output, installed dependency tree, cache, or successful-run archive is
included. The original experiment directory is not required by the main commands.

## Clone and prepare

Requires Linux, NVIDIA GPU/compatible driver, Docker + NVIDIA Container Toolkit,
Git and Git LFS. Measured reference: RTX 5090 32 GiB, Isaac Sim 5.1.0, pinned
Isaac Lab submodule, torch 2.7.0+cu128, CUDA toolkit 12.8 and cuRobo commit
`ebb71702f3f70e767f40fd8e050674af0288abe8`. See `runtime.lock.json` and
`docker/requirements.txt`. This does not install anything into host Python or an
existing Cyclo Lab container.

```bash
git clone --branch feature-mimic-gen https://github.com/cyclo-leaderboard/cyclo_lab.git
cd cyclo_lab
git lfs install --local
git lfs pull
git submodule update --init
```

Initialize direct submodules, not Arena's nested Isaac Lab checkout. Both Git LFS
objects and submodules are necessary: cloning source files alone is not sufficient.
NVIDIA container/license acceptance and any registry login are the operator's
responsibility. The launcher sets the same EULA variables as the repository.

```bash
task759_run="$PWD/scripts/sim2real/imitation_learning/experiments/task_000759_skillgen_curobo/run.sh"
bash "$task759_run" build-base
bash "$task759_run" build
bash "$task759_run" doctor
```

`build-base` builds this checkout's existing Cyclo Lab Dockerfile into the separate
tag `cyclo-lab-task759-base:5.1.0`; it does NOT run Compose or any bridge/service.
`build` adds cuRobo in a separate image `cyclo-lab-task759:20261106`. This first
build can be large/slow and needs network access and disk space. A pre-existing
compatible base can be selected with `TASK759_BASE_IMAGE`; verify it with doctor.
For a different GPU, set the correct architecture during the cuRobo build, e.g.
`TASK759_CUDA_ARCH=<your_compute_capability> bash "$task759_run" build`.
Default `12.0` is specifically the tested RTX 5090, not a universal GPU value.

## Generate

Start with one angle before running the entire set:

```bash
bash "$task759_run" generate --name smoke0 --yaws 0
bash "$task759_run" generate --name all11
```

Supported delta angles, relative to the source object's 26.43-degree orientation:
`0 +30 -30 +45 -45 +60 -60 +75 -75 +90 -90`.
These are actual 240 Hz physics executions with 60 Hz targets and normal 15 fps
videos, not playback-speed changes. The launcher holds the source lift target and
uses local finite damping 600; the lift is not locked and table collisions remain
enabled. Approximately ten-second cycles were observed in tuned historical runs,
not guaranteed by the generator on every run or computer.

For plan-only debugging, followed by an independent physical execution:

```bash
bash "$task759_run" plan --name planned0 --yaws 0
bash "$task759_run" simulate --name planned0
```

To run only the historical **official SkillGen core** pose adaptation (not the
final pick/place generator): `bash "$task759_run" skillgen-poses --name skillgen1`.
It writes adapted EEF poses/report under `output/skillgen1`, never modifies the
bundled input, and makes no physical-grasp-success claim.

The new orchestrator creates all intermediate inputs in a fresh output directory;
it never requires the old `hybrid11_selection.json`, `rapid10_report`, or previous
successful trajectories. It refuses to overwrite run names/physical attempts.
Default GPU guard refuses to compete with another Isaac process; no process is
stopped. `--allow-shared-gpu` is an explicit opt-in only. Final settings, including
special +60/+75 clearance, -75 protected approach, -90 refined grasp, and +75
wrist/tilt balance, are in `final_settings.json`.

Outputs: `output/<name>/plan_manifest.json`, `generation_report.json`, per-angle
planning/simulation logs, `run_<angle>/result.json`, measured `execution.npz`, full
`executed_trajectory.npz`, and `observer.mp4`. Failures are retained and return a
nonzero exit status. Only if all eleven pass does the runner export `report/`
and concatenate `report/final_11_angles_1x.mp4`, with top-right angle labels.
`TASK759_OUTPUT_DIR=/absolute/path` redirects outputs without moving source code.

## Reproducibility scope

- Same implementation/configuration/seed, not bit-identical plans or a guaranteed
  11/11 result. CUDA numerical planning, PhysX, GPU, driver and versions can differ.
- Scene, cucumber shape/friction, gripper collision proxies and basket are
  approximate. Arm self-collision is disabled in the source articulation config.
  This is not real-robot validation or a global/natural-motion optimality proof.
- The seed originated from a visually successful real video; FK is derived from
  measured joints. Its use as a prior is not proof that replay succeeds in physics.
- Historical selected durations were 9.05–11.90 s (mean 9.89 s). Historical final
  eleven were tuned selections, not an unbiased one-shot success rate. Those
  result files are intentionally not shipped or claimed as fresh validations.
- Existing `scripts/run_hybrid11*.sh`, `sim_*worker.sh`, `compare_rapid.py` and old
  report scripts are research history and may expect absent historical outputs.
  Use this directory's **root `run.sh`** for a fresh generation workflow.
- Task901 is separately registered at
  [tasks/catalog/task_000901](../../tasks/catalog/task_000901/README.md).
  Its scene assets are included. Its Mimic pilot needs its own original reference
  archive; the cucumber seed is not interchangeable with that task's reference.

No simulator is started for static checking:

```bash
python3 scripts/sim2real/imitation_learning/experiments/task_000759_skillgen_curobo/scripts/portable.py doctor --static
TASK759_DRY_RUN=1 bash "$task759_run" generate --name inspect --yaws 0
```
