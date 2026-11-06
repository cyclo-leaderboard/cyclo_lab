# Task environment layout

Every numbered task has a tracked `task.json` manifest.
The manifest identifies its scene family (`showroom`, `white_room`, or
`wall_room`), robot, canonical Gym ID, optional variants, and optional base
task. `task_catalog.py` validates the manifests and resolves a task number
without importing Isaac Sim.

```
environments/<scene>/plugin.py        tracked shared scene loader
source/cyclo_lab/data/environments/<scene>/  shared scene USD and textures
tasks/catalog/task_000000/
  task.json                         tracked number, scene, robot, and variants
  plugin.py                         tracked Gym registration and task hooks
  runtime/                          tracked task environment configuration and required assets
  scripts/                          optional tracked task-specific workflows
  README.md                         optional task-specific usage notes
tasks/local/task_000000/             machine-local experiments (ignored)
```

All canonical numbered task manifests and plugins live under `tasks/catalog/`.
Shared scene code and assets stay in `source/cyclo_lab/`; each task's runtime
configuration selects a scene and adds its own objects, robot settings, and
supported variants. Experiment numbers 000305 through 000329 are excluded
from the catalog; Task000300 remains. Task000901 is the
eleventh task, marked `status=setup`. Its current environment registration
contains the SG2, cameras, table, tags, cucumber, and onion; manipulation
actions and rewards are still being configured.

For a new task, copy `task_template.json` into
`catalog/task_XXXXXX/task.json`, set the scene and robot, then create the
task-specific `plugin.py` and environment config under that directory's
`runtime/` tree. Put tracked task-specific workflow scripts under `scripts/`.
Register only supported variants. Put shared USD files in the tracked scene
family and task-only assets beside the task. Keep experiments under `local/`.
Then run:

```
python3 scripts/sim2real/imitation_learning/task_catalog.py --list
python3 scripts/sim2real/imitation_learning/task_catalog.py 000400
python3 scripts/sim2real/imitation_learning/task_catalog.py 000853 --variant mimic_generate_right
python3 scripts/sim2real/imitation_learning/task_catalog.py --validate
```

`bringup.py`, `record_demos.py`, and `mimic/generate_dataset.py` accept the same
six-digit number through `--task`; use `--task-variant` for non-default variants.
They still accept full Gym IDs. A MimicGen run should select a registered
`mimic_generate_*` variant explicitly when the task's default is its recording
environment.

The catalog, task implementations, shared scene source, and required assets
are Git-visible. Everything under `tasks/local/` and `environments/local/`
remains ignored. USD, STL, and task texture assets use Git LFS through the
repository `.gitattributes`.
