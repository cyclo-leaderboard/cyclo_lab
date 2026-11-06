# Latest Task901 profile publication: 2026-11-06

Validated the latest **automation code**, not the stale November3 episode010
archive or the separate first-frame-only 0.689674 m camera fit.

- CPU USD composition/authoring tests: **4 passed** using Isaac5.1 USD/PhysxSchema
  libraries, with no Kit application, GPU or ROS service started.
- Both latest room layers compose a **0.695 m** tabletop; tabletop bounds and all
  five tags move down **0.005 m** together; table collision APIs remain present.
- Three latest color plate layers compose **384** enabled convex colliders each;
  metadata is d185/b160/a73, 1 mm shell, 15 g.
- Actual shipped SG2 soft-tip USD was composed; the real shared collision builder
  created **20** convex pieces and **50** opposing-tip exclusions. The latest
  task-local authoring function retained that layout and applied recess5mm:
  measured maximum vertex shift **0.004979542657688603 m**. All **35** visual mesh
  point arrays stayed identical. Four joint2/4 limits were authored as1.1rad;
  their original USD mimic frequency/damping were preserved.
- The contact helper is byte-identical to its source SHA256 in physics_profile.json.
- All **20** Task901 Python files parse; `git diff --check` passes.
- Catalog validates11 tasks (1setup),0previews; Task759's4 CPU mocked routing/guard
  regression tests pass. These mocked tests are not physical successes.

The separate `validate_latest_scene.py` performs the same4 USD tests plus a
24-step registered-environment GPU physics smoke test. **It was not run for this
publication:** a user-owned Task901 replay/probe was occupying the GPU. No user
process/container was stopped or restarted. No new videos, grasp/insertion
success rate, hardware replay or end-to-end Mimic success is claimed.

Only Task901 files changed. Shared SG2/BG2 defaults, Task759, original local
checkout and experiment outputs were not edited. Historical episode6 Mimic IDs
retain their original table/contact profile explicitly so existing seed poses
and expected slots are not silently invalidated.
