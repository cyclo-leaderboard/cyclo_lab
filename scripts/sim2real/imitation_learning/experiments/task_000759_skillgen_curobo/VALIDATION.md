# Publication validation, 2026-11-06 KST

Checked in a relocated publication worktree, not the original experiment output.

- All 31 included Python sources parse; shell sources pass `bash -n`.
- Four CPU-only orchestration tests pass (angle routing, special placement flags,
  overwrite refusal, GPU guard). Mocked test successes are NOT physical results.
- The Task901 catalog validator passes: 11 tasks, one setup task, zero previews.
- Numerical inputs and local task assets match `input/manifest.json` SHA256.
- Runtime doctor passes with the locked torch/CUDA/cuRobo and numerical packages.
- Fresh 0-degree planning and physical pick/place/return pass: 9.2167 s, max arm
  tracking error .02112 rad, no preclose push above 5 mm, actual lift range 3.13 mm.
  Lift damping/stiffness 600/10000; table collision enabled; no object scripting.
- Its MP4 fully decodes: 138 frames, 15 fps, 9.2 s. Videos and generated outputs
  are excluded from Git, per user request.
- The initial official SkillGen-core script also executes successfully: one demo,
  boundaries [30,66), 37 EEF poses per skill and five adapted angles. Its export
  location was changed to writable output, not bundled input.
- The already-running optional planning finished all eleven pickup prefixes before
  cancellation was processed. No further physical angle tests were started after
  the user requested publication. This is NOT a new eleven-angle success rate.

The relocated runtime test reuses the known installed Isaac/CUDA/cuRobo image and
read-only pinned dependency sources. The separate Dockerfile's new dependency
build completed successfully from the reference `robotis/cyclo-lab:latest` base:
CUDA toolkit installation, pinned requirements, pinned cuRobo compilation and
torch/cuRobo imports all passed. Building the whole Cyclo Lab base from scratch
on a different PC has NOT been certified here. Follow README setup and doctor.
Same code/configuration is provided; cross-GPU identical physical outcomes are
not promised. No original Cyclo Lab, Task901 render, ROS or robot process was stopped.
