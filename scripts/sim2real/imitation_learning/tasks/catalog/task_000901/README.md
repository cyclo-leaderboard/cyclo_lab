# Task000901 portable scene and bimanual Mimic pilot

The scene, plate-holder variant, object geometry/textures, local plugin and Mimic
adapter are in this directory. Run `git lfs pull` after cloning: USD/mesh/NPZ/image
assets use Git LFS and pointer text cannot be rendered as an asset.

## Latest scene defaults (2026-11-06)

The Environment and PlateHolder IDs now use [physics_profile.json](physics_profile.json),
ported from the latest `real_sim_replay/automation` runner (source SHA256 recorded):

- Table top **0.695 m** (previous 0.700 m); table collision/visual, tags and tabletop
  props/plates/holder move together by -5 mm. Walls and robot root stay unchanged.
- Real-STL soft tips, 20 existing convex pieces and maximum **5 mm** inward pocket;
  the visual finger mesh stays unchanged. Self collision on; **retain all 50 opposing
  tip filtered pairs**, matching the latest runner, not the earlier tip-contact test.
- Distal friction 4.0/3.5; joint2/4 limit 1.1 rad; original USD mimic 1500/5 retained.
- Latest plate template d185/b160/a73: outer185 mm, flat160 mm, wall73 degrees,
  height24 mm, shell1 mm, mass15 g, 384 convexes; plate friction0.6/0.45.

These are current experiment parameters, not measured rubber compliance or a
guarantee of insertion success. The earlier **0.689674 m** first-frame-only camera
fit is a separate experiment and is **not** the current replay default. Calibration
camera mounts/K remain unchanged by this physics-only port.

The existing **Bimanual-Mimic Seed/Generate IDs intentionally retain their archived
episode6 profile** (0.700 m, opposing-tip exclusions removed). Their seed poses and
success targets were authored in that older scene; silently switching its physics
would invalidate the reference. Fresh latest-profile demonstrations must be recorded
and annotated before claiming a new latest-profile Mimic generation success.

Finite headless validation, without ROS or robot commands:

```bash
/isaac-sim/python.sh scripts/sim2real/imitation_learning/tasks/catalog/task_000901/validate_latest_scene.py --output /tmp/task901_latest_validation.json
```

This runs four CPU USD geometry/authoring tests and a 24-step registered-environment
physics smoke check, not a grasp/insertion rollout. Run only when the GPU is free.

한국어: 기본 장면·PlateHolder는 **상판69.5cm / 팁 홈최대5mm / 팁 예외50쌍 유지**로
갱신했다. 기존 Mimic 에피소드6 시드는 원래70cm/팁 접촉 기준을 명시적으로 보존한다.
공통 SG2 설정과 오이 Task759는 변경하지 않는다. 구버전 시드를 새 환경에서
성공 검증한 것처럼 취급하지 않는다.

From the repository's configured Isaac Sim/Lab Python environment:

```bash
/isaac-sim/python.sh scripts/sim2real/imitation_learning/task_catalog.py --validate
/isaac-sim/python.sh scripts/sim2real/imitation_learning/tasks/catalog/task_000901/render_calibration.py
```

The second command is a headless, finite calibration render, not a robot/ROS
command. It writes `/tmp/task901_calibration/renders` inside the container. Run it
only when the GPU is available. It registers
`Cyclo-Real-WhiteRoom-Task000901-PlateHolder-FFW-SG2-v0`. Other registered IDs are in
`task.json`; scene status remains `setup`, not a production success certification.

**The scene is included; the original Mimic reference episode is not.**
`mimic/prepare_reference.py --source /path/to/episode_006 --output /path/to/reference`
expects the measured-state replay archive listed in [mimic/README.md](mimic/README.md).
The old `/home/robotis-ai/...` default is provenance from the workstation, not an
asset downloaded by cloning. Supply the corresponding archive explicitly before
using that pilot. Code/scene reproducibility does not recover unrecorded original
teleoperation actions or guarantee insertion success on a different GPU.

한국어: 장면·물체·홀더·플러그인은 이 저장소에 포함되어 있다. 다른 PC에서는
Git LFS/submodule과 Isaac Sim/Lab 환경을 준비한 뒤 위 명령으로 장면을 확인한다.
Task901 Mimic pilot의 원본 에피소드 archive는 별도이므로 단순 clone으로 원본
데이터가 생기지는 않는다. `prepare_reference.py --source`로 실제 입력 경로를
지정해야 한다. 오이 실험은 `experiments/task_000759_skillgen_curobo`에 분리되어 있다.
