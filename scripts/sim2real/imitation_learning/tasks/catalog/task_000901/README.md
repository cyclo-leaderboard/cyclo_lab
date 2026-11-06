# Task000901 portable scene and bimanual Mimic pilot

The scene, plate-holder variant, object geometry/textures, local plugin and Mimic
adapter are in this directory. Run `git lfs pull` after cloning: USD/mesh/NPZ/image
assets use Git LFS and pointer text cannot be rendered as an asset.

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
