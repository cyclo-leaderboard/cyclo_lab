# Task000759 오이 실험 재현

이 폴더는 원본 시연을 참고하는 cuRobo 기반 독립 실험이다. 초기 공식 SkillGen
호출 코드는 `scripts/skillgen_poses.py`에 남아 있지만, 최종 생성기는 커스텀
파지 후보·관절 최적화·구간별 재타이밍·실제 파지 상태 기반 배치/복귀를 쓴다.
기존 Cyclo Lab/로봇/ROS 프로세스를 바꾸거나 실행하지 않는다.

원본 에피소드 93 하나에서 추출한 작은 수치 시드, FK, annotation, 장면/URDF/
오이 USD와 출처를 포함했다. 영상·데이터셋 전체·이전 성공 output·캐시·설치된
cuRobo 디렉터리는 포함하지 않는다. 원래 `/home/robotis-ai/agents/...` 폴더가
없어도 기본 명령으로 새 출력을 생성할 수 있다.

## 다른 PC 준비

Linux, NVIDIA GPU/드라이버, Docker/NVIDIA Container Toolkit, Git/Git LFS가
필요하다. 검증 기준 환경은 RTX 5090 32 GiB, Isaac Sim 5.1.0, 고정 Isaac Lab
submodule, torch 2.7.0+cu128, CUDA 12.8, 고정 cuRobo commit이다. 상세 버전은
`runtime.lock.json`과 `docker/requirements.txt`를 따른다.

```bash
git clone --branch feature-mimic-gen https://github.com/cyclo-leaderboard/cyclo_lab.git
cd cyclo_lab
git lfs install --local
git lfs pull
git submodule update --init
task759_run="$PWD/scripts/sim2real/imitation_learning/experiments/task_000759_skillgen_curobo/run.sh"
bash "$task759_run" build-base
bash "$task759_run" build
bash "$task759_run" doctor
bash "$task759_run" generate --name smoke0 --yaws 0
```

단순 clone만으로 완료되지 않는다. LFS 자산·submodule과 Docker 빌드가 필요하다.
NVIDIA 라이선스/컨테이너 레지스트리 접근은 실행자가 준비한다. 첫 빌드는
네트워크·디스크가 필요하고 오래 걸릴 수 있다. 기존 `cyclo_lab` 컨테이너를
재시작하지 않고 별도 이미지 `cyclo-lab-task759-base:5.1.0`,
`cyclo-lab-task759:20261106`를 만든다. 호스트 Python에는 설치하지 않는다.
다른 GPU에서는 `TASK759_CUDA_ARCH=<해당 GPU compute capability>`를
설정하고 `build`해야 한다. 기본 12.0은 RTX 5090용이다.

0도 확인 후 전체 11개를 생성한다:

```bash
bash "$task759_run" generate --name all11
```

기본 각도는 원본 오이 방향 대비 `0, ±30, ±45, ±60, ±75, ±90`도다.
계획만 하려면 `plan --name planned0 --yaws 0`, 이어서 물리 실행하려면
`simulate --name planned0`을 사용한다. 생성에 필요한 중간 scene/접근 궤적을
새 output 안에서 만들며, 이전 `hybrid11_*` 성공 선정표를 필요로 하지 않는다.
기존 run 이름·물리 시도는 덮어쓰지 않는다.

초기 **공식 SkillGen core**만 따로 실행하려면
`bash "$task759_run" skillgen-poses --name skillgen1`을 쓴다. EEF 변환 pose와
보고서를 `output/skillgen1`에 만들며 입력 파일은 수정하지 않는다. 최종 파지·
배치 생성기와 별개이고, 이 명령만으로 물리적 파지 성공을 주장하지 않는다.

`output/<이름>/` 아래에 계획 로그, `plan_manifest.json`,
`generation_report.json`, 각도별 `run_*/result.json`, 실제 상태·전체 궤적 NPZ,
`observer.mp4`를 저장한다. 11개가 모두 성공하면 `report/`와 각도 표시된
`report/final_11_angles_1x.mp4`도 만든다. 실패는 보존하고 종료 상태가 실패다.
출력 위치는 `TASK759_OUTPUT_DIR=/절대/경로`로 바꿀 수 있다.

다른 Isaac 작업이 GPU를 쓰면 기본적으로 거부하며 그 프로세스를 끄지 않는다.
GPU 공유를 의도할 때만 `--allow-shared-gpu`를 명시한다. 목표 60 Hz, 물리
240 Hz, 정상 속도 영상 15 fps다. 리프트 목표는 고정이고 damping 600을
실험 내부 복사 config에만 적용한다. 리프트를 잠그거나 책상 충돌을 끄지 않는다.

## 보장 범위와 이전 결과 구분

같은 코드·설정·시드로 실행하는 구성이지, 다른 GPU/드라이버/PhysX/CUDA에서도
동일한 궤적·성공률을 보장하는 구성은 아니다. 과거 평균 9.89초, 범위 9.05–11.90초,
최종 선정 11개 성공은 튜닝한 이전 실험 결과이며 이번 클론의 새 검증 결과가
아니다. 해당 output은 포함하지 않는다. 장면·오이·마찰·그리퍼 충돌 형상은
근사값이고, 원본 articulation의 팔 self-collision은 꺼져 있다. 실물 검증이나
전역 최적·사람 같은 동작을 수학적으로 보장하지 않는다.

`final_settings.json`에 +60/+75 접근 여유, -75 보호된 진입, -90 새 파지 후보,
+75 손목 이동/기울기 균형 등 알려진 최종 설정을 기록했다. 과거 실험용
`scripts/run_hybrid11*.sh`, worker/비교 스크립트는 이전 output을 요구할 수 있다.
새 재현에는 반드시 이 폴더 최상위 `run.sh`를 사용한다.

Task901 장면은 별도 [환경 폴더](../../tasks/catalog/task_000901/README.md)에 있다.
그 환경 자산은 포함하지만 Mimic pilot의 원본 reference archive는 별도 입력이다.
오이 시드를 Task901의 reference로 사용할 수 없다.

GPU 없이 정적 파일 검증만 하려면:

```bash
python3 scripts/sim2real/imitation_learning/experiments/task_000759_skillgen_curobo/scripts/portable.py doctor --static
TASK759_DRY_RUN=1 bash "$task759_run" generate --name inspect --yaws 0
```
