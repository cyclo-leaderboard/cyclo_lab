#!/usr/bin/env bash
# Standalone launcher: never starts/restarts cyclo_lab, ROS, Zenoh, or a real robot.
set -euo pipefail
package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$package_dir/../../../../.." && pwd)"
task759_image=${TASK759_IMAGE:-cyclo-lab-task759:20261106}
task759_base_image=${TASK759_BASE_IMAGE:-cyclo-lab-task759-base:5.1.0}
task759_arch=${TASK759_CUDA_ARCH:-12.0}
task759_output=${TASK759_OUTPUT_DIR:-$package_dir/output}
usage() {
    printf '%s\n' 'Usage: bash run.sh {build-base|build|doctor|plan|generate|simulate|skillgen-poses} [arguments]' \
      'First run: build-base; build; doctor; generate --name trial1 --yaws 0' \
      'Change GPU arch with TASK759_CUDA_ARCH (default 12.0, RTX 5090).' \
      'Use TASK759_DRY_RUN=1 to print the Docker command without running it.'
}
execute() {
    if [[ ${TASK759_DRY_RUN:-0} == 1 ]]; then printf '%q ' "$@"; printf '\n'; else "$@"; fi
}
command=${1:-help};if (($#));then shift;fi
if [[ $command == help || $command == --help || $command == -h ]]; then usage;exit 0;fi
[[ -f "$repo_dir/docker/Dockerfile.base" ]] || { echo 'Cannot locate Cyclo Lab repository root.' >&2;exit 1; }
case "$command" in
  build-base)
    for module in IsaacLab IsaacLab-Arena zenoh_ros2_sdk; do
      [[ -d "$repo_dir/third_party/$module/source" || -f "$repo_dir/third_party/$module/setup.py" || -f "$repo_dir/third_party/$module/pyproject.toml" ]] || {
        echo "Missing submodule $module. Run: git submodule update --init" >&2;exit 1;
      }
    done
    execute docker build -f "$repo_dir/docker/Dockerfile.base" -t "$task759_base_image" \
      --build-arg ISAACSIM_BASE_IMAGE_ARG=nvcr.io/nvidia/isaac-sim --build-arg ISAACSIM_VERSION_ARG=5.1.0 \
      --build-arg ISAACSIM_ROOT_PATH_ARG=/isaac-sim --build-arg CYCLOLAB_PATH_ARG=/workspace/cyclo_lab \
      --build-arg DOCKER_USER_HOME_ARG=/root "$repo_dir" ;;
  build)
    execute docker build -f "$package_dir/docker/Dockerfile" -t "$task759_image" \
      --build-arg "CYCLO_BASE_IMAGE=$task759_base_image" --build-arg "TORCH_CUDA_ARCH_LIST=$task759_arch" "$repo_dir" ;;
  doctor|plan|generate|simulate|skillgen-poses)
    mkdir -p "$task759_output"
    docker_args=(run --rm --gpus all --network bridge --shm-size=2g --memory=20g --memory-swap=20g --cpus=8
      --entrypoint /bin/bash -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ALLOW_ROOT=1
      -e PYTHONDONTWRITEBYTECODE=1 -e ROS_DOMAIN_ID=241 -e NVIDIA_DRIVER_CAPABILITIES=all
      --mount "type=bind,source=$repo_dir,target=/workspace/cyclo_lab,readonly"
      --mount "type=bind,source=$package_dir,target=/experiment,readonly"
      --mount "type=bind,source=$task759_output,target=/experiment/output"
      --mount type=volume,source=task759-cuda-cache,target=/root/.cache
      -w /experiment "$task759_image" /isaac-sim/python.sh -u /experiment/scripts/portable.py)
    execute docker "${docker_args[@]}" "$command" "$@" ;;
  *) usage;exit 2 ;;
esac
