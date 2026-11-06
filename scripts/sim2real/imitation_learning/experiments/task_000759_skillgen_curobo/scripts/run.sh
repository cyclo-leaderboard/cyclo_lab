#!/usr/bin/env bash
# Host-side launcher. Never changes or restarts the existing cyclo_lab container.
set -euo pipefail
experiment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
container_name=task000759_skillgen_20261105
base_image=sha256:1aaaf8c2722691ede57952ff56b921d30e60f73dc13fd9540fd5fdd3a8b7f964
case "${1:-status}" in
  create)
    if docker container inspect "$container_name" >/dev/null 2>&1; then
      echo "Experiment container already exists; use start."; exit 1
    fi
    docker run -d --name "$container_name" --gpus all --network bridge \
      --shm-size=2g --memory=16g --cpus=8 --entrypoint /bin/bash \
      -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e PYTHONDONTWRITEBYTECODE=1 \
      -e ROS_DOMAIN_ID=241 -e NVIDIA_DRIVER_CAPABILITIES=all \
      -v /home/robotis-ai/cyclo_lab:/workspace/cyclo_lab:ro \
      -v /home/robotis-ai/cyclo_intelligence/docker/workspace/dataset/Dongkkka/task_000759_move_vegetable_v30:/dataset:ro \
      -v /home/robotis-ai/ai_worker/ffw_description:/robot_description:ro \
      -v "$experiment_dir":/experiment:rw -w /experiment "$base_image" -c 'sleep infinity'
    ;;
  start) docker start "$container_name" ;;
  stop) docker stop --timeout 20 "$container_name" ;;
  status) docker inspect --format '{{.Name}} {{.State.Status}} network={{.HostConfig.NetworkMode}}' "$container_name" ;;
  install) docker exec "$container_name" bash /experiment/scripts/install_isolated.sh ;;
  poses) docker exec "$container_name" /isaac-sim/python.sh -u /experiment/scripts/skillgen_poses.py --headless ;;
  plan) shift; docker exec -e CUDA_HOME=/usr/local/cuda-12.8 "$container_name" /isaac-sim/python.sh -u /experiment/scripts/plan.py "$@" ;;
  plan-goalset) shift; docker exec -e CUDA_HOME=/usr/local/cuda-12.8 "$container_name" /isaac-sim/python.sh -u /experiment/scripts/plan_goalset.py "$@" ;;
  plan-pickplace) shift; docker exec -e CUDA_HOME=/usr/local/cuda-12.8 "$container_name" /isaac-sim/python.sh -u /experiment/scripts/plan_pickplace.py "$@" ;;
  refine-transport) shift; docker exec -e CUDA_HOME=/usr/local/cuda-12.8 "$container_name" /isaac-sim/python.sh -u /experiment/scripts/refine_transport.py "$@" ;;
  evaluate-pickplace) shift; docker exec -e CUDA_HOME=/usr/local/cuda-12.8 "$container_name" /isaac-sim/python.sh -u /experiment/scripts/evaluate_pickplace.py "$@" ;;
  simulate) shift; docker exec "$container_name" /isaac-sim/python.sh -u /experiment/scripts/simulate.py --headless "$@" ;;
  *) echo 'Usage: bash scripts/run.sh {create|start|stop|status|install|poses|plan|plan-goalset|plan-pickplace|refine-transport|evaluate-pickplace|simulate} [arguments]'; exit 2 ;;
esac
