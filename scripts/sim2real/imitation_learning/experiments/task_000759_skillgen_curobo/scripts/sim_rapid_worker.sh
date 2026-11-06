#!/usr/bin/env bash
set -uo pipefail
export CUDA_HOME=/usr/local/cuda-12.8
export TASK759_TIMING_PROFILE=rapid
for yaw in "$@"; do
    printf -v signed '%+03d' "$yaw"
    if [[ "$yaw" == 75 || "$yaw" == -90 ]]; then export TASK759_LOW_ROTATION=1; else export TASK759_LOW_ROTATION=0; fi
    /isaac-sim/python.sh -u /experiment/scripts/simulate.py --headless --video --interpolate-targets --velocity-feedforward --plan-place --return-home --profile hybrid --yaw "$yaw" --name "rapid10_v3_${signed}" --scene /experiment/output/rapid10_pickups_v3/scene.json --trajectory "/experiment/output/rapid10_pickups_v3/hybrid_${signed}/trajectory.npz" > "/experiment/output/rapid10_v3_${signed}.log" 2>&1
done
