#!/usr/bin/env bash
set -uo pipefail
export CUDA_HOME=/usr/local/cuda-12.8
export TASK759_TIMING_PROFILE=rapid
profiles=$1
prefix=$2
shift 2
for yaw in "$@"; do
    printf -v signed '%+03d' "$yaw"
    if [[ "$yaw" == 75 || "$yaw" == -90 ]]; then export TASK759_LOW_ROTATION=1; else export TASK759_LOW_ROTATION=0; fi
    /isaac-sim/python.sh -u /experiment/scripts/simulate.py --headless --video --interpolate-targets --velocity-feedforward --plan-place --return-home --lift-damping 600 --profile hybrid --yaw "$yaw" --name "${prefix}_${signed}" --scene "/experiment/output/${profiles}/scene.json" --trajectory "/experiment/output/${profiles}/hybrid_${signed}/trajectory.npz" > "/experiment/output/${prefix}_${signed}.log" 2>&1
done
