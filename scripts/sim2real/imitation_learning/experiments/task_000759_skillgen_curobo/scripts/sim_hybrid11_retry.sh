#!/usr/bin/env bash
set -uo pipefail
export CUDA_HOME=/usr/local/cuda-12.8
profiles=$1
prefix=$2
shift 2
for yaw in "$@"; do
    printf -v signed '%+03d' "$yaw"
    /isaac-sim/python.sh -u /experiment/scripts/simulate.py --headless --video --interpolate-targets --velocity-feedforward --plan-place --return-home --motion-speed 1.2 --profile hybrid --yaw "$yaw" --name "${prefix}_${signed}" --scene "/experiment/output/${profiles}/scene.json" --trajectory "/experiment/output/${profiles}/hybrid_${signed}/trajectory.npz" > "/experiment/output/${prefix}_sim_${signed}.log" 2>&1
done
