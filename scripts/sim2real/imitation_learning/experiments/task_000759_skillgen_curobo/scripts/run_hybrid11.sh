#!/usr/bin/env bash
set -euo pipefail
export CUDA_HOME=/usr/local/cuda-12.8
export TASK759_CANDIDATE_MAP='{"0":[36,0,37],"30":[37,13,1],"45":[111,113,109],"60":[113,111,197],"75":[197,195,193,185,189],"90":[197,195,193],"-30":[73,75,145],"-45":[145,147,149],"-60":[145,217,219],"-75":[229,217,219,225],"-90":[229,225,217,233]}'
/isaac-sim/python.sh -u /experiment/scripts/plan_pickplace.py --yaws 0 30 -30 45 -45 60 -60 75 -75 90 -90 --output hybrid11_initial --pickup-only --prior-weight .005 --attempts 5 > /experiment/output/hybrid11_planning.log 2>&1
/isaac-sim/python.sh -u /experiment/scripts/make_profile_pickups.py --input hybrid11_initial --yaws 0 30 -30 45 -45 60 -60 75 -75 90 -90 --profiles hybrid --motion-speed 1.2 --output hybrid11_profiles > /experiment/output/hybrid11_profiles.log 2>&1
