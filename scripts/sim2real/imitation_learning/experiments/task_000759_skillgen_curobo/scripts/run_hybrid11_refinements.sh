#!/usr/bin/env bash
set -uo pipefail
bash /experiment/scripts/sim_hybrid11_retry.sh hybrid11_alternative_profiles hybrid11_alt -75
bash /experiment/scripts/sim_hybrid11_retry.sh hybrid11_profiles hybrid11_retry2 0
bash /experiment/scripts/sim_hybrid11_retry.sh hybrid11_symmetric_profiles hybrid11_sym -90
