#!/usr/bin/env bash
set -euo pipefail
test "$(hostname)" != "$(cat /experiment/host_hostname 2>/dev/null || true)"
test -d /dataset && test -d /experiment
export DEBIAN_FRONTEND=noninteractive
export PIP_CONSTRAINT=/experiment/scripts/constraints.txt
/isaac-sim/python.sh -m pip install pyarrow==19.0.1 ninja yourdfpy
if ! test -x /usr/local/cuda-12.8/bin/nvcc; then
  . /etc/os-release
  cuda_repo="ubuntu${VERSION_ID//./}"
  wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${cuda_repo}/x86_64/cuda-keyring_1.1-1_all.deb" -O /tmp/task759-cuda-keyring.deb
  dpkg -i /tmp/task759-cuda-keyring.deb
  apt-get update
  apt-get install -y --no-install-recommends cuda-nvcc-12-8 cuda-cudart-dev-12-8 libcurand-dev-12-8 build-essential
fi
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST='12.0'
export MAX_JOBS=4
if ! test -d /experiment/deps/curobo/.git; then
  git clone https://github.com/NVlabs/curobo.git /experiment/deps/curobo
fi
git -C /experiment/deps/curobo checkout ebb71702f3f70e767f40fd8e050674af0288abe8
/isaac-sim/python.sh -m pip install -e /experiment/deps/curobo --no-build-isolation
/isaac-sim/python.sh -c 'import curobo, torch; print("ISOLATED_CUROBO_OK", curobo.__file__, torch.__version__, torch.cuda.get_device_name(0))'
