# Build the repository's pinned Isaac Sim/Lab base first; no shared container is changed.
ARG CYCLO_BASE_IMAGE=cyclo-lab-task759-base:5.1.0
FROM ${CYCLO_BASE_IMAGE}
USER root
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
ARG TORCH_CUDA_ARCH_LIST=12.0
ARG CUROBO_COMMIT=ebb71702f3f70e767f40fd8e050674af0288abe8
ENV CUDA_HOME=/usr/local/cuda-12.8
ENV PATH=/usr/local/cuda-12.8/bin:${PATH}
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV MAX_JOBS=4
ENV PYTHONDONTWRITEBYTECODE=1
COPY scripts/sim2real/imitation_learning/experiments/task_000759_skillgen_curobo/docker/ /opt/task759-build/
RUN source /etc/os-release && \
    wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${VERSION_ID//./}/x86_64/cuda-keyring_1.1-1_all.deb" -O /tmp/task759-cuda-keyring.deb && \
    dpkg -i /tmp/task759-cuda-keyring.deb && apt-get update && \
    apt-get install -y --no-install-recommends cuda-nvcc-12-8 cuda-cudart-dev-12-8 libcurand-dev-12-8 build-essential git && \
    /isaac-sim/python.sh -c 'import torch; assert torch.__version__ == "2.7.0+cu128", torch.__version__' && \
    PIP_CONSTRAINT=/opt/task759-build/constraints.txt /isaac-sim/python.sh -m pip install -r /opt/task759-build/requirements.txt && \
    git clone https://github.com/NVlabs/curobo.git /opt/task759/curobo && \
    git -C /opt/task759/curobo checkout --detach "${CUROBO_COMMIT}" && \
    PIP_CONSTRAINT=/opt/task759-build/constraints.txt /isaac-sim/python.sh -m pip install -e /opt/task759/curobo --no-build-isolation && \
    /isaac-sim/python.sh -c 'import torch,curobo; assert torch.__version__ == "2.7.0+cu128"; print(curobo.__file__)'
LABEL org.cyclo.task759.curobo="ebb71702f3f70e767f40fd8e050674af0288abe8"
WORKDIR /experiment
ENTRYPOINT ["/bin/bash"]
