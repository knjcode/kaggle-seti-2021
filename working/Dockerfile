FROM nvidia/cuda:11.1.1-devel-ubuntu18.04

ENV CUDNN_VERSION=8.0.4.30-1+cuda11.1
ENV NCCL_VERSION=2.8.3-1+cuda11.1

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

ARG PYTHON_VERSION=3.8
RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         wget \
         ca-certificates \
         libfabric-dev \
         libgl1-mesa-dev \
         libglib2.0-0 \
         libcudnn8=${CUDNN_VERSION} \
         libnccl2=${NCCL_VERSION} \
         libnccl-dev=${NCCL_VERSION} \
         libnuma-dev \
         libjpeg-dev \
         libpng-dev \
         librdmacm1 \
         libibverbs1 \
         ibverbs-providers &&\
     rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH

# Install Open MPI
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://www.open-mpi.org/software/ompi/v4.1/downloads/openmpi-4.1.1.tar.gz && \
    tar zxf openmpi-4.1.1.tar.gz && \
    cd openmpi-4.1.1 && \
    ./configure --enable-orterun-prefix-by-default --with-cuda && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

RUN pip install -U --no-cache-dir pip && pip install -U --no-cache-dir \
        albumentations \
        logzero \
        madgrad matplotlib mpi4py \
        pandas pillow \
        scikit-learn \
        timm \
        tensorboard transformers

# Install OpenSSH for MPI to communicate between containers
RUN apt-get update && apt-get install -y --no-install-recommends openssh-client openssh-server zip unzip && \
    mkdir -p /var/run/sshd && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /opt/conda/bin/python /usr/local/bin/python
RUN ln -s /opt/conda/bin/pip /usr/local/bin/pip

RUN  /opt/conda/bin/conda install -y pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia && \
     /opt/conda/bin/conda clean -y --all

# Install Horovod, temporarily using CUDA stubs
RUN ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
    HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_PYTORCH=1 \
         pip install --no-cache-dir horovod && \
    horovodrun --check-build && \
    ldconfig
