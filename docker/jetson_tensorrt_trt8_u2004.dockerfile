# Base image starts with CUDA
ARG BASE_IMG=nvcr.io/nvidia/l4t-tensorrt:r8.5.2.2-devel
FROM ${BASE_IMG} as base
ENV BASE_IMG=nvcr.io/nvidia/l4t-tensorrt:r8.5.2.2-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN rm /etc/apt/sources.list && \
    echo "deb https://mirrors.ustc.edu.cn/ubuntu-ports/ focal main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.ustc.edu.cn/ubuntu-ports/ focal-security main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.ustc.edu.cn/ubuntu-ports/ focal-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.ustc.edu.cn/ubuntu-ports/ focal-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    apt-get update

RUN apt install -y \
    build-essential \
    manpages-dev \
    wget \
    zlib1g \
    software-properties-common \
    git \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    ca-certificates \
    curl \
    llvm \
    libncurses5-dev \
    xz-utils tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    mecab-ipadic-utf8 \
    libopencv-dev \
    libeigen3-dev \
    libgtest-dev \
    libassimp-dev \
    libbenchmark-dev

# cmake
RUN cd /tmp && \
    wget https://gp.zz990099.cn/https://github.com/Kitware/CMake/releases/download/v3.22.3/cmake-3.22.3-linux-aarch64.tar.gz && \
    tar -xzvf cmake-3.22.3-linux-aarch64.tar.gz && \
    mv cmake-3.22.3-linux-aarch64 /opt/cmake-3.22.3 && \
    rm cmake-3.22.3-linux-aarch64.tar.gz && \
    rm /usr/local/bin/cmake && \
    rm /usr/local/bin/ctest && \
    ln -s /opt/cmake-3.22.3/bin/cmake /usr/local/bin/cmake && \
    ln -s /opt/cmake-3.22.3/bin/ctest /usr/local/bin/ctest

# glog
RUN cd /tmp && \
    wget https://gp.zz990099.cn/https://github.com/google/glog/archive/refs/tags/v0.7.0.tar.gz && \
    tar -xzvf v0.7.0.tar.gz && \
    cd glog-0.7.0 && \
    mkdir build && cd build && \
    cmake .. && make -j && \
    make install

# cv-cuda
RUN cd /tmp && \
    wget https://gp.zz990099.cn/https://github.com/CVCUDA/CV-CUDA/releases/download/v0.12.0-beta/cvcuda-lib-0.12.0_beta-cuda11-aarch64-linux.deb && \
    wget https://gp.zz990099.cn/https://github.com/CVCUDA/CV-CUDA/releases/download/v0.12.0-beta/cvcuda-dev-0.12.0_beta-cuda11-aarch64-linux.deb && \
    dpkg -i cvcuda-lib-0.12.0_beta-cuda11-aarch64-linux.deb && \
    dpkg -i cvcuda-dev-0.12.0_beta-cuda11-aarch64-linux.deb && \
    rm cvcuda-lib-0.12.0_beta-cuda11-aarch64-linux.deb && \
    rm cvcuda-dev-0.12.0_beta-cuda11-aarch64-linux.deb
