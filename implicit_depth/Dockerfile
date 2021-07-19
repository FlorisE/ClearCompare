FROM nvidia/cudagl:11.4.0-devel-ubuntu20.04 
ARG PYTHON_VERSION=3.7

SHELL ["/bin/bash", "--login", "-c"]

RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        ca-certificates \
        libjpeg-dev \
        libpng-dev \
        htop \
        python3-dev \
        libopenblas-dev \
        libopenexr-dev \
        libcudnn8 \
        tmux \
        vim \
        rsync \
        unzip \
        wget \
        libgtk2.0-dev \
        python3-pip \
        zlib1g-dev \
        openexr \
        xorg-dev \
        libgl1-mesa-dev \
        libglu1-mesa-dev \
        libglfw3-dev \
        at && \
    rm -rf /var/lib/apt/lists/*

ENV PATH /opt/conda/bin:$PATH
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.profile

RUN conda init bash

RUN conda create --name lidf python=3.7
RUN echo "conda activate lidf" >> ~/.bashrc

RUN conda activate lidf && \
    conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include ninja cython typing && \
    conda install -y pytorch torchvision cudatoolkit=11.1 -c pytorch -c nvidia && \
    conda install -y pytorch-scatter -c rusty1s && \
    conda install -c conda-forge -c fvcore fvcore

WORKDIR /workspace
RUN chmod -R a+w .

RUN apt-get update && apt-get install -y --no-install-recommends \
         software-properties-common && \
    apt-key adv --keyserver keys.gnupg.net --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE && \
    add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u && \
    apt-get update && apt-get install -y --no-install-recommends \
         librealsense2-dkms \
         librealsense2-utils \
         librealsense2-dev \
         librealsense2-dbg

ADD https://api.github.com/repos/FlorisE/ClearCompare/git/refs/heads/main clearcompare_version.json
RUN git clone https://github.com/FlorisE/ClearCompare

RUN cd /workspace/ClearCompare/implicit_depth/live_demo/realsense && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make

ADD https://api.github.com/repos/FlorisE/implicit_depth/git/refs/heads/main implicit_depth_version.json
RUN git clone https://github.com/FlorisE/implicit_depth && \
    cd implicit_depth && \
    conda activate lidf && \
    pip install --upgrade pip && \
    pip install torch && \
    pip install -r requirements.txt

RUN conda clean -ya

COPY checkpoints checkpoints
COPY cleargrasp cleargrasp
