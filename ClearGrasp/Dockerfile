FROM nvidia/cudagl:11.1.1-devel-ubuntu18.04 
ARG PYTHON_VERSION=3.7

SHELL ["/bin/bash", "--login", "-c"]

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    git \
    libhdf5-100 \
    libhdf5-serial-dev \
    libhdf5-dev \
    libhdf5-cpp-100 \
    libopenexr-dev \
    zlib1g-dev \
    openexr \
    xorg-dev \
    libglfw3-dev

# Install Anaconda
ENV PATH /opt/conda/bin:$PATH
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.profile

RUN conda init bash

RUN conda create --name cleargrasp python=3.6
RUN echo "conda activate cleargrasp" >> ~/.bashrc

WORKDIR /workspace
RUN chmod -R a+w .

# Install realsense
RUN apt-get update && apt-get install -y --no-install-recommends \
         software-properties-common && \
    apt-key adv --keyserver keys.gnupg.net --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE && \
    add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u && \
    apt-get update && apt-get install -y --no-install-recommends \
         librealsense2-dkms \
         librealsense2-utils \
         librealsense2-dev \
         librealsense2-dbg

# Install cleargrasp
RUN apt-get install -y --no-install-recommends \
        git \
        cmake \
        libgl1-mesa-dev \
        libglu1-mesa-dev
ADD https://api.github.com/repos/FlorisE/cleargrasp/git/refs/heads/master cleargrasp_version.json
RUN git clone https://github.com/FlorisE/cleargrasp && \
    cd cleargrasp/live_demo && \
    cp config/config.yaml.sample config/config.yaml && \
    cd realsense && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make

# Build depth2depth
RUN cd /workspace/cleargrasp/api/depth2depth/gaps && \
    export CPATH="/usr/include/hdf5/serial/" && \
    make

# Install Python requirements
RUN conda activate cleargrasp && \
    cd cleargrasp && \
    pip install -r requirements.txt

COPY cleargrasp-checkpoints /workspace/cleargrasp/data/cleargrasp/cleargrasp-checkpoints

# Install conda libraries
RUN conda activate cleargrasp && \
    conda install -y pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia

# Clean package cache
RUN rm -rf /var/lib/apt/lists/*
