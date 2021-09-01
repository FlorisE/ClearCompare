FROM nvidia/cudagl:11.1.1-devel-ubuntu18.04 
ARG PYTHON_VERSION=3.7

SHELL ["/bin/bash", "--login", "-c"]

WORKDIR /workspace

RUN apt-get update

RUN apt-get install -y --no-install-recommends \
        curl \
        git \
        libhdf5-100 \
        libhdf5-serial-dev \
        libhdf5-dev \
        libhdf5-cpp-100 \
        libgl1-mesa-dev \
        libglu1-mesa-dev \
        cmake \
        libopenexr-dev \
        openexr \
        libglfw3-dev\
        libsm6 \
        libxrender1 \
        libxext-dev

# Install Anaconda
ENV PATH /opt/conda/bin:$PATH
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.profile

RUN conda init bash

RUN conda create --name depth2depth python=3.6
RUN echo "conda activate depth2depth" >> ~/.bashrc

COPY depth2depth /workspace/depth2depth

# Build depth2depth
RUN cd /workspace/depth2depth && \
    export CPATH="/usr/include/hdf5/serial/" && \
    make

RUN conda activate depth2depth && \
    cd /workspace/depth2depth && \
    pip install -r requirements.txt

# Clean package cache
RUN rm -rf /var/lib/apt/lists/*
