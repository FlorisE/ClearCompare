FROM nvidia/cudagl:11.4.0-devel-ubuntu20.04 
ARG PYTHON_VERSION=3.8

SHELL ["/bin/bash", "--login", "-c"]

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        git \
        libxcb-xinerama0 \
        libsm6 \
        wget \
        unzip


WORKDIR /workspace
RUN chmod -R a+w .

# Install Anaconda
ENV PATH /opt/conda/bin:$PATH
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.profile

RUN conda init bash

RUN conda create --name keypose python=3.7
RUN echo "conda activate keypose" >> ~/.bashrc

# Checkout only the KeyPose repository
RUN git init && \
    git remote add origin https://github.com/google-research/google-research && \
    git config core.sparseCheckout true && \
    echo "keypose" >> .git/info/sparse-checkout && \
    git pull --depth=1 origin master

RUN cd keypose && \
    conda activate keypose && \
    pip3 install -r requirements.txt

RUN keypose/download_models.sh

RUN rm -rf /var/lib/apt/lists/*
