FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

ARG PYTHON_VERSION=3.7.5

RUN apt-get update && apt-get upgrade -y && apt install -y \
    wget \
    tar \
    zip \
    unzip \
    vim \
    git \
    libxrender1 \
    && apt-get clean

RUN set -ex \
    && wget --quiet --no-check-certificate https://repo.continuum.io/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh \
        && /bin/bash Miniconda3-py37_4.10.3-Linux-x86_64.sh -f -b -p /opt/miniconda \
        && export PATH="/opt/miniconda/bin:$PATH" \
    && rm -rf \
        /Miniconda3-py37_4.10.3-Linux-x86_64.sh \
    && conda update --all \
    && conda install -y \
        rdkit \
        -c rdkit \
    && conda install -y \
        jupyterlab \
        matplotlib \
        tensorboard \
        -c conda-forge \
    && conda install -y \
        pytorch \
        torchvision \
        torchaudio \
        cudatoolkit=10.2 \
        -c pytorch \
    && conda install -y \
        pyg \
        -c pyg \
        -c conda-forge \
    && conda clean -yq -a

ENV PATH /opt/miniconda/bin:$PATH

ADD . /requirements
RUN pip install -U -r /requirements/requirements.txt
