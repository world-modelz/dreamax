#FROM nvidia/cuda:11.5.1-cudnn8-devel-ubuntu20.04
#FROM nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

# declare the image name
ENV IMG_NAME=11.4.0-devel-ubuntu20.04 \
    # declare what jaxlib tag to use
    # if a CI/CD system is expected to pass in these arguments
    # the dockerfile should be modified accordingly
    JAXLIB_VERSION=0.3.2

# install python3-pip
RUN apt update && apt install git curl wget vim less unzip htop iftop iotop build-essential autotools-dev nfs-common pdsh cmake g++ gcc ca-certificates ssh python3-dev libpython3-dev python3-pip -y
RUN apt-get install ffmpeg libsm6 libxext6 libx11-6 -y
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 && pip install --upgrade pip

# install dependencies via pip
RUN pip3 install numpy scipy six wheel jaxlib==${JAXLIB_VERSION}+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_releases.html jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_releases.html
RUN pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

COPY requirements.txt /tmp
RUN pip3 install -r /tmp/requirements.txt && rm /tmp/requirements.txt