FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04
SHELL ["/bin/bash", "-c"]
# Set the timezone info because otherwise tzinfo blocks install 
# flow and ignores the non-interactive frontend command 🤬🤬🤬
RUN ln -snf /usr/share/zoneinfo/America/New_York /etc/localtime && echo "/usr/share/zoneinfo/America/New_York" > /etc/timezone

# Core system packages
RUN apt-get update --fix-missing
RUN apt install -y software-properties-common wget curl gpg gcc git make

# Install miniconda to /miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

# Additional dev packages
RUN apt install -y --no-install-recommends libssl-dev libmodule-install-perl libboost-all-dev libgl1-mesa-dev libopenblas-dev


RUN conda install python=3.7 -y
RUN conda config --set channel_priority strict
RUN conda install cudatoolkit-dev=11.1 -c conda-forge -y
RUN conda install pytorch=1.8 torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia

# Setup and install kaolin with version v0.1
RUN git clone --depth 1 --branch v0.1 https://github.com/NVIDIAGameWorks/kaolin
WORKDIR kaolin
ENV TORCH_CUDA_ARCH_LIST="Ampere;Turing;Pascal"
RUN python setup.py develop
RUN cd ..

WORKDIR /project
RUN pip install open3d
