FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

# Install some basic utilities
RUN apt update && apt install -y \
   curl \
   ca-certificates \
   sudo \
   git \
   bzip2 \
   libx11-6 \
   libgl1-mesa-glx \
   libglib2.0-0 \
   
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
   && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda and Python 3.8
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py38_4.8.3-Linux-x86_64.sh \
   && chmod +x ~/miniconda.sh \
   && ~/miniconda.sh -b -p ~/miniconda \
   && rm ~/miniconda.sh \
   && conda install -y python==3.8.3 \
   && conda clean -ya

# CUDA 11.1-specific steps
RUN conda install -y -c conda-forge cudatoolkit=11.1.1 \
   && conda install -y -c pytorch \
      "pytorch=1.8.1=py3.8_cuda11.1_cudnn8.0.5_0" \
      "torchvision=0.9.1=py38_cu111" \
   && conda clean -ya

RUN pip install Flask \
   jupyterlab \
   numpy>=1.15.0 \
   Pillow>=5.0.0 \
   opencv-contrib-python>=3.4.0 \
   cupy-cuda111

EXPOSE 8888

WORKDIR /workspace

ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0"]