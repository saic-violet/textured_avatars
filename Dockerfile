FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-devel

RUN apt-get update && apt-get install -y \
    software-properties-common \
    swig \
    libsm6 \
    libxext6 \
    libxrender-dev

RUN pip install \
    opencv-python

RUN apt-get install -y libsm6 libxext6 libxrender-dev
