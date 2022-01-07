#
# This example Dockerfile illustrates a method to install
# additional packages on top of NVIDIA's PyTorch container image.
#
# To use this Dockerfile, use the `docker build` command.
# See https://docs.docker.com/engine/reference/builder/
# for more information.
#
# Software available with the container version 20.11 is listed at:
# https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html
#
FROM nvcr.io/nvidia/pytorch:20.11-py3

# Python OpenCV should be available, if not, run the following.
RUN apt-get update
RUN apt-get upgrade -y --no-install-recommends
RUN apt-get update

RUN pip install --upgrade pip
RUN pip install opencv-python==3.4.8.29

# Add a user if desirable
#RUN useradd -ms /bin/bash -u <uid> <user_name>
#USER <user_name>
#WORKDIR /home/<user_name>
