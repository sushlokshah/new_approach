FROM carlasim/carla:0.9.13

# Remove CUDA and NVIDIA apt repos for docker image to be able to build, use root user to do this
USER root

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update 
RUN apt-get install -y python3-pip wget unzip 

RUN mkdir /home/carla/data_generation
COPY download_repo_from_github.sh /home/carla/data_generation/download_repo_from_github.sh
RUN chmod +x /home/carla/data_generation/download_repo_from_github.sh
RUN bash /home/carla/data_generation/download_repo_from_github.sh

# Give the user carla ownership of the data_generation folder in order to be able to write to it and create folders
RUN chown -R carla:carla /home/carla/data_generation

# Set the user to non-root user when running the container
USER carla

# Upgrade pip to latest version
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install --upgrade pip

RUN pip3 install carla==0.9.13
RUN pip3 install numpy
# This takes a long time because it builds the opencv from source, but it is necessary to use the cv2 library in python
# TODO use micromamba or miniconda to install opencv (not so trivial inside docker)
# Once the docker was built once with the next line, you can add docker instructions below it and it will use the cached image
# If you change lines above it, it will rebuild the image and not cache opencv-python (long build time)
RUN pip3 install opencv-python --verbose

VOLUME [ "/home/carla/data_generation/dataset" ]
