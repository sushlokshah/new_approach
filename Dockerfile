FROM carlasim/carla:0.9.13

# Remove CUDA and NVIDIA apt repos for docker image to be able to build, use root user to do this
USER root

# Set the shell to bash shell
SHELL ["/bin/bash", "-c", "-l"]

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update 
RUN apt-get install -y curl tar

# Give the user carla ownership of the data_generation folder in order to be able to write to it and create folders
RUN mkdir /home/carla/data_generation
RUN mkdir /home/carla/data_generation/new_approach
RUN mkdir /home/carla/data_generation/dataset
RUN chown -R carla:carla /home/carla/data_generation
RUN chown -R carla:carla /home/carla/data_generation/new_approach
RUN chown -R carla:carla /home/carla/data_generation/dataset

USER carla

# Install micromaba
RUN curl -Ls https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
COPY micromamba_bashrc.txt /home/carla/micromamba_bashrc.txt 
RUN ./bin/micromamba shell init -s bash -p /home/carla/micromamba && cat micromamba_bashrc.txt >> .bashrc && source .bashrc
ENV MAMBA_ROOT_PREFIX=/home/carla/micromamba

# Install the conda environment
COPY env_data_generation.yaml /home/carla/data_generation/env_data_generation.yaml
RUN ./bin/micromamba create -f /home/carla/data_generation/env_data_generation.yaml

COPY . /home/carla/data_generation/new_approach

# Add the pip installed packages to python path so that user root can also use them
ENV PYTHONPATH "${PYTHONPATH}:/home/carla/.local/lib/python3.6/site-packages"