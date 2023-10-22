# New_approach (suggest name)

## Directory Structure:
```
    alt_cuda_corr
    core
    datasets
    -- config
    -- data_generation
    envs
    general_file
    pretrained_models
    README.md
    runs
    utils

```
## download pretrained_models
1. link: https://drive.google.com/drive/folders/1PuXwJSyMlG_eYOCDbqB6hU9otF-oK5-4?usp=share_link
2. directory structure:
```
pretrained_models
    -- gmflownet_mix-sintel.pth
    -- gmflownet_mix-things.pth
    -- gmflownet-kitti.pth
    -- gmflownet-thing.pth
    -- new_model.pth
``` 

## dataset generation
1. carla setup instructions(Debian CARLA installation)
    The Debain package is available for both Ubuntu 18.04 and Ubuntu 20.04, however the officially supported platform is Ubuntu 18.04.
    1. Set up the Debian repository in the system:
    ```
        sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 1AF1527DE64CB8D9
        sudo add-apt-repository "deb [arch=amd64] http://dist.carla.org/carla $(lsb_release -sc) main"
    ```
    2. Install CARLA and check for the installation in the /opt/ folder:
    ```
        sudo apt-get update # Update the Debian package index
        sudo apt-get install carla-simulator=0.9.12 # Install CARLA
    ```
2. Data generation:
    1. Run the CARLA simulator:
    ```
        CUDA_VISIBLE_DEVICES=0 bash /opt/carla-simulator/CarlaUE4.sh -RenderOffScreen
    ```
    2. generate traffic
    edit path from file generate_traffic.py on line number 116 i.e 
    ```
    client.start_recorder('/home/sushlok/new_approach/datasets/data_generation/recording02.log')
    ```
    with your absolute path and run the following commands

    ```
        python3 datasets/data_generation/generate_traffic.py
    ```
    run this program for 2-3 mins. it will generate a recording file with all the agents in the scene.

    3. append the same path in data_generation.py file on line number 81 i.e
    ```
    client.replay_file("/home/sushlok/new_approach/datasets/data_generation/recording02.log",0,1000,0)
    ```
    edit the path with your absolute path and run the following command
    ```
        python3 datasets/data_generation/data_generation.py
    ```

    ```
    4. weather conditions can be changed by commenting and uncommenting the following list from data_generation.py code
    ```
    weather_list = [
        # carla.WeatherParameters.SoftRainNight,
        # carla.WeatherParameters.ClearNoon,
        carla.WeatherParameters.CloudyNoon,
        # carla.WeatherParameters.HardRainNight,
        # carla.WeatherParameters.WetNoon,
        # carla.WeatherParameters.WetCloudyNoon,
        # carla.WeatherParameters.MidRainyNoon,
        # carla.WeatherParameters.HardRainNoon,
        # carla.WeatherParameters.SoftRainNoon,
        carla.WeatherParameters.ClearSunset,
        carla.WeatherParameters.CloudySunset,
        # carla.WeatherParameters.WetSunset,
        # carla.WeatherParameters.WetCloudySunset,
        # carla.WeatherParameters.MidRainSunset,
        # carla.WeatherParameters.HardRainSunset,
        # carla.WeatherParameters.SoftRainSunset,
        # carla.WeatherParameters.ClearNight,
        # carla.WeatherParameters.CloudyNight,
        # carla.WeatherParameters.WetNight,
        # carla.WeatherParameters.WetCloudyNight
        
    ]

    weather_name_list = [
        #  'SoftRainNight',
        #  'ClearNoon',
         'CloudyNoon',
        # 'HardRainNight',
        # 'WetNoon',
        # 'WetCloudyNoon',
        # 'MidRainyNoon',
        # 'HardRainNoon',
        # 'SoftRainNoon',
         'ClearSunset',
         'CloudySunset',
        # 'WetSunset',
        # "WetCloudySunset",
        # "MidRainSunset",
        # 'HardRainSunset',
        # 'SoftRainSunset',
        # 'ClearNight',
        # 'CloudyNight',
        # 'WetNight',
        # 'WetCloudyNight'
    ]
    ```

3. final directory structure:
```
datasets
    -- carla
    -- config
    -- data_generation
```

4. create conda environment 
```
    conda env create -f envs/gmflownet.yml
```

5. update configuration file with absolute paths and run type in config.yml file and run the training code
```
    python3 train.py

```

## Dataset generation using Docker

1. Build the docker container. This will take a while because of opencv-python dependency. Navigate to this git repo and run the following command:
```
    docker build -t user/carla_data_generation:TAG_NUMBER .
    docker build -t sushlok/carla_data_generation:0 . # example
```

2. Create a docker volume to store the dataset. Run the following command:
```
    docker volume create --name carla_dataset --opt type=none --opt device=/path/to/local/dataset/folder --opt o=bind
```

3. Inside the terminal running the docker container, run the following commands:
```
    xhost +

    docker run --privileged --name carlaserver --mount source=carla_dataset,target=/path/to/local/dataset/folder -v /tmp/.X11-unix:/tmp/.X11-unix -it --gpus all -p 2000-2002:2000-2002 sushlok/carla-data_generation:0 ./CarlaUE4.sh -RenderOffScreen
```

4. Pull the latest version of this repo inside the docker container using root user. Inside another terminal run the following commands:
```
    docker exec -u 0 -it carlaserver bash # we need -u 0 for root access
    # Inside the docker container
    cd data_generation
    ./download_repo_from_github.sh # When asked enter A for replacing all files
```

5. Run the generate_traffic and data_generation scripts inside the docker container using the regular carla user. Enter the docker container and run the following commands (make sure not to overwrite previous recordings):
```
    docker exec -it carlaserver bash
    # Inside the docker container
    cd data_generation/new_approach_master/datasets/data_generation
    python3 generate_traffic.py --log_path /path/to/local/dataset/folder/recording02.log
    python3 data_generation.py --log_path /path/to/local/dataset/folder/recording02.log
```


### Training using vscode devcontainer
1. Install docker vscode extension. (first install docker and nvidia-docker on system before downloading vscode extension)
2. `.devcontainer` folder contains the docker file and configuration file.
3. The post processing task is included in the `devcontainer.json` file which will execute `install_requirements.sh` file.
4. After building the enviroment activate gmflownet env by using 

```
conda activate gmflownet

```
5. Edit paths from config file related to dataset
6. Set training procedure, for ex. `train: true` 
7. finally run train.py code using
```
python train.py
```
8. for running sweep with tune, absolute path must be provided in the config.yml file
