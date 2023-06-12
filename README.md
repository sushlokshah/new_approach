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
