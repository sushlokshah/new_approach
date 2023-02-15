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
    2. generate traffic and record data: (current setup 1000 frames per camera configuration and total 8 camera comfiguration)
    ```
        python3 datasets/data_generation/generate_traffic.py
        python3 datasets/data_generation/carla_data_recorder.py
    ```
    3. weather conditions can be changed by commenting and uncommenting the following list from carla_data_recorder.py code
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
