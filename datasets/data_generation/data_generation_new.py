import glob
import os
import sys
import time
import math
import weakref
from queue import Queue
from queue import Empty
import yaml
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import numpy as np
import cv2 as cv

import argparse
import logging
import random
from pprint import pprint

def save_data(camera_sensor_queue, flow_sensor_queue,path, log_path):
    print("saving data")
    log_path_dir = os.path.dirname(log_path)
    
    # create the absolute path for the image
    cam_data = list(camera_sensor_queue.queue)
    flow_data_list = list(flow_sensor_queue.queue)
    for i in range(1,len(list(camera_sensor_queue.queue))):
        frame_name, image_data= cam_data[i]
        frame_name, flow_data = flow_data_list[i]
        flow = np.frombuffer(flow_data.raw_data, dtype=np.float32)
        
        # if flow mag is very small then skip
        # if np.mean(np.abs(flow)) < 0.5:
        #     continue
        print("flow mag:", np.mean(np.abs(flow)))
        if np.mean(np.abs(flow)) < 0.01:
            continue
        img_rel_filepath_old = path + "/image_0/{}.png".format(image_data.frame)
        img_rel_filepath_current = path + "/image_1/{}.png".format(image_data.frame)
        img_abs_filepath_old = os.path.join(log_path_dir, img_rel_filepath_old)
        img_abs_path_dir_old = os.path.dirname(img_abs_filepath_old)
        img_abs_filepath_current = os.path.join(log_path_dir, img_rel_filepath_current)
        img_abs_path_dir_current = os.path.dirname(img_abs_filepath_current)
        # create dir if not exists
        if not os.path.exists(img_abs_path_dir_old):
            print("creating dir: ", img_abs_path_dir_old)
            os.makedirs(img_abs_path_dir_old)
            
        if not os.path.exists(img_abs_path_dir_current):
            print("creating dir: ", img_abs_path_dir_current)
            os.makedirs(img_abs_path_dir_current)
        
        
        # flow rel filepath
        flow_rel_filepath = path + "/flow_npz/{}.npz".format(image_data.frame)
        # create dirs if not exist
        flow_abs_filepath = os.path.join(log_path_dir, flow_rel_filepath)
        flow_abs_path_dir = os.path.dirname(flow_abs_filepath)
        if not os.path.exists(flow_abs_path_dir):
            print("creating dir: ", flow_abs_path_dir)
            os.makedirs(flow_abs_path_dir)
            
        # flow vis
        flow_vis_rel_filepath = path + "/flow_vis/{}.png".format(image_data.frame)
        flow_vis_abs_filepath = os.path.join(log_path_dir, flow_vis_rel_filepath)
        flow_vis_abs_path_dir = os.path.dirname(flow_vis_abs_filepath)
        if not os.path.exists(flow_vis_abs_path_dir):
            print("creating dir: ", flow_vis_abs_path_dir)
            os.makedirs(flow_vis_abs_path_dir)

        

        current_data = image_data.raw_data
        old_data = cam_data[i-1][1].raw_data
        current_buffer = np.frombuffer(current_data, dtype=np.uint8)
        old_buffer = np.frombuffer(old_data, dtype=np.uint8)
        
        current_buffer = current_buffer.reshape(image_data.height, image_data.width, 4)
        old_buffer = old_buffer.reshape(image_data.height, image_data.width, 4)
        
        current_img = cv.cvtColor(current_buffer, cv.COLOR_BGRA2BGR)
        old_img = cv.cvtColor(old_buffer, cv.COLOR_BGRA2BGR)
        
        print("saving image0: ", img_abs_filepath_old)
        print("saving image1: ", img_abs_filepath_current)
        print("saving flow: ", flow_abs_filepath)
        print("saving flow_vis: ", flow_vis_abs_filepath)
        
        cv.imwrite(img_abs_filepath_old, old_img)
        cv.imwrite(img_abs_filepath_current, current_img)
        
        np.savez(flow_abs_filepath, flow=flow)
        
        # flow vis
        flow_vis = flow_data.get_color_coded_flow()
        data = flow_vis.raw_data
        # print(sensor_data,data.shape)
        buffer = np.frombuffer(data, dtype=np.uint8)
        buffer = buffer.reshape((flow_vis.height, flow_vis.width, 4))
        flow_vis_img = cv.cvtColor(buffer, cv.COLOR_BGRA2BGR)
        cv.imwrite(flow_vis_abs_filepath, flow_vis_img)
        
    


def sensor_callback(sensor_data, camera_queue, flow_queue, sensor_name, log_path):
    if sensor_name[:11] != "flow_camera":
        sensor_queue = camera_queue
    else:
        sensor_queue = flow_queue
    sensor_queue.put((sensor_data.frame, sensor_data))

def main(world, weather_param, path, sensor, log_path, env_state):
    
    # client = carla.Client('127.0.0.1', 2000)
    # client.set_timeout(10.0)

    # global print_info_once
    # if print_info_once:
    #     print("trying to use log file for image generation: ", log_path)
    #     client.show_recorder_file_info(log_path, True)
    #     print_info_once = False
    try:
        world = client.get_world()
        ego_vehicle = None
        sensors = []

        # --------------
        # Query the recording
        # --------------

        # Show the most important events in the recording.
        # print(client.show_recorder_file_info("/home/sushlok/new_approach/datasets/data_generation/recording01.log",False))
        # Show actors not moving 1 meter in 10 seconds.
        # print(client.show_actors("/home/sushlok/new_approach/datasets/data_generation/recording01.log",10,1))
        # Show collisions between any type of actor.
        # print(client.show_recorder_collisions("~/tutorial/recorder/recording04.log",'v','a'))

        # --------------
        # Reenact a fragment of the recording
        # --------------

        # client.replay_file("/home/sushlok/new_approach/datasets/data_generation/recording02.log", 0, 1000, 0)
        client.replay_file(log_path, 0, 1000, 0)

        world = client.get_world()
        original_settings = world.get_settings()
        settings = world.get_settings()

        # We set CARLA syncronous mode
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True
        world.apply_settings(settings)
        world.set_weather(weather_param)

        # get list of all the actors from the recording and finding the required vehicle
        print(world.get_actors())
        for actor in world.get_actors():
            if actor.attributes.get('role_name') == 'hero':
                print("found ego_vehicle")
                ego_vehicle = actor
                print(ego_vehicle)
                break

        # sensors.append(ego_vehicle)
        print('created %s' % ego_vehicle.type_id)

        ego_vehicle.set_autopilot(True)

        camera_sensor_queue = Queue()
        flow_sensor_queue = Queue()

        # create dir if not exist
        if not os.path.exists(path):
            os.makedirs(path)
        sensor_list = []
        num_images = env_state['num_camera_images']
        
        blueprint_library = world.get_blueprint_library()

        rgb_camera_bp = blueprint_library.find('sensor.camera.rgb')
        flow_camera_bp = blueprint_library.find('sensor.camera.optical_flow')

        rgb_camera_bp.set_attribute("image_size_x", str(1920))
        rgb_camera_bp.set_attribute("image_size_y", str(1080))
        rgb_camera_bp.set_attribute("fov", str(105))

        flow_camera_bp.set_attribute("image_size_x", str(1920))
        flow_camera_bp.set_attribute("image_size_y", str(1080))
        flow_camera_bp.set_attribute("fov", str(105))

        # # create dir if not exist
        # if not os.path.exists(weather + "/rgb_camera_{}".format(i)):
        #     os.makedirs(weather + "/rgb_camera_{}".format(i))

        # if not os.path.exists(weather + "/flow_camera_{}".format(i)):
        #     os.makedirs(weather + "/flow_camera_{}".format(i))

        camera_transform = carla.Transform(carla.Location(x=sensor['pos_x'], y = sensor['pos_y'],z=sensor['pos_z']), carla.Rotation(roll=sensor['roll'], pitch=sensor['pitch'], yaw=sensor['yaw']))
        camera = world.spawn_actor(rgb_camera_bp, camera_transform, attach_to=ego_vehicle)
        sensors.append(camera)
        print('created %s' % camera.type_id)

        flow_camera_transform = carla.Transform(carla.Location(x=sensor['pos_x'], y = sensor['pos_y'],z=sensor['pos_z']), carla.Rotation(roll=sensor['roll'], pitch=sensor['pitch'], yaw=sensor['yaw']))
        flow_camera = world.spawn_actor(flow_camera_bp, flow_camera_transform, attach_to=ego_vehicle)
        sensors.append(flow_camera)
        print('created %s' % flow_camera.type_id)

        camera.listen(lambda data: sensor_callback(data, camera_sensor_queue, flow_sensor_queue, "rgb_camera_{}".format(sensor['name']), log_path))
        sensor_list.append(camera)

        flow_camera.listen(lambda data: sensor_callback(data, camera_sensor_queue, flow_sensor_queue,"flow_camera_{}".format(sensor['name']), log_path))
        sensor_list.append(flow_camera)

        print(sensor_list)
        count = 0
        # Main loop
        while count < num_images:
            # Tick the server
            print(count)
            world.tick()
            w_frame = world.get_snapshot().frame
            print("\nWorld's frame: %d" % w_frame)
            count += 1
            # Now, we wait to the sensors data to be received.
            # As the queue is blocking, we will wait in the queue.get() methods
            # until all the information is processed and we continue with the next frame.
            # We include a timeout of 1.0 s (in the get method) and if some information is
            # not received in this time we continue.
            # try:
            #     for _ in range(len(sensor_list)):
            #         s_frame = camera_sensor_queue.get(True, 0.1)
            #         s_frame_1 = flow_sensor_queue.get(True, 0.1)
            #         print("    Frame: %d   Sensor: %s" % (s_frame[0], s_frame[1]))
            #         print("    Frame: %d   Sensor: %s" % (s_frame_1[0], s_frame_1[1]))

            # except Empty:
            #     print("    Some of the sensor information is missed")
        
        # print(list(camera_sensor_queue.queue))
        # print(list(flow_sensor_queue.queue))
        
        save_data(camera_sensor_queue, flow_sensor_queue, path + "/" + sensor['name'], log_path)

    finally:
        print("distoring the ego_vehicle")
        # world.apply_settings(original_settings)
        print(sensors)
        for sensor in sensors:
            try:
                sensor.stop()
                sensor.destroy()
            except:
                pass
        # ego_vehicle.destroy()
        print('done.')


if __name__ == "__main__":

    weather_parameters_dict = {'SoftRainNight': carla.WeatherParameters.SoftRainNight,
                                'ClearNoon': carla.WeatherParameters.ClearNoon,
                                'CloudyNoon': carla.WeatherParameters.CloudyNoon,
                                'HardRainNight': carla.WeatherParameters.HardRainNight,
                                'WetNoon': carla.WeatherParameters.WetNoon,
                                'WetCloudyNoon': carla.WeatherParameters.WetCloudyNoon,
                                'MidRainyNoon': carla.WeatherParameters.MidRainyNoon,
                                'HardRainNoon': carla.WeatherParameters.HardRainNoon,
                                'SoftRainNoon': carla.WeatherParameters.SoftRainNoon,
                                'ClearSunset': carla.WeatherParameters.ClearSunset,
                                'CloudySunset': carla.WeatherParameters.CloudySunset,
                                'WetSunset': carla.WeatherParameters.WetSunset,
                                'WetCloudySunset': carla.WeatherParameters.WetCloudySunset,
                                'MidRainSunset': carla.WeatherParameters.MidRainSunset,
                                'HardRainSunset': carla.WeatherParameters.HardRainSunset,
                                'SoftRainSunset': carla.WeatherParameters.SoftRainSunset,
                                'ClearNight': carla.WeatherParameters.ClearNight,
                                'CloudyNight': carla.WeatherParameters.CloudyNight,
                                'WetNight': carla.WeatherParameters.WetNight,
                                'WetCloudyNight': carla.WeatherParameters.WetCloudyNight}
    
    parser = argparse.ArgumentParser(description='CARLA Sensor Data Recorder')
    parser.add_argument('--log_path', type=str, help='Path to the log file')
    parser.add_argument('--env_state', type=str, help='config file for the environment and recorder car')
    args = parser.parse_args()
    log_path = args.log_path
    
    # convert env_state.yaml to dict
    env_state = {}
    with open(args.env_state, 'r') as f:
        env_state = yaml.load(f, Loader=yaml.FullLoader)
        
        
    weather_list = env_state['weather_list']

    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)
    world = client.get_world()

    for weather in weather_list:
        weather_param = weather_parameters_dict[weather]
        path = "/home/sushlok/new_approach/datasets/data_generation/datasets2/" + weather
        os.makedirs(path, exist_ok=True)
        for key, sensor in env_state['sensor_config'].items():
            try:
                main(world, weather_param, path, sensor, log_path, env_state)
            except RuntimeError as e:
                print(e)
                continue
