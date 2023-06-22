import glob
import os
import sys
import time
import math
import weakref
from queue import Queue
from queue import Empty

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


def sensor_callback(sensor_data, sensor_queue, sensor_name, log_path):
    sensor_queue.put((sensor_data.frame, sensor_name))

    log_path_dir = os.path.dirname(log_path)
    # create the absolute path for the image
    img_rel_filepath = weather + "/" + sensor_name + "/{}_{}.png".format(sensor_data.frame, sensor_data.timestamp)
    img_abs_filepath = os.path.join(log_path_dir, img_rel_filepath)
    img_abs_path_dir = os.path.dirname(img_abs_filepath)
    # create dir if not exists
    if not os.path.exists(img_abs_path_dir):
        print("creating dir: ", img_abs_path_dir)
        os.makedirs(img_abs_path_dir)

    if (sensor_name[:10] == "rgb_camera"):
        data = sensor_data.raw_data
        buffer = np.frombuffer(data, dtype=np.uint8)
        buffer = buffer.reshape(sensor_data.height, sensor_data.width, 4)
        img = cv.cvtColor(buffer, cv.COLOR_BGRA2BGR)

        print("image filename:", img_rel_filepath)
        print("log_path_dir:", log_path_dir)
        print("saving image at: ", img_abs_filepath)

        # create dir if not exists
        if not os.path.exists(log_path_dir):
            os.makedirs(log_path_dir)

        cv.imwrite(img_abs_filepath, img)

    elif (sensor_name[:11] == "flow_camera"):
        flow = np.frombuffer(sensor_data.raw_data, dtype=np.float32)
        # print(flow)

        # create dir if not exists
        # if not os.path.exists(weather + "/" + sensor_name + "/flow_npz/"):
        #     os.mkdir(weather + "/" + sensor_name + "/flow_npz/")
        flow_rel_filepath = weather + "/" + sensor_name + "/flow_npz/{}_{}.npz".format(sensor_data.frame, sensor_data.timestamp)
        # create dirs if not exist
        flow_abs_filepath = os.path.join(log_path_dir, flow_rel_filepath)
        flow_abs_path_dir = os.path.dirname(flow_abs_filepath)
        if not os.path.exists(flow_abs_path_dir):
            print("creating dir: ", img_abs_path_dir)
            os.makedirs(flow_abs_path_dir)

        np.savez(flow_abs_filepath, flow=flow)
        image = sensor_data.get_color_coded_flow()
        data = image.raw_data
        # print(sensor_data,data.shape)
        buffer = np.frombuffer(data, dtype=np.uint8)
        buffer = buffer.reshape(sensor_data.height, sensor_data.width, 4)
        img = cv.cvtColor(buffer, cv.COLOR_BGRA2BGR)
        cv.imwrite(img_abs_filepath, img)


def main(world, weather_param, weather, num_camera, i, num_imgs, log_path):
    # client = carla.Client('127.0.0.1', 2000)
    # client.set_timeout(10.0)

    global print_info_once
    if print_info_once:
        print("trying to use log file for image generation: ", log_path)
        client.show_recorder_file_info(log_path, True)
        print_info_once = False
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

        sensor_queue = Queue()

        # create dir if not exist
        if not os.path.exists(weather):
            os.makedirs(weather)
        sensor_list = []

        if (abs(i) <= num_camera//4):
            position_x = 1.7
            position_z = 3
        elif (abs(i) > num_camera//4):
            position_x = -1.7
            position_z = 3

        blueprint_library = world.get_blueprint_library()

        rgb_camera_bp = blueprint_library.find('sensor.camera.rgb')
        flow_camera_bp = blueprint_library.find('sensor.camera.optical_flow')

        rgb_camera_bp.set_attribute("image_size_x", str(1920))
        rgb_camera_bp.set_attribute("image_size_y", str(1080))
        rgb_camera_bp.set_attribute("fov", str(105))

        flow_camera_bp.set_attribute("image_size_x", str(1920))
        flow_camera_bp.set_attribute("image_size_y", str(1080))
        flow_camera_bp.set_attribute("fov", str(105))

        # create dir if not exist
        if not os.path.exists(weather + "/rgb_camera_{}".format(i)):
            os.makedirs(weather + "/rgb_camera_{}".format(i))

        if not os.path.exists(weather + "/flow_camera_{}".format(i)):
            os.makedirs(weather + "/flow_camera_{}".format(i))

        camera_transform = carla.Transform(carla.Location(x=position_x, z=position_z), carla.Rotation(yaw=(360/num_camera)*i))
        camera = world.spawn_actor(rgb_camera_bp, camera_transform, attach_to=ego_vehicle)
        sensors.append(camera)
        print('created %s' % camera.type_id)

        flow_camera_transform = carla.Transform(carla.Location(x=position_x, z=position_z), carla.Rotation(yaw=(360/num_camera)*i))
        flow_camera = world.spawn_actor(flow_camera_bp, flow_camera_transform, attach_to=ego_vehicle)
        sensors.append(flow_camera)
        print('created %s' % flow_camera.type_id)

        camera.listen(lambda data: sensor_callback(data, sensor_queue, "rgb_camera_{}".format(i), log_path))
        sensor_list.append(camera)

        flow_camera.listen(lambda data: sensor_callback(data, sensor_queue, "flow_camera_{}".format(i), log_path))
        sensor_list.append(flow_camera)

        print(sensor_list)
        count = 0
        # Main loop
        while count < num_imgs:
            # Tick the server
            world.tick()
            w_frame = world.get_snapshot().frame
            print("\nWorld's frame: %d" % w_frame)
            count += 1
            # Now, we wait to the sensors data to be received.
            # As the queue is blocking, we will wait in the queue.get() methods
            # until all the information is processed and we continue with the next frame.
            # We include a timeout of 1.0 s (in the get method) and if some information is
            # not received in this time we continue.
            try:
                for _ in range(len(sensor_list)):
                    s_frame = sensor_queue.get(True, 1.0)
                    print("    Frame: %d   Sensor: %s" % (s_frame[0], s_frame[1]))

            except Empty:
                print("    Some of the sensor information is missed")

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

    parser = argparse.ArgumentParser(description='CARLA Sensor Data Recorder')
    parser.add_argument('--log_path', type=str, help='Path to the log file')
    args = parser.parse_args()
    log_path = args.log_path

    global print_info_once
    print_info_once = True

    weather_list = [
        # carla.WeatherParameters.SoftRainNight,
        carla.WeatherParameters.ClearNoon,
        carla.WeatherParameters.CloudyNoon,
        # carla.WeatherParameters.HardRainNight,
        # carla.WeatherParameters.WetNoon,
        # carla.WeatherParameters.WetCloudyNoon,
        carla.WeatherParameters.MidRainyNoon,
        # carla.WeatherParameters.HardRainNoon,
        carla.WeatherParameters.SoftRainNoon,
        # carla.WeatherParameters.ClearSunset,
        # carla.WeatherParameters.CloudySunset,
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
        # 'SoftRainNight',
        'ClearNoon',
        'CloudyNoon',
        # 'HardRainNight',
        # 'WetNoon',
        # 'WetCloudyNoon',
        'MidRainyNoon',
        # 'HardRainNoon',
        'SoftRainNoon',
        #  'ClearSunset',
        #  'CloudySunset',
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

    num_camera = 8
    # -7 to 7
    num_imgs = 500
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)
    world = client.get_world()
    # transform = world.get_map().get_spawn_points()[-1]
    # print(transform)
    for j in range(len(weather_list)):
        weather_param = weather_list[j]
        weather = "datasets/carla/" + weather_name_list[j]
        print(weather)
        # i = -(num_camera//2 - 1)
        for i in range(-(num_camera//2 - 1), num_camera//2 + 1):
            try:
                main(world, weather_param, weather, num_camera, i, num_imgs, log_path)
            except RuntimeError as e:
                print(e)
                continue
