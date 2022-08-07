#!/usr/bin/env python

import glob
import os
import sys
from queue import Queue
from queue import Empty
import random
import time
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

# Sensor callback.

def sensor_callback(sensor_data, sensor_queue, sensor_name):
    sensor_queue.put((sensor_data.frame, sensor_name))
    
    if(sensor_name[:10] == "rgb_camera"):    
        data = sensor_data.raw_data
        buffer = np.frombuffer(data, dtype=np.uint8)
        buffer = buffer.reshape(sensor_data.height, sensor_data.width, 4)
        img = cv.cvtColor(buffer, cv.COLOR_BGRA2BGR)
        
        cv.imwrite(weather + "/" + sensor_name + "/{}_{}.png".format(sensor_data.frame,sensor_data.timestamp),img)
        
    elif(sensor_name[:11] == "flow_camera"):
        flow  = np.frombuffer(sensor_data.raw_data, dtype=np.float32)
        # print(flow)
        
        # create dir if not exists
        if not os.path.exists(weather + "/" + sensor_name + "/flow_npz/"):
            os.mkdir(weather + "/" + sensor_name + "/flow_npz/")
            
        np.savez(weather + "/" + sensor_name + "/flow_npz/{}_{}.npz".format(sensor_data.frame,sensor_data.timestamp),flow = flow)
        image = sensor_data.get_color_coded_flow()
        data = image.raw_data
        # print(sensor_data,data.shape)
        buffer = np.frombuffer(data, dtype=np.uint8)
        buffer = buffer.reshape(sensor_data.height, sensor_data.width, 4)
        img = cv.cvtColor(buffer, cv.COLOR_BGRA2BGR)
        cv.imwrite(weather + "/" + sensor_name + "/{}_{}.png".format(sensor_data.frame,sensor_data.timestamp),img)

        
def main(world, weather_param,weather,num_camera,i,num_imgs,transform):
    # We start creating the client
    actor_list = []
    world.set_weather(weather_param)
    try:
        
        original_settings = world.get_settings()
        settings = world.get_settings()

        # We set CARLA syncronous mode
        settings.fixed_delta_seconds = 0.2
        settings.synchronous_mode = True
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()
        
        vehicle_bp = blueprint_library.find('vehicle.tesla.cybertruck')
        
        vehicle = world.spawn_actor(vehicle_bp, transform)
        actor_list.append(vehicle)
        print('created %s' % vehicle.type_id) 
        
        vehicle.set_autopilot(True)
        
        sensor_queue = Queue()
        
        # create dir if not exist
        if not os.path.exists(weather):
            os.makedirs(weather)
        sensor_list = []
        
        if(abs(i) <= num_camera//4):
            position_x = 1.7
            position_z = 3
        elif(abs(i) > num_camera//4):
            position_x = -1.7
            position_z = 3
            
        rgb_camera_bp = blueprint_library.find('sensor.camera.rgb')
        flow_camera_bp = blueprint_library.find('sensor.camera.optical_flow')
        # create dir if not exist
        if not os.path.exists(weather + "/rgb_camera_{}".format(i)):
            os.makedirs(weather + "/rgb_camera_{}".format(i))
        
        if not os.path.exists(weather + "/flow_camera_{}".format(i)):
            os.makedirs(weather + "/flow_camera_{}".format(i))
            
        camera_transform = carla.Transform(carla.Location(x=position_x, z=position_z), carla.Rotation(yaw=(360/num_camera)*i))
        camera = world.spawn_actor(rgb_camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)
        print('created %s' % camera.type_id)
        
        flow_camera_transform = carla.Transform(carla.Location(x=position_x, z=position_z), carla.Rotation(yaw=(360/num_camera)*i))
        flow_camera = world.spawn_actor(flow_camera_bp, flow_camera_transform, attach_to=vehicle)
        actor_list.append(flow_camera)
        print('created %s' % flow_camera.type_id)

        camera.listen(lambda data: sensor_callback(data, sensor_queue, "rgb_camera_{}".format(i)))
        sensor_list.append(camera)

        flow_camera.listen(lambda data: sensor_callback(data, sensor_queue, "flow_camera_{}".format(i)))
        sensor_list.append(flow_camera)
        
        print(sensor_list)
        count = 0
        # Main loop
        while count < num_imgs :
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
        print("distoring the vehicle")
        # world.apply_settings(original_settings)
        print(actor_list)
        for sensor in actor_list:
            try:
                sensor.destroy()
            except:
                pass
            


if __name__ == "__main__":

    weather_list = [
        # carla.WeatherParameters.SoftRainNight,
        # carla.WeatherParameters.ClearNoon,
        # carla.WeatherParameters.CloudyNoon,
        # carla.WeatherParameters.HardRainNight,
        # carla.WeatherParameters.WetNoon,
        # carla.WeatherParameters.WetCloudyNoon,
        # carla.WeatherParameters.MidRainyNoon,
        carla.WeatherParameters.HardRainNoon,
        # carla.WeatherParameters.SoftRainNoon,
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
        # 'ClearNoon',
        # 'CloudyNoon',
        # 'HardRainNight',
        # 'WetNoon',
        # 'WetCloudyNoon',
        'MidRainyNoon',
        # 'HardRainNoon',
        # 'SoftRainNoon',
        # 'ClearSunset',
        # 'CloudySunset',
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

    num_camera = 24
        # -7 to 7 
    num_imgs = 10
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)
    world = client.get_world()
    transform = world.get_map().get_spawn_points()[-1]
    print(transform)
    for j in range(len(weather_list)):
        weather_param = weather_list[j]
        weather = "datasets/carla/" + weather_name_list[j]
        print(weather)
        # i = -(num_camera//2 - 1)
        for i in range(-(num_camera//2 - 1), num_camera//2 + 1):
            try:
                main(world,weather_param,weather,num_camera,i,num_imgs,transform)
            except RuntimeError as e:
                print(e)
                continue          
