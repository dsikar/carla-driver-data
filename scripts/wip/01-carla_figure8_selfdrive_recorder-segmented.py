# -*- coding: utf-8 -*-
"""carla_figure8_selfdrive_recorder.py

Modified to use configuration file and include lane segmentation overlay.
"""

import carla
import math
import random
import time
import os
import cv2
import numpy as np
from datetime import datetime
from config_utils import load_config

def get_figure8_waypoints(world):
    road_sequence = [42, 267, 43, 35, 861, 36, 760, 37, 1602, 38, 1091, 39, 1184, 40, 1401, 
                     41, 6, 45, 145, 46, 1072, 47, 774, 48, 901, 49, 1173, 50]
    carla_map = world.get_map()
    waypoints = carla_map.generate_waypoints(1.0)

    all_waypoints = []
    for road_id in road_sequence:
        road_waypoints = [wp for wp in waypoints if wp.road_id == road_id and wp.lane_id == -3]
        road_waypoints.sort(key=lambda x: x.s)
        all_waypoints.extend(road_waypoints)

    return all_waypoints

def set_spectator_camera_following_car(world, vehicle):
    spectator = world.get_spectator()
    vehicle_transform = vehicle.get_transform()
    forward_vector = vehicle_transform.get_forward_vector()
    offset_location = vehicle_transform.location + carla.Location(
        x=-5 * forward_vector.x,
        y=-5 * forward_vector.y,
        z=3
    )
    spectator.set_transform(carla.Transform(
        offset_location,
        carla.Rotation(pitch=-15, yaw=vehicle_transform.rotation.yaw, roll=0)
    ))
    return spectator

def compute_control(vehicle, target_wp, target_speed):
    control = carla.VehicleControl()
    current_transform = vehicle.get_transform()
    current_velocity = vehicle.get_velocity()
    speed = 3.6 * math.sqrt(current_velocity.x**2 + current_velocity.y**2)

    forward = current_transform.get_forward_vector()
    target_vector = target_wp.transform.location - current_transform.location
    forward_dot = forward.x * target_vector.x + forward.y * target_vector.y
    right_dot = -forward.y * target_vector.x + forward.x * target_vector.y
    steering = math.atan2(right_dot, forward_dot) / math.pi
    control.steer = max(-1.0, min(1.0, steering))

    speed_error = target_speed - speed
    if speed_error > 0:
        control.throttle = min(0.3, speed_error / target_speed)
        control.brake = 0.0
    else:
        control.throttle = 0.0
        control.brake = min(0.3, -speed_error / target_speed)

    return control

def setup_cameras(world, vehicle, config):
    """Setup RGB and segmentation cameras attached to the vehicle with parameters from config."""
    blueprint_library = world.get_blueprint_library()
    
    # Get camera parameters from config
    image_width = config['camera']['image_width']
    image_height = config['camera']['image_height']
    fov = config['camera']['fov']
    cam_pos = config['camera']['position']
    cam_rot = config['camera']['rotation']
    
    camera_transform = carla.Transform(
        carla.Location(x=cam_pos['x'], y=cam_pos['y'], z=cam_pos['z']),
        carla.Rotation(pitch=cam_rot['pitch'], yaw=cam_rot['yaw'], roll=cam_rot['roll'])
    )

    # RGB Camera
    rgb_bp = blueprint_library.find('sensor.camera.rgb')
    rgb_bp.set_attribute('image_size_x', str(image_width))
    rgb_bp.set_attribute('image_size_y', str(image_height))
    rgb_bp.set_attribute('fov', str(fov))
    rgb_camera = world.spawn_actor(rgb_bp, camera_transform, attach_to=vehicle)

    # Segmentation Camera
    seg_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
    seg_bp.set_attribute('image_size_x', str(image_width))
    seg_bp.set_attribute('image_size_y', str(image_height))
    seg_bp.set_attribute('fov', str(fov))
    seg_camera = world.spawn_actor(seg_bp, camera_transform, attach_to=vehicle)

    return rgb_camera, seg_camera

def process_images(rgb_image, seg_image, config, steering_angle, image_counter=[0]):
    """Process and save the RGB image with segmented lanes overlaid, including steering data."""
    output_dir = config['output']['directory']
    os.makedirs(output_dir, exist_ok=True)

    # Process RGB image
    rgb_array = np.frombuffer(rgb_image.raw_data, dtype=np.uint8)
    rgb_array = np.reshape(rgb_array, (rgb_image.height, rgb_image.width, 4))[:, :, :3]

    # Process segmentation image (extract lane markings, ID 6)
    seg_array = np.frombuffer(seg_image.raw_data, dtype=np.uint8)
    seg_array = np.reshape(seg_array, (seg_image.height, seg_image.width, 4))[:, :, :3]
    lane_mask = (seg_array[:, :, 2] == 6).astype(np.uint8) * 255  # Lane markings in CARLA are ID 6

    # Overlay lanes on RGB image (green for visibility)
    overlaid_array = rgb_array.copy()
    overlaid_array[lane_mask == 255] = [0, 255, 0]  # Green lanes

    # Save the overlaid image with steering angle
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image_filename = f"{timestamp}_steering_{steering_angle:.4f}_lanes.jpg"
    cv2.imwrite(os.path.join(output_dir, image_filename), cv2.cvtColor(overlaid_array, cv2.COLOR_RGB2BGR))

    # Optional: Save raw RGB for reference
    #rgb_filename = f"{timestamp}_steering_{steering_angle:.4f}_raw.jpg"
    #cv2.imwrite(os.path.join(output_dir, rgb_filename), cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR))

    image_counter[0] += 1
    if image_counter[0] % 100 == 0:
        print(f"Recorded {image_counter[0]} images at {datetime.now().strftime('%H:%M:%S')}")

    return image_counter[0]

def drive_figure_eight(world, vehicle, waypoints, rgb_camera, seg_camera, config):
    """Drive the vehicle along the figure-8 track and record camera data with lane segmentation."""
    output_dir = config['output']['directory']
    target_speed = config['simulation']['target_speed']
    image_width = config['camera']['image_width']
    image_height = config['camera']['image_height']
    fov = config['camera']['fov']
    cam_pos = config['camera']['position']
    cam_rot = config['camera']['rotation']
    
    os.makedirs(output_dir, exist_ok=True)
    image_counter = [0]

    # Buffer to hold latest images
    latest_rgb = [None]
    latest_seg = [None]
    rgb_camera.listen(lambda image: latest_rgb.__setitem__(0, image))
    seg_camera.listen(lambda image: latest_seg.__setitem__(0, image))

    start_time = datetime.now()
    try:
        for i, wp in enumerate(waypoints):
            print(f"Current target waypoint {i + 1}/{len(waypoints)}: {wp.transform.location}")
            world.debug.draw_point(
                wp.transform.location,
                size=0.2,
                color=carla.Color(255, 0, 0),
                life_time=5.0
            )

            while True:
                control = compute_control(vehicle, wp, target_speed)
                vehicle.apply_control(control)
                set_spectator_camera_following_car(world, vehicle)

                # Process images if both are available
                if latest_rgb[0] is not None and latest_seg[0] is not None:
                    process_images(latest_rgb[0], latest_seg[0], config, control.steer, image_counter)

                current_location = vehicle.get_transform().location
                distance_to_waypoint = current_location.distance(wp.transform.location)
                if distance_to_waypoint < 2.0:
                    break

                if world.get_settings().synchronous_mode:
                    world.tick()
                else:
                    time.sleep(0.1)

    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
        control = carla.VehicleControl(throttle=0, brake=1)
        vehicle.apply_control(control)
    finally:
        end_time = datetime.now()
        rgb_camera.stop()
        seg_camera.stop()
        with open(os.path.join(output_dir, 'log.txt'), 'w') as f:
            f.write(f"Simulation Log with lane segmentation:wq\n")
            f.write(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Images Recorded: {image_counter[0]}\n")
            f.write(f"Camera Position: x={cam_pos['x']}, y={cam_pos['y']}, z={cam_pos['z']}\n")
            f.write(f"Camera Rotation: pitch={cam_rot['pitch']}, yaw={cam_rot['yaw']}, roll={cam_rot['roll']}\n")
            f.write(f"Camera Parameters: resolution={image_width}x{image_height}, fov={fov}\n")
        print(f"Simulation ended. Recorded {image_counter[0]} images. Log saved to {output_dir}/log.txt")

def draw_permanent_waypoint_lines(world, waypoints, color=carla.Color(0, 255, 0), thickness=0.02):
    for i in range(len(waypoints) - 1):
        wp1 = waypoints[i]
        wp2 = waypoints[i + 1]
        world.debug.draw_line(
            wp1.transform.location,
            wp2.transform.location,
            thickness=thickness,
            color=color,
            life_time=0
        )

def main(config_path='config.json'):
    # Load configuration
    config = load_config(config_path)
    
    # Connect to CARLA server
    client = carla.Client(config['simulation']['server_host'], config['simulation']['server_port'])
    client.set_timeout(10.0)
    
    # Load specified town
    world = client.load_world(config['simulation']['town'])

    # Set simulation settings
    settings = world.get_settings()
    settings.synchronous_mode = config['simulation']['synchronous_mode']
    settings.fixed_delta_seconds = config['simulation']['fixed_delta_seconds']
    world.apply_settings(settings)

    try:
        waypoints = get_figure8_waypoints(world)
        print(f"Number of waypoints retrieved: {len(waypoints)}")

        # Spawn vehicle
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter(config['vehicle']['model'])[0]
        spawn_point = waypoints[0].transform
        spawn_point.location.z += config['vehicle']['spawn_height_offset']
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        print(f"Vehicle spawned at {spawn_point.location}")

        # Setup cameras and start driving
        rgb_camera, seg_camera = setup_cameras(world, vehicle, config)
        drive_figure_eight(world, vehicle, waypoints, rgb_camera, seg_camera, config)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'rgb_camera' in locals():
            rgb_camera.destroy()
            print("RGB Camera destroyed.")
        if 'seg_camera' in locals():
            seg_camera.destroy()
            print("Segmentation Camera destroyed.")
        if 'vehicle' in locals():
            vehicle.destroy()
            print("Vehicle destroyed.")
        settings.synchronous_mode = False
        world.apply_settings(settings)

if __name__ == "__main__":
    # main('config_1920x1080.json')  # Use default config.json
    main('config_640x480_segmented.json')
    # To use a custom config: main('custom_config.json')