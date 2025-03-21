#!/usr/bin/env python3

"""
Figure 8 Data Collection

This script combines driving around a figure-8 track in Town04 with camera data collection.
It drives the vehicle around the figure-8 track and saves camera images to a timestamped directory.
"""

import glob
import os
import sys
import time
import math
import queue
import numpy as np
import cv2
from datetime import datetime
import carla

# # Add CARLA path to enable importing its Python API
# try:
#     sys.path.append(glob.glob('/opt/carla-simulator/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
#         sys.version_info.major,
#         sys.version_info.minor,
#         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
# except IndexError:
#     pass

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FOV = 110
CAMERA_LOCATION = carla.Location(x=0.5, z=1.8)
CAMERA_ROTATION = carla.Rotation(pitch=-15)

def get_figure8_waypoints(world, offset=0.0):
    """
    Generates waypoints for a figure-8 shaped path in Town04.
    
    Args:
        world: CARLA world object
        offset: Optional lateral offset from road center
        
    Returns:
        List of waypoints forming a figure-8 path
    """
    # Town04 has a figure-8 shaped highway with these road IDs
    road_id_sequence = [42, 267, 43, 35, 861, 36, 760, 37, 1602, 38, 1091, 39, 1184, 40, 1401, \
                                                                       41, 6, 45, 145, 46, 1072, 47, 774, 48, 901, 49, 1173, 50]
 
    # Get a map and all waypoints with a specific fixed distance between them
    carla_map = world.get_map()
    precision = 1.0 # meters
    waypoints = []
    
    for road_id in road_id_sequence:
        road_waypoints = carla_map.generate_waypoints(precision)
        filtered_waypoints = [wp for wp in road_waypoints if wp.road_id == road_id]
        filtered_waypoints.sort(key=lambda wp: wp.s)
        waypoints.extend(filtered_waypoints)
    
    # Apply lateral offset if requested
    if abs(offset) > 0.0:
        for i, wp in enumerate(waypoints):
            waypoints[i] = wp.next(0.0)[0].get_left_lane() if offset < 0 else wp.next(0.0)[0].get_right_lane()
    
    return waypoints

def compute_control(vehicle, waypoint, next_waypoint, max_speed=50.0):
    """
    Computes vehicle control commands using PID approach to reach the target waypoint.
    
    Args:
        vehicle: CARLA vehicle actor
        waypoint: Current target waypoint
        next_waypoint: Next waypoint for lookahead
        max_speed: Maximum speed in km/h
        
    Returns:
        CARLA VehicleControl object
    """
    # PID parameters
    Kp_steering = 0.5
    Kd_steering = 0.1
    Kp_throttle = 0.5
    Ki_throttle = 0.05
    Kd_throttle = 0.1
    
    # Get vehicle state
    vehicle_transform = vehicle.get_transform()
    vehicle_location = vehicle_transform.location
    vehicle_velocity = vehicle.get_velocity()
    vehicle_speed = 3.6 * math.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2 + vehicle_velocity.z**2)  # km/h
    
    # Calculate steering
    target_vector = waypoint.transform.location - vehicle_location
    target_vector_len = math.sqrt(target_vector.x**2 + target_vector.y**2)
    
    forward_vector = vehicle_transform.get_forward_vector()
    dot = target_vector.x * forward_vector.x + target_vector.y * forward_vector.y
    cross = forward_vector.x * target_vector.y - forward_vector.y * target_vector.x
    
    steering = math.atan2(cross, dot)
    steering_derivative = 0  # In a real setting, this would be the difference from the last steering angle
    
    # Apply PID to steering
    steering = Kp_steering * steering + Kd_steering * steering_derivative
    steering = max(-1.0, min(1.0, steering))  # Clamp to [-1, 1]
    
    # Calculate throttle/brake
    target_speed = max_speed
    speed_error = target_speed - vehicle_speed
    
    # Simple P controller for throttle and brake
    throttle = Kp_throttle * speed_error
    throttle = max(0.0, min(1.0, throttle))  # Clamp to [0, 1]
    
    # If we need to slow down, apply brakes instead of throttle
    brake = 0.0
    if speed_error < -5:  # Only brake if we're significantly over speed
        brake = min(1.0, abs(speed_error) * 0.05)
        throttle = 0.0
    
    control = carla.VehicleControl()
    control.steer = steering
    control.throttle = throttle
    control.brake = brake
    
    return control, steering

def setup_camera(world, vehicle):
    """
    Sets up and attaches a camera to the vehicle.
    
    Args:
        world: CARLA world object
        vehicle: CARLA vehicle actor
        
    Returns:
        camera: CARLA sensor actor
        image_queue: Queue for camera images
    """
    # Set up image queue
    image_queue = queue.Queue()
    
    # Create camera blueprint
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(CAMERA_WIDTH))
    camera_bp.set_attribute('image_size_y', str(CAMERA_HEIGHT))
    camera_bp.set_attribute('fov', str(CAMERA_FOV))
    
    # Attach camera to vehicle
    camera_transform = carla.Transform(CAMERA_LOCATION, CAMERA_ROTATION)
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    
    # Create callback to process camera images
    def process_image(image):
        image.convert(carla.ColorConverter.Raw)
        image_queue.put(image)
    
    # Register callback and return camera
    camera.listen(process_image)
    return camera, image_queue

def set_spectator_camera_following_car(world, vehicle, distance=6):
    """
    Sets the spectator camera to follow the vehicle.
    
    Args:
        world: CARLA world object
        vehicle: CARLA vehicle actor
        distance: Following distance
    """
    spectator = world.get_spectator()
    
    transform = vehicle.get_transform()
    forward_vec = transform.get_forward_vector()
    
    spectator_transform = carla.Transform(
        transform.location - distance * forward_vec + carla.Location(z=2.0),
        carla.Rotation(pitch=-15, yaw=transform.rotation.yaw)
    )
    
    spectator.set_transform(spectator_transform)

def save_image(image, output_dir, steering_angle):
    """
    Saves a camera image to the specified output directory.
    
    Args:
        image: CARLA camera image
        output_dir: Output directory path
        steering_angle: Current steering angle
    """
    # Convert CARLA raw image to numpy array
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (CAMERA_HEIGHT, CAMERA_WIDTH, 4))
    array = array[:, :, :3]  # Remove alpha channel
    
    # Create filename with timestamp and steering angle
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image_filename = f"{timestamp}_steering_{steering_angle:.4f}.jpg"
    
    # Save image
    cv2.imwrite(os.path.join(output_dir, image_filename), array)

def drive_figure_eight(world, vehicle, output_dir):
    """
    Drive vehicle around figure-8 track while saving camera images.
    
    Args:
        world: CARLA world object
        vehicle: CARLA vehicle actor
        output_dir: Output directory path
    """
    # Generate waypoints for figure-8
    waypoints = get_figure8_waypoints(world)
    
    # Set up camera and image queue
    camera, image_queue = setup_camera(world, vehicle)
    
    try:
        current_wp_index = 0
        
        # Drive until we've completed the figure-8 loop
        while current_wp_index < len(waypoints):
            # Update spectator camera
            set_spectator_camera_following_car(world, vehicle)
            
            # Get current and next waypoints
            current_wp = waypoints[current_wp_index]
            next_wp_index = min(current_wp_index + 10, len(waypoints) - 1)
            next_wp = waypoints[next_wp_index]
            
            # Compute and apply control
            control, steering = compute_control(vehicle, current_wp, next_wp)
            vehicle.apply_control(control)
            
            # Check if we've reached the waypoint
            vehicle_location = vehicle.get_transform().location
            dist = math.sqrt((vehicle_location.x - current_wp.transform.location.x)**2 + 
                             (vehicle_location.y - current_wp.transform.location.y)**2)
            
            if dist < 2.0:  # If within 2 meters of waypoint, move to next
                current_wp_index += 1
            
            # Process any images in the queue
            while not image_queue.empty():
                img = image_queue.get()
                save_image(img, output_dir, steering)
            
            # Sleep to prevent hogging CPU
            time.sleep(0.01)
            
        # Complete a full circle once more to collect more data
        # Reset to beginning
        current_wp_index = 0
        while current_wp_index < len(waypoints):
            set_spectator_camera_following_car(world, vehicle)
            
            current_wp = waypoints[current_wp_index]
            next_wp_index = min(current_wp_index + 10, len(waypoints) - 1)
            next_wp = waypoints[next_wp_index]
            
            control, steering = compute_control(vehicle, current_wp, next_wp)
            vehicle.apply_control(control)
            
            vehicle_location = vehicle.get_transform().location
            dist = math.sqrt((vehicle_location.x - current_wp.transform.location.x)**2 + 
                             (vehicle_location.y - current_wp.transform.location.y)**2)
            
            if dist < 2.0:
                current_wp_index += 1
            
            while not image_queue.empty():
                img = image_queue.get()
                save_image(img, output_dir, steering)
            
            time.sleep(0.01)
            
    finally:
        # Clean up the camera when done
        if camera:
            camera.destroy()

def main():
    """
    Main function that sets up the CARLA environment and runs the figure-8 data collection.
    """
    client = None
    world = None
    vehicle = None
    
    try:
        # Connect to CARLA
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        
        # Get world and load Town04
        world = client.load_world('Town04')
        
        # Disable rendering for better performance
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        
        # Set up the vehicle
        vehicle_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
        spawn_point = world.get_map().get_spawn_points()[0]
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        
        # Let the vehicle settle
        world.tick()
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"carla_dataset_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Drive and collect data
        drive_figure_eight(world, vehicle, output_dir)
        
        print(f"Data collection complete. Images saved to {output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Clean up
        if world and world.get_settings().synchronous_mode:
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
        
        if vehicle:
            vehicle.destroy()

if __name__ == "__main__":
    main()