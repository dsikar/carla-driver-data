# -*- coding: utf-8 -*-
"""18-town04-figure-of-8-non-NN-Self-Drive-modified-with-popup.ipynb

# Figure of 8 with LLM prediction and image popup
"""

import carla
import math
import random
import time
import carla_helpers as helpers
import numpy as np
import cv2
from pathlib import Path

# Connect to the client and get the world object
client = carla.Client('localhost', 2000)
world = client.get_world()

# load Town04 map
world = client.load_world('Town04')

"""## Helper functions"""

def set_spectator_camera_following_car(world, vehicle):
    """
    Position the spectator 3 meters above and 5 meters behind the vehicle.
    """
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

def compute_control(vehicle, target_wp, target_speed=10):
    """
    Compute vehicle control using LLM prediction from camera image with popup display.
    Requests a prediction as soon as the input folder is empty, and adjusts steering
    as soon as a prediction is available in the output folder.
    Maintains a constant speed of 10 km/h.
    """
    control = carla.VehicleControl()
    
    # Setup camera if not already present
    if not hasattr(vehicle, 'camera'):
        blueprint_library = vehicle.get_world().get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        camera_bp.set_attribute('fov', '120')
        camera_spawn_point = carla.Transform(carla.Location(x=1.8, y=0, z=1.8), carla.Rotation(-58, 0, 0))
        vehicle.camera = vehicle.get_world().spawn_actor(camera_bp, camera_spawn_point, attach_to=vehicle)
        vehicle.image = None
        vehicle.last_prediction = "OK"  # Default to "OK" (steer straight) until first prediction
        
        def process_image(image):
            try:
                img = np.frombuffer(image.raw_data, dtype=np.uint8)
                img = img.reshape((480, 640, 4))
                vehicle.image = img[:, :, :3]
            except Exception as e:
                print(f"Error processing image: {e}")
                
        vehicle.camera.listen(process_image)
    
    # Wait for image
    timeout = time.time() + 2.0
    while vehicle.image is None and time.time() < timeout:
        vehicle.get_world().tick()
        time.sleep(0.05)
    
    # Define input and output directories
    input_dir = Path('/home/daniel/git/DeepSeek-VL/input_images')
    output_dir = Path('/home/daniel/git/DeepSeek-VL/output_predictions')
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_file = input_dir / 'prediction.jpg'
    prediction_file = output_dir / 'prediction.txt'

    # Step 1: Check if a new prediction is available in the output folder
    prediction = vehicle.last_prediction  # Default to the last prediction
    if prediction_file.exists():
        try:
            with open(prediction_file, 'r') as f:
                prediction = f.read().strip()
            prediction_file.unlink()  # Delete the prediction file after reading
            vehicle.last_prediction = prediction  # Update the last prediction
            print(f"New prediction received: {prediction}")
        except Exception as e:
            print(f"Error reading prediction: {e}")

    # Step 2: Request a new prediction if the input folder is empty
    if vehicle.image is not None:
        # Check if the input folder is empty (i.e., no prediction.jpg exists)
        if not input_file.exists():
            try:
                # Save the current image to the input folder
                resized = cv2.resize(vehicle.image, (384, 384))
                cv2.imwrite(str(input_file), cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
                print(f"New image saved for prediction: {input_file}")
            except Exception as e:
                print(f"Error saving image: {e}")
        
        # Step 3: Update the display with the current prediction
        def convert_canonical(pred):
            if pred == "Left":
                return "Steer Left"
            if pred == "Right":
                return "Steer Right"
            return "Steer straight"
        
        display_img = vehicle.image.copy()
        cv2.putText(display_img, 
                   f"Prediction: {convert_canonical(prediction)}",
                   (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   1,
                   (0, 255, 0),
                   2)
        
        cv2.imshow("Camera Feed & Prediction", display_img)
        cv2.waitKey(1)
        
        # Step 4: Apply the prediction to steering
        def convert_prediction(pred):
            try:
                if pred == "Left":  # Green light on the left, steer left
                    return -0.08
                if pred == "Right":  # Green light on the right, steer right
                    return 0.08
                return 0.0
            except ValueError:
                return 0.0
            
        try:
            control.steer = max(-1.0, min(1.0, float(convert_prediction(prediction))))
        except ValueError:
            control.steer = 0.0

        # Log the current state
        current_velocity = vehicle.get_velocity()
        speed = 3.6 * math.sqrt(current_velocity.x**2 + current_velocity.y**2)
        print(f"Prediction: {prediction}, Steering: {control.steer}, Speed: {speed:.2f} km/h")
    
        # Basic speed control
        speed_error = target_speed - speed
        if speed_error > 0:
            control.throttle = min(0.3, speed_error / target_speed)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(0.3, -speed_error / target_speed)
            
        return control
    
    # Fallback control if no image
    display_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(display_img, 
               "No image available",
               (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX,
               1,
               (0, 0, 255),
               2)
    cv2.imshow("Camera Feed & Prediction", display_img)
    cv2.waitKey(1)
    
    control.throttle = 0.0
    control.brake = 0.3
    control.steer = 0.0
    return control

def drive_figure_eight(world, vehicle, waypoints, target_speed=10):
    """
    Drive the vehicle along the figure-8 track using LLM predictions.
    """
    try:
        for i, wp in enumerate(waypoints):
            print(f"Current target waypoint {i + 1}/{len(waypoints)}: {wp.transform.location}")
            waypoints = helpers.get_town04_figure8_waypoints(world, lane_id=-2, waypoint_distance=3.0)
            draw_permanent_waypoint_lines(world, waypoints, color=carla.Color(0, 1, 0), thickness=2.0)
            
            while True:
                control = compute_control(vehicle, wp, target_speed)
                vehicle.apply_control(control)
                set_spectator_camera_following_car(world, vehicle)
                
                if world.get_settings().synchronous_mode:
                    world.tick()
                else:
                    time.sleep(0.1)
                    
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
        control = carla.VehicleControl(throttle=0, brake=1)
        vehicle.apply_control(control)
    finally:
        if hasattr(vehicle, 'camera'):
            vehicle.camera.destroy()
        cv2.destroyAllWindows()

def draw_permanent_waypoint_lines(world, waypoints, color=carla.Color(0, 1, 0), thickness=0.1):
    """
    Draw permanent lines linking every waypoint on the road.
    
    Parameters:
    - world: CARLA world object.
    - waypoints: List of waypoints to link.
    - color: Color of the lines (default is neon green).
    - thickness: Thickness of the lines (default is 0.1 meters).
    """
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

"""## Main execution"""

def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    
    try:
        carla_map = world.get_map()
        waypoints = helpers.get_town04_figure8_waypoints(world, -2)
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
        spawn_point = waypoints[0].transform
        spawn_point.location.z += 2.0
        
        try:
            vehicle = world.spawn_actor(vehicle_bp, spawn_point)
            print(f"Vehicle spawned at {spawn_point.location}")
        except Exception as e:
            print(f"Failed to spawn vehicle: {e}")
            return
            
        try:
            drive_figure_eight(world, vehicle, waypoints, target_speed=10)
        except Exception as e:
            print(f"Error during driving: {e}")
            
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    finally:
        if 'vehicle' in locals():
            if hasattr(vehicle, 'camera'):
                vehicle.camera.destroy()
            vehicle.destroy()
            cv2.destroyAllWindows()
            print("Vehicle, camera, and window destroyed.")

main()