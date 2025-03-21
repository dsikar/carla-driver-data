# -*- coding: utf-8 -*-
"""carla_figure8_selfdrive_recorder.py

Class-based implementation of a script to drive a vehicle around a figure-8 track 
in CARLA's Town04 map while recording camera images and steering angles for 
autonomous driving training.
"""

import carla
import math
import random
import time
import os
import cv2
import numpy as np
from datetime import datetime

class CarlaRecorder:
    def __init__(self, 
                 host='localhost', 
                 port=2000, 
                 output_dir='carla_dataset',
                 image_width=640, 
                 image_height=480, 
                 fov=90,
                 camera_x=1.5, 
                 camera_y=0.0, 
                 camera_z=1.8,
                 camera_pitch=-5.0, 
                 camera_yaw=0.0, 
                 camera_roll=0.0,
                 target_speed=10,
                 fixed_delta_seconds=0.05,
                 waypoint_distance_threshold=2.0):
        """
        Initialize the CARLA recorder with configuration parameters.
        
        Parameters:
        - host: CARLA server host
        - port: CARLA server port
        - output_dir: Directory for saving dataset
        - image_width: Camera resolution width in pixels
        - image_height: Camera resolution height in pixels
        - fov: Camera field of view in degrees
        - camera_x/y/z: Camera position coordinates
        - camera_pitch/yaw/roll: Camera rotation angles
        - target_speed: Vehicle driving speed in km/h
        - fixed_delta_seconds: Time step for synchronous mode
        - waypoint_distance_threshold: Distance to consider a waypoint reached
        """
        # Camera configuration
        self.image_width = image_width
        self.image_height = image_height
        self.fov = fov
        self.camera_transform = carla.Transform(
            carla.Location(x=camera_x, y=camera_y, z=camera_z),
            carla.Rotation(pitch=camera_pitch, yaw=camera_yaw, roll=camera_roll)
        )
        
        # Recording parameters
        self.output_dir = output_dir
        self.target_speed = target_speed
        self.fixed_delta_seconds = fixed_delta_seconds
        self.waypoint_distance_threshold = waypoint_distance_threshold
        
        # CARLA components
        self.client = None
        self.world = None
        self.vehicle = None
        self.camera = None
        self.waypoints = None
        
        # Session tracking
        self.image_counter = [0]  # List for mutable reference
        self.start_time = None
        self.end_time = None
        self.original_settings = None
        
        # Connect to CARLA
        self.connect_to_carla(host, port)
        
    def connect_to_carla(self, host, port):
        """Connect to the CARLA server and load Town04 map."""
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        # Load Town04 map
        self.world = self.client.load_world('Town04')
        print("Connected to CARLA server and loaded Town04 map")
        
    def enable_synchronous_mode(self):
        """Enable synchronous mode for consistent timing."""
        settings = self.world.get_settings()
        self.original_settings = carla.WorldSettings(
            settings.synchronous_mode,
            settings.fixed_delta_seconds
        )
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.fixed_delta_seconds
        self.world.apply_settings(settings)
        print(f"Enabled synchronous mode with time step {self.fixed_delta_seconds}")
        
    def restore_settings(self):
        """Restore original world settings."""
        if self.original_settings:
            self.world.apply_settings(self.original_settings)
            print("Restored original world settings")
    
    def get_figure8_waypoints(self):
        """Get waypoints for figure-8 track in Town04."""
        # Road sequence for a complete figure-8 in Town04
        road_sequence = [42, 267, 43, 35, 861, 36, 760, 37, 1602, 38, 1091, 39, 1184, 40, 1401, 
                         41, 6, 45, 145, 46, 1072, 47, 774, 48, 901, 49, 1173, 50]
        
        carla_map = self.world.get_map()
        waypoints = carla_map.generate_waypoints(1.0)
        
        all_waypoints = []
        for road_id in road_sequence:
            road_waypoints = [wp for wp in waypoints if wp.road_id == road_id and wp.lane_id == -3]
            road_waypoints.sort(key=lambda x: x.s)
            all_waypoints.extend(road_waypoints)
        
        self.waypoints = all_waypoints
        print(f"Loaded {len(self.waypoints)} waypoints for figure-8 track")
        return self.waypoints
    
    def setup_vehicle(self):
        """Spawn a vehicle at the first waypoint."""
        if not self.waypoints:
            self.get_figure8_waypoints()
            
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
        
        # Get spawn point from the first waypoint
        spawn_point = self.waypoints[0].transform
        spawn_point.location.z += 2.0  # Prevent spawn collision
        
        # Spawn the vehicle
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        print(f"Vehicle spawned at {spawn_point.location}")
        return self.vehicle
    
    def setup_camera(self):
        """Setup an RGB camera attached to the vehicle."""
        if not self.vehicle:
            raise RuntimeError("Vehicle not initialized. Call setup_vehicle first.")
            
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.image_width))
        camera_bp.set_attribute('image_size_y', str(self.image_height))
        camera_bp.set_attribute('fov', str(self.fov))
        
        self.camera = self.world.spawn_actor(camera_bp, self.camera_transform, attach_to=self.vehicle)
        print(f"Camera attached to vehicle with resolution {self.image_width}x{self.image_height}, FOV {self.fov}")
        print(f"Camera position: {self.camera_transform.location}, rotation: {self.camera_transform.rotation}")
        return self.camera
    
    def process_image(self, image):
        """Process and save the captured image with steering data."""
        # Convert raw image data to numpy array
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # Remove alpha channel
        
        # Get current steering angle from vehicle
        steering_angle = self.vehicle.get_control().steer
        
        # Save image with timestamp and steering angle in filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_filename = f"{timestamp}_steering_{steering_angle:.4f}.jpg"
        cv2.imwrite(os.path.join(self.output_dir, image_filename), array)
        
        # Update and report progress
        self.image_counter[0] += 1
        if self.image_counter[0] % 100 == 0:
            print(f"Recorded {self.image_counter[0]} images at {datetime.now().strftime('%H:%M:%S')}")
    
    def set_spectator_camera_following_car(self):
        """Position the spectator camera to follow the vehicle."""
        if not self.vehicle:
            raise RuntimeError("Vehicle not initialized")
            
        spectator = self.world.get_spectator()
        vehicle_transform = self.vehicle.get_transform()
        
        # Get the vehicle's forward vector
        forward_vector = vehicle_transform.get_forward_vector()
        
        # Position spectator 5 meters behind and 3 meters above the vehicle
        offset_location = vehicle_transform.location + carla.Location(
            x=-5 * forward_vector.x,
            y=-5 * forward_vector.y,
            z=3
        )
        
        # Set spectator transform with a slight downward tilt
        spectator.set_transform(carla.Transform(
            offset_location,
            carla.Rotation(pitch=-15, yaw=vehicle_transform.rotation.yaw, roll=0)
        ))
        return spectator
    
    def compute_control(self, target_wp):
        """Compute vehicle control to reach the target waypoint at the specified speed."""
        if not self.vehicle:
            raise RuntimeError("Vehicle not initialized")
            
        control = carla.VehicleControl()
        
        # Get current state
        current_transform = self.vehicle.get_transform()
        current_velocity = self.vehicle.get_velocity()
        speed = 3.6 * math.sqrt(current_velocity.x**2 + current_velocity.y**2)  # km/h
        
        # Calculate steering angle
        forward = current_transform.get_forward_vector()
        target_vector = target_wp.transform.location - current_transform.location
        forward_dot = forward.x * target_vector.x + forward.y * target_vector.y
        right_dot = -forward.y * target_vector.x + forward.x * target_vector.y
        steering = math.atan2(right_dot, forward_dot) / math.pi
        control.steer = max(-1.0, min(1.0, steering))  # Clamp to [-1, 1]
        
        # Speed control (PID-like)
        speed_error = self.target_speed - speed
        if speed_error > 0:
            # If below target speed, apply throttle
            control.throttle = min(0.3, speed_error / self.target_speed)
            control.brake = 0.0
        else:
            # If above target speed, apply brake
            control.throttle = 0.0
            control.brake = min(0.3, -speed_error / self.target_speed)
        
        return control
    
    def distance_to_waypoint(self, waypoint):
        """Calculate the distance from vehicle to waypoint."""
        if not self.vehicle:
            raise RuntimeError("Vehicle not initialized")
            
        loc = self.vehicle.get_location()
        wp_loc = waypoint.transform.location
        return math.sqrt((loc.x - wp_loc.x)**2 + (loc.y - wp_loc.y)**2)
    
    def drive_figure_eight(self):
        """Drive the vehicle along the figure-8 track and record camera data."""
        if not self.vehicle or not self.camera or not self.waypoints:
            raise RuntimeError("Vehicle, camera, and waypoints must be initialized first")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize recording
        self.image_counter = [0]
        self.start_time = datetime.now()
        print(f"Recording started at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Set up camera callback to process and save images
        self.camera.listen(lambda image: self.process_image(image))
        
        try:
            # Navigate through all waypoints
            for i, wp in enumerate(self.waypoints):
                print(f"Current target waypoint {i + 1}/{len(self.waypoints)}: {wp.transform.location}")
                
                # Visualize current target waypoint
                self.world.debug.draw_point(
                    wp.transform.location,
                    size=0.2,
                    color=carla.Color(255, 0, 0),
                    life_time=5.0
                )
                
                # Drive toward the current waypoint
                while True:
                    # Compute and apply control
                    control = self.compute_control(wp)
                    self.vehicle.apply_control(control)
                    
                    # Update spectator camera
                    self.set_spectator_camera_following_car()
                    
                    # Check if we've reached the waypoint
                    distance = self.distance_to_waypoint(wp)
                    if distance < self.waypoint_distance_threshold:
                        break
                    
                    # Advance simulation
                    if self.world.get_settings().synchronous_mode:
                        self.world.tick()
                    else:
                        time.sleep(0.1)
            
            print("Completed figure-8 track")
            
        except KeyboardInterrupt:
            print("Recording interrupted by user")
        except Exception as e:
            print(f"Error during recording: {e}")
        finally:
            # Stop the vehicle
            self.vehicle.apply_control(carla.VehicleControl(throttle=0, brake=1))
            
            # Record end time and stop camera
            self.end_time = datetime.now()
            self.camera.stop()
            
            # Create session log
            self.create_log_file()
    
    def create_log_file(self):
        """Write session log file with recording details."""
        if not self.start_time or not self.end_time:
            print("Warning: Start or end time not set, log file may be incomplete")
            if not self.start_time:
                self.start_time = datetime.now()
            if not self.end_time:
                self.end_time = datetime.now()
                
        with open(os.path.join(self.output_dir, 'log.txt'), 'w') as f:
            f.write(f"CARLA Figure-8 Recording Session Log\n")
            f.write(f"=====================================\n")
            f.write(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"End Time: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {(self.end_time - self.start_time).total_seconds()} seconds\n")
            f.write(f"Total Images Recorded: {self.image_counter[0]}\n\n")
            
            f.write(f"Camera Parameters\n")
            f.write(f"----------------\n")
            f.write(f"Resolution: {self.image_width}x{self.image_height}\n")
            f.write(f"Field of View: {self.fov}\n")
            f.write(f"Position: x={self.camera_transform.location.x}, y={self.camera_transform.location.y}, z={self.camera_transform.location.z}\n")
            f.write(f"Rotation: pitch={self.camera_transform.rotation.pitch}, yaw={self.camera_transform.rotation.yaw}, roll={self.camera_transform.rotation.roll}\n\n")
            
            f.write(f"Vehicle\n")
            f.write(f"-------\n")
            f.write(f"Model: Tesla Model 3\n")
            f.write(f"Target Speed: {self.target_speed} km/h\n")
            
        print(f"Session ended. Recorded {self.image_counter[0]} images.")
        print(f"Log saved to {os.path.join(self.output_dir, 'log.txt')}")
    
    def draw_waypoint_lines(self, color=carla.Color(0, 255, 0), thickness=0.02):
        """Draw lines connecting waypoints for visualization."""
        if not self.waypoints:
            self.get_figure8_waypoints()
            
        for i in range(len(self.waypoints) - 1):
            wp1 = self.waypoints[i]
            wp2 = self.waypoints[i + 1]
            self.world.debug.draw_line(
                wp1.transform.location,
                wp2.transform.location,
                thickness=thickness,
                color=color,
                life_time=0  # Permanent line
            )
        print(f"Drew {len(self.waypoints)-1} waypoint connections")
    
    def cleanup(self):
        """Clean up resources and restore original settings."""
        if hasattr(self, 'camera') and self.camera:
            self.camera.stop()
            self.camera.destroy()
            print("Camera destroyed")
        
        if hasattr(self, 'vehicle') and self.vehicle:
            self.vehicle.destroy()
            print("Vehicle destroyed")
        
        self.restore_settings()
    
    def run(self):
        """Execute the full recording sequence."""
        try:
            # Setup
            self.enable_synchronous_mode()
            if not self.waypoints:
                self.get_figure8_waypoints()
            if not self.vehicle:
                self.setup_vehicle()
            if not self.camera:
                self.setup_camera()
            
            # Optional waypoint visualization
            # self.draw_waypoint_lines()
            
            # Allow time for setup to complete
            print("Setup complete. Starting recording in 3 seconds...")
            for _ in range(60):  # 3 seconds at 20 FPS
                self.world.tick()
            
            # Drive and record
            self.drive_figure_eight()
            
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            self.cleanup()


# Example usage
if __name__ == "__main__":
    # Create recorder with default parameters
    recorder = CarlaRecorder()
    recorder.run()
    
    # Example with custom parameters:
    # recorder = CarlaRecorder(
    #     image_width=1920,
    #     image_height=1080,
    #     fov=120,
    #     target_speed=15,
    #     output_dir="carla_dataset_hd"
    # )
    # recorder.run()