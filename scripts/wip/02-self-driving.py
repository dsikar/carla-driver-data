#########
# MODEL #
#########

## Helper functions
import torch
import torch.nn as nn
import torch.nn.functional as F

class NVIDIANet(nn.Module):
    def __init__(self, num_outputs=1, dropout_rate=0.1):
        super(NVIDIANet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)
        
        # Dropout layer
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Dense layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1152, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, num_outputs)
        
    def forward(self, x):
        # Input normalization
        x = x / 255.0
        
        # Convolutional layers with ELU activation and dropout
        x = F.elu(self.conv1(x))
        x = self.dropout(x)
        
        x = F.elu(self.conv2(x))
        x = self.dropout(x)
        
        x = F.elu(self.conv3(x))
        x = self.dropout(x)
        
        x = F.elu(self.conv4(x))
        x = self.dropout(x)
        
        x = F.elu(self.conv5(x))
        x = self.dropout(x)
        
        # Flatten and dense layers
        x = self.flatten(x)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = self.fc4(x)
        
        return x

def load_model(model, model_path, device='cuda'):
    """Load a saved model"""
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

#############
# SPECTATOR #
#############

import math
import carla

def set_spectator_camera_following_car(world, vehicle):

    spectator = world.get_spectator()
    transform = vehicle.get_transform()
    location = transform.location
    rotation = transform.rotation

    offset_location = location - carla.Location(x=35 * math.cos(math.radians(rotation.yaw)),
                                              y=35 * math.sin(math.radians(rotation.yaw)))
    offset_location.z += 20

    spectator.set_transform(carla.Transform(offset_location,
                                          carla.Rotation(pitch=-15, yaw=rotation.yaw, roll=rotation.roll)))
    return spectator
    
##############
# SELF-DRIVE #
##############
    
import carla
import numpy as np
import torch
import cv2
import queue
import threading
import time
from config_utils import load_config

class CarlaSteering:
    def __init__(self, model_path='model.pth', host='localhost', port=2000, 
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Initialize CARLA client
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        # Set synchronous mode with fixed time step
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1  # 10 FPS - slower rate for better synchronization
        self.world.apply_settings(settings)
        
        # Initialize model
        self.model = NVIDIANet()
        self.model = load_model(self.model, model_path, device)
        
        # Image queue for processing
        self.image_queue = queue.Queue()
        self.current_image = None
        
        # Control parameters
        self.max_steering_angle = 1.0
        self.steering_smoothing = 0.5
        self.last_steering = 0.0
        
    def setup_vehicle(self):
        """Spawn and setup the ego vehicle with sensors"""
        # Load Town04
        self.client.load_world('Town04')
        self.world = self.client.get_world()
        
        # Get specific spawn point from route 35, lane -3
        waypoints = self.world.get_map().generate_waypoints(2.0)  # 2.0 is the distance between waypoints
        route_42_waypoints = [w for w in waypoints if w.road_id == 42 and w.lane_id == -3]
        if not route_42_waypoints:
            raise ValueError("Could not find waypoints for route 42, lane -3")
        
        # Get the first waypoint and create a spawn point
        first_waypoint = route_42_waypoints[0]
        # Create spawn point and lift it by 1 unit to avoid collision
        spawn_location = first_waypoint.transform.location
        spawn_location.z += 1  # Lift vehicle 1 unit up to avoid collision
        spawn_point = carla.Transform(
            spawn_location,
            first_waypoint.transform.rotation
        )
        
        # Spawn vehicle
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('model3')[0]
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        
        # Spawn camera
        image_width = 640 #800
        image_height = 480 #600
        fov = 90
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        # pass the image width and height to the blueprint
        camera_bp.set_attribute('image_size_x', str(image_width))
        camera_bp.set_attribute('image_size_y', str(image_height))  
        camera_bp.set_attribute('fov', str(fov))
        
        # Attach camera to vehicle
        camera_spawn_point = carla.Transform(carla.Location(x=1.5, z=1.8), carla.Rotation(pitch=-5, yaw=0, roll=0))
        self.camera = self.world.spawn_actor(camera_bp, camera_spawn_point, attach_to=self.vehicle)
        self.camera.listen(self.process_image)
        
    def process_image(self, image):
        """Callback to process images from CARLA camera"""
        image_width = 640 #800
        image_height = 480 #600
        # Convert CARLA image to numpy array
        img = np.array(image.raw_data).reshape(image_height, image_width, 4)
        img = img[:, :, :3]  # Remove alpha channel
        
        # Store in queue
        self.image_queue.put(img)
        
    def preprocess_image(self, img):
        """Preprocess image for neural network"""
        # Store original image for display
        self.original_img = img.copy()
        
        # Crop
        cropped = img[260:440, :]
        
        # Resize
        resized = cv2.resize(cropped, (200, 66))
        
        # Convert to YUV
        yuv = cv2.cvtColor(resized, cv2.COLOR_RGB2YUV)
        
        # Store preprocessed image for display
        self.preprocessed_img = yuv.copy()
        
        # Prepare for PyTorch (CHW format)
        yuv = yuv.transpose((2, 0, 1))
        yuv = np.ascontiguousarray(yuv)
        
        return torch.from_numpy(yuv).float().unsqueeze(0).to(self.device)
        
    def display_images(self):
        """Display original and preprocessed images side by side"""
        if hasattr(self, 'original_img') and hasattr(self, 'preprocessed_img'):
            # Resize original image to be similar height as preprocessed
            display_height = 264  # 4x preprocessed height
            aspect_ratio = self.original_img.shape[1] / self.original_img.shape[0]
            display_width = int(display_height * aspect_ratio)
            original_resized = cv2.resize(self.original_img, (display_width, display_height))
            
            # Resize preprocessed image for better visibility
            image_width = 640 #800
            # image_height = 480 #600
            preprocessed_display = cv2.resize(self.preprocessed_img, (image_width, 264))  # 4x original size
            
            # Create a black canvas for side-by-side display
            canvas_width = display_width + image_width + 20  # +20 for padding
            canvas = np.zeros((display_height, canvas_width, 3), dtype=np.uint8)
            
            # Place images on canvas
            canvas[:, :display_width] = original_resized
            canvas[:, display_width+20:] = preprocessed_display
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(canvas, 'Original Camera Feed', (10, 30), font, 1, (255, 255, 255), 2)
            cv2.putText(canvas, 'Neural Network Input (YUV)', (display_width+30, 30), font, 1, (255, 255, 255), 2)
            
            # Show the canvas
            cv2.imshow('Camera Views', canvas)
            cv2.waitKey(1)
        
    def predict_steering(self, image):
        """Make steering prediction from image"""
        with torch.no_grad():
            steering_pred = self.model(image)
            
        # Get steering angle from prediction
        steering_angle = float(steering_pred.cpu().numpy()[0, 0])
        
        # Clip and smooth steering
        steering_angle = np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle)
        # smoothed_steering = (self.steering_smoothing * self.last_steering + 
                           #(1 - self.steering_smoothing) * steering_angle)
        self.last_steering = steering_angle # smoothed_steering
        
        return steering_angle # smoothed_steering

    def apply_control(self, steering, target_speed=10):
        """Apply control to vehicle with neural network steering and proportional speed control"""
        control = carla.VehicleControl()
        # Keep neural network steering
        control.steer = steering

        # Add speed-based control from compute_control
        current_velocity = self.vehicle.get_velocity()
        speed = 3.6 * math.sqrt(current_velocity.x**2 + current_velocity.y**2)  # km/h
        speed_error = target_speed - speed
        
        if speed_error > 0:
            control.throttle = min(0.3, speed_error / target_speed)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(0.3, -speed_error / target_speed)

        self.vehicle.apply_control(control)        
        
    def run(self):
        """Main control loop"""
        try:
            self.setup_vehicle()
            print("Vehicle and sensors initialized. Starting control loop...")
            
            while True:
                # Clear the image queue to always use latest frame
                while not self.image_queue.empty():
                    _ = self.image_queue.get()
                
                # Tick the world and wait for new image
                self.world.tick()
                
                try:
                    # Wait for new image with timeout
                    img = self.image_queue.get(timeout=0.1)
                    
                    # Preprocess image
                    processed_img = self.preprocess_image(img)
                    
                    # Get steering prediction
                    steering = self.predict_steering(processed_img)
                    
                    # Apply control
                    self.apply_control(steering)
                    
                    # Update spectator camera
                    set_spectator_camera_following_car(self.world, self.vehicle)
                    
                    # Display camera feeds
                    self.display_images()
                    
                    #print(f"Applied steering angle: {steering:.3f}")
                    
                except queue.Empty:
                    print("Warning: Frame missed!")
                    
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            # Cleanup
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
            
            if hasattr(self, 'camera'):
                self.camera.destroy()
            if hasattr(self, 'vehicle'):
                self.vehicle.destroy()

if __name__ == '__main__':
    try:
        controller = CarlaSteering(model_path='model.pth')
        controller.run()
    except Exception as e:
        print(f"An error occurred: {e}")


    
 
