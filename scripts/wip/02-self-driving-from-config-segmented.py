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
import argparse

class CarlaSteering:
    def __init__(self, config, model_path='model.pth'):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load simulation settings
        sim_config = config['simulation']
        self.client = carla.Client(sim_config['server_host'], sim_config['server_port'])
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        # Set synchronous mode with fixed time step
        settings = self.world.get_settings()
        settings.synchronous_mode = sim_config['synchronous_mode']
        settings.fixed_delta_seconds = sim_config['fixed_delta_seconds_self_drive']
        self.world.apply_settings(settings)
        
        # Initialize model
        self.model = NVIDIANet()
        self.model = load_model(self.model, model_path, self.device)
        
        # Load control parameters
        control_config = config.get('control', {})
        self.max_steering_angle = control_config.get('max_steering_angle', 1.0)
        self.steering_smoothing = control_config.get('steering_smoothing', 0.5)
        self.last_steering = 0.0
        self.target_speed = sim_config.get('target_speed', 10)
        
        # Load camera settings
        camera_config = config['camera']
        self.image_width = camera_config['image_width']
        self.image_height = camera_config['image_height']
        
        # Load image processing parameters
        img_proc_config = config['image_processing']
        self.crop_top = img_proc_config['crop_top']
        self.crop_bottom = img_proc_config['crop_bottom']
        self.resize_width = img_proc_config['resize_width']
        self.resize_height = img_proc_config['resize_height']
        
        # Image queues for RGB and segmentation
        self.rgb_queue = queue.Queue()
        self.seg_queue = queue.Queue()
        self.current_rgb = None
        self.current_seg = None
        
    def setup_vehicle(self):
        """Spawn and setup the ego vehicle with RGB and segmentation sensors"""
        # Load town from config
        sim_config = self.config['simulation']
        self.client.load_world(sim_config['town'])
        self.world = self.client.get_world()
        
        # Load vehicle settings
        vehicle_config = self.config['vehicle']
        waypoints = self.world.get_map().generate_waypoints(2.0)
        route_waypoints = [w for w in waypoints if w.road_id == vehicle_config['spawn_road_id'] 
                         and w.lane_id == vehicle_config['spawn_lane_id']]
        if not route_waypoints:
            raise ValueError(f"Could not find waypoints for road {vehicle_config['spawn_road_id']}, "
                           f"lane {vehicle_config['spawn_lane_id']}")
        
        # Get spawn point
        first_waypoint = route_waypoints[0]
        spawn_location = first_waypoint.transform.location
        spawn_location.z += vehicle_config['spawn_height_offset']
        spawn_point = carla.Transform(spawn_location, first_waypoint.transform.rotation)
        
        # Spawn vehicle
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter(vehicle_config['model'].split('.')[-1])[0]
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        
        # Spawn RGB camera
        camera_config = self.config['camera']
        rgb_bp = blueprint_library.find('sensor.camera.rgb')
        rgb_bp.set_attribute('image_size_x', str(self.image_width))
        rgb_bp.set_attribute('image_size_y', str(self.image_height))
        rgb_bp.set_attribute('fov', str(camera_config['fov']))
        
        pos = camera_config['position']
        rot = camera_config['rotation']
        camera_spawn_point = carla.Transform(
            carla.Location(x=pos['x'], y=pos['y'], z=pos['z']),
            carla.Rotation(pitch=rot['pitch'], yaw=rot['yaw'], roll=rot['roll'])
        )
        self.rgb_camera = self.world.spawn_actor(rgb_bp, camera_spawn_point, attach_to=self.vehicle)
        self.rgb_camera.listen(self.process_rgb_image)
        
        # Spawn segmentation camera
        seg_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        seg_bp.set_attribute('image_size_x', str(self.image_width))
        seg_bp.set_attribute('image_size_y', str(self.image_height))
        seg_bp.set_attribute('fov', str(camera_config['fov']))
        self.seg_camera = self.world.spawn_actor(seg_bp, camera_spawn_point, attach_to=self.vehicle)
        self.seg_camera.listen(self.process_seg_image)
        
    def process_rgb_image(self, image):
        """Callback to process RGB images from CARLA camera"""
        img = np.array(image.raw_data).reshape(self.image_height, self.image_width, 4)
        img = img[:, :, :3]  # Remove alpha channel
        self.rgb_queue.put(img)
        
    def process_seg_image(self, image):
        """Callback to process segmentation images from CARLA camera"""
        seg_array = np.array(image.raw_data).reshape(self.image_height, self.image_width, 4)
        seg_array = seg_array[:, :, :3]  # Remove alpha channel
        lane_mask = (seg_array[:, :, 2] == 6).astype(np.uint8) * 255  # Lane markings (ID 6)
        self.seg_queue.put(lane_mask)
        
    def preprocess_image(self, rgb_img, lane_mask):
        """Preprocess RGB image with lane segmentation overlay for neural network"""
        self.original_rgb = rgb_img.copy()
        
        # Overlay lane mask on RGB image (green lanes)
        overlaid_img = rgb_img.copy()
        overlaid_img[lane_mask == 255] = [0, 255, 0]  # Green for lanes
        
        # Crop and resize using config values
        cropped = overlaid_img[self.crop_top:self.crop_bottom, :]
        resized = cv2.resize(cropped, (self.resize_width, self.resize_height))
        
        # Convert to YUV
        yuv = cv2.cvtColor(resized, cv2.COLOR_RGB2YUV)
        self.preprocessed_img = yuv.copy()
        
        # Prepare for PyTorch
        yuv = yuv.transpose((2, 0, 1))
        yuv = np.ascontiguousarray(yuv)
        
        return torch.from_numpy(yuv).float().unsqueeze(0).to(self.device)
        
    def display_images(self):
        """Display original RGB and preprocessed images side by side"""
        if hasattr(self, 'original_rgb') and hasattr(self, 'preprocessed_img'):
            display_height = 264
            aspect_ratio = self.original_rgb.shape[1] / self.original_rgb.shape[0]
            display_width = int(display_height * aspect_ratio)
            original_resized = cv2.resize(self.original_rgb, (display_width, display_height))
            
            preprocessed_display = cv2.resize(self.preprocessed_img, (self.image_width, 264))
            
            canvas_width = display_width + self.image_width + 20
            canvas = np.zeros((display_height, canvas_width, 3), dtype=np.uint8)
            
            canvas[:, :display_width] = original_resized
            canvas[:, display_width+20:] = preprocessed_display
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(canvas, 'Original RGB Feed', (10, 30), font, 1, (255, 255, 255), 2)
            cv2.putText(canvas, 'Neural Network Input (YUV with Lanes)', (display_width+30, 30), font, 1, (255, 255, 255), 2)
            
            cv2.imshow('Camera Views', canvas)
            cv2.waitKey(1)
        
    def predict_steering(self, image):
        """Make steering prediction from image"""
        with torch.no_grad():
            steering_pred = self.model(image)
            
        steering_angle = float(steering_pred.cpu().numpy()[0, 0])
        steering_angle = np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle)
        self.last_steering = steering_angle
        
        return steering_angle

    def apply_control(self, steering):
        """Apply control to vehicle with neural network steering and proportional speed control"""
        control = carla.VehicleControl()
        control.steer = steering

        current_velocity = self.vehicle.get_velocity()
        speed = 3.6 * math.sqrt(current_velocity.x**2 + current_velocity.y**2)  # km/h
        speed_error = self.target_speed - speed
        
        if speed_error > 0:
            control.throttle = min(0.3, speed_error / self.target_speed)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(0.3, -speed_error / self.target_speed)

        self.vehicle.apply_control(control)        
        
    def run(self):
        """Main control loop"""
        try:
            self.setup_vehicle()
            print("Vehicle and sensors initialized. Starting control loop...")
            
            while True:
                # Clear queues to ensure latest images
                while not self.rgb_queue.empty():
                    _ = self.rgb_queue.get()
                while not self.seg_queue.empty():
                    _ = self.seg_queue.get()
                
                self.world.tick()
                
                try:
                    rgb_img = self.rgb_queue.get(timeout=0.1)
                    lane_mask = self.seg_queue.get(timeout=0.1)
                    processed_img = self.preprocess_image(rgb_img, lane_mask)
                    steering = self.predict_steering(processed_img)
                    self.apply_control(steering)
                    set_spectator_camera_following_car(self.world, self.vehicle)
                    self.display_images()
                    
                except queue.Empty:
                    print("Warning: Frame missed!")
                    
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
            
            if hasattr(self, 'rgb_camera'):
                self.rgb_camera.destroy()
            if hasattr(self, 'seg_camera'):
                self.seg_camera.destroy()
            if hasattr(self, 'vehicle'):
                self.vehicle.destroy()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run CARLA self-driving simulation.')
    parser.add_argument('--config', type=str, default='config.json', 
                        help='Path to the configuration JSON file (default: config.json)')
    parser.add_argument('--model', type=str, default='model.pth', 
                        help='Path to the trained model file (default: model.pth)')
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        controller = CarlaSteering(config, model_path=args.model)
        controller.run()
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
# python 02-self-driving-from-config-segmented.py \
# --config /home/daniel/git/carla-driver-data/scripts/wip/config_640x480_segmented.json \
# --model /home/daniel/git/carla-driver-data/scripts/wip/best_steering_model_20250318-155258.pth  
 