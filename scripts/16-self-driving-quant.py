# https://chat.deepseek.com/a/chat/s/4349f49c-08f2-4977-8599-a04d1a9529bb
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

import carla
import numpy as np
import torch
import cv2
import queue
import threading
import time
import math

class CarlaSteering:
    def __init__(self, model_path='best_steering_model_20250329-153940.pth', host='localhost', port=2000, 
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
        
        # Image queues for processing
        self.rgb_queue = queue.Queue()
        self.seg_queue = queue.Queue()
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
        waypoints = self.world.get_map().generate_waypoints(2.0)
        route_42_waypoints = [w for w in waypoints if w.road_id == 42 and w.lane_id == -1]
        if not route_42_waypoints:
            raise ValueError("Could not find waypoints for route 42, lane -3")
        
        # Get the first waypoint and create a spawn point
        first_waypoint = route_42_waypoints[0]
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
        
        # Camera parameters
        image_width = 640
        image_height = 480
        fov = 90
        
        # RGB Camera
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(image_width))
        camera_bp.set_attribute('image_size_y', str(image_height))  
        camera_bp.set_attribute('fov', str(fov))
        camera_spawn_point = carla.Transform(carla.Location(x=1.5, z=1.8), carla.Rotation(pitch=-5, yaw=0, roll=0))
        self.rgb_camera = self.world.spawn_actor(camera_bp, camera_spawn_point, attach_to=self.vehicle)
        self.rgb_camera.listen(self.process_rgb_image)
        
        # Semantic Segmentation Camera
        seg_camera_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        seg_camera_bp.set_attribute('image_size_x', str(image_width))
        seg_camera_bp.set_attribute('image_size_y', str(image_height))
        seg_camera_bp.set_attribute('fov', str(fov))
        self.seg_camera = self.world.spawn_actor(seg_camera_bp, camera_spawn_point, attach_to=self.vehicle)
        self.seg_camera.listen(self.process_seg_image)
        
    def process_rgb_image(self, image):
        """Callback to process RGB images from CARLA camera"""
        img = np.array(image.raw_data).reshape(image.height, image.width, 4)
        img = img[:, :, :3]  # Remove alpha channel
        self.rgb_queue.put(img)
        
    def process_seg_image(self, image):
        """Callback to process segmentation images from CARLA camera"""
        seg_array = np.array(image.raw_data).reshape(image.height, image.width, 4)
        seg_array = seg_array[:, :, :3]  # Remove alpha channel
        
        # Extract road (class 7) and lane markings (class 6)
        road_mask = (seg_array[:, :, 2] == 7).astype(np.uint8) * 255
        lane_mask = (seg_array[:, :, 2] == 6).astype(np.uint8) * 255
        
        self.seg_queue.put((road_mask, lane_mask))
        
    def create_composite_image(self, rgb_img, road_mask, lane_mask):
        """Create a composite image with segmentation overlays"""
        composite = rgb_img.copy()
        
        # Apply road mask (green overlay)
        road_overlay = np.zeros_like(composite)
        road_overlay[:, :, 1] = road_mask  # Green channel
        composite = cv2.addWeighted(composite, 1.0, road_overlay, 0.3, 0.0)
        
        # Apply lane mask (yellow overlay)
        lane_overlay = np.zeros_like(composite)
        lane_overlay[:, :, 1] = lane_mask  # Green channel
        lane_overlay[:, :, 2] = lane_mask  # Red channel
        composite = cv2.addWeighted(composite, 1.0, lane_overlay, 0.5, 0.0)
        
        return composite
        
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
        """Display original, preprocessed and segmentation images"""
        if hasattr(self, 'original_img') and hasattr(self, 'preprocessed_img') and hasattr(self, 'composite_img'):
            # Resize original image to be similar height as preprocessed
            display_height = 264  # 4x preprocessed height
            aspect_ratio = self.original_img.shape[1] / self.original_img.shape[0]
            display_width = int(display_height * aspect_ratio)
            original_resized = cv2.resize(self.original_img, (display_width, display_height))
            
            # Resize preprocessed image for better visibility
            preprocessed_display = cv2.resize(self.preprocessed_img, (display_width, 264))
            
            # Resize composite image
            composite_display = cv2.resize(self.composite_img, (display_width, display_height))
            
            # Create a black canvas for display
            canvas_width = display_width * 3 + 40  # 3 images + padding
            canvas = np.zeros((display_height, canvas_width, 3), dtype=np.uint8)
            
            # Place images on canvas
            canvas[:, :display_width] = original_resized
            canvas[:, display_width+20:2*display_width+20] = preprocessed_display
            canvas[:, 2*display_width+40:] = composite_display
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(canvas, 'Original', (10, 30), font, 0.7, (255, 255, 255), 2)
            cv2.putText(canvas, 'Preprocessed', (display_width+30, 30), font, 0.7, (255, 255, 255), 2)
            cv2.putText(canvas, 'Segmentation', (2*display_width+50, 30), font, 0.7, (255, 255, 255), 2)
            
            # Show the canvas
            cv2.imshow('Camera Views', canvas)
            cv2.waitKey(1)


        
    def predict_steering_quant(self, image):
        """Make steering prediction from image and quantize the output"""
        with torch.no_grad():
            steering_pred = self.model(image)
        
        steering_angle = float(steering_pred.cpu().numpy()[0, 0])
        
        # Define the quantization levels (same as in compute_control)
        steering_levels = [round(x*0.02, 3) for x in range(-32, 33)]  # -0.64 to +0.64
        
        # Clamp the steering angle to the valid range first
        clamped_angle = np.clip(steering_angle, -0.65, 0.65)
        
        # Quantize to nearest level
        quantized_angle = min(steering_levels, key=lambda x: abs(x - clamped_angle))
        
            # Debug output shows the chaos
        print(f"Steering - Raw: {steering_angle:.4f} → Final: {quantized_angle:.4f}")

        self.last_steering = quantized_angle
        
        return quantized_angle
        
    def apply_control(self, steering, target_speed=10):
        """Apply control to vehicle with neural network steering and proportional speed control"""
        control = carla.VehicleControl()
        control.steer = steering

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
                # Clear the queues to always use latest frames
                while not self.rgb_queue.empty():
                    _ = self.rgb_queue.get()
                while not self.seg_queue.empty():
                    _ = self.seg_queue.get()
                
                # Tick the world and wait for new images
                self.world.tick()
                
                try:
                    # Wait for new images with timeout
                    rgb_img = self.rgb_queue.get(timeout=0.1)
                    road_mask, lane_mask = self.seg_queue.get(timeout=0.1)
                    
                    # Create composite image with segmentation
                    self.composite_img = self.create_composite_image(rgb_img, road_mask, lane_mask)
                    
                    # Preprocess image (using RGB only for prediction)
                    processed_img = self.preprocess_image(rgb_img)
                    
                    # Get steering prediction
                    steering = self.predict_steering_quant(processed_img)
                    
                    # Apply control
                    self.apply_control(steering)
                    
                    # Update spectator camera
                    set_spectator_camera_following_car(self.world, self.vehicle)
                    
                    # Display camera feeds
                    self.display_images()
                    
                except queue.Empty:
                    print("Warning: Frame missed!")
                    
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            # Cleanup
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
            
            if hasattr(self, 'rgb_camera'):
                self.rgb_camera.destroy()
            if hasattr(self, 'seg_camera'):
                self.seg_camera.destroy()
            if hasattr(self, 'vehicle'):
                self.vehicle.destroy()

def set_spectator_camera_following_car(world, vehicle):
    """Set spectator camera to follow the vehicle"""
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

if __name__ == '__main__':
    try:
        model_path='best_steering_model_20250329-194129.pth'
        controller = CarlaSteering(model_path=model_path)
        controller.run()
    except Exception as e:
        print(f"An error occurred: {e}")
