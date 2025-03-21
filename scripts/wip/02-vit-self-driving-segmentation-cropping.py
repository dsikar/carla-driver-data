import carla
import cv2
import numpy as np
import queue
import time
import os
import math
from pathlib import Path

# Configuration
IMAGE_DIR = Path("/home/daniel/dev/claude-dr/transformer-regressor/input_dir")
PREDICTION_FILE = Path("/home/daniel/dev/claude-dr/transformer-regressor/output_dir/prediction.txt")
ORIGINAL_IMAGE_SIZE = (480, 640)  # Original size from camera
TARGET_IMAGE_SIZE = (640, 480)    # Target size after preprocessing

class CarlaSteering:
    def __init__(self, host='localhost', port=2000):
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        self.world.apply_settings(settings)
        
        self.image_queue = queue.Queue()
        self.last_steering = 0.0
        self.steering_smoothing = 0.7
        self.max_steering_angle = 0.15
        self.awaiting_prediction = False
        
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        if PREDICTION_FILE.exists():
            PREDICTION_FILE.unlink()
            print(f"Deleted existing {PREDICTION_FILE} at initialization")

    def setup_vehicle_and_cameras(self):
        self.client.load_world('Town04')
        self.world = self.client.get_world()
        
        waypoints = self.world.get_map().generate_waypoints(2.0)
        route_42_waypoints = [w for w in waypoints if w.road_id == 42 and w.lane_id == -3]
        if not route_42_waypoints:
            raise ValueError("Could not find waypoints for route 42, lane -3")
        
        spawn_point = carla.Transform(
            carla.Location(route_42_waypoints[0].transform.location.x,
                         route_42_waypoints[0].transform.location.y,
                         route_42_waypoints[0].transform.location.z + 1),
            route_42_waypoints[0].transform.rotation
        )
        
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('model3')[0]
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        
        rgb_bp = blueprint_library.find('sensor.camera.rgb')
        rgb_bp.set_attribute('image_size_x', str(ORIGINAL_IMAGE_SIZE[1]))
        rgb_bp.set_attribute('image_size_y', str(ORIGINAL_IMAGE_SIZE[0]))
        rgb_bp.set_attribute('fov', '90')
        camera_spawn_point = carla.Transform(carla.Location(x=1.5, z=1.8), carla.Rotation(pitch=-5))
        self.rgb_camera = self.world.spawn_actor(rgb_bp, camera_spawn_point, attach_to=self.vehicle)
        
        seg_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        seg_bp.set_attribute('image_size_x', str(ORIGINAL_IMAGE_SIZE[1]))
        seg_bp.set_attribute('image_size_y', str(ORIGINAL_IMAGE_SIZE[0]))
        seg_bp.set_attribute('fov', '90')
        self.seg_camera = self.world.spawn_actor(seg_bp, camera_spawn_point, attach_to=self.vehicle)
        
        self.latest_rgb = None
        self.latest_seg = None
        self.rgb_camera.listen(lambda image: self.process_rgb_image(image))
        self.seg_camera.listen(lambda image: self.process_seg_image(image))

    def process_rgb_image(self, image):
        rgb_array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(ORIGINAL_IMAGE_SIZE[0], ORIGINAL_IMAGE_SIZE[1], 4)[:, :, :3]
        self.latest_rgb = rgb_array

    def process_seg_image(self, image):
        seg_array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(ORIGINAL_IMAGE_SIZE[0], ORIGINAL_IMAGE_SIZE[1], 4)[:, :, :3]
        self.latest_seg = seg_array

    def preprocess_image(self, image):
        """Crop top 210 pixels and resize to 640x480."""
        cropped = image[210:480, :, :]  # Crop to 270x640
        resized = cv2.resize(cropped, (TARGET_IMAGE_SIZE[1], TARGET_IMAGE_SIZE[0]), interpolation=cv2.INTER_LINEAR)
        return resized

    def process_combined_image(self):
        if self.latest_rgb is None or self.latest_seg is None:
            print("No complete image data yet (RGB or Seg missing)")
            return None
            
        # Perform segmentation on original images (480x640)
        lane_mask = (self.latest_seg[:, :, 2] == 6).astype(np.uint8) * 255
        overlaid_image = self.latest_rgb.copy()
        overlaid_image[lane_mask == 255] = [0, 255, 0]  # Green overlay for lanes
        
        # Preprocess the combined image after segmentation
        processed_image = self.preprocess_image(overlaid_image)
        return processed_image

    def get_latest_prediction(self, image_filename):
        if not PREDICTION_FILE.exists():
            print(f"{PREDICTION_FILE} does not exist yet")
            return None
        
        with open(PREDICTION_FILE, "r") as f:
            lines = f.readlines()
            print(f"Read {PREDICTION_FILE}: {len(lines)} predictions found")
            if lines:
                try:
                    steering = float(lines[-1].strip())
                    print(f"Found prediction for {image_filename}: steering = {steering:.4f}")
                    PREDICTION_FILE.unlink()
                    print(f"Cleared {PREDICTION_FILE} after reading prediction")
                    image_path = IMAGE_DIR / image_filename
                    if image_path.exists():
                        image_path.unlink()
                        print(f"Deleted processed image {image_filename}")
                    return steering
                except ValueError:
                    print(f"Invalid prediction format in {PREDICTION_FILE}: {lines[-1].strip()}")
            else:
                print(f"Empty prediction file {PREDICTION_FILE}")
        return None

    def apply_control(self, steering, target_speed=8):
        control = carla.VehicleControl()
        control.steer = np.clip(steering, -1.0, 1.0)
        
        current_velocity = self.vehicle.get_velocity()
        speed = 3.6 * math.sqrt(current_velocity.x**2 + current_velocity.y**2)
        speed_error = target_speed - speed
        
        if speed_error > 0:
            control.throttle = min(0.3, speed_error / target_speed)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(0.3, -speed_error / target_speed)
        
        self.vehicle.apply_control(control)
        print(f"Control applied: steer={control.steer:.4f}, throttle={control.throttle:.4f}, brake={control.brake:.4f}")

    def run(self):
        try:
            self.setup_vehicle_and_cameras()
            print("Vehicle and cameras initialized. Starting control loop...")
            frame_count = 0
            
            while True:
                self.world.tick()
                print(f"World tick completed at {time.time():.2f}")
                
                # Generate frame regardless of prediction status
                combined_img = self.process_combined_image()
                if combined_img is not None:
                    frame_count += 1
                    image_filename = f"frame_{frame_count:06d}.jpg"
                    image_path = IMAGE_DIR / image_filename
                    cv2.imwrite(str(image_path), cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR))
                    print(f"Generated {image_filename} with lane segmentation (preprocessed to {TARGET_IMAGE_SIZE})")
                    
                    if not self.awaiting_prediction:
                        self.image_queue.put((image_filename, combined_img))
                        self.awaiting_prediction = True
                        print(f"State: Sent {image_filename} for prediction")
                
                if not self.image_queue.empty():
                    image_filename, _ = self.image_queue.queue[0]
                    print(f"Checking prediction for {image_filename}")
                    
                    steering = None
                    start_time = time.time()
                    while time.time() - start_time < 10.0:
                        steering = self.get_latest_prediction(image_filename)
                        if steering is not None:
                            break
                        time.sleep(0.1)
                    
                    if steering is not None:
                        self.image_queue.get()
                        self.awaiting_prediction = False
                        steering = np.clip(steering, -self.max_steering_angle, self.max_steering_angle)
                        self.last_steering = steering
                        print(f"Confirmed steering prediction: {steering:.4f}")
                        print(f"State: Prediction received, ready to send next image")
                    else:
                        steering = self.last_steering
                        print(f"No prediction received after timeout for {image_filename}, using last steering: {steering:.4f}")
                    
                    self.apply_control(steering)
                    print(f"Applied steering: {steering:.4f}")
                else:
                    print("State: Queue empty, applying last steering")
                    self.apply_control(self.last_steering)
                    print(f"Applied steering: {self.last_steering:.4f}")
                
                spectator = self.world.get_spectator()
                transform = self.vehicle.get_transform()
                location = transform.location
                rotation = transform.rotation
                offset_location = location - carla.Location(
                    x=35 * math.cos(math.radians(rotation.yaw)),
                    y=35 * math.sin(math.radians(rotation.yaw))
                )
                offset_location.z += 20
                spectator.set_transform(carla.Transform(
                    offset_location,
                    carla.Rotation(pitch=-15, yaw=rotation.yaw)
                ))
                
        finally:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
            for actor in (self.rgb_camera, self.seg_camera, self.vehicle):
                if actor is not None:
                    actor.destroy()

if __name__ == "__main__":
    controller = CarlaSteering()
    controller.run()