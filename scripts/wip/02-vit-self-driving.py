import carla
import cv2
import numpy as np
import queue
import time
import os
import math

# Configuration
IMAGE_DIR = "/home/daniel/dev/claude-dr/transformer-regressor/input_dir/"
PREDICTION_FILE = "/home/daniel/dev/claude-dr/transformer-regressor/output_dir/prediction.txt"
IMAGE_SIZE = (480, 640)

class CarlaSteering:
    def __init__(self, host='localhost', port=2000):
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        # Synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        self.world.apply_settings(settings)
        
        self.image_queue = queue.Queue()
        self.last_steering = 0.0
        self.steering_smoothing = 0.7
        self.max_steering_angle = 0.15
        
        # Ensure shared directories/files exist
        if not os.path.exists(IMAGE_DIR):
            os.makedirs(IMAGE_DIR)
        if os.path.exists(PREDICTION_FILE):
            os.remove(PREDICTION_FILE)  # Clear old predictions

    def setup_vehicle(self):
        # [Same as your original setup_vehicle()]
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
        
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(IMAGE_SIZE[1]))
        camera_bp.set_attribute('image_size_y', str(IMAGE_SIZE[0]))
        camera_bp.set_attribute('fov', '90')
        camera_spawn_point = carla.Transform(carla.Location(x=1.5, z=1.8), carla.Rotation(pitch=-5))
        self.camera = self.world.spawn_actor(camera_bp, camera_spawn_point, attach_to=self.vehicle)
        self.camera.listen(self.process_image)

    def process_image(self, image):
        img = np.array(image.raw_data).reshape(IMAGE_SIZE[0], IMAGE_SIZE[1], 4)[:, :, :3]
        self.image_queue.put(img)

    def get_latest_prediction(self, image_filename):
        if not os.path.exists(PREDICTION_FILE):
            return None
        with open(PREDICTION_FILE, "r") as f:
            lines = f.readlines()
            for line in reversed(lines):  # Check latest predictions first
                if line.startswith(image_filename):
                    return float(line.split(",")[1])
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

    def run(self):
        try:
            self.setup_vehicle()
            print("Vehicle initialized. Starting control loop...")
            frame_count = 0
            
            while True:
                self.world.tick()
                
                if not self.image_queue.empty():
                    img = self.image_queue.get()
                    frame_count += 1
                    image_filename = f"frame_{frame_count:06d}.jpg"
                    image_path = os.path.join(IMAGE_DIR, image_filename)
                    
                    # Save image for inference
                    cv2.imwrite(image_path, img)
                    print(f"Saved {image_filename}")
                    
                    # Wait for prediction
                    steering = None
                    for _ in range(50):  # Wait up to 5 seconds (0.1s * 50)
                        steering = self.get_latest_prediction(image_filename)
                        if steering is not None:
                            break
                        time.sleep(0.1)
                    
                    if steering is None:
                        print(f"No prediction for {image_filename}, using last steering")
                        steering = self.last_steering
                    else:
                        # Smooth steering
                        steering = np.clip(steering, -self.max_steering_angle, self.max_steering_angle)
                        # steering = (self.steering_smoothing * self.last_steering + 
                        #           (1 - self.steering_smoothing) * steering)
                        self.last_steering = steering
                    
                    self.apply_control(steering)
                    print(f"Applied steering: {steering:.4f}")
                
                # Update spectator camera (from your original code)
                spectator = self.world.get_spectator()
                transform = self.vehicle.get_transform()
                location = transform.location
                rotation = transform.rotation
                offset_location = location - carla.Location(x=35 * math.cos(math.radians(rotation.yaw)),
                                                          y=35 * math.sin(math.radians(rotation.yaw)))
                offset_location.z += 20
                spectator.set_transform(carla.Transform(offset_location,
                                                      carla.Rotation(pitch=-15, yaw=rotation.yaw)))
                
        finally:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
            if hasattr(self, 'camera'):
                self.camera.destroy()
            if hasattr(self, 'vehicle'):
                self.vehicle.destroy()

if __name__ == "__main__":
    controller = CarlaSteering()
    controller.run()