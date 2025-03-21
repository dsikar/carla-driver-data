import carla
import numpy as np
import cv2
import time
from pathlib import Path

class CarlaSteeringSimplified:
    def __init__(self, host='localhost', port=2000):
        # Initialize CARLA client
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        # File paths
        self.input_dir = Path('/home/daniel/git/DeepSeek-VL/input_images')
        self.output_dir = Path('/home/daniel/git/DeepSeek-VL/output_predictions')
        
        # Ensure directories exist
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Image storage
        self.image = None

    def setup_vehicle_and_camera(self):
        """Spawn vehicle and camera, load Town04"""
        try:
            # Load Town04
            print("Loading Town04...")
            self.client.load_world('Town04')
            self.world = self.client.get_world()
            
            # Clean up existing actors
            for actor in self.world.get_actors():
                actor.destroy()
            
            # Get spawn point (simplified to first available waypoint)
            waypoints = self.world.get_map().generate_waypoints(2.0)
            spawn_point = waypoints[0].transform
            spawn_point.location.z += 1  # Lift slightly
            
            # Spawn vehicle
            blueprint_library = self.world.get_blueprint_library()
            vehicle_bp = blueprint_library.filter('model3')[0]
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            print("Vehicle spawned")
            
            # Spawn camera
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '640')
            camera_bp.set_attribute('image_size_y', '480')
            camera_bp.set_attribute('fov', '90')
            
            camera_spawn_point = carla.Transform(carla.Location(x=1.5, z=1.8))
            self.camera = self.world.spawn_actor(camera_bp, camera_spawn_point, attach_to=self.vehicle)
            self.camera.listen(self.process_image)
            print("Camera attached")
            
        except Exception as e:
            print(f"Error in setup: {e}")
            raise

    def process_image(self, image):
        """Callback to process one image"""
        try:
            img = np.frombuffer(image.raw_data, dtype=np.uint8)
            img = img.reshape((480, 640, 4))
            self.image = img[:, :, :3]  # Remove alpha channel
            print("Image captured")
        except Exception as e:
            print(f"Error processing image: {e}")

    def save_image(self):
        """Save the captured image"""
        if self.image is not None:
            try:
                resized = cv2.resize(self.image, (200, 200))
                filename = self.input_dir / 'test_image.jpg'
                cv2.imwrite(str(filename), cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
                print(f"Image saved to {filename}")
                return filename
            except Exception as e:
                print(f"Error saving image: {e}")
                return None
        return None

    def get_prediction(self):
        """Read prediction from file"""
        prediction_file = self.output_dir / '200x200.txt'
        timeout = 10.0  # Wait up to 10 seconds
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if prediction_file.exists():
                try:
                    with open(prediction_file, 'r') as f:
                        content = f.read().strip()
                    prediction_file.unlink()  # Delete after reading
                    print(f"Prediction received: {content}")
                    return content
                except Exception as e:
                    print(f"Error reading prediction: {e}")
                    return None
            time.sleep(0.1)
        
        print("Prediction timeout!")
        return None

    def run(self):
        """Main execution"""
        try:
            # Setup
            self.setup_vehicle_and_camera()
            
            # Wait briefly for first image
            print("Waiting for image...")
            for _ in range(20):  # Wait up to 2 seconds (at 10 FPS)
                self.world.tick()
                if self.image is not None:
                    break
                time.sleep(0.1)
            
            # Save image and get prediction
            if self.image is not None:
                self.save_image()
                prediction = self.get_prediction()
                if prediction:
                    print(f"Process completed with prediction: {prediction}")
                else:
                    print("No prediction received")
            else:
                print("No image captured")
            
        except Exception as e:
            print(f"Error in run: {e}")
        finally:
            # Cleanup
            if hasattr(self, 'camera'):
                self.camera.destroy()
            if hasattr(self, 'vehicle'):
                self.vehicle.destroy()
            print("Cleanup complete")

if __name__ == '__main__':
    try:
        controller = CarlaSteeringSimplified()
        controller.run()
    except Exception as e:
        print(f"Script error: {e}")