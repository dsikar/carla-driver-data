# File: carla_camera.py

import carla
import time
import threading
from collections import namedtuple

# Replace dataclass with namedtuple or regular class
CameraTransform = namedtuple('CameraTransform', ['x_offset', 'y_offset', 'z_offset', 'rotation'])

# Alternative using regular class if you need mutability:
'''
class CameraTransform:
    def __init__(self, x_offset, y_offset, z_offset, rotation):
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.z_offset = z_offset
        self.rotation = rotation

    def __repr__(self):
        return f"CameraTransform(x_offset={self.x_offset}, y_offset={self.y_offset}, z_offset={self.z_offset}, rotation={self.rotation})"
'''

class CameraManager:
    def __init__(self):
        self._camera_config = self._get_default_config()
        self._running = False
        self._camera_thread = None
        self._config_lock = threading.Lock()
    
    def _get_default_config(self):
        return CameraTransform(
            x_offset=-4.0,
            y_offset=0.0,
            z_offset=2.5,
            rotation=carla.Rotation(0, 0, 0)
        )
    
    @property
    def config(self):
        with self._config_lock:
            return self._camera_config
    
    def set_rear_view(self):
        with self._config_lock:
            self._camera_config = CameraTransform(
                x_offset=-4.0,
                y_offset=0.0,
                z_offset=2.5,
                rotation=carla.Rotation(0, 0, 0)
            )
    
    def set_front_view(self):
        with self._config_lock:
            self._camera_config = CameraTransform(
                x_offset=1.5,
                y_offset=0.0,
                z_offset=1.5,
                rotation=carla.Rotation(0, 180, 0)
            )
    
    def set_right_profile(self):
        with self._config_lock:
            self._camera_config = CameraTransform(
                x_offset=0.0,
                y_offset=4.0,
                z_offset=2.5,
                rotation=carla.Rotation(0, 90, 0)
            )
    
    def set_left_profile(self):
        with self._config_lock:
            self._camera_config = CameraTransform(
                x_offset=0.0,
                y_offset=-4.0,
                z_offset=2.5,
                rotation=carla.Rotation(0, -90, 0)
            )
    
    def set_top_down(self):
        with self._config_lock:
            self._camera_config = CameraTransform(
                x_offset=0.0,
                y_offset=0.0,
                z_offset=8.0,
                rotation=carla.Rotation(-90, 0, 0)
            )
    
    def set_custom_view(self, x, y, z, pitch=0, yaw=0, roll=0):
        with self._config_lock:
            self._camera_config = CameraTransform(
                x_offset=x,
                y_offset=y,
                z_offset=z,
                rotation=carla.Rotation(pitch, yaw, roll)
            )
    
    def _follow_vehicle(self, world, vehicle, spectator):
        """Internal method to handle camera following logic"""
        while self._running:
            try:
                with self._config_lock:
                    config = self._camera_config
                
                # Get vehicle's transform
                vehicle_transform = vehicle.get_transform()
                
                # Create relative location based on current camera configuration
                relative_location = carla.Location(
                    x=config.x_offset,
                    y=config.y_offset,
                    z=config.z_offset
                )
                
                # Get world location for camera
                camera_world_loc = vehicle_transform.transform(relative_location)
                
                # Combine vehicle's rotation with camera rotation offset
                camera_rotation = carla.Rotation(
                    pitch=vehicle_transform.rotation.pitch + config.rotation.pitch,
                    yaw=vehicle_transform.rotation.yaw + config.rotation.yaw,
                    roll=vehicle_transform.rotation.roll + config.rotation.roll
                )
                
                # Create and set the new transform
                transform = carla.Transform(camera_world_loc, camera_rotation)
                spectator.set_transform(transform)
                
                time.sleep(0.01)
            except Exception as e:
                print("Camera update error: {}".format(e))  # Python 3.6 string formatting
                break
    
    def start_following(self, world, vehicle, spectator):
        """Start following the vehicle with the camera"""
        if self._camera_thread is not None and self._camera_thread.is_alive():
            print("Camera is already following")
            return
        
        self._running = True
        self._camera_thread = threading.Thread(
            target=self._follow_vehicle,
            args=(world, vehicle, spectator)
        )
        self._camera_thread.start()
        print("Camera following started")
    
    def stop_following(self):
        """Stop following the vehicle"""
        self._running = False
        if self._camera_thread is not None:
            self._camera_thread.join()
            self._camera_thread = None
        print("Camera following stopped")
    
    def __del__(self):
        self.stop_following()