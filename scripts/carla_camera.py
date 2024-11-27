import carla
import time
import threading
from collections import namedtuple

CameraTransform = namedtuple('CameraTransform', ['x_offset', 'y_offset', 'z_offset', 'rotation'])

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
    
    def set_back_right_diagonal(self):
        with self._config_lock:
            self._camera_config = CameraTransform(
                x_offset=-4.0,  # Behind vehicle
                y_offset=-4.0,  # Left side
                z_offset=2.5,   # Height
                rotation=carla.Rotation(0, 45, 0)  # Look right
            )
    
    def set_back_left_diagonal(self):
        with self._config_lock:
            self._camera_config = CameraTransform(
                x_offset=-4.0,  # Behind vehicle
                y_offset=4.0,   # Right side
                z_offset=2.5,   # Height
                rotation=carla.Rotation(0, -45, 0)  # Look left
            )
    
    def set_front_right_diagonal(self):
        with self._config_lock:
            self._camera_config = CameraTransform(
                x_offset=4.0,   # Front of vehicle
                y_offset=-4.0,  # Left side
                z_offset=2.5,   # Height
                rotation=carla.Rotation(0, 135, 0)  # Look right
            )
    
    def set_front_left_diagonal(self):
        with self._config_lock:
            self._camera_config = CameraTransform(
                x_offset=4.0,   # Front of vehicle
                y_offset=4.0,   # Right side
                z_offset=2.5,   # Height
                rotation=carla.Rotation(0, -135, 0)  # Look left
            )
    
    def set_right_profile(self):
        with self._config_lock:
            self._camera_config = CameraTransform(
                x_offset=0.0,   # No forward/back offset
                y_offset=4.0,   # Offset to right side
                z_offset=2.0,   # Standard height
                rotation=carla.Rotation(0, -90, 0)  # Face left to look at vehicle
            )
    
    def set_left_profile(self):
        with self._config_lock:
            self._camera_config = CameraTransform(
                x_offset=0.0,   # No forward/back offset
                y_offset=-4.0,  # Offset to left side
                z_offset=2.0,   # Standard height
                rotation=carla.Rotation(0, 90, 0)  # Face right to look at vehicle
            )
    
    def set_back_view_tilted(self):
        with self._config_lock:
            self._camera_config = CameraTransform(
                x_offset=-4.0,  # Behind vehicle
                y_offset=0.0,
                z_offset=2.5,
                rotation=carla.Rotation(-30, 0, 0)  # 30 degrees down tilt
            )
    
    def set_front_view_tilted(self):
        with self._config_lock:
            self._camera_config = CameraTransform(
                x_offset=4.0,   # Front of vehicle
                y_offset=0.0,
                z_offset=2.5,
                rotation=carla.Rotation(-30, 180, 0)  # 30 degrees down tilt, 180 to face vehicle
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
                
                vehicle_transform = vehicle.get_transform()
                relative_location = carla.Location(
                    x=config.x_offset,
                    y=config.y_offset,
                    z=config.z_offset
                )
                
                camera_world_loc = vehicle_transform.transform(relative_location)
                camera_rotation = carla.Rotation(
                    pitch=vehicle_transform.rotation.pitch + config.rotation.pitch,
                    yaw=vehicle_transform.rotation.yaw + config.rotation.yaw,
                    roll=vehicle_transform.rotation.roll + config.rotation.roll
                )
                
                transform = carla.Transform(camera_world_loc, camera_rotation)
                spectator.set_transform(transform)
                
                time.sleep(0.01)
            except Exception as e:
                print("Camera update error: {}".format(e))
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