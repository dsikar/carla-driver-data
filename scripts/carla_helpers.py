# carla_helpers.py

import carla
import math

def get_spectator_transform(world):
    """
    Get and print the current spectator transform in CARLA world.
    
    Args:
        world: CARLA world object
    
    Returns:
        tuple: (location, rotation) of the spectator
    """
    spectator = world.get_spectator()
    transform = spectator.get_transform()
    
    location = transform.location
    rotation = transform.rotation
    
    print("\n=== Spectator Transform ===")
    print(f"Location: x={location.x:.2f}, y={location.y:.2f}, z={location.z:.2f}")
    print(f"Rotation: pitch={rotation.pitch:.2f}, yaw={rotation.yaw:.2f}, roll={rotation.roll:.2f}")
    
    return location, rotation

def print_camera_params(camera):
    """
    Print camera parameters including FOV, resolution, and position.
    
    Args:
        camera: CARLA camera actor
    """
    camera_bp = camera.attributes
    transform = camera.get_transform()
    
    print("\n=== Camera Parameters ===")
    print(f"FOV: {camera_bp.get('fov', 'N/A')}")
    print(f"Resolution: {camera_bp.get('image_size_x', 'N/A')}x{camera_bp.get('image_size_y', 'N/A')}")
    print(f"Location: x={transform.location.x:.2f}, y={transform.location.y:.2f}, z={transform.location.z:.2f}")
    print(f"Rotation: pitch={transform.rotation.pitch:.2f}, yaw={transform.rotation.yaw:.2f}, roll={transform.rotation.roll:.2f}")

def get_world_info(world):
    """
    Print information about the current CARLA world.
    
    Args:
        world: CARLA world object
    """
    map_name = world.get_map().name
    weather = world.get_weather()
    
    print("\n=== World Information ===")
    print(f"Map: {map_name}")
    print(f"Number of actors: {len(world.get_actors())}")
    print("\nWeather:")
    print(f"Sun Altitude: {weather.sun_altitude_angle:.2f}°")
    print(f"Sun Azimuth: {weather.sun_azimuth_angle:.2f}°")
    print(f"Cloudiness: {weather.cloudiness}%")
    print(f"Precipitation: {weather.precipitation}%")
    print(f"Fog Density: {weather.fog_density}%")

def save_transform(transform, filename="transform.txt"):
    """
    Save current transform to a file for later use.
    
    Args:
        transform: CARLA transform object
        filename (str): Name of file to save to
    """
    with open(filename, 'w') as f:
        f.write(f"Location: {transform.location.x},{transform.location.y},{transform.location.z}\n")
        f.write(f"Rotation: {transform.rotation.pitch},{transform.rotation.yaw},{transform.rotation.roll}\n")
    print(f"\nTransform saved to {filename}")

def load_transform(filename="transform.txt"):
    """
    Load transform from a file.
    
    Args:
        filename (str): Name of file to load from
    
    Returns:
        carla.Transform: Loaded transform object
    """
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            loc = [float(x) for x in lines[0].split(': ')[1].strip().split(',')]
            rot = [float(x) for x in lines[1].split(': ')[1].strip().split(',')]
            
            location = carla.Location(x=loc[0], y=loc[1], z=loc[2])
            rotation = carla.Rotation(pitch=rot[0], yaw=rot[1], roll=rot[2])
            
            return carla.Transform(location, rotation)
    except Exception as e:
        print(f"Error loading transform: {e}")
        return None

def format_transform(transform):
    """
    Format a transform object into a clean string representation.
    
    Args:
        transform: CARLA transform object
    
    Returns:
        str: Formatted string with transform information
    """
    return (f"Location(x={transform.location.x:.2f}, y={transform.location.y:.2f}, z={transform.location.z:.2f}), "
            f"Rotation(pitch={transform.rotation.pitch:.2f}, yaw={transform.rotation.yaw:.2f}, roll={transform.rotation.roll:.2f})")

# Example usage:
"""
import carla
from carla_helpers import *

# Connect to CARLA
client = carla.Client('localhost', 2000)
world = client.get_world()

# Get spectator position
location, rotation = get_spectator_transform(world)

# Get world info
get_world_info(world)

# Save current spectator transform
spectator = world.get_spectator()
save_transform(spectator.get_transform())

# Load saved transform
new_transform = load_transform()
if new_transform:
    print("\nLoaded transform:")
    print(format_transform(new_transform))
"""

def set_spectator_transform(world, location, rotation=None):
    """
    Set the spectator's transform in the CARLA world.
    
    Args:
        world: CARLA world object
        location: Can be one of:
            - carla.Location object
            - carla.Transform object
            - tuple/list of 3 floats (x, y, z)
        rotation (optional): Can be one of:
            - carla.Rotation object
            - tuple/list of 3 floats (pitch, yaw, roll)
            - None (if location is a Transform or if no rotation needed)
    
    Returns:
        carla.Transform: The new transform that was applied
    
    Example usage:
        # Using Transform
        transform = carla.Transform(carla.Location(x=10, y=20, z=30), 
                                  carla.Rotation(pitch=0, yaw=180, roll=0))
        set_spectator_transform(world, transform)
        
        # Using separate location and rotation
        set_spectator_transform(world, 
                              carla.Location(x=10, y=20, z=30),
                              carla.Rotation(pitch=0, yaw=180, roll=0))
        
        # Using tuples
        set_spectator_transform(world, (10, 20, 30), (0, 180, 0))
    """
    spectator = world.get_spectator()
    
    # Handle Transform object
    if isinstance(location, carla.Transform):
        transform = location
    
    # Handle tuple/list input for location
    elif isinstance(location, (tuple, list)):
        if len(location) != 3:
            raise ValueError("Location tuple/list must have exactly 3 elements (x, y, z)")
        location = carla.Location(x=float(location[0]), 
                                y=float(location[1]), 
                                z=float(location[2]))
        
        # Handle rotation if provided as tuple/list
        if isinstance(rotation, (tuple, list)):
            if len(rotation) != 3:
                raise ValueError("Rotation tuple/list must have exactly 3 elements (pitch, yaw, roll)")
            rotation = carla.Rotation(pitch=float(rotation[0]), 
                                    yaw=float(rotation[1]), 
                                    roll=float(rotation[2]))
        
        transform = carla.Transform(location, rotation or carla.Rotation())
    
    # Handle carla.Location with optional rotation
    elif isinstance(location, carla.Location):
        transform = carla.Transform(location, rotation or carla.Rotation())
    
    else:
        raise ValueError("Location must be a carla.Transform, carla.Location, or tuple/list of 3 floats")

    # Set the transform
    spectator.set_transform(transform)
    
    # Print the new transform
    print("\n=== Spectator Transform Set ===")
    print(f"Location: x={transform.location.x:.2f}, y={transform.location.y:.2f}, z={transform.location.z:.2f}")
    print(f"Rotation: pitch={transform.rotation.pitch:.2f}, yaw={transform.rotation.yaw:.2f}, roll={transform.rotation.roll:.2f}")
    
    return transform

# Example usage:
"""
import carla
from carla_helpers import set_spectator_transform

# Connect to CARLA
client = carla.Client('localhost', 2000)
world = client.get_world()

# Example 1: Using carla objects
location = carla.Location(x=10, y=20, z=30)
rotation = carla.Rotation(pitch=0, yaw=180, roll=0)
set_spectator_transform(world, location, rotation)

# Example 2: Using tuples
set_spectator_transform(world, (10, 20, 30), (0, 180, 0))

# Example 3: Using Transform
transform = carla.Transform(carla.Location(x=10, y=20, z=30), 
                          carla.Rotation(pitch=0, yaw=180, roll=0))
set_spectator_transform(world, transform)

# Example 4: Just location, no rotation
set_spectator_transform(world, (10, 20, 30))
"""

def print_cardinal_directions():
    """
    Print CARLA's coordinate system and cardinal directions.
    """
    print("\n=== CARLA Cardinal Directions ===")
    print("Coordinate System:")
    print("X+ = North")
    print("X- = South")
    print("Y+ = East")
    print("Y- = West")
    print("Z+ = Up")
    print("Z- = Down")
    print("\nRotations (yaw):")
    print("0° = North")
    print("90° = East")
    print("180° = South")
    print("270° or -90° = West")

def get_cardinal_direction(yaw):
    """
    Convert a yaw angle to the closest cardinal direction.
    
    Args:
        yaw (float): Yaw angle in degrees
    
    Returns:
        str: Cardinal direction (N, NE, E, SE, S, SW, W, or NW)
    """
    # Normalize yaw to 0-360
    yaw = yaw % 360
    
    # Define direction ranges (each 45° sector)
    if 337.5 <= yaw or yaw < 22.5:
        return "N"
    elif 22.5 <= yaw < 67.5:
        return "NE"
    elif 67.5 <= yaw < 112.5:
        return "E"
    elif 112.5 <= yaw < 157.5:
        return "SE"
    elif 157.5 <= yaw < 202.5:
        return "S"
    elif 202.5 <= yaw < 247.5:
        return "SW"
    elif 247.5 <= yaw < 292.5:
        return "W"
    else:  # 292.5 <= yaw < 337.5
        return "NW"

def get_spectator_direction(world):
    """
    Print the current direction the spectator is facing.
    
    Args:
        world: CARLA world object
    """
    spectator = world.get_spectator()
    transform = spectator.get_transform()
    direction = get_cardinal_direction(transform.rotation.yaw)
    
    print(f"\nSpectator is facing: {direction}")
    print(f"Exact yaw angle: {transform.rotation.yaw:.1f}°")

# Example usage:
"""
import carla
from carla_helpers import print_cardinal_directions, get_spectator_direction

client = carla.Client('localhost', 2000)
world = client.get_world()

# Print coordinate system reference
print_cardinal_directions()

# Get current spectator direction
get_spectator_direction(world)
"""

def get_map_geo_reference(world):
    """
    Get and print the geographic reference parameters of the CARLA map.
    
    Args:
        world: CARLA world object
    
    Returns:
        dict: Geographic reference parameters
    """
    carla_map = world.get_map()
    geo_ref = carla_map.transform_to_geolocation
    
    print("\n=== Map Geographic Reference ===")
    print(f"Latitude: {geo_ref.latitude} degrees")
    print(f"Longitude: {geo_ref.longitude} degrees")
    print(f"Altitude: {geo_ref.altitude} meters")
    
    return {
        "latitude": geo_ref.latitude,
        "longitude": geo_ref.longitude,
        "altitude": geo_ref.altitude
    }

def convert_to_geo(world, location):
    """
    Convert CARLA world location to geographic coordinates.
    
    Args:
        world: CARLA world object
        location: carla.Location object or tuple/list of (x, y, z)
    
    Returns:
        tuple: (latitude, longitude, altitude)
    """
    if isinstance(location, (tuple, list)):
        location = carla.Location(x=location[0], y=location[1], z=location[2])
    
    geo_location = world.get_map().transform_to_geolocation(location)
    
    print("\n=== Coordinate Conversion ===")
    print(f"CARLA Location: x={location.x:.2f}, y={location.y:.2f}, z={location.z:.2f}")
    print(f"Geographic Location:")
    print(f"Latitude:  {geo_location.latitude:.6f}°")
    print(f"Longitude: {geo_location.longitude:.6f}°")
    print(f"Altitude:  {geo_location.altitude:.2f}m")
    
    return geo_location.latitude, geo_location.longitude, geo_location.altitude

def get_spectator_geo_location(world):
    """
    Get the geographic coordinates of the spectator's current position.
    
    Args:
        world: CARLA world object
    
    Returns:
        tuple: (latitude, longitude, altitude)
    """
    spectator = world.get_spectator()
    location = spectator.get_transform().location
    return convert_to_geo(world, location)

# Example usage:
"""
import carla
from carla_helpers import get_map_geo_reference, convert_to_geo, get_spectator_geo_location

# Connect to CARLA
client = carla.Client('localhost', 2000)
world = client.get_world()

# Get map's geographic reference
geo_ref = get_map_geo_reference(world)

# Convert a specific location
location = carla.Location(x=100, y=200, z=0)
lat, lon, alt = convert_to_geo(world, location)

# Get spectator's geographic position
spec_lat, spec_lon, spec_alt = get_spectator_geo_location(world)
"""

################
# SUN POSITION #
################

# import carla # assumed imported already

def set_sun_position(world, altitude, azimuth):
    """
    Set sun position with debug print
    
    Args:
        world: CARLA world object
        altitude (float): Sun's height angle in degrees. Range: -90 to 90
                         -90 = below horizon (night)
                         0 = at horizon
                         90 = directly overhead
        azimuth (float): Sun's horizontal angle in degrees. Range: 0 to 360
                        0/360 = North
                        90 = East
                        180 = South
                        270 = West
    
    Returns:
        carla.WeatherParameters: The weather object with updated sun position
        
    Additional Parameters (fixed in this function):
        cloudiness: Range 0 to 100
        precipitation: Range 0 to 100
        precipitation_deposits: Range 0 to 100
        fog_density: Range 0 to 100
    """
    weather = carla.WeatherParameters(
        sun_altitude_angle=altitude,
        sun_azimuth_angle=azimuth,
        cloudiness=0,
        precipitation=0,
        precipitation_deposits=0,
        fog_density=0
    )
    world.set_weather(weather)
    print(f"Sun Position Set - Altitude: {altitude}°, Azimuth: {azimuth}°")
    return weather

# Time of day presets
def morning_east(world):
    """Sunrise from the east"""
    return set_sun_position(world, altitude=15, azimuth=90)  # Low in the east

def noon_overhead(world):
    """Sun directly overhead"""
    return set_sun_position(world, altitude=90, azimuth=180)  # Highest point

def afternoon_west(world):
    """Afternoon sun from west"""
    return set_sun_position(world, altitude=45, azimuth=270)  # Medium height in west

def sunset_west(world):
    """Sunset in the west"""
    return set_sun_position(world, altitude=10, azimuth=270)  # Low in the west

def night_scene(world):
    """Night time"""
    return set_sun_position(world, altitude=-90, azimuth=270)  # Below horizon

# Example usage:
"""
# Connect to CARLA
client = carla.Client('localhost', 2000)
world = client.get_world()

# Test different positions
morning_east(world)    # Sunrise
time.sleep(2)          # Wait to observe
noon_overhead(world)   # Noon
time.sleep(2)
afternoon_west(world)  # Afternoon
time.sleep(2)
sunset_west(world)     # Sunset
time.sleep(2)
night_scene(world)     # Night
"""

# SPAWN POINT HELPERS

def set_spectator_to_spawn(world, spawn_points, index):
   if not spawn_points or index < 0 or index >= len(spawn_points):
       raise ValueError(f"Index {index} out of bounds for spawn points list of length {len(spawn_points)}")
       
   spectator = world.get_spectator()
   spawn_point = spawn_points[index]
   spectator_transform = carla.Transform(
       spawn_point.location + carla.Location(z=2.5, x=-8),
       carla.Rotation(pitch=-15, yaw=spawn_point.rotation.yaw)
   )
   spectator.set_transform(spectator_transform)

def print_transform(transform):
   print(f"x={transform.location.x:.2f}, y={transform.location.y:.2f}, z={transform.location.z:.2f}")
   print(f"roll={transform.rotation.roll:.2f}, pitch={transform.rotation.pitch:.2f}, yaw={transform.rotation.yaw:.2f}")   

def print_transform_data(transform):
   forward = transform.get_forward_vector()
   right = transform.get_right_vector()
   up = transform.get_up_vector()
   
   print("Forward vector (x,y,z):", f"({forward.x:.2f}, {forward.y:.2f}, {forward.z:.2f})")
   print("Right vector (x,y,z):", f"({right.x:.2f}, {right.y:.2f}, {right.z:.2f})")
   print("Up vector (x,y,z):", f"({up.x:.2f}, {up.y:.2f}, {up.z:.2f})")
   
   print("\nTransform matrix:")
   for row in transform.get_matrix():
       print([f"{x:.2f}" for x in row])
       
   print("\nInverse matrix:")
   for row in transform.get_inverse_matrix():
       print([f"{x:.2f}" for x in row]) 