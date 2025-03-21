import carla
import math
import random
import time
import carla_helpers as helpers
import time

# Connect to the client and get the world object
client = carla.Client('localhost', 2000)
world = client.get_world()

# load Town04 map
world = client.load_world('Town01')

