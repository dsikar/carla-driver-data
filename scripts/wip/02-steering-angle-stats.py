import os
import numpy as np
# read all file names in directory ~/git/carla-driver-data/carla_dataset, in the format 20250314_093000_573640_steering_0.0000.jpg
# and extract the steering angle from the file name
# store the steering angles in an array

files = os.listdir('/home/daniel/git/carla-driver-data/carla_dataset')
steering_angles = []
for file in files:
    if file.endswith('.jpg'):
        steering_angle = float(file.split('_')[-1].split('.jpg')[0])
        steering_angles.append(steering_angle)

# print basic stats
print(f"Number of images: {len(steering_angles)}")
print(f"Mean steering angle: {np.mean(steering_angles):.4f}")   
# maximum and minimum steering angles
print(f"Max steering angle: {max(steering_angles):.4f}")
print(f"Min steering angle: {min(steering_angles):.4f}")
# standard deviation
print(f"Standard deviation: {np.std(steering_angles):.4f}") 

# plot histogram
import matplotlib.pyplot as plt
plt.hist(steering_angles, bins=21, color='blue', edgecolor='black')
plt.xlabel('Steering angle')
plt.ylabel('Frequency')
plt.title('Histogram of steering angles')
plt.show()
# save histogram
plt.savefig('/home/daniel/git/carla-driver-data/steering_angle_histogram.png')