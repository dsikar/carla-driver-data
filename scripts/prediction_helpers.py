import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
import pandas as pd
import re

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

class SteeringPredictor:
    def __init__(self, model_path, min_timestamp, device='cuda', max_steering_angle=25.0):
        """
        Initialize the steering predictor
        
        Args:
            model_path (str): Path to the saved model
            min_timestamp (str): Minimum timestamp in format "YYYYMMDD_HHMMSS_microseconds"
            device (str): Device to run inference on ('cuda' or 'cpu')
            max_steering_angle (float): Maximum steering angle in degrees
        """
        self.device = device
        self.max_steering_angle = max_steering_angle
        self.last_steering = 0.0
        self.min_timestamp = min_timestamp
        
        # Initialize and load model
        self.model = NVIDIANet()
        self.model = load_model(self.model, model_path, device)
        
    def extract_timestamp_and_angle(self, filename):
        """Extract timestamp and angle from filename"""
        pattern = r"frame_(\d{8}_\d{6}_\d+)(?:_(-?\d+\.\d+))?\.jpg"
        match = re.match(pattern, filename)
        
        if match:
            timestamp = match.group(1)
            angle = float(match.group(2)) if match.group(2) else None
            return timestamp, angle
        return None, None

    def is_timestamp_valid(self, timestamp):
        """Check if timestamp is greater than or equal to minimum timestamp"""
        return timestamp >= self.min_timestamp

    def preprocess_image(self, img):
        """Preprocess image for neural network"""
        self.original_img = img.copy()
        
        # Crop
        cropped = img[260:440, :]
        
        # Resize
        resized = cv2.resize(cropped, (200, 66))
        
        # Convert to YUV
        yuv = cv2.cvtColor(resized, cv2.COLOR_RGB2YUV)
        
        # Store preprocessed image
        self.preprocessed_img = yuv.copy()
        
        # Prepare for PyTorch (CHW format)
        yuv = yuv.transpose((2, 0, 1))
        yuv = np.ascontiguousarray(yuv)
        
        return torch.from_numpy(yuv).float().unsqueeze(0).to(self.device)
        
    def predict_steering(self, image):
        """Make steering prediction from image"""
        with torch.no_grad():
            steering_pred = self.model(image)
            
        steering_angle = float(steering_pred.cpu().numpy()[0, 0])
        steering_angle = np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle)
        self.last_steering = steering_angle 
        
        return steering_angle
    
    def process_image_file(self, image_path):
        """Process a single image file and predict steering angle"""
        timestamp, actual_angle = self.extract_timestamp_and_angle(Path(image_path).name)
        
        if timestamp is None or not self.is_timestamp_valid(timestamp):
            return None, None
            
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        processed_img = self.preprocess_image(img)
        predicted_angle = self.predict_steering(processed_img)
        
        return predicted_angle, actual_angle
    
import os
import glob
from typing import List, Tuple

def get_carla_data_files(data_dir: str, min_timestamp: str = "20241209_173218_948893") -> List[Tuple[str, float]]:
    """
    Get all valid training files from the Carla dataset directory and their steering angles.
    
    Args:
        data_dir: Path to the carla_dataset directory
        min_timestamp: Minimum timestamp to include (as string)
    
    Returns:
        List of tuples containing (file_path, steering_angle)
    """
    # Get all jpg files in directory
    pattern = os.path.join(data_dir, "*.jpg")
    all_files = glob.glob(pattern)
    
    valid_files = []
    for file_path in all_files:
        # Get filename without extension
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        
        # Check if filename matches expected pattern
        if len(parts) >= 5 and 'steering' in filename:
            # Extract timestamp and steering
            timestamp = '_'.join(parts[0:3])  # Combine timestamp parts
            try:
                steering = float(parts[-1].replace('.jpg', ''))
                
                # Only include files with timestamp >= min_timestamp
                if timestamp >= min_timestamp:
                    valid_files.append((file_path, steering))
            except ValueError:
                continue  # Skip if steering value can't be converted to float
    
    # Sort by timestamp
    valid_files.sort(key=lambda x: os.path.basename(x[0]).split('_')[0:3])
    
    return valid_files    

import cv2
def prepare_image_for_neural_network(image_path, crop_top=260, crop_bottom=440):
    """
    Load image, crop, resize, and convert to YUV for neural network processing.
    
    Args:
        image_path: Path to the input image
        
    Returns:
        numpy array in YUV format, size 66x200x3
    """
    # Read and convert image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Crop
    cropped = img_rgb[crop_top:crop_bottom, :]
    
    # Resize to neural network input size (66x200)
    resized = cv2.resize(cropped, (200, 66))
    
    # Convert to YUV
    yuv = cv2.cvtColor(resized, cv2.COLOR_RGB2YUV)

    return yuv