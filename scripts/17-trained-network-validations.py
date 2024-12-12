from prediction_helpers import *

data_dir = "carla_dataset"
file_pairs = get_carla_data_files(data_dir)
file_pairs = file_pairs[:1]
print(file_pairs)

# Load model
# prepend the path to the model

model_path = 'best_steering_model_v1.pth'
model = NVIDIANet()
model = load_model(model, model_path)

image_path, steering_angle = file_pairs[0]
image = prepare_image_for_neural_network(image_path)

# Load model
model_path = '/home/daniel/git/carla-driver-data/scripts/best_steering_model_v1.pth'
model = NVIDIANet()
model = load_model(model, model_path)

# data_dir = "carla_dataset"
# file_pairs = get_carla_data_files(data_dir)
# file = file_pairs[0]
# image_path, steering_angle = file
# image = prepare_image_for_neural_network(image_path)

# Convert to torch tensor and adjust dimensions for PyTorch (CHW instead of HWC)
image_tensor = torch.from_numpy(image).float()
image_tensor = image_tensor.permute(2, 0, 1)  # Change from HWC to CHW format

image_tensor = np.ascontiguousarray(image_tensor)
image_tensor = torch.from_numpy(image_tensor).float().unsqueeze(0).to('cuda')
steering_pred = model(image_tensor)
print(f"Actual: {steering_angle}, Predicted: {steering_pred.item()}")

