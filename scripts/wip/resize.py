from PIL import Image

# Open the original image
input_path = "carla_dataset_1920x1080/20250317_175517_203463_steering_0.0000.jpg"
original_image = Image.open(input_path)

# Resize the image to 200x200
resized_image = original_image.resize((200, 200))
output_path = "200x200.jpg"
# Save the resized image
resized_image.save(output_path)

