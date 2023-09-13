import os
import cv2
import numpy as np

# Define the path to the AID dataset
dataset_dir = "data\AID_Split"

# Initialize lists to store pixel values
pixel_values = []

# Loop through the images in the dataset
for root, dirs, files in os.walk(dataset_dir):
    for filename in files:
        if filename.endswith(".jpg"):  # Assuming images are in JPG format
            image_path = os.path.join(root, filename)
            image = cv2.imread(image_path)
            if image is not None:
                pixel_values.append(image / 255.0)  # Normalize pixel values to [0, 1]

# Calculate the mean and standard deviation
pixel_values = np.array(pixel_values)
mean = np.mean(pixel_values, axis=(0, 1, 2))
std = np.std(pixel_values, axis=(0, 1, 2))

print("Mean (R, G, B):", mean)
print("Standard Deviation (R, G, B):", std)