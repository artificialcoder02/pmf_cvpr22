import os
import shutil
import random

# Define the path to the main source data directory
source_data_dir = (r"C:\Users\rctuh\Desktop\ISRO\pmf_cvpr22\data\eurosat\2750")

# Define the paths to the destination folders
output_base_dir = (r"C:\Users\rctuh\Desktop\ISRO\pmf_cvpr22\data\EURO_SPLIT")
train_dir = os.path.join(output_base_dir, "train")
test_dir = os.path.join(output_base_dir, "test")
val_dir = os.path.join(output_base_dir, "val")

# Define the split ratios
train_ratio = 0.7
test_ratio = 0.2
val_ratio = 0.1

# Create destination folders if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Iterate through the class folders
for class_folder in os.listdir(source_data_dir):
    class_path = os.path.join(source_data_dir, class_folder)

    # List all files (images) in the class folder
    all_files = os.listdir(class_path)

    # Shuffle the list of files randomly
    random.shuffle(all_files)

    # Calculate the split points for this class
    total_files = len(all_files)
    train_split = int(total_files * train_ratio)
    test_split = int(total_files * (train_ratio + test_ratio))

    # Split the files into train, test, and validation sets for this class
    train_files = all_files[:train_split]
    test_files = all_files[train_split:test_split]
    val_files = all_files[test_split:]

    # Copy the files to their respective folders
    for filename in train_files:
        src = os.path.join(class_path, filename)
        dst = os.path.join(train_dir, class_folder, filename)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)

    for filename in test_files:
        src = os.path.join(class_path, filename)
        dst = os.path.join(test_dir, class_folder, filename)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)

    for filename in val_files:
        src = os.path.join(class_path, filename)
        dst = os.path.join(val_dir, class_folder, filename)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)

print("Dataset split and copied into train, test, and val folders.")
