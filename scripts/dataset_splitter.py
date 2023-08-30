import os
import random
import shutil

def split_dataset(data_folder, train_percent, val_percent):
    class_folders = [folder for folder in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, folder))]
    
    for class_folder in class_folders:
        class_path = os.path.join(data_folder, class_folder)
        images = os.listdir(class_path)
        random.shuffle(images)
        
        num_images = len(images)
        num_train = int(num_images * train_percent)
        num_val = int(num_images * val_percent)
        
        train_images = images[:num_train]
        val_images = images[num_train:num_train+num_val]
        test_images = images[num_train+num_val:]
        
        train_folder = os.path.join(data_folder, 'train', class_folder)
        val_folder = os.path.join(data_folder, 'val', class_folder)
        test_folder = os.path.join(data_folder, 'test', class_folder)
        
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(val_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)
        
        for img in train_images:
            src_path = os.path.join(class_path, img)
            dest_path = os.path.join(train_folder, img)
            shutil.copy(src_path, dest_path)
        
        for img in val_images:
            src_path = os.path.join(class_path, img)
            dest_path = os.path.join(val_folder, img)
            shutil.copy(src_path, dest_path)
        
        for img in test_images:
            src_path = os.path.join(class_path, img)
            dest_path = os.path.join(test_folder, img)
            shutil.copy(src_path, dest_path)

if __name__ == "__main__":
    data_folder = r"C:\Users\rctuh\Desktop\ISRO\pmf_cvpr22\data\EuroSAT\2750"  # Directory containing class folders
    train_percent = 0.7  # Percentage of data for training
    val_percent = 0.15   # Percentage of data for validation and testing
    
    # Create train, val, and test folders
    os.makedirs(os.path.join(data_folder, 'train'), exist_ok=True)
    os.makedirs(os.path.join(data_folder, 'val'), exist_ok=True)
    os.makedirs(os.path.join(data_folder, 'test'), exist_ok=True)
    
    print("Your dataset splitting has started")
    # Split and copy the data
    split_dataset(data_folder, train_percent, val_percent)
    print("Your dataset has been splitted")
