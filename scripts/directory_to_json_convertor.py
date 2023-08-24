import os
import random
import json

def create_episode(class_folders, num_support, num_query):
    episode = {"Support": [], "Query": []}
    
    for class_folder in class_folders:
        class_support = random.sample(os.listdir(class_folder), num_support)
        class_query = random.sample(os.listdir(class_folder), num_query)
        
        episode["Support"].append([os.path.join(class_folder, img) for img in class_support])
        episode["Query"].append([os.path.join(class_folder, img) for img in class_query])
    
    return episode

def main(data_folder, num_support, num_query, num_classes, num_shots, num_episodes, output_filename):
    class_folders = [os.path.join(data_folder, folder) for folder in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, folder))]
    
    episodes = []
    for _ in range(num_episodes):
        episode = create_episode(class_folders, num_shots, num_query)
        episodes.append(episode)
    
    total_support_files = num_classes * num_shots
    total_query_files = num_classes * num_query
    
    print("Number of files in the support set:", total_support_files)
    print("Number of files in the query set:", total_query_files)
    
    with open(output_filename, 'w') as json_file:
        json.dump(episodes, json_file, indent=4)

if __name__ == "__main__":
    data_folder = r"C:\Users\rctuh\Desktop\ISRO\pmf_cvpr22\data\EuroSAT\2750"  # Directory containing class folders
    num_classes = 5  # Number of classes in each episode
    num_shots = 5    # Number of support images per class
    num_query = 15   # Number of query images per class
    num_episodes = 100  # Number of episodes
    output_filename = "episodes.json"  # Output JSON filename
    
    main(data_folder, num_shots, num_query, num_classes, num_shots, num_episodes, output_filename)

