import json
import os
import random

# Define the source folder path
source_folder = r"C:\Users\rctuh\Desktop\ISRO\pmf_cvpr22\data\EURO_SPLIT\val"

# Define class names
class_names = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]

# Number of classes and shots per class
num_classes_support = 5
num_shots_support = 5
num_shots_query = 15

# Create a list to store the data
data = []

while len(class_names) >= num_classes_support:
    class_data = {
        "Support": [],
        "Query": []
    }

    # Randomly select 5 classes for support
    selected_classes_support = random.sample(class_names, num_classes_support)
    
    # Remove selected classes from the available class list
    for c in selected_classes_support:
        class_names.remove(c)

    for i in range(num_classes_support):
        # Sample 5 images as support set from each selected class
        support_folder = os.path.join(source_folder, selected_classes_support[i])
        support_images = os.listdir(support_folder)
        random.shuffle(support_images)
        support_images = support_images[:num_shots_support]
        class_data["Support"].append([os.path.join(selected_classes_support[i], img) for img in support_images])

    # Initialize query set
    query_images = []

    # Sample 15 images as query set from each selected class
    for selected_class_query in selected_classes_support:
        support_folder = os.path.join(source_folder, selected_class_query)
        support_images = os.listdir(support_folder)
        random.shuffle(support_images)
        query_images.extend([os.path.join(selected_class_query, img) for img in support_images[:num_shots_query]])

    class_data["Query"].append(query_images)
    data.append(class_data)

# Save the data to a JSON file
output_file = "5way5shot_data.json"
with open(output_file, "w") as json_file:
    json.dump(data, json_file, indent=4)

print(f"JSON data saved to {output_file}")
