import torch
import torchvision.transforms as transforms
from PIL import Image
from models import get_model
from dotmap import DotMap
import os

# Load your pre-trained DINO_deitsmall16_pretrain model
# Replace 'YourModel.pth' with the actual path to your model's weights
args = DotMap()
args.deploy = 'vanilla'
args.arch = 'dino_small_patch16'
model = get_model(args)
device = torch.device('cuda:0')
model = model.to(device)
model.load_state_dict(torch.load(r'C:\Users\rctuh\Desktop\ISRO\pmf_cvpr22\outputs\EURO_OPT\best.pth'), strict=False)
model.eval()  # Set the model to evaluation mode

# Define the transformation for preprocessing the input image
def test_transform(image_path):
    def _convert_image_to_rgb(im):
        return im.convert('RGB')

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        _convert_image_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load and preprocess the user-provided image
    query = Image.open(image_path)
    query = preprocess(query).unsqueeze(0).to(device)  # Add a batch dimension
    return query

# Function to load support images from a folder
def load_support_images(folder_path):
    support_images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.png')):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)
            preprocessed_image = preprocess(image)
            support_images.append(preprocessed_image)
    return torch.stack(support_images, dim=0).unsqueeze(0)

# Modified inference function
def inference_with_folder_support(query_path, labels, support_folder_path):
    '''
    query_path: Path to the user-provided image
    labels: list of class names
    support_folder_path: path to the folder containing support images
    '''
    labels = labels.split(',')

    # Load and preprocess the user-provided image
    query = test_transform(query_path)

    with torch.no_grad():
        # query image
        query = query.to(device) # (1, 3, H, W)

        # Load support images from the specified folder
        support_images = load_support_images(support_folder_path)

        supp_x = support_images  # (1, n_supp, 3, H, W)
        supp_y = torch.zeros(support_images.size(1)).long().unsqueeze(0).to(device)  # Dummy support labels

        with torch.cuda.amp.autocast(True):
            output = model(supp_x, supp_y, query)  # (1, 1, n_labels)

        probs = output.softmax(dim=-1).squeeze(0).cpu().numpy()  # Remove unnecessary dimensions

        return {k: float(v) for k, v in zip(labels, probs)}

def main():
    query_path = r'C:\Users\rctuh\Desktop\ISRO\pmf_cvpr22\data\EURO_SPLIT\train\AnnualCrop\AnnualCrop_2.jpg'
    labels = "AnnualCrop,Forest,HerbaceousVegetation,Highway,Industrial,Pasture,PermanentCrop,Residential,River,SeaLake"
    support_folder_path = r'c:\Users\rctuh\Desktop\ISRO\pmf_cvpr22\data\support'  # Replace with the actual path to your support images folder
    preprocess = test_transform(query_path)
    result = inference_with_folder_support(query_path, labels, support_folder_path)
    print('Predicted probabilities:', result)

if __name__ == "__main__":
    main()
