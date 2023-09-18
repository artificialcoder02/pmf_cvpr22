import torch
import torchvision.transforms as transforms
from PIL import Image
from models import get_model
from dotmap import DotMap

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
    query = preprocess(query).unsqueeze(0).to(device)  # (1, 3, H, W)
    return query

def inference(query_path, labels):
    '''
    query_path: Path to the user-provided image
    labels: list of class names
    '''
    labels = labels.split(',')

    # Load and preprocess the user-provided image
    query = test_transform(query_path)

    with torch.no_grad():
        # Simulate fetching support images (in this example, we use the same query image as support)
        supp_x = query.repeat(1, len(labels), 1, 1, 1)  # Repeat the query for each label
        supp_y = torch.arange(len(labels)).unsqueeze(0).repeat(1, 1).to(device)  # Label indices

        print('Using the provided image as support.')

        output = model(supp_x, supp_y, query)  # (1, 1, n_labels)
        probs = output.softmax(dim=-1).detach().cpu().numpy()

        return {k: float(v) for k, v in zip(labels, probs[0, 0])}

def main():
    query_path = r'C:\Users\rctuh\Desktop\ISRO\pmf_cvpr22\data\eurosat\2750\AnnualCrop\AnnualCrop_1.jpg' # Replace with the actual image path
    labels = ("AnnualCrop,Forest,HerbaceousVegetation,Highway,Industrial,Pasture,PermanentCrop,Residential,River,SeaLake")
  # Replace with appropriate labels

    result = inference(query_path, labels)
    print('Predicted probabilities:', result)

if __name__ == "__main__":
    main()
