import torch
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

#labels = ('AnnualCrop,Forest,HerbaceousVegetation,Highway,Industrial,Pasture,PermanentCrop,Residential,River,SeaLake')

# Function to print tensor details
def print_tensor_details(tensor, name):
    print(f"Tensor: {name}, Shape: {tensor.shape}")

# Print details of all the tensors in the model
def print_model_tensors(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print_tensor_details(param.data, name)

''' def print_labels(labels):
        labels = labels.split(',')
        print(labels) '''

print_model_tensors(model)

#print_labels(labels)