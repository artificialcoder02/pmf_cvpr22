import torch
import torchvision.transforms as transforms
from models import get_model
from dotmap import DotMap

#args
args = DotMap()
args.deploy = 'vanilla'
args.arch = 'dino_small_patch16'
args.no_pretrain = True
args.resume = r'C:\Users\rctuh\Desktop\ISRO\pmf_cvpr22\outputs\EURO_OPT\best.pth'

# model
device = 'cuda' #torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(args)
model.to(device)
#model.load_state_dict(torch.load(PATH))
model.eval()
#checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu')
model.load_state_dict(r'C:\Users\rctuh\Desktop\ISRO\pmf_cvpr22\outputs\EURO_OPT\best.pth', strict=True)

# image transforms
def test_transform():
    def _convert_image_to_rgb(im):
        return im.convert('RGB')

    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        _convert_image_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        ])

preprocess = test_transform()

@torch.no_grad()
def denormalize(x, mean, std):
    # 3, H, W
    t = x.clone()
    t.mul_(std).add_(mean)
    return torch.clamp(t, 0, 1)

image = Image.open(r'C:\Users\rctuh\Desktop\ISRO\pmf_cvpr22\data\eurosat\2750\AnnualCrop\AnnualCrop_1.jpg')  # Load your image
image = transform(image)  # Apply the transformations

with torch.no_grad():
    image = image.unsqueeze(0)  # Add a batch dimension
    output = model(image)
    _, predicted_class = torch.max(output, 1)  # Get the class with the highest probability

print(f"Predicted Class: {predicted_class.item()}")
