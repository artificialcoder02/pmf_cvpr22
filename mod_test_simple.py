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
    query = preprocess(query).unsqueeze(0).to(device)  # Add a batch dimension
    return query

@torch.no_grad()
def denormalize(x, mean, std):
    # 3, H, W
    t = x.clone()
    t.mul_(std).add_(mean)
    return torch.clamp(t, 0, 1)

def Inference(query,labels):
    with torch.no_grad():
        # query image
        query = preprocess(query).unsqueeze(0).unsqueeze(0).to(device) # (1, 1, 3, H, W)

        supp_x = []
        supp_y = []

        # search support images
        for idx, y in enumerate(labels):
            gis = GoogleImagesSearch(args.api_key, args.cx)
            _search_params['q'] = y
            gis.search(search_params=_search_params, custom_image_name='my_image')
            gis._custom_image_name = 'my_image' # fix: image name sometimes too long

            for j, x in enumerate(gis.results()):
                x.download('./')
                x_im = Image.open(x.path)

                # vis
                axs[idx, j].imshow(x_im)
                axs[idx, j].set_title(f'{y}{j}:{x.url}')
                axs[idx, j].axis('off')

                x_im = preprocess(x_im) # (3, H, W)
                supp_x.append(x_im)
                supp_y.append(idx)

        print('Searching for support images is done.')

        supp_x = torch.stack(supp_x, dim=0).unsqueeze(0).to(device) # (1, n_supp*n_labels, 3, H, W)
        supp_y = torch.tensor(supp_y).long().unsqueeze(0).to(device) # (1, n_supp*n_labels)

        with torch.cuda.amp.autocast(True):
            output = model(supp_x, supp_y, query) # (1, 1, n_labels)

        probs = output.softmax(dim=-1).detach().cpu().numpy()

        return {k: float(v) for k, v in zip(labels, probs[0, 0])}, fig




def main():
    query_path = r'C:\Users\rctuh\Desktop\ISRO\pmf_cvpr22\data\EURO_SPLIT\train\AnnualCrop\AnnualCrop_2.jpg'
    labels = ("AnnualCrop,Forest,HerbaceousVegetation,Highway,Industrial,Pasture,PermanentCrop,Residential,River,SeaLake")

    result = inference(query_path, labels)
    print('Predicted probabilities:', result)

if __name__ == "__main__":
    main()



















''' def inference(query_path, labels):
    
    query_path: Path to the user-provided image
    labels: list of class names
   
    labels = labels.split(',')

    # Load and preprocess the user-provided image
    query = test_transform(query_path)

    with torch.no_grad():
        # query image
        query = query.to(device) # (1, 3, H, W)

        # Dummy support data (you can modify this to use actual support data)
        supp_x = query.unsqueeze(0)  # (1, 1, 3, H, W)
        supp_y = torch.zeros(1).long().unsqueeze(0).to(device)  # Dummy support labels

        with torch.cuda.amp.autocast(True):
            output = model(supp_x, supp_y, query)  # (1, 1, n_labels)

        probs = output.softmax(dim=-1).squeeze(0).cpu().numpy()  # Remove unnecessary dimensions

        return {k: float(v) for k, v in zip(labels, probs)} '''