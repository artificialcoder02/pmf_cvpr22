import numpy as np
import torchvision.transforms as transforms

def eurosat_dataset_setting(nSupport, img_size=128):
    """
    Return EuroSAT dataset setting

    :param int nSupport: number of support examples
    """
    mean = [0.485, 0.456, 0.406]  # ImageNet mean values
    std = [0.229, 0.224, 0.225]  # ImageNet standard deviation values
    normalize = transforms.Normalize(mean=mean, std=std)
    
    trainTransform = transforms.Compose([
        transforms.RandomResizedCrop((img_size, img_size), scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    
    valTransform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])
    
    inputW, inputH, nbCls = img_size, img_size, 10  # Assuming EuroSAT has 10 classes
    
    trainDir = "./data/EuroSAT/train"
    valDir = "./data/EuroSAT/val"
    testDir = "./data/EuroSAT/test"
    episodeJson = "./data/EuroSAT/episode_info.json"  # Replace with actual path
    
    return trainTransform, valTransform, inputW, inputH, trainDir, valDir, testDir, episodeJson, nbCls
