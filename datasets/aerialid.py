import numpy as np
import torchvision.transforms as transforms

def dataset_setting(nSupport, img_size=600):
    """
    Return EuroSAT dataset setting

    :param int nSupport: number of support examples
    """
    mean = [0.415, 0.456, 0.462] 
    std = [0.197, 0.209, 0.237]  
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
    
    #Change the values here

    trainDir = r"C:\Users\rctuh\Desktop\ISRO\pmf_cvpr22\data\EURO_SPLIT\train"
    valDir = r"C:\Users\rctuh\Desktop\ISRO\pmf_cvpr22\data\EURO_SPLIT\val"
    testDir = r"C:\Users\rctuh\Desktop\ISRO\pmf_cvpr22\data\EURO_SPLIT\test"
    episodeJson = r"C:\Users\rctuh\Desktop\ISRO\pmf_cvpr22\data\EURO_SPLIT\5way5shot_data.json"  # Replace with actual path
    
    return trainTransform, valTransform, inputW, inputH, trainDir, valDir, testDir, episodeJson, nbCls
