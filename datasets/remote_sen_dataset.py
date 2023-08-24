import numpy as np
import torchvision.transforms as transforms

def dataset_setting(nSupport, img_size=80):
    """
    Return dataset setting

    :param int nSupport: number of support examples
    """
    mean = [x in [71.78008428 , 60.76412707 , 49.99450878]]
    std = [x in [2.16575857 , 1.76581818 ,  1.78591218]]
    normalize = transforms.Normalize(mean=mean, std=std)
    trainTransform = transforms.Compose([#transforms.RandomCrop(img_size, padding=8),
                                         transforms.RandomResizedCrop((img_size, img_size), scale=(0.05, 1.0)),
                                         transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                         transforms.RandomHorizontalFlip(),
                                         #lambda x: np.asarray(x),
                                         transforms.ToTensor(),
                                         normalize
                                        ])

    valTransform = transforms.Compose([#transforms.CenterCrop(80),
                                       transforms.Resize((img_size, img_size)),
                                       #lambda x: np.asarray(x),
                                       transforms.ToTensor(),
                                       normalize])

    inputW, inputH, nbCls = img_size, img_size, 6 

    trainDir = r'C:\Users\rctuh\Desktop\ISRO\pmf_cvpr22\data\Mini-ImageNet\train'
    valDir = r'C:\Users\rctuh\Desktop\ISRO\pmf_cvpr22\data\Mini-ImageNet\val'
    testDir = r'C:\Users\rctuh\Desktop\ISRO\pmf_cvpr22\data\Mini-ImageNet\test'
    #episodeJson = r'C:\Users\rctuh\Desktop\ISRO\pmf_cvpr22\data\Mini-ImageNet\val1000Episode_5_way_1_shot.json' if nSupport == 1 \
            #else r'C:\Users\rctuh\Desktop\ISRO\pmf_cvpr22\data\Mini-ImageNet\val1000Episode_5_way_5_shot.json'

    return trainTransform, valTransform, inputW, inputH, trainDir, valDir, testDir, nbCls
