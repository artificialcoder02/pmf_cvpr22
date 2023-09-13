import cv2

# Load an example image (replace with your image path)
image_path = r"data\AID_Split\Airport\airport_1.jpg"
image = cv2.imread(image_path)

if image is not None:
    num_channels = image.shape[2] if len(image.shape) == 3 else 1
    
    if num_channels == 1:
        print("Image is grayscale with 1 channel.")
    elif num_channels == 3:
        print("Image is in color with 3 channels (RGB).")
    else:
        print(f"Image has {num_channels} channels.")
else:
    print("Failed to load the image.")
 



