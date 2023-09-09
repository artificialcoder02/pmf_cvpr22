''' import torch

print("Torch version:",torch.__version__)

print("Is CUDA enabled?",torch.cuda.is_available()) '''

import torch

# Load the model checkpoint
checkpoint = torch.load(r'C:\Users\rctuh\Desktop\ISRO\pmf_cvpr22\outputs\Cifar_exp1\checkpoint.pth')

# Access the model's state dictionary
model_state_dict = checkpoint['model']

# Create and open a text file for writing
output_file = open(r'outputs/Cifar_exp1/outputs.txt', 'w')


# Iterate through the model's state dictionary and write values to the text file
for key, value in model_state_dict.items():
    output_file.write(f'Key: {key}\n')
    output_file.write(f'Value:\n{value}\n')
    output_file.write('\n' + '-'*50 + '\n')  # Separation line for clarity

# Close the text file
output_file.close()