import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
import numpy as np

# Function to load and visualize label images
def visualize_label_images(folder_path, title):
    plt.figure(figsize=(15, 5))
    files = os.listdir(folder_path)
    num_images = min(len(files), 5)  # Limit to 5 images for visualization
    for i in range(num_images):
        file_path = os.path.join(folder_path, files[i])
        label_img = Image.open(file_path)
        label_img = label_img.convert('L')  # Convert to grayscale
        label_array = np.array(label_img)  # Convert PIL image to NumPy array
        label_tensor = torch.tensor(label_array)  # Convert NumPy array to PyTorch tensor
        plt.subplot(1, num_images, i + 1)
        plt.imshow(label_tensor, cmap='gray')
        plt.title(f'{title} {i+1}')
        plt.axis('off')
    plt.show()

# Load and visualize train label images
train_label_folder = r'C:\Users\Admin\Desktop\mini_project(prototype)\ACGPN_TrainData'
visualize_label_images(os.path.join(train_label_folder, 'train_label'), 'Train Label')

# Load and visualize test label images
test_label_folder = r'C:\Users\Admin\Desktop\mini_project(prototype)\ACGPN_TestData'
visualize_label_images(os.path.join(test_label_folder, 'test_label'), 'Test Label')
