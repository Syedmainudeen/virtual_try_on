import os
import matplotlib.pyplot as plt
from PIL import Image
import torch

# Function to load and visualize color images
def visualize_color_images(folder_path, title):
    plt.figure(figsize=(15, 5))
    files = os.listdir(folder_path)
    num_images = min(len(files), 5)  # Limit to 5 images for visualization
    for i in range(num_images):
        file_path = os.path.join(folder_path, files[i])
        color_img = Image.open(file_path)
        plt.subplot(1, num_images, i + 1)
        plt.imshow(color_img)
        plt.title(f'{title} {i+1}')
        plt.axis('off')
    plt.show()

# Load and visualize color images from train_color folder
train_color_folder = r'C:\Users\Admin\Desktop\mini_project(prototype)\ACGPN_TrainData\train_color'
visualize_color_images(train_color_folder, 'Train Color Images')

# Load and visualize color images from test_color folder
test_color_folder = r'C:\Users\Admin\Desktop\mini_project(prototype)\ACGPN_TestData\test_color'
visualize_color_images(test_color_folder, 'Test Color Images')
