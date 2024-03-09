import os
from custom_dataset import CustomDataset
from torchvision import transforms

# Define the paths for train and test data
train_root_dir = r'C:\Users\Admin\Desktop\mini_project(prototype)\ACGPN_TrainData'
test_root_dir = r'C:\Users\Admin\Desktop\mini_project(prototype)\ACGPN_TestData'

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Create instances of the custom dataset
train_dataset = CustomDataset(root_dir=train_root_dir, transform=transform)
test_dataset = CustomDataset(root_dir=test_root_dir, transform=transform)

# Accessing the first 7 items in the train dataset
print("Train dataset:")
for i in range(7):
    item = train_dataset[i]
    print(f"Item {i+1}: {item}")

# Accessing the first 7 items in the test dataset
print("\nTest dataset:")
for i in range(7):
    item = test_dataset[i]
    print(f"Item {i+1}: {item}")

