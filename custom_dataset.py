import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from pose_parse import parse_json_files

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = self.load_data()

    def load_data(self):
        data = []

        train_subfolders = ['train_img', 'train_label', 'train_color', 'train_colormask', 'train_edge', 'train_mask']
        test_subfolders = ['test_img', 'test_label', 'test_color', 'test_colormask', 'test_edge', 'test_mask']

        for subfolder in train_subfolders:
            subfolder_path = os.path.join(self.root_dir, subfolder)
            if os.path.exists(subfolder_path):
                filenames = os.listdir(subfolder_path)
                for filename in filenames:
                    entry = {}
                    entry[subfolder] = os.path.join(subfolder_path, filename)
                    data.append(entry)
                print(f"Loaded {len(filenames)} files from {subfolder}")

        for subfolder in test_subfolders:
            subfolder_path = os.path.join(self.root_dir, subfolder)
            if os.path.exists(subfolder_path):
                filenames = os.listdir(subfolder_path)
                for filename in filenames:
                    entry = {}
                    entry[subfolder] = os.path.join(subfolder_path, filename)
                    data.append(entry)
                print(f"Loaded {len(filenames)} files from {subfolder}")

        # Load pose data for train and test labels
        train_pose_folder_path = os.path.join(self.root_dir, 'ACGPN_TrainData', 'train_pose')
        test_pose_folder_path = os.path.join(self.root_dir, 'ACGPN_TestData', 'test_pose')
        train_pose_data = parse_json_files(r'C:\Users\Admin\Desktop\mini_project(prototype)\ACGPN_TrainData\train_pose')
        test_pose_data = parse_json_files(r'C:\Users\Admin\Desktop\mini_project(prototype)\ACGPN_TestData\test_pose')

        for entry in data:
            label_path = entry.get('train_label') or entry.get('test_label')
            if label_path:
                pose_filename = os.path.basename(label_path).split('.')[0] + '.json'
                if 'train_label' in entry:
                    entry['pose'] = train_pose_data.get(pose_filename)
                elif 'test_label' in entry:
                    entry['pose'] = test_pose_data.get(pose_filename)
                print("Pose for", label_path, ":", entry['pose'])

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self.data):
            raise IndexError("Index out of range")
    
        entry = self.data[idx]
        img_path = entry.get('train_img') or entry.get('test_img')
        label_path = entry.get('train_label') or entry.get('test_label')
        color_path = entry.get('train_color') or entry.get('test_color')
        colormask_path = entry.get('train_colormask') or entry.get('test_colormask')
        edge_path = entry.get('train_edge') or entry.get('test_edge')
        mask_path = entry.get('train_mask') or entry.get('test_mask')
        pose = entry.get('pose')

        # Initialize variables to store data
        img = label = color = colormask = edge = mask = None

        # Load images if paths exist
        if img_path and os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
        if label_path and os.path.exists(label_path):
            label = Image.open(label_path).convert('RGB')
        if color_path and os.path.exists(color_path):
            color = Image.open(color_path).convert('RGB')
        if colormask_path and os.path.exists(colormask_path):
            colormask = Image.open(colormask_path).convert('RGB')
        if edge_path and os.path.exists(edge_path):
            edge = Image.open(edge_path).convert('RGB')
        if mask_path and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('RGB')

        # Apply transformations if available
        if self.transform:
            if img:
                img = self.transform(img)
            if label:
                label = self.transform(label)
            if color:
                color = self.transform(color)
            if colormask:
                colormask = self.transform(colormask)
            if edge:
                edge = self.transform(edge)
            if mask:
                mask = self.transform(mask)

        return img, label, color, colormask, edge, mask, pose

# Example usage:
if __name__ == "__main__":
    # Define the root directory
    root_dir = r'C:\Users\Admin\Desktop\mini_project(prototype)'

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

    # Create instances of the custom dataset for train and test
    train_root_dir = r'C:\Users\Admin\Desktop\mini_project(prototype)\ACGPN_TrainData'
    test_root_dir = r'C:\Users\Admin\Desktop\mini_project(prototype)\ACGPN_TestData'
    train_dataset = CustomDataset(root_dir=train_root_dir, transform=transform)
    test_dataset = CustomDataset(root_dir=test_root_dir, transform=transform)

    # Accessing all items in the train dataset
    print("\nTrain Dataset Items:")
    for idx in range(len(train_dataset)):
        item = train_dataset[idx]
        print("Item", idx, ":", item)

    # Accessing all items in the test dataset
    print("\nTest Dataset Items:")
    for idx in range(len(test_dataset)):
        item = test_dataset[idx]
        print("Item", idx, ":", item)
