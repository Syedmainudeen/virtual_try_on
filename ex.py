from custom_dataset import CustomDataset
# Define the paths for train and test data
train_root_dir = r'C:\Users\Admin\Desktop\mini_project(prototype)\ACGPN_TrainData'
test_root_dir = r'C:\Users\Admin\Desktop\mini_project(prototype)\ACGPN_TestData'

# Create instances of the custom dataset
train_dataset = CustomDataset(root_dir=train_root_dir)
test_dataset = CustomDataset(root_dir=test_root_dir)

# Check the loaded data for the first sample in the train dataset
print("First train dataset item:", train_dataset.data[0])

# Check the loaded data for the first sample in the test dataset
print("First test dataset item:", test_dataset.data[0])
