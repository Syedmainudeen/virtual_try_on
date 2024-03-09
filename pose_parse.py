import os
import json

def parse_json_files(folder_path):
    data = {}
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if file_path.endswith(".json"):
            with open(file_path) as f:
                json_data = json.load(f)
                data[filename] = json_data
    return data

# Example usage for train_pose:
train_pose_folder_path = r'C:\Users\Admin\Desktop\mini_project(prototype)\ACGPN_TrainData\train_pose'
parsed_train_data = parse_json_files(train_pose_folder_path)
print("Train Data:", parsed_train_data)

# Example usage for test_pose:
test_pose_folder_path = r'C:\Users\Admin\Desktop\mini_project(prototype)\ACGPN_TestData\test_pose'
parsed_test_data = parse_json_files(test_pose_folder_path)
print("Test Data:", parsed_test_data)
