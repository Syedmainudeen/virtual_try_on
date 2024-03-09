import os
import torch
from torch.optim import Adam
from torch.nn import BCELoss
from torch.utils.data import DataLoader
from torchvision import transforms

from train_utils import load_data, train
from G_D import Generator, Discriminator
from logger import Logger
from visualization import TensorBoardVisualizer
from custom_dataset import CustomDataset

if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    project_dir = r'C:\Users\Admin\Desktop\mini_project(prototype)'
    train_root_dir = os.path.join(project_dir, 'ACGPN_TrainData')
    test_root_dir = os.path.join(project_dir, 'ACGPN_TestData')
    log_dir = os.path.join(project_dir, 'logs')
    checkpoint_dir = os.path.join(project_dir, 'checkpoints')
    num_epochs = 10
    batch_size = 16
    lr = 0.0002
    beta1 = 0.5
    beta2 = 0.999
    early_stop_patience = 5
    num_workers = 4

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if not os.path.exists(train_root_dir):
        print("Error: Training directory", train_root_dir, "does not exist.")
        exit()

    if not os.path.exists(test_root_dir):
        print("Error: Testing directory", test_root_dir, "does not exist.")
        exit()

    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    optimizer_G = Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_D = Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))
    criterion_g = BCELoss()
    criterion_d = BCELoss()  # Define discriminator loss criterion

    train_loader, test_loader = load_data(
        os.path.join(train_root_dir), 
        os.path.join(test_root_dir), 
        transform_train=transform_train, 
        transform_test=transform_test, 
        batch_size=batch_size,
        num_workers=num_workers
    )

    train(
        generator, discriminator, optimizer_G, optimizer_D, criterion_g, criterion_d,
        train_loader, test_loader, num_epochs, log_dir, checkpoint_dir, device, early_stop_patience=5
    )
