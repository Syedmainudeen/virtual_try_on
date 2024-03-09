# train_utils.py

import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from G_D import Generator, Discriminator
from torch.optim import Adam
from torch.nn import BCELoss
from logger import Logger
from visualization import TensorBoardVisualizer
from custom_dataset import CustomDataset

def load_data(train_root_dir, test_root_dir, transform_train=None, transform_test=None, batch_size=16, num_workers=4):
    # Load the custom dataset for training
    train_dataset = CustomDataset(root_dir=train_root_dir, transform=transform_train)
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty or does not contain any samples.")
    
    # Load the custom dataset for testing
    test_dataset = CustomDataset(root_dir=test_root_dir, transform=transform_test)
    if len(test_dataset) == 0:
        raise ValueError("Testing dataset is empty or does not contain any samples.")

    # Split the dataset into train and test
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


def evaluate_model(model, criterion, data_loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, clothes, labels, _ in data_loader:
            images = images.to(device)
            clothes = clothes.to(device)
            labels = labels.to(device)
            outputs = model(images, clothes)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
    return total_loss / len(data_loader.dataset)

def save_checkpoint(model, optimizer, epoch, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

def train(generator, discriminator, optimizer_G, optimizer_D, criterion_g, criterion_d,
        train_loader, test_loader, num_epochs, log_dir, checkpoint_dir, device, early_stop_patience=5):
    # Update log directory path
    log_dir = r'C:\Users\Admin\Desktop\mini_project(prototype)\logs'
    
    logger = Logger(log_dir)
    visualizer = TensorBoardVisualizer(log_dir)
    best_loss = np.inf
    no_improvement_count = 0

    for epoch in range(num_epochs):
        print(f"\n** Starting Epoch {epoch+1} **")

        generator.train()
        discriminator.train()

        for i, (person_images, clothes_images, labels, _) in enumerate(train_loader):
            person_images = person_images.to(device)
            clothes_images = clothes_images.to(device)

            optimizer_D.zero_grad()
            real_labels = torch.ones(person_images.size(0), 1).to(device)
            fake_labels = torch.zeros(person_images.size(0), 1).to(device)

            real_outputs = discriminator(torch.cat((person_images, clothes_images), 1))
            d_loss_real = criterion_d(real_outputs, real_labels)
            d_loss_real.backward()

            fake_person_images = generator(person_images, clothes_images)

            fake_outputs = discriminator(torch.cat((fake_person_images.detach(), clothes_images), 1))
            d_loss_fake = criterion_d(fake_outputs, fake_labels)
            d_loss_fake.backward()

            optimizer_D.step()

            fake_outputs = discriminator(torch.cat((fake_person_images, clothes_images), 1))
            g_loss = criterion_g(fake_outputs, real_labels)

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

            logger.log_scalar("Generator Loss", g_loss.item(), epoch * len(train_loader) + i)
            logger.log_scalar("Discriminator Loss Real", d_loss_real.item(), epoch * len(train_loader) + i)
            logger.log_scalar("Discriminator Loss Fake", d_loss_fake.item(), epoch * len(train_loader) + i)

            if i % 100 == 0:
                print(f"Iteration {i}/{len(train_loader)}: Generator Loss: {g_loss.item()}, Discriminator Loss Real: {d_loss_real.item()}, Discriminator Loss Fake: {d_loss_fake.item()}")

            # Log generated images to TensorBoard
            if i % 500 == 0:
                with torch.no_grad():
                    fake_person_images = generator(person_images, clothes_images)
                    visualizer.log_generated_images(fake_person_images.cpu(), epoch, i)

        # Evaluate model on test set
        test_loss = evaluate_model(generator, criterion_g, test_loader, device)

        # Log test loss
        logger.log_scalar("Test Loss", test_loss, epoch)

        # Save model checkpoint if test loss improves
        if test_loss < best_loss:
            save_checkpoint(generator, optimizer_G, epoch, checkpoint_dir)
            best_loss = test_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= early_stop_patience:
                print(f"No improvement for {early_stop_patience} epochs. Early stopping...")
                break

    # Close the logger and visualization writer
    logger.close()
    visualizer.writer.close()
