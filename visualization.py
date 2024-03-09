import torch
import torchvision
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

def visualize_samples(dataset, num_samples=5):
    fig, axes = plt.subplots(num_samples, 7, figsize=(15, 15))
    for i in range(num_samples):
        img, label, color, colormask, edge, mask, pose = dataset[i]
        axes[i, 0].imshow(img.permute(1, 2, 0))
        axes[i, 1].imshow(label.permute(1, 2, 0))
        axes[i, 2].imshow(color.permute(1, 2, 0))
        axes[i, 3].imshow(colormask.permute(1, 2, 0))
        axes[i, 4].imshow(edge.permute(1, 2, 0))
        axes[i, 5].imshow(mask.permute(1, 2, 0))
        axes[i, 6].text(0.5, 0.5, str(pose), ha='center')
        axes[i, 0].set_title('Image')
        axes[i, 1].set_title('Label')
        axes[i, 2].set_title('Color')
        axes[i, 3].set_title('Colormask')
        axes[i, 4].set_title('Edge')
        axes[i, 5].set_title('Mask')
        axes[i, 6].set_title('Pose')
        for ax in axes[i]:
            ax.axis('off')
    plt.tight_layout()
    plt.show()

class TensorBoardVisualizer:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)  # Update log directory path

    def log_generated_images(self, images, epoch, step):
        """Logs generated images to TensorBoard."""
        global_step = (epoch * len(images)) + step
        grid = torchvision.utils.make_grid(images, nrow=4, normalize=True)
        self.writer.add_image('Generated Images', grid, global_step=global_step)
