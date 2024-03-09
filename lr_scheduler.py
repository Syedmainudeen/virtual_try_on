# lr_scheduler.py

import torch

class CosineAnnealingLR:
    def __init__(self, optimizer, max_epochs, initial_lr):
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.initial_lr = initial_lr

    def step(self, epoch):
        lr = self.initial_lr * 0.5 * (1 + torch.cos(epoch * 1.0 / self.max_epochs * 3.14159))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
