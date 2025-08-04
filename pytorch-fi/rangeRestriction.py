import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import torch.nn as nn
import torch.optim as optim
import train

max_values = {}

def hook_fn(module, input, output):
    # output shape is (batch_size, num_channels, H, W)
    # Compute max values across each channel for the current batch
    current_max_values = torch.amax(output, dim=(0, 2, 3))
    # output.max(dim=-1)[0].max(dim=-1)[0]  # Max over H and W dimensions
    if module not in max_values:
        max_values[str(module)] = current_max_values  # Initialize if not already done
    else:
        max_values[str(module)] = torch.max(max_values[str(module)], current_max_values)  # Update max values
    # print(current_max_values.shape, "vs ",max_values[module].shape)

def get_range(model, dataloader, device):
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            layer.register_forward_hook(hook_fn)

    accuracy = train.test_accuracy(model, dataloader, device)
    print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f}%')

    for layer, values in max_values.items():
        print(f"Max values for {layer}: shape {values.shape} ")

    return max_values
