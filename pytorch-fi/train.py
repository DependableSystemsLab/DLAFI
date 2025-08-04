import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import torch.nn as nn
import torch.optim as optim

def test_accuracy(model, dataloader, device):
    correct = 0
    total = 0
    model.eval()  # Set the model to evaluation mode
    cnt = 0
    with torch.no_grad():  # No need to track gradients for validation
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            cnt += 1 
            if (cnt == 5):
                break

    accuracy = 100 * correct / total
    return accuracy

