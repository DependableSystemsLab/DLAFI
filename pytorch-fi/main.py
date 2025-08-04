import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import torch.nn as nn
import torch.optim as optim
import plot
import rangeRestriction
import FI
import copy
import train

model_path = './checkpoint/cifar_resnet18.pth'


# Define a transform to normalize the data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load CIFAR-10 training and test sets
trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True,
                                        download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=16,
                         shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./dataset', train=False,
                                       download=True, transform=transform)
testloader = DataLoader(testset, batch_size=4,
                        shuffle=False, num_workers=2)

# Class labels in CIFAR-10
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Load a pre-trained ResNet model
model = resnet18(pretrained=True)

# Adjust the model for CIFAR-10's 10 classes
model.fc = nn.Linear(model.fc.in_features, 10)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define a Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    print("Loaded the trained model from disk.")
else:

    # Train the model
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i%100==0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
            
                torch.save(model.state_dict(), model_path)

    print('Finished Training')

accuracy = train.test_accuracy(model, testloader, device)
print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f}%')

hook_model = copy.deepcopy(model)
max_values = rangeRestriction.get_range(hook_model, testloader,device)
print(len(max_values))
# plot.plot_max_values_2d(max_values)

hook_FI_model = copy.deepcopy(model)
analysis_values = FI.Analysis(hook_FI_model,-1,max_values, testloader,device)
print(f"analysis_values: {analysis_values}")
