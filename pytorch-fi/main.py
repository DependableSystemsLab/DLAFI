import onnx
import torch
import numpy as np
import onnx
from onnx2pytorch import ConvertModel
import torchvision
from  torchvision import transforms, models
import os
import torch.nn as nn
import torchvision.transforms.functional as F
import copy
import FI
import random
import pickle
import argparse

torch.manual_seed(0)


def get_labels(model, inputs):
    correct = 0
    total = 0
    cnt = 0
    model.eval()  # Set the model to evaluation mode
    labels = []
    for image in inputs:
        # print("input ",image.shape)
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        labels.append(predicted.item())
        # print(predicted.item())
        cnt += 1

    return labels

def load_pb_to_pytorch(pb_file):
    # Load the serialized ONNX tensor from the .pb file

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    tensor_proto = onnx.TensorProto()
    with open(pb_file, 'rb') as f:
        tensor_proto.ParseFromString(f.read())
    
    # Convert the ONNX tensor to a NumPy array
    array = onnx.numpy_helper.to_array(tensor_proto)
    
    # Convert the NumPy array to a PyTorch tensor
    torch_tensor = torch.from_numpy(array)
    
    input_tensor = F.resize(torch_tensor, size=256)  # Resize to 256x256

    # Center crop the tensor
    input_tensor = F.center_crop(input_tensor, output_size=224)  # Crop to 224x224

    # Normalize the tensor (if not already done before this function)
    input_tensor = F.normalize(input_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    return input_tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FI analysis on selected models and images.")
    parser.add_argument('--models', nargs='+', default=["resnet50", "resnet18", "alexnet", "shufflenet_v2", "inceptionnet_v1"],
                        help='List of model names to run (default: all)')
    parser.add_argument('--num-images', type=int, default=10,
                        help='Number of input images to use (default: 10)')
    parser.add_argument('--num-iters', type=int, default=100,
                        help='Number of iterations for FI (default: 10)')
    parser.add_argument('--sa-dim', type=int, default=16,
                        help='Systolic array dimension (default: 16 for a 16x16 array)')


    args = parser.parse_args()

    dataset = []
    loaded_models = {}
    for model_name in args.models:
        print(f"+++++++++++++++++++++++Loading model: {model_name}")
        if model_name == "resnet50":
            loaded_models[model_name] = models.resnet50()
        elif model_name == "resnet18":
            loaded_models[model_name] = models.resnet18()
        elif model_name == "alexnet":
            loaded_models[model_name] = models.alexnet()
        elif model_name == "shufflenet_v2":
            loaded_models[model_name] = models.shufflenet_v2_x1_0()
        elif model_name == "inceptionnet_v1":
            loaded_models[model_name] = models.googlenet(pretrained=True).eval()
        else:
            print(f"Unknown model: {model_name}")
            continue
        loaded_models[model_name].load_state_dict(torch.load(f"./models/{model_name}.pth"))
        loaded_models[model_name].eval()

    for i in range(args.num_images):
        pb_file_path = f'./imagenet_samples/input_{i}.pb'
        torch_tensor = load_pb_to_pytorch(pb_file_path)
        dataset.append(torch_tensor)


    sa_dim = args.sa_dim
    num_iters = args.num_iters
    for model_name in loaded_models.keys():
        print(model_name)     
        pytorch_model = loaded_models[model_name]
        true_labels = get_labels(pytorch_model,dataset)
        hook_FI_model = copy.deepcopy(pytorch_model)
        sdc_total = 0
        for ft in range(2):
            for it in range(num_iters):
                sa_x = random.randint(0, sa_dim - 1)
                sa_y = random.randint(0, sa_dim - 1)
                fi_bit = random.randint(0, 31)
                fi_type = f"StuckAt{ft}"
                modelFI = FI.Analysis(hook_FI_model, fi_type,fi_bit,sa_x)
                labels = get_labels(modelFI,dataset)
                sdc = sum(1 for i in range(len(labels)) if labels[i] != true_labels[i]) / len(labels)
                print(f"Iteration {it+1}/{num_iters} for {model_name} with FI type {fi_type}, FI bit {fi_bit}, SA position ({sa_x}, {sa_y}): SDC = {sdc:.4f}")
                sdc_total += sdc
        sdc_avg = sdc_total / (num_iters * 2)
        print(f"Average SDC for {model_name} : {sdc_avg:.4f}")
    
    