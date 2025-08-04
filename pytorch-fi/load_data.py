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

my_models = ["resnet50-v1-12", "shufflenet-v2-10", "squeezenet1.0-12" ,"vgg16-12","bvlcalexnet-12","googlenet-12","inception-v1-12"]
# my_models = ["shufflenet-v2-10"]



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

def remove_dropout(model):
    # Recursively visit all modules and replace Dropout layers with Identity
    for name, module in model.named_children():
        if isinstance(module, nn.Dropout):
            setattr(model, name, nn.Identity())
            print("removed drop out!")
        else:
            remove_dropout(module)

def load_pb_to_pytorch(pb_file):
    # Load the serialized ONNX tensor from the .pb file

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # transform = transforms.Compose([
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])


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


def convert_onnx_to_pytorch(onnx_model_path):
    # Load the ONNX model
    onnx_model = onnx.load(onnx_model_path)

    # Convert to PyTorch model using onnx2pytorch
    pytorch_model = ConvertModel(onnx_model)

    return pytorch_model


# Usage example
if __name__ == "__main__":
    dataset = []
    loaded_models = {}
    loaded_models["squeezenet1_0"] = models.squeezenet1_0(pretrained=True).eval()
    loaded_models["shufflenet_v2_x1_0"] = models.shufflenet_v2_x1_0(pretrained=True).eval()
    loaded_models["googlenet"] = models.googlenet(pretrained=True).eval()
    loaded_models["alexnet"] = models.alexnet(pretrained=True).eval()
    loaded_models["resnet50"] = models.resnet50(pretrained=True).eval()
    loaded_models["vgg16"] = models.vgg16(pretrained=True).eval()
    # loaded_models["inception_v3"] = models.inception_v3(pretrained=True).eval()
    # print(loaded_models.keys())
    # exit()
    for i in range(10):
        pb_file_path = f'../new_dataset/input_{i}.pb'  # Specify your .pb file path
        torch_tensor = load_pb_to_pytorch(pb_file_path)
        print(torch_tensor.shape)
        dataset.append(torch_tensor)
    answer = {}
    with open('answer.pkl', 'rb') as f:
        answer = pickle.load(f)
    print(answer)
    for model_name in loaded_models.keys():   
        answer[model_name] = []
        print(model_name)     
        pytorch_model = loaded_models[model_name]
        true_labels = get_labels(pytorch_model,dataset)
        print(true_labels)
        hook_FI_model = copy.deepcopy(pytorch_model)
        for fitype in range(2):
            # for fibit in range(12,32):
            for ppp in range(6):
                pattern = random.randint(0, 15)
                for fff in range(4):
                    fibit = random.randint(12, 31)
                    fit = f"StuckAt{fitype}"
                    modelFI = FI.Analysis(hook_FI_model,-1,None, None,None,fit,fibit,pattern)
                    labels = get_labels(modelFI,dataset)
                    sdc = 0
                    for i in range(10):
                        if(labels[i] !=true_labels[i]):
                            sdc += 1
                    # print(labels)
                    print(fit, fibit,pattern,sdc)
                    answer[model_name].append((fit, fibit,pattern,sdc))
                    # with open('answer.pkl', 'wb') as f:
                    #     pickle.dump(answer, f)
        # print(f"analysis_values: {analysis_values}")     
    
    