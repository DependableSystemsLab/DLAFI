import random
import numpy as np
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import torch.nn as nn
import torch.optim as optim
import struct
import train

def float_to_fixed(value, integer_bits=16, fractional_bits=16):
    """
    Converts a floating-point number to fixed-point representation.
    """
    scale_factor = 2 ** fractional_bits
    return int(value * scale_factor)

def fixed_to_float(value, integer_bits=16, fractional_bits=16):
    """
    Converts a fixed-point number back to floating-point representation.
    """
    scale_factor = 2 ** fractional_bits
    return value / scale_factor

def inject_fault_fixed_point(matrix, integer_bits=16, fractional_bits=16):
    # Ensure the matrix is not empty
    if not matrix or not matrix[0]:
        return None
    
    # Select a random element
    row = random.randint(0, len(matrix) - 1)
    col = random.randint(0, len(matrix[0]) - 1)
    original_value = matrix[row][col]
    
    # Convert to fixed-point
    fixed_value = float_to_fixed(original_value, integer_bits, fractional_bits)
    
    # Convert to binary representation
    bits = format(fixed_value, '032b')  # 32-bit representation
    
    # Flip a random bit
    bit_to_flip = random.randint(0, 31)
    if bits[bit_to_flip] == '0':
        new_bits = bits[:bit_to_flip] + '1' + bits[bit_to_flip + 1:]
    else:
        new_bits = bits[:bit_to_flip] + '0' + bits[bit_to_flip + 1:]
    
    # Convert back to fixed-point integer
    new_fixed_value = int(new_bits, 2)
    
    # Convert back to floating-point
    new_value = fixed_to_float(new_fixed_value, integer_bits, fractional_bits)
    
    # Inject the fault
    matrix[row][col] = new_value
    
    return (row, col), original_value, new_value



def BitFlipFloatingPoint(original_value, bit, integer_bits=16, fractional_bits=16):

    binary_repr = struct.unpack('>I', struct.pack('>f', original_value.item()))[0]
    flipped_repr = binary_repr ^ (1 << bit)
    new_value = struct.unpack('>f', struct.pack('>I', flipped_repr))[0]
    return new_value

def StuckAtFloatingPoint(original_value, bit, dbit, integer_bits=16, fractional_bits=16):

    binary_repr = struct.unpack('>I', struct.pack('>f', original_value.item()))[0]
    flipped_repr = binary_repr | (1 << bit)
    if(dbit == 0):
        flipped_repr -= (1 << bit)
    new_value = struct.unpack('>f', struct.pack('>I', flipped_repr))[0]
    return new_value
    

def BitFlipFixedPoint(original_value, bit, integer_bits=16, fractional_bits=16):
    fixed_value = float_to_fixed(original_value, integer_bits, fractional_bits)
    bits = format(fixed_value, '032b')  # 32-bit representation
    new_bits = bits[:bit] + ('1' if bits[bit] == '0' else '0') + bits[bit + 1:]
    # Convert back to fixed-point integer
    new_fixed_value = int(new_bits, 2)

    # Convert back to floating-point
    new_value = fixed_to_float(new_fixed_value, integer_bits, fractional_bits)
    return new_value

def StuckAtFixedPoint(original_value, bit, dbit, integer_bits=16, fractional_bits=16):
    fixed_value = float_to_fixed(original_value, integer_bits, fractional_bits)
    bits = format(fixed_value, '032b')  # 32-bit representation

    new_bits = bits[:bit] + str(dbit) + bits[bit + 1:]

    # Convert back to fixed-point integer
    new_fixed_value = int(new_bits, 2)

    # Convert back to floating-point
    new_value = fixed_to_float(new_fixed_value, integer_bits, fractional_bits)
    return new_value

def ChannelFI(matrix, OutC, fault_type="BitFlip", FI_bit=-1, integer_bits=16, fractional_bits=16,fault_model="float"):
    modified_elements = []
    for i in range(matrix.shape[0]):
        for w in range(matrix.shape[1]):
            if(w%16!=OutC):
                continue
            for j in range(matrix.shape[2]):
                for k in range(matrix.shape[3]):
                    depth = w
                    row = j
                    col = k
                    original_value = matrix[i][depth][row][col]

                    if(FI_bit == -1):
                        FI_bit = random.randint(0, 31)
                    if(fault_type == "BitFlip"):
                        if(fault_model == "float"):
                            new_value = BitFlipFloatingPoint(original_value,FI_bit,integer_bits,fractional_bits)
                        else:
                            new_value = BitFlipFloatingPoint(original_value,FI_bit,integer_bits,fractional_bits)
                    elif(fault_type == "StuckAt0"):
                        if(fault_model == "float"):
                            new_value = StuckAtFloatingPoint(original_value,FI_bit,0,integer_bits,fractional_bits)
                        else:
                            new_value = StuckAtFixedPoint(original_value,FI_bit,0,integer_bits,fractional_bits)
                    elif(fault_type == "StuckAt1"):
                        if(fault_model == "float"):
                            new_value = StuckAtFloatingPoint(original_value,FI_bit,1,integer_bits,fractional_bits)
                        else:
                            new_value = StuckAtFixedPoint(original_value,FI_bit,1,integer_bits,fractional_bits)

                    dif = abs(new_value - original_value)
                    modified_elements.append(((i, depth, row, col), original_value, new_value))
                    # if(dif > 0.01):
                    #     # print("Dif=",dif,FI_bit ,new_value, original_value)
                    #     modified_elements.append(((i, depth, row, col), original_value, new_value))
                    matrix[i][depth][row][col] = new_value
    return matrix, modified_elements


def inject_fault_fixed_point_4d(matrix, pattern="Channel", fault_type="BitFlip", FI_bit=-1, integer_bits=16, fractional_bits=16,fault_model="float",pattern_index=-1):

    if(pattern == "Single"):
        depth = random.randint(0, matrix.shape[1]-1)
        row = random.randint(0, matrix.shape[2]-1)
        col = random.randint(0, matrix.shape[3]-1)
        return SingleFI(matrix, row, col, depth, fault_type=fault_type, FI_bit=FI_bit, integer_bits=integer_bits, fractional_bits=fractional_bits,fault_model=fault_model)
    elif(pattern == "Channel"):
        depth = random.randint(0, matrix.shape[1]-1)
        if(pattern_index!= -1):
            depth = pattern_index
        return ChannelFI(matrix, depth, fault_type=fault_type, FI_bit=FI_bit, integer_bits=integer_bits, fractional_bits=fractional_bits,fault_model=fault_model)


def ColumnFI(matrix, OutC, fault_type="BitFlip", FI_bit=-1, integer_bits=16, fractional_bits=16,fault_model="float"):
    modified_elements = []
    if len(matrix.shape) == 2:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if(j%16!=OutC):
                    continue
                
                col = j
                original_value = matrix[i][col]

                if(FI_bit == -1):
                    FI_bit = random.randint(0, 31)
                if(fault_type == "BitFlip"):
                    if(fault_model == "float"):
                        new_value = BitFlipFloatingPoint(original_value,FI_bit,integer_bits,fractional_bits)
                    else:
                        new_value = BitFlipFloatingPoint(original_value,FI_bit,integer_bits,fractional_bits)
                elif(fault_type == "StuckAt0"):
                    if(fault_model == "float"):
                        new_value = StuckAtFloatingPoint(original_value,FI_bit,0,integer_bits,fractional_bits)
                    else:
                        new_value = StuckAtFixedPoint(original_value,FI_bit,0,integer_bits,fractional_bits)
                elif(fault_type == "StuckAt1"):
                    if(fault_model == "float"):
                        new_value = StuckAtFloatingPoint(original_value,FI_bit,1,integer_bits,fractional_bits)
                    else:
                        new_value = StuckAtFixedPoint(original_value,FI_bit,1,integer_bits,fractional_bits)

                dif = abs(new_value - original_value)
                modified_elements.append(((i, 0, col), original_value, new_value))
                # if(dif > 0.01):
                #     # print("Dif=",dif,FI_bit ,new_value, original_value)
                #     modified_elements.append(((i, depth, row, col), original_value, new_value))
                matrix[i][col] = new_value
        return matrix, modified_elements    
    else:
        for i in range(matrix.shape[0]):
            for k in range(matrix.shape[1]):
                for j in range(matrix.shape[2]):
                    if(j%16!=OutC):
                        continue
                    row = k
                    col = j
                    original_value = matrix[i][row][col]

                    if(FI_bit == -1):
                        FI_bit = random.randint(0, 31)
                    if(fault_type == "BitFlip"):
                        if(fault_model == "float"):
                            new_value = BitFlipFloatingPoint(original_value,FI_bit,integer_bits,fractional_bits)
                        else:
                            new_value = BitFlipFloatingPoint(original_value,FI_bit,integer_bits,fractional_bits)
                    elif(fault_type == "StuckAt0"):
                        if(fault_model == "float"):
                            new_value = StuckAtFloatingPoint(original_value,FI_bit,0,integer_bits,fractional_bits)
                        else:
                            new_value = StuckAtFixedPoint(original_value,FI_bit,0,integer_bits,fractional_bits)
                    elif(fault_type == "StuckAt1"):
                        if(fault_model == "float"):
                            new_value = StuckAtFloatingPoint(original_value,FI_bit,1,integer_bits,fractional_bits)
                        else:
                            new_value = StuckAtFixedPoint(original_value,FI_bit,1,integer_bits,fractional_bits)

                    dif = abs(new_value - original_value)
                    modified_elements.append(((i, row, col), original_value, new_value))
                    # if(dif > 0.01):
                    #     # print("Dif=",dif,FI_bit ,new_value, original_value)
                    #     modified_elements.append(((i, depth, row, col), original_value, new_value))
                    matrix[i][row][col] = new_value
        return matrix, modified_elements
        

def inject_fault_fixed_point_2d(matrix, pattern="Column", fault_type="BitFlip", FI_bit=-1, integer_bits=16, fractional_bits=16,fault_model="float",pattern_index=-1):

    if(pattern == "Column"):
        depth = random.randint(0, matrix.shape[1]-1)
        if(pattern_index!= -1):
            depth = pattern_index
        return ColumnFI(matrix, depth, fault_type=fault_type, FI_bit=FI_bit, integer_bits=integer_bits, fractional_bits=fractional_bits,fault_model=fault_model)



def Analysis(model, layerNum,maxvalues, dataloader,device,fault_type="StuckAt0", FI_bit=30,pattern_index=0):
    # print(maxvalues.keys())
    def fault_injection_hook(module, input, output):
            with torch.no_grad():
                output, modified_elements = inject_fault_fixed_point_4d(output, pattern="Channel", fault_type=fault_type, FI_bit=FI_bit,fault_model="float",pattern_index=pattern_index)
                # print("modified elements",len(modified_elements))
    def fault_injection_hook_2d(module, input, output):
            with torch.no_grad():
                output, modified_elements = inject_fault_fixed_point_2d(output, pattern="Column", fault_type=fault_type, FI_bit=FI_bit,fault_model="float",pattern_index=pattern_index)
                # print("modified elements",len(modified_elements))

    cnt = 0
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            # print(maxvalues[str(layer)])
            layer.register_forward_hook(fault_injection_hook)
        if isinstance(layer, nn.Linear):
            # print(maxvalues[str(layer)])
            layer.register_forward_hook(fault_injection_hook_2d)
    return model
    # accuracy = train.test_accuracy(model, dataloader, device)
    # print("Fault injection on the convolutional layers. Acc=",accuracy)
    # return 0
    
    