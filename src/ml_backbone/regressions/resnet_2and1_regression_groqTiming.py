from resnets import *
import torch
import os
import sys
import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F
groq_path = "/home/jhirschm/groqflow"
sys.path.append(groq_path) # Assumes on rdsrv420 with groqflow installed correctly
from groqflow import groqit
def remove_module_prefix(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        return new_state_dict

def decode_2hot_to_phases(output_vector, n_classes, phase_range=(0, 2 * np.pi)):
    """
    Decodes the 2-hot encoded output vector into the original phase values.

    Args:
    output_vector (torch.Tensor): The prediction vector (shape: [batch_size, n_classes]).
    n_classes (int): Number of classes (must match the encoder's n_classes).
    phase_range (tuple): Tuple indicating the range of phase values (min_phase, max_phase).

    Returns:
    phases1 (torch.Tensor): Decoded first phase values (shape: [batch_size]).
    phases2 (torch.Tensor): Decoded second phase values (shape: [batch_size]).
    """
    min_phase, max_phase = phase_range

    # Apply sigmoid to the output vector
    # print(output_vector)
    probabilities = 1/(1+np.exp(-1*(output_vector)))
    # print(probabilities)

    
    # Split the vector into two halves
    half_n_classes = n_classes // 2
    first_half = probabilities[:, :half_n_classes]
    second_half = probabilities[:, half_n_classes:]

    # Get the max index from each half (corresponding to the predicted phases)
    idx1 = np.argmax(first_half, axis=1)
    idx2 = np.argmax(second_half, axis=1)

    # Convert indices back to phase values
    phases1 = (idx1.astype(float) / half_n_classes) * (max_phase - min_phase) + min_phase
    phases2 = (idx2.astype(float) / half_n_classes) * (max_phase - min_phase) + min_phase

    output_phases = np.ndarray((phases1.shape[0], 2))
    output_phases[:, 0] = phases1
    output_phases[:, 1] = phases2
    return output_phases

def earth_mover_distance(y_pred, y_true):
    """
    Calculates the Earth Mover's Distance (EMD) between the cumulative sums
    of the predicted and true distributions.

    Args:
    y_pred (torch.Tensor): The predicted logits or probabilities.
    y_true (torch.Tensor): The true distributions (typically one-hot encoded).

    Returns:
    torch.Tensor: The computed EMD loss.
    """
    # Ensure both inputs are probabilities
    y_pred_prob = torch.sigmoid(y_pred)
    
    # Compute the cumulative sums
    cumsum_y_true = torch.cumsum(y_true, dim=-1)
    cumsum_y_pred = torch.cumsum(y_pred_prob, dim=-1)
    
    # Calculate the squared difference between cumulative sums
    emd_loss = torch.mean(torch.square(cumsum_y_true - cumsum_y_pred), dim=-1)
    
    return torch.mean(emd_loss)

def phase_to_2hot(phases1, phases2, n_classes, phase_range=(0, 2 * torch.pi)):
    """
    Converts batches of phase values into a batch of 2-hot encoded vectors.

    Args:
    phases1 (torch.Tensor): Batch of first phase values (shape: [batch_size]).
    phases2 (torch.Tensor): Batch of second phase values (shape: [batch_size]).
    n_classes (int): Number of classes for the 2-hot encoding.
    phase_range (tuple): Tuple indicating the range of phase values (min_phase, max_phase).

    Returns:
    torch.Tensor: Batch of 2-hot encoded vectors (shape: [batch_size, n_classes]).
    """
    min_phase, max_phase = phase_range

    # Ensure the phases are within the specified range
    assert torch.all((min_phase <= phases1) & (phases1 <= max_phase)), f"Some values in phases1 are out of the specified range ({min_phase}, {max_phase}). Values: {phases1}"
    assert torch.all((min_phase <= phases2) & (phases2 <= max_phase)), f"Some values in phases2 are out of the specified range ({min_phase}, {max_phase}). Values: {phases2}"

    # Normalize the phases to a range from 0 to n_classes
    phases1_norm = (phases1 - min_phase) / (max_phase - min_phase)
    phases2_norm = (phases2 - min_phase) / (max_phase - min_phase)

    # Convert normalized phases to class indices for each half of the vector
    half_n_classes = n_classes // 2
    idx1 = (phases1_norm * half_n_classes).long() % half_n_classes
    idx2 = (phases2_norm * half_n_classes).long() % half_n_classes + half_n_classes

    # Create 2-hot encoded vectors
    batch_size = phases1.size(0)
    one_hot_vectors = torch.zeros(batch_size, n_classes, device=phases1.device)
    one_hot_vectors[torch.arange(batch_size), idx1] = 1
    one_hot_vectors[torch.arange(batch_size), idx2] = 1

    return one_hot_vectors

def phases_to_1hot(phase, num_classes, phase_range=(0, 2 * torch.pi)):
    '''
    Converts a batch of phase values to a batch of one-hot encoded vectors.
    '''
    min_phase, max_phase = phase_range
    phases_norm = (phase - min_phase) / (max_phase - min_phase)
    idx = (phases_norm * num_classes).long() % num_classes
    one_hot_vectors = torch.zeros(phase.size(0), num_classes, device=phase.device)
    one_hot_vectors[torch.arange(phase.size(0)), idx] = 1
    return one_hot_vectors

def phases_to_1hot_wrapping(phase, num_classes, phase_range=(0, torch.pi)):
    '''
    Converts a batch of phase values to a batch of one-hot encoded vectors.
    '''
    min_phase, max_phase = phase_range
    # phases = torch.fmod(torch.abs(phase), torch.pi)
    phases = torch.arccos(torch.cos(phase)) #not including sign now
    phases_norm = (phases - min_phase) / (max_phase - min_phase)

    idx = (phases_norm * num_classes).long() % num_classes
    one_hot_vectors = torch.zeros(phase.size(0), num_classes, device=phase.device)
    one_hot_vectors[torch.arange(phase.size(0)), idx] = 1
    return one_hot_vectors

def onehot_to_phase(one_hot_vector, num_classes, phase_range=(0, 2 * torch.pi)):
    '''
    Converts a batch of one-hot encoded vectors to a batch of phase values.
    '''
    min_phase, max_phase = phase_range
    idx = torch.argmax(one_hot_vector, dim=1)
    phases_norm = idx.float() / num_classes
    phases = phases_norm * (max_phase - min_phase) + min_phase
    return phases

def get_phase(outputs, num_classes, max_val=2*torch.pi):
    # Convert the model outputs to probabilities using softmax
    probabilities = F.softmax(outputs, dim=1)
    
    # Calculate a weighted sum of the class indices
    indices = torch.arange(num_classes, device=outputs.device, dtype=outputs.dtype)
    phase_values = torch.sum(probabilities * indices, dim=1)

    # Map the weighted sum to a phase value between 0 and 2*pi
    phase_values = phase_values * (max_val / num_classes)
    
    # Add an extra dimension to match expected output shape
    phase_values = torch.unsqueeze(phase_values, 1)
    phase_values = phase_values.to(torch.float32)
    return phase_values


def main():
    torch.manual_seed(0)



   


    
    num_classes = 4000#1024
    model_2pulse = resnet34(num_classes=num_classes)

    num_classes = 2000
    # model = resnet152(num_classes=num_classes)
    model_1pulse = resnet18(num_classes=num_classes)



    batch_size = 1
    input_channels = 1
    height, width = 16, 512
    inputs = {"x": torch.randn(batch_size, input_channels, height, width)}

    gmodel1 = groqit(model_2pulse, inputs, groqview=True)

    estimate = gmodel1.estimate_performance()
    print("Your 2 Pulse Regression build's estimated performance is:")
    print(f"{estimate.latency:.7f} {estimate.latency_units}")
    print(f"{estimate.throughput:.1f} {estimate.throughput_units}")

    gmodel2 = groqit(model_1pulse, inputs, groqview=True)

    estimate = gmodel2.estimate_performance()
    print("Your 1 Pulse Regression build's estimated performance is:")
    print(f"{estimate.latency:.7f} {estimate.latency_units}")
    print(f"{estimate.throughput:.1f} {estimate.throughput_units}")


    print("Example estimate_performance.py finished")
    


   
    print(summary(model, input_size=(1, 1, 512, 16)))
if __name__ == "__main__":
    main()