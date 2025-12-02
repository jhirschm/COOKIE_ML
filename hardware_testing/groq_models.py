import os
import sys

# from groqflow import groqit
import torch
import numpy as np
import torch.nn as nn
from typing import Dict, List

# import groq API convolution
from groq_convolution import conv1d
from groq_convolution.main import test_1dconv

from groq_convolution.compile_lpu_convolution import compile_g_api_1dconv

current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the utils directory relative to the current file's directory
utils_dir = os.path.abspath(os.path.join(current_dir, "../src/", "ml_backbone"))
denoise_dir = os.path.abspath(os.path.join(current_dir, "../src/", "denoising"))

sys.path.append(utils_dir)
sys.path.append(denoise_dir)

from denoising_util import *
from ximg_to_ypdf_autoencoder import Ximg_to_Ypdf_Autoencoder, Zero_PulseClassifier


def extract_encoder_weights(state_dict: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
    """
    Extract encoder.*.weight fields from a state_dict and return them as a list.

    Args:
        state_dict: Dictionary containing model parameters (from model.state_dict())

    Returns:
        List of weight tensors from encoder layers, ordered by layer index
    """
    encoder_weights = []
    # Filter for keys that start with "encoder." and end with ".weight"
    for key in sorted(state_dict.keys()):
        if key.startswith("encoder.") and key.endswith(".weight"):
            encoder_weights.append(state_dict[key])
    return encoder_weights


layer_configurations = [
    {
        "kernel_size": 4,
        "image_size": 1 * 320,
        "in_channel_num": 1,
        "out_channel_num": 5,
        "batch_num": 1,
        "stride": 1,
        "padding": 0,
    }
]

# Encoding layers
encoder_layers = [
    [
        nn.Conv1d(  # convolutional layer
            layer_conf["in_channel_num"],
            layer_conf["out_channel_num"],
            kernel_size=layer_conf["kernel_size"],
            stride=layer_conf["stride"],
            padding=layer_conf["padding"],
            bias=False,
        ),
        nn.Identity(),  # activation function
    ]
    for layer_conf in layer_configurations
]

# encoder_layers = [
#     [nn.Conv2d(1, 16, kernel_size=3, padding=2), nn.ReLU()],
#     [nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU()],
#     [nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU()],
# ]


decoder_layers = None
# decoder_layers = np.array([
#     [nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1), nn.ReLU()],
#     [nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1), nn.ReLU()],
#     [nn.ConvTranspose2d(16, 1, kernel_size=3, padding=2), nn.Sigmoid()]  # Example with Sigmoid activation
#     # [nn.ConvTranspose2d(16, 1, kernel_size=3, padding=2), None],  # Example without activation
# ])


autoencoder = Ximg_to_Ypdf_Autoencoder(
    encoder_layers, decoder_layers=decoder_layers, outputEncoder=True
)


# Extract kernel matrices for Groq implementation
# Get all parameters as a dictionary
state_dict = autoencoder.state_dict()
kernels = extract_encoder_weights(state_dict)

for key, value in state_dict.items():
    print(key, value.shape)
    print(value)

# compile the program for Groq hardware implementation
kernel = (
    kernels[0].detach().cpu().numpy().astype(np.float16)
)  # Convert to numpy float16
layer_configuration = layer_configurations[0]

compiled_program = compile_g_api_1dconv(layer_configuration, kernel)


inputs = {"x": torch.rand(1, 1, 512, 16)}


# test_1dconv(layer_configurations[0])


# gmodel = groqit(autoencoder, inputs, groqview=True, rebuild="always", build_name="model "+str(1))
#   # Get performance estimates in terms of latency and throughput
# estimate = gmodel.estimate_performance()
# print("Your build's estimated performance is:")
# print(f"{estimate.latency:.7f} {estimate.latency_units}")
# print(f"{estimate.throughput:.1f} {estimate.throughput_units}")
# print("Example estimate_performance.py finished")
# gmodel.groqview()
