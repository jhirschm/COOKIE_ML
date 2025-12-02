import os
import sys

# from groqflow import groqit
import torch
import numpy as np
import torch.nn as nn

# import groq API convolution
from groq_convolution import conv1d
from groq_convolution.main import test_1dconv

current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the utils directory relative to the current file's directory
utils_dir = os.path.abspath(os.path.join(current_dir, "../src/", "ml_backbone"))
denoise_dir = os.path.abspath(os.path.join(current_dir, "../src/", "denoising"))

sys.path.append(utils_dir)
sys.path.append(denoise_dir)

from denoising_util import *
from ximg_to_ypdf_autoencoder import Ximg_to_Ypdf_Autoencoder, Zero_PulseClassifier

test_cases = [
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

# Example usage
encoder_layers = [
    [
        nn.Conv1d(
            test_case["in_channel_num"],
            test_case["out_channel_num"],
            kernel_size=test_case["kernel_size"],
            stride=test_case["stride"],
            padding=test_case["padding"],
        ),
        nn.ReLU(),
    ]
    for test_case in test_cases
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

inputs = {"x": torch.rand(1, 1, 512, 16)}


test_case = {
    "kernel_size": 4,
    "image_size": 1 * 320,
    "in_channel_num": 1,
    "out_channel_num": 5,
    "batch_num": 1,
    "stride": 1,
    "padding": 0,
}

test_1dconv(test_case)


# gmodel = groqit(autoencoder, inputs, groqview=True, rebuild="always", build_name="model "+str(1))
#   # Get performance estimates in terms of latency and throughput
# estimate = gmodel.estimate_performance()
# print("Your build's estimated performance is:")
# print(f"{estimate.latency:.7f} {estimate.latency_units}")
# print(f"{estimate.throughput:.1f} {estimate.throughput_units}")
# print("Example estimate_performance.py finished")
# gmodel.groqview()
