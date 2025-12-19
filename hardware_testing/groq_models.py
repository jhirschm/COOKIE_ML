import os
import sys

# from groqflow import groqit
import torch
import numpy as np
import torch.nn as nn
from typing import Dict, List
from enum import Enum


# import tsp runner
from groq_convolution.main import get_tsp_runner

from compile_lpu_model import compile_encoder_with_g_api, compile_encoder_with_compiler
import groq.api as g


class CompilerType(Enum):
    gAPI = "gAPI"
    Compiler = "Compiler"


compiler_type = CompilerType.gAPI


current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the utils directory relative to the current file's directory
utils_dir = os.path.abspath(os.path.join(current_dir, "../src/", "ml_backbone"))
denoise_dir = os.path.abspath(os.path.join(current_dir, "../src/", "denoising"))

sys.path.append(utils_dir)
sys.path.append(denoise_dir)

from denoising_util import *
from ximg_to_ypdf_autoencoder import Ximg_to_Ypdf_Autoencoder, Zero_PulseClassifier


def extract_encoder_weights(state_dict: Dict[str, torch.Tensor]) -> List[np.ndarray]:
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
            weight = state_dict[key].detach().cpu().numpy().astype(np.float16)
            encoder_weights.append(weight)

    return encoder_weights


layer1_configuration = {
    "batch_num": 1,
    "image_size": 512,
    "conv_kernel_size": 3,
    "conv_in_channel_num": 1,
    "conv_out_channel_num": 16,
    "conv_stride": 1,
    "conv_padding": 1,
    "conv_activation_function": "ReLU",
    "pooling_in_channel_num": 16,
    "pooling_kernel_size": 2,
    "pooling_stride": 2,
    "pooling_padding": 0,
}

layer2_configuration = {
    "batch_num": 1,
    "image_size": layer1_configuration["image_size"],
    "conv_kernel_size": 3,
    "conv_in_channel_num": layer1_configuration["pooling_in_channel_num"],
    "conv_out_channel_num": 10,
    "conv_stride": 1,
    "conv_padding": 1,
    "conv_activation_function": "ReLU",
    "pooling_in_channel_num": 10,
    "pooling_kernel_size": 2,
    "pooling_stride": 2,
    "pooling_padding": 0,
}
layer_configurations = [layer1_configuration, layer2_configuration]


# Torch Encoding layers
encoder_layers = []
for layer_conf in layer_configurations:

    sub_layers = []
    sub_layers.append(
        [
            nn.Conv1d(  # convolutional layer
                layer_conf["conv_in_channel_num"],
                layer_conf["conv_out_channel_num"],
                kernel_size=layer_conf["conv_kernel_size"],
                stride=layer_conf["conv_stride"],
                padding=layer_conf["conv_padding"],
                bias=False,
            ),
            (
                nn.ReLU() if layer_conf["conv_activation_function"] == "ReLU" else None
            ),  # activation function
        ]
    )

    sub_layers.append(
        [
            nn.MaxPool1d(  # pooling layer
                kernel_size=layer_conf["pooling_kernel_size"],
                stride=layer_conf["pooling_stride"],
                padding=layer_conf["pooling_padding"],
            ),
            None,
        ]
    )

    encoder_layers.extend(sub_layers)


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


# compile the program for Groq hardware implementation


image = np.random.randn(
    layer_configurations[0]["batch_num"],
    layer_configurations[0]["conv_in_channel_num"],
    layer_configurations[0]["image_size"],
).astype(np.float32)

image_fp16 = image.astype(np.float16)

if compiler_type == CompilerType.gAPI:

    output_tensor_name = "encoder_result"

    compiled_program = compile_encoder_with_g_api(
        layer_configurations, kernels, image_fp16, output_tensor_name
    )

    inputs = {"image": image_fp16}

elif compiler_type == CompilerType.Compiler:

    output_tensor_name = "output000"

    image_torch = torch.from_numpy(image)
    compiled_program = compile_encoder_with_compiler(autoencoder, image_torch)
    inputs = {"arg000": image}

    print(inputs["arg000"].shape)
else:
    raise ValueError(f"Invalid compiler type: {compiler_type}")


# run the program on TSP
runner = get_tsp_runner(compiled_program["iop_file"])
results_groq = runner(**inputs)
output_tensor = results_groq[output_tensor_name]

with torch.no_grad():
    image_torch = torch.from_numpy(image)
    result_torch = autoencoder(image_torch)
    result_torch = result_torch.detach().numpy()

if np.allclose(output_tensor, result_torch, atol=0.01):
    print(f"Groq result matches torch result in test case {layer_configurations}.")

    """Returns allclose result along with statistics."""
    diff = np.abs(output_tensor - result_torch)
    relative_diff = np.abs(diff / (np.abs(result_torch) + 1e-8))

    stats = {
        "mean_abs_error": np.mean(diff),
        "max_abs_error": np.max(diff),
        "mean_rel_error": np.mean(relative_diff),
        "max_rel_error": np.max(relative_diff),
        "rmse": np.sqrt(np.mean(diff**2)),
    }

    print(stats)
else:
    print("Groq ouput: ")
    print(output_tensor[:, -16:-1])
    print("Torch output:")
    print(result_torch[0, :, -16:-1])

    max_error = np.max(np.abs(output_tensor - result_torch))
    print(max_error)

    raise RuntimeError(
        f"Groq result differs from torch result in test case {layer_configurations}"
    )

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
