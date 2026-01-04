import os
import sys
import time
import json

# from groqflow import groqit
import torch
import numpy as np
import torch.nn as nn
from typing import Dict, List
from enum import Enum


# import tsp runner
from gstruct.runner import GroqRunner

from compile_lpu_model import (
    compile_encoder_with_gapi,
    compile_encoder_with_compiler,
    compile_encoder_with_ttl,
    CompilerType,
)
import groq.api as g


compiler_type = CompilerType.ttl


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


# Load encoder configuration from JSON file
config_path = os.path.join(current_dir, "encoder_config.json")

with open(config_path, "r") as f:
    config = json.load(f)

input_size = config["input_size"]
batch_num = config["batch_num"]
encoder_config = config["encoder"]
layer_configurations = []

# Add batch_num to each layer configuration
for layer_conf in encoder_config["layer_configurations"]:
    layer_conf_with_batch = layer_conf.copy()
    layer_conf_with_batch["batch_num"] = batch_num
    layer_configurations.append(layer_conf_with_batch)


# Torch Encoding layers
encoder_layers = []
for layer_conf in layer_configurations:

    sub_layers = []
    sub_layers.append(
        [
            nn.Conv1d(  # convolutional layer
                layer_conf["in_channel_num"],
                layer_conf["out_channel_num"],
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
    encoder_layers,
    decoder_layers=decoder_layers,
    outputEncoder=True,
    dtype=torch.float16,
)


# Extract kernel matrices for Groq implementation
# Get all parameters as a dictionary
state_dict = autoencoder.state_dict()
kernels = extract_encoder_weights(state_dict)


# compile the program for Groq hardware implementation


image = np.random.randn(
    layer_configurations[0]["batch_num"],
    layer_configurations[0]["in_channel_num"],
    input_size,
).astype(np.float32)

image_fp16 = image.copy().astype(np.float16)
program_name = "encoder"

if compiler_type == CompilerType.gAPI:

    output_tensor_name = "encoder_result"
    input_tensor_name = "image"

    compiled_program = compile_encoder_with_gapi(
        layer_configurations, kernels, image_fp16, output_tensor_name, program_name
    )

    inputs = {input_tensor_name: image_fp16}

elif compiler_type == CompilerType.Compiler:

    output_tensor_name = "output000"
    input_tensor_name = "arg000"

    image_torch = torch.from_numpy(image_fp16)
    compiled_program = compile_encoder_with_compiler(
        autoencoder, image_torch, program_name
    )
    inputs = {input_tensor_name: image_fp16}

elif compiler_type == CompilerType.ttl:

    output_tensor_name = "encoder_result"
    input_tensor_name = "image"

    compiled_program = compile_encoder_with_ttl(
        layer_configurations, kernels, input_size, output_tensor_name, program_name
    )
    inputs = {input_tensor_name: image_fp16}

    program_name = "unnamed"

else:
    raise ValueError(f"Invalid compiler type: {compiler_type}")


# run the program on TSP
runner = GroqRunner(timing_report=True)
runner.upload_iop_file(compiled_program["iop_file"], program_name=program_name)


# measure the performance of the hardware implementation
iteration_num = 500
elapsed_time = 0
for _ in range(iteration_num):

    test_image = np.random.randn(
        layer_configurations[0]["batch_num"],
        layer_configurations[0]["in_channel_num"],
        input_size,
    ).astype(np.float16)

    start_time = time.perf_counter()
    results_groq = runner.invoke({input_tensor_name: test_image})
    end_time = time.perf_counter()
    elapsed_time += end_time - start_time

elapsed_time = elapsed_time / iteration_num

results_groq = runner.invoke(inputs)

timings = runner.get_timings()
print("\n=== Timing Results ===")
for key, value in timings.items():
    if key in ["upload_iop_file", "create_buffers"]:
        print(f"{key}: {value} microseconds")
    else:
        print(f"{key}: {value/iteration_num} microseconds")
print("=" * 25)

print(
    f"Groq runner total execution time: {elapsed_time:.6f} seconds ({elapsed_time * 1000000:.3f} microseconds)"
)

output_tensor = results_groq[output_tensor_name]
print("output_tensor.shape: ", output_tensor.shape)

with torch.no_grad():
    image_torch = torch.from_numpy(image_fp16)
    print("image_torch.shape: ", image_torch.dtype)
    result_torch = autoencoder(image_torch)
    result_torch = result_torch.detach().numpy()


if np.allclose(output_tensor, result_torch, atol=0.02, rtol=0.1):
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
    print(output_tensor[0, :, -16:])
    print("Torch output:")
    print(result_torch[0, :, -16:])

    # Find all differing elements using the same tolerance as allclose
    atol = 0.02
    rtol = 0.5
    diff = np.abs(output_tensor - result_torch)
    relative_diff = np.abs(diff / (np.abs(result_torch) + 1e-8))

    # Elements that differ beyond tolerance
    differing_mask = (diff > atol) & (relative_diff > rtol)
    differing_indices = np.where(differing_mask)

    max_error = np.max(diff)
    max_error_index = np.unravel_index(np.argmax(diff), diff.shape)

    print(f"max_error: {max_error}, max_error_index: {max_error_index}")
    print(f"Total differing elements: {np.sum(differing_mask)}")
    print("\nAll differing elements:")
    print("-" * 80)

    # Limit output to first 100 differing elements to avoid overwhelming output
    num_differing = len(differing_indices[0])
    max_to_show = min(100, num_differing)

    for i in range(max_to_show):
        idx = tuple(dim[i] for dim in differing_indices)
        groq_val = output_tensor[idx]
        torch_val = result_torch[idx]
        abs_diff = diff[idx]
        rel_diff = relative_diff[idx]
        print(
            f"Index {idx}: groq={groq_val:.6f}, torch={torch_val:.6f}, "
            f"abs_diff={abs_diff:.6f}, rel_diff={rel_diff:.6f}"
        )

    if num_differing > max_to_show:
        print(f"\n... and {num_differing - max_to_show} more differing elements")

    print("-" * 80)

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
