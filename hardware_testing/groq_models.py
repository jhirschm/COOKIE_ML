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
    compile_autoencoder_with_ttl,
    compile_zero_classifier_with_ttl,
    compile_overall_model_with_ttl,
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


def extract_autoencoder_weights(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, List[np.ndarray]]:
    """
    Extract encoder.*.weight and decoder.*.weight fields from a state_dict and return them as a dictionary.

    Args:
        state_dict: Dictionary containing model parameters (from model.state_dict())

    Returns:
        Dictionary with two keys:
        - "encoder_weights": List of numpy arrays (float16) containing encoder layer weights, ordered by layer index
        - "decoder_weights": List of numpy arrays (float16) containing decoder layer weights, ordered by layer index
    """
    encoder_weights = []
    decoder_weights = []
    # Filter for keys that start with "encoder." and end with ".weight"
    for key in sorted(state_dict.keys()):
        if key.startswith("encoder.") and key.endswith(".weight"):
            weight = state_dict[key].detach().cpu().numpy().astype(np.float16)
            encoder_weights.append(weight)
        if key.startswith("decoder.") and key.endswith(".weight"):
            weight = state_dict[key].detach().cpu().numpy().astype(np.float16)
            decoder_weights.append(weight)

    return {"encoder_weights": encoder_weights, "decoder_weights": decoder_weights}


def extract_classifier_weights(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, List[np.ndarray]]:
    """
    Extract convolutional layer weights and fully connected layer weights and biases from a state_dict.

    Args:
        state_dict: Dictionary containing model parameters (from model.state_dict())

    Returns:
        Dictionary with three keys:
        - "conv_weights": List of numpy arrays (float16) containing convolutional layer weights, ordered by layer index
        - "fc_weights": List of numpy arrays (float16) containing fully connected layer weights, ordered by layer index
        - "fc_biases": List of numpy arrays (float16) containing fully connected layer biases, ordered by layer index
    """
    conv_weights = []
    fc_weights = []
    fc_biases = []

    # Filter for keys that contain conv-related patterns and end with ".weight"
    # This handles Conv1d, Conv2d, ConvTranspose1d, ConvTranspose2d layers
    for key in sorted(state_dict.keys()):
        key_lower = key.lower()
        if ("conv" in key_lower or "convtranspose" in key_lower) and key.endswith(
            ".weight"
        ):
            weight = state_dict[key].detach().cpu().numpy().astype(np.float16)
            conv_weights.append(weight)

    # Filter for keys that contain fc/linear patterns
    # Extract weights and biases separately
    for key in sorted(state_dict.keys()):
        key_lower = key.lower()
        if ("fc" in key_lower or "linear" in key_lower) and key.endswith(".weight"):
            weight = state_dict[key].detach().cpu().numpy().astype(np.float16)
            fc_weights.append(weight)
        elif ("fc" in key_lower or "linear" in key_lower) and key.endswith(".bias"):
            bias = state_dict[key].detach().cpu().numpy().astype(np.float16)
            fc_biases.append(bias)

    return {
        "conv_weights": conv_weights,
        "fc_weights": fc_weights,
        "fc_biases": fc_biases,
    }


def get_activation_function(activation_function_name):
    if activation_function_name == "ReLU":
        return nn.ReLU()
    elif activation_function_name == "Sigmoid":
        return nn.Sigmoid()
    elif activation_function_name == "Tanh":
        return nn.Tanh()
    else:
        return None


# Load encoder configuration from JSON file
config_path = os.path.join(current_dir, "model_config.json")

with open(config_path, "r") as f:
    config = json.load(f)

input_size = config["input_size"]
batch_num = config["batch_num"]
encoder_config = config["encoder"]
decoder_config = config["decoder"]
zero_mask_classifier_config = config["zero_mask_classifier"]
layer_configurations_encoder = []
layer_configurations_decoder = []
conv_layer_configurations_zero_mask_classifier = []
fc_layer_configurations_zero_mask_classifier = []

# Add batch_num to each layer configuration
for layer_conf in encoder_config["layer_configurations"]:
    layer_conf_with_batch = layer_conf.copy()
    layer_conf_with_batch["batch_num"] = batch_num
    layer_configurations_encoder.append(layer_conf_with_batch)

for layer_conf in decoder_config["layer_configurations"]:
    layer_conf_with_batch = layer_conf.copy()
    layer_conf_with_batch["batch_num"] = batch_num
    layer_configurations_decoder.append(layer_conf_with_batch)

for layer_conf in zero_mask_classifier_config["conv_layer_configurations"]:
    layer_conf_with_batch = layer_conf.copy()
    layer_conf_with_batch["batch_num"] = batch_num
    conv_layer_configurations_zero_mask_classifier.append(layer_conf_with_batch)

for layer_conf in zero_mask_classifier_config["fc_layer_configurations"]:
    layer_conf_with_batch = layer_conf.copy()
    layer_conf_with_batch["batch_num"] = batch_num
    fc_layer_configurations_zero_mask_classifier.append(layer_conf_with_batch)

# Torch Encoding layers
encoder_layers = []
for layer_conf in layer_configurations_encoder:

    activation_function = get_activation_function(
        layer_conf["conv_activation_function"]
    )

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
            (activation_function),  # activation function
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

# Torch Decoding layers
decoder_layers = []
for layer_conf in layer_configurations_decoder:

    activation_function = get_activation_function(
        layer_conf["conv_activation_function"]
    )

    decoder_layers.append(
        [
            nn.ConvTranspose1d(  # convolutional layer
                layer_conf["in_channel_num"],
                layer_conf["out_channel_num"],
                kernel_size=layer_conf["conv_kernel_size"],
                stride=layer_conf["conv_stride"],
                padding=layer_conf["conv_padding"],
                bias=False,
            ),
            (activation_function),  # activation function
        ]
    )

# Torch Zero Mask Classifier layers
zero_mask_classifier_conv_layers = []
for layer_conf in conv_layer_configurations_zero_mask_classifier:

    activation_function = get_activation_function(
        layer_conf["conv_activation_function"]
    )

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
            (activation_function),  # activation function
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

    zero_mask_classifier_conv_layers.extend(sub_layers)


"""
# Calculate the output size after conv layers
def get_conv_output_size(input_size, conv_layers):
    x = torch.randn(input_size)
    model = nn.Sequential(
        *[
            layer
            for layer_pair in conv_layers
            for layer in layer_pair
            if layer is not None
        ]
    )
    x = model(x)
    return x.shape


output_size = get_conv_output_size((1, 16, 512), zero_mask_classifier_conv_layers)
print(f"Output size after conv layers: {output_size}")
"""
zero_mask_classifier_fc_layers = []
for layer_conf in fc_layer_configurations_zero_mask_classifier:

    activation_function = get_activation_function(layer_conf["activation_function"])

    zero_mask_classifier_fc_layers.append(
        [
            nn.Linear(layer_conf["input_size"], layer_conf["output_size"]),
            (activation_function),
        ]
    )

print("zero_mask_classifier_fc_layers: ", zero_mask_classifier_fc_layers)
# zero_mask_classifier_fc_layers = []


autoencoder = Ximg_to_Ypdf_Autoencoder(
    encoder_layers,
    decoder_layers=decoder_layers,
    outputEncoder=False,
    dtype=torch.float16,
)

zero_mask_classifier = Zero_PulseClassifier(
    zero_mask_classifier_conv_layers,
    zero_mask_classifier_fc_layers,
    dtype=torch.float16,
)

print("zero_mask_classifier: ", zero_mask_classifier.state_dict().keys())


# Extract kernel matrices for Groq implementation
# Get all parameters as a dictionary
state_dict = autoencoder.state_dict()
kernels = extract_autoencoder_weights(state_dict)

state_dict = zero_mask_classifier.state_dict()
classifier_weights = extract_classifier_weights(state_dict)


# compile the program for Groq hardware implementation


image = np.random.randn(
    layer_configurations_encoder[0]["batch_num"],
    layer_configurations_encoder[0]["in_channel_num"],
    input_size,
).astype(np.float32)

image_fp16 = image.copy().astype(np.float16)
program_name = "encoder"

if compiler_type == CompilerType.gAPI:

    output_tensor_name = "encoder_result"
    input_tensor_name = "image"

    compiled_program = compile_encoder_with_gapi(
        layer_configurations_encoder,
        kernels["encoder_weights"],
        image_fp16,
        output_tensor_name,
        program_name,
    )

    inputs = {input_tensor_name: image_fp16}

elif compiler_type == CompilerType.Compiler:

    output_tensor_name = "output000"
    input_tensor_name = "arg000"
    image_torch = torch.from_numpy(image_fp16)

    """
    compiled_program = compile_encoder_with_compiler(
        autoencoder, image_torch, program_name
    )
    """

    compiled_program = compile_encoder_with_compiler(
        zero_mask_classifier, image_torch, program_name
    )

    inputs = {input_tensor_name: image_fp16}

elif compiler_type == CompilerType.ttl:

    output_tensor_name = "encoder_result"
    input_tensor_name = "image"
    """
    compiled_program = compile_autoencoder_with_ttl(
        layer_configurations_encoder,
        layer_configurations_decoder,
        kernels,
        input_size,
        output_tensor_name,
        program_name,
    )
    """

    """
    output_tensor_name = "classifier_result"
    input_tensor_name = "image"

    compiled_program = compile_zero_classifier_with_ttl(
        conv_layer_configurations_zero_mask_classifier,
        fc_layer_configurations_zero_mask_classifier,
        classifier_weights,
        input_size,
        output_tensor_name,
        program_name,
    )
    """

    output_tensor_name = "overall_model_result"
    input_tensor_name = "image"

    compiled_program = compile_overall_model_with_ttl(
        layer_configurations_encoder,
        layer_configurations_decoder,
        kernels,
        conv_layer_configurations_zero_mask_classifier,
        fc_layer_configurations_zero_mask_classifier,
        classifier_weights,
        input_size,
    )
    inputs = {input_tensor_name: image_fp16}

    program_name = "unnamed"

else:
    raise ValueError(f"Invalid compiler type: {compiler_type}")


# run the program on TSP
runner = GroqRunner(timing_report=True)
runner.upload_iop_file(compiled_program["iop_file"], program_name=program_name)

"""
# measure the performance of the hardware implementation
iteration_num = 500
elapsed_time = 0
for _ in range(iteration_num):

    test_image = np.random.randn(
        layer_configurations_encoder[0]["batch_num"],
        layer_configurations_encoder[0]["in_channel_num"],
        input_size,
    ).astype(np.float16)

    start_time = time.perf_counter()
    results_groq = runner.invoke({input_tensor_name: test_image})
    end_time = time.perf_counter()
    elapsed_time += end_time - start_time

elapsed_time = elapsed_time / iteration_num


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
"""

results_groq = runner.invoke(inputs)


output_tensor = results_groq[output_tensor_name]
print("output_tensor.shape: ", output_tensor.shape)


with torch.no_grad():
    image_torch = torch.from_numpy(image_fp16)
    print("image_torch.shape: ", image_torch.dtype)

    result_torch = autoencoder(image_torch)
    result_torch = result_torch.detach().numpy()
    """

    result_torch = zero_mask_classifier(image_torch)
    result_torch = result_torch.detach().numpy()
    """

print("output_tensor: ", output_tensor[:, :, 0:3])

if np.allclose(output_tensor, result_torch, atol=0.02, rtol=0.1):
    print(
        f"Groq result matches torch result in test case {layer_configurations_encoder}."
    )

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
        f"Groq result differs from torch result in test case {layer_configurations_encoder}"
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
