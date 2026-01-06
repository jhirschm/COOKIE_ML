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
    compile_lstm_pulsenum_classifier_with_ttl,
    CompilerType,
)
import groq.api as g


class TargetModel(Enum):

    autoencoder = "autoencoder"
    zero_pulse_classifier = "zero_pulse_classifier"
    lstm_pulsenum_classifier = "lstm_pulsenum_classifier"


compiler_type = CompilerType.ttl
target_model = TargetModel.autoencoder


current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the utils directory relative to the current file's directory
utils_dir = os.path.abspath(os.path.join(current_dir, "../src/", "ml_backbone"))
denoise_dir = os.path.abspath(os.path.join(current_dir, "../src/", "denoising"))

sys.path.append(utils_dir)
sys.path.append(denoise_dir)

from denoising_util import *
from ximg_to_ypdf_autoencoder import Ximg_to_Ypdf_Autoencoder, Zero_PulseClassifier
from extract_weights import extract_autoencoder_weights, extract_classifier_weights
from gen_zero_classifier_torch_layers import gen_zero_classifier_torch_layers
from gen_autoencoder_torch_layers import gen_autoencoder_torch_layers


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

# Torch Encoder and Decoder layers
encoder_layers, decoder_layers = gen_autoencoder_torch_layers(
    layer_configurations_encoder,
    layer_configurations_decoder,
)

# Torch Zero Mask Classifier layers
zero_mask_classifier_conv_layers, zero_mask_classifier_fc_layers = (
    gen_zero_classifier_torch_layers(
        conv_layer_configurations_zero_mask_classifier,
        fc_layer_configurations_zero_mask_classifier,
    )
)


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

    if target_model == TargetModel.autoencoder:
        compiled_program = compile_encoder_with_compiler(
            autoencoder, image_torch, program_name
        )

    elif target_model == TargetModel.zero_pulse_classifier:

        compiled_program = compile_encoder_with_compiler(
            zero_mask_classifier, image_torch, program_name
        )

    elif target_model == TargetModel.lstm_pulsenum_classifier:
        raise ValueError(
            f"LSTM pulse number classifier not supported for compiler type {compiler_type}"
        )
    else:
        raise ValueError(f"Invalid target model: {target_model}")

    inputs = {input_tensor_name: image_fp16}

elif compiler_type == CompilerType.ttl:

    if target_model == TargetModel.autoencoder:
        output_tensor_name = "encoder_result"
        input_tensor_name = "image"

        compiled_program = compile_autoencoder_with_ttl(
            layer_configurations_encoder,
            layer_configurations_decoder,
            kernels,
            input_size,
            output_tensor_name,
            program_name,
        )

    elif target_model == TargetModel.zero_pulse_classifier:
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

    elif target_model == TargetModel.lstm_pulsenum_classifier:
        output_tensor_name = "lstm_pulsenum_classifier"
        input_tensor_name = "image"

        compiled_program = compile_lstm_pulsenum_classifier_with_ttl(
            layer_configurations_encoder,
            layer_configurations_decoder,
            kernels,
            conv_layer_configurations_zero_mask_classifier,
            fc_layer_configurations_zero_mask_classifier,
            classifier_weights,
            input_size,
        )

    else:
        raise ValueError(f"Invalid target model: {target_model}")

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
