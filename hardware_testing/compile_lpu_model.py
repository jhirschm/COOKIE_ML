"""
Module for compiling convolution programs for Groq LPU (Language Processing Unit).

This module provides functions to compile encoder models using different compilation
methods (gAPI, gMLIR, or Compiler) for execution on Groq hardware accelerators.
"""

import numpy as np
import groq.api as g
from enum import Enum

from typing import Any, Union, List, Dict

from groq_convolution.compile_lpu_convolution import (
    compile_g_api,
    compile_with_compiler,
    get_iop_stats,
)
from groq_convolution.conv1d import GroqConv1D, ResourceScopeName
from groq_convolution.groq_pooling import GroqMaxPooling1D
from groq_convolution.constants import VECTOR_SIZE

import torch


class CompilerType(Enum):
    """Enumeration of available compiler types for Groq LPU compilation.

    Attributes:
        gAPI: Use the Groq API (gAPI) compiler for compilation.
        gMLIR: Use the gMLIR compiler for compilation.
        Compiler: Use the standard Groq compiler for compilation.
    """

    gAPI = "gAPI"
    gMLIR = "gMLIR"
    Compiler = "Compiler"


def compile_encoder_with_compiler(
    model: torch.nn.Module,
    image: torch.Tensor,
    program_name: str = "encoder",
) -> Union[dict[str, Union[str, Any]], Any]:
    """Compile an encoder model using the Groq Compiler.

    This function compiles a PyTorch encoder model for execution on Groq hardware
    using the standard Groq compiler. The compiled program is saved to the
    "encoderCompiler" output directory.

    Args:
        model: PyTorch neural network module representing the encoder to compile.
        image: Example input tensor used for shape inference during compilation.
            Should match the expected input shape of the model.
        program_name: Name of the compiled program. Defaults to "encoder".

    Returns:
        Dictionary containing compilation results with keys:
            - "iop_file": Path to the compiled IOP file
            - "output_dir": Directory where compiled files are saved
            - "program_name": Name of the compiled program
            Additional keys may be present depending on the compiler output.

    Raises:
        Exception: If compilation fails, the underlying exception is raised
            with error details.
    """
    # Set file names used below
    output_dir = "encoderCompiler"

    return compile_with_compiler(
        model, image, program_name, output_dir, gen_vis_data=True
    )


def compile_encoder_with_g_api(
    layer_configurations: List[Dict[str, int]],
    kernels: List[np.ndarray],
    input: np.ndarray,
    output_tensor_name: str = "encoder_result",
    program_name: str = "encoder",
) -> Union[dict[str, Union[str, Any]], Any]:
    """Compile an encoder model using the Groq API (gAPI) compiler.

    This function compiles a multi-layer encoder consisting of convolutional and
    pooling layers for execution on Groq hardware using the gAPI compiler.
    Each layer configuration should define both convolution and pooling parameters.

    Args:
        layer_configurations: List of dictionaries, each containing configuration
            for one encoder layer. Each dictionary should include:
            - "batch_num": Batch size
            - "conv_in_channel_num": Input channels for convolution
            - "conv_out_channel_num": Output channels for convolution
            - "conv_kernel_size": Convolution kernel size
            - "conv_stride": Convolution stride
            - "conv_padding": Convolution padding
            - "conv_activation_function": Activation function name (e.g., "ReLU")
            - "pooling_in_channel_num": Input channels for pooling
            - "pooling_kernel_size": Pooling kernel size
            - "pooling_stride": Pooling stride
            - "pooling_padding": Pooling padding
        kernels: List of numpy arrays representing convolution kernels/weights
            for each layer. Should match the order of layer_configurations.
        input: Example input numpy array used for shape inference. Should be
            float16 dtype and match the expected input shape (batch, channels, length).
        output_tensor_name: Name of the output tensor in the compiled program.
            Defaults to "encoder_result".
        program_name: Name of the compiled program. Defaults to "encoder".

    Returns:
        Dictionary containing compilation results with keys:
            - "iop_file": Path to the compiled IOP file
            - "output_dir": Directory where compiled files are saved ("./encoderGAPI")
            - "program_name": Name of the compiled program
            Additional keys may be present depending on the compiler output.

    Raises:
        Exception: If compilation fails, the underlying exception is raised
            with error details and a full traceback is printed.

    Note:
        The function automatically creates GroqConv1D and GroqMaxPooling1D layers
        for each configuration and compiles them with overlapped scopes for
        optimized execution on Groq hardware.
    """

    with g.ProgramContext(program_id=program_name) as pc:

        input_mt = g.input_tensor(
            shape=input.shape,
            dtype=g.float16,
            name="image",
            layout="H1(W), -1, S2",
            split_sizes=VECTOR_SIZE,
        )

        tsp_layers = []
        for layer_configuration, kernel in zip(layer_configurations, kernels):
            tsp_layer = GroqConv1D(
                conv_kernel=kernel,
                batch_num=layer_configuration["batch_num"],
                padding=layer_configuration["conv_padding"],
                activation_function=layer_configuration.get(
                    "conv_activation_function", "none"
                ),
                overlapped_scopes=True,
                return_at_scope=ResourceScopeName.UNPACK_CONV_RES,
            )
            tsp_layers.append(tsp_layer)

            tsp_layer = GroqMaxPooling1D(
                in_channel_num=layer_configuration["pooling_in_channel_num"],
                kernel_size=layer_configuration["pooling_kernel_size"],
                stride=layer_configuration["pooling_stride"],
                batch_num=layer_configuration["batch_num"],
                padding=layer_configuration["pooling_padding"],
                overlapped_scopes=True,
            )
            tsp_layers.append(tsp_layer)

        try:
            compiled_program = compile_g_api(
                tsp_layers,
                input_mt,
                output_dir="./encoderGAPI",
                program_name="encoder",
                output_tensor_name="encoder_result",
            )

            # Get iop stats
            iop_stats_output = get_iop_stats(
                compiled_program["output_dir"], compiled_program["program_name"]
            )
            print(iop_stats_output)

            return compiled_program

        except Exception as e:
            print(f"Error message: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback

            traceback.print_exc()
            raise e
