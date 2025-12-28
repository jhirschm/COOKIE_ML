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
from groq_convolution.gapi_conv1d import GroqConv1D, ResourceScopeName
from groq_convolution.gapi_pooling import GroqMaxPooling1D
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


def compile_encoder_with_gapi(
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


def compile_encoder_with_gstruct(
    layer_configurations: List[Dict[str, int]],
    kernels: List[np.ndarray],
    input: np.ndarray,
    output_tensor_name: str = "encoder_result",
    program_name: str = "encoder",
) -> Union[dict[str, Union[str, Any]], Any]:

    from gstruct.ops import conv1d as gstruct_conv1d, Conv1dStageName
    from gstruct.ops import maxpool1d as gstruct_maxpool1d
    from gstruct import tiled_memref, dtypes, groq_buffer, gstruct_to_mlir, mlir_to_iop

    output_dir = "./encoderGstruct"

    try:

        in_channel_num = layer_configurations[0]["conv_in_channel_num"]
        out_channel_num = layer_configurations[0]["conv_out_channel_num"]
        batch_num = layer_configurations[0]["batch_num"]
        image_size = layer_configurations[0]["image_size"]

        split_num = (image_size + VECTOR_SIZE - 1) // VECTOR_SIZE

        tinput = tiled_memref(
            (batch_num, in_channel_num, image_size),
            dtypes.f16,
            ends=(split_num * 320 - image_size,),
        )
        input_buffer = groq_buffer.input("image", tinput)

        input = input_buffer
        print("input.shape: ", input.out_tmemrefs[0])

        idx = 0

        for layer_configuration, kernel in zip(layer_configurations, kernels):

            if idx == 0:
                return_at_stage = Conv1dStageName.FINAL_TRANSPOSE
            else:
                return_at_stage = Conv1dStageName.INPUT_MASKING

            print("return_at_stage: ", return_at_stage)

            output_tensor = gstruct_conv1d(
                input=input,
                conv_kernel=kernel,
                in_channel_num=layer_configuration["conv_in_channel_num"],
                out_channel_num=layer_configuration["conv_out_channel_num"],
                padding=layer_configuration["conv_padding"],
                batch_num=layer_configuration["batch_num"],
                stride=layer_configuration["conv_stride"],
                return_at_stage=return_at_stage,
            )

            # if idx == 1:
            #    print("??? output_tensor.shape: ", output_tensor[0].out_tmemrefs[0])

            idx += 1

            # print("conv_unpacked.shape: ", output_tensor.out_tmemrefs[0])
            """
            output_tensor = gstruct_maxpool1d(
                image=output_tensor,
                kernel_size=layer_configuration["pooling_kernel_size"],
                channel_num=layer_configuration["pooling_in_channel_num"],
                stride=layer_configuration["pooling_stride"],
                batch_num=layer_configuration["batch_num"],
                padding=layer_configuration["pooling_padding"],
            )
            print("maxpool: ", output_tensor.out_tmemrefs[0])
            """
            input = output_tensor

        output_buffer = groq_buffer.output(output_tensor_name, output_tensor)
        # output_buffer2 = groq_buffer.output("eee", output_tensor[1])
        mlirtext = gstruct_to_mlir([output_buffer])  # , output_buffer2])
        iop_file = mlir_to_iop(
            mlirtext, program_name, output_dir, is_opt=False
        )  # ; assert False

        program_name = "unnamed"
        compiled_program = {
            "iop_file": iop_file,
            "output_dir": output_dir,
            "program_name": program_name,
        }

    except Exception as e:
        print(layer_configurations)
        print(f"Error message: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback

        traceback.print_exc()
        return None

    return compiled_program
