"""
Use to compile convolution programs for LPU
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
    gAPI = "gAPI"
    gMLIR = "gMLIR"
    Compiler = "Compiler"


def compile_encoder_with_compiler(
    model: torch.nn.Module,
    image: torch.Tensor,
) -> Union[dict[str, Union[str, Any]], Any]:

    # Set file names used below
    output_dir = "encoderCompiler"

    program_name = "encoder"

    return compile_with_compiler(
        model, image, program_name, output_dir, gen_vis_data=True
    )


def compile_encoder_with_g_api(
    layer_configurations: List[Dict[str, int]],
    kernels: List[np.ndarray],
    input: np.ndarray,
    output_tensor_name: str = "encoder_result",
) -> Union[dict[str, Union[str, Any]], Any]:

    with g.ProgramContext() as pc:

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
