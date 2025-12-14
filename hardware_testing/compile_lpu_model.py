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
)
from groq_convolution.conv1d import GroqConv1D, VECTOR_SIZE

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
                padding=layer_configuration["padding"],
                overlapped_scopes=True,
            )
            tsp_layers.append(tsp_layer)

        try:
            return compile_g_api(
                tsp_layers,
                input_mt,
                output_dir="./encoderGAPI",
                program_name="encoder",
                output_tensor_name="encoder_result",
            )
        except Exception as e:
            print(f"Error message: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback

            traceback.print_exc()
            raise e
