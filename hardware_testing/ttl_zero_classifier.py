from ttl.ops import conv1d as ttl_conv1d, Conv1dStageName, linear as ttl_linear
from ttl.ops import maxpool1d as ttl_maxpool1d
from ttl.ops import gapi_input
from ttl import Layout, dtypes
from ttl import gapi
from ttl import GroqProgram

from typing import List, Dict, Optional, Any
import numpy as np

from ttl.constants import VECTOR_SIZE


def zero_classifier_model_to_ttl(
    conv_layer_configurations: List[Dict[str, int]],
    fc_layer_configurations: List[Dict[str, int]],
    weights: Dict[str, List[np.ndarray]],
    input_size: Optional[int] = None,
    input_tensor: Optional[GroqProgram] = None,
) -> GroqProgram:

    in_channel_num = conv_layer_configurations[0]["in_channel_num"]
    batch_num = conv_layer_configurations[0]["batch_num"]

    conv_kernels = weights["conv_weights"]
    fc_weights = weights["fc_weights"]
    fc_biases = weights["fc_biases"]

    if input_tensor is None:
        split_num = (input_size + VECTOR_SIZE - 1) // VECTOR_SIZE

        tinput = Layout(
            (batch_num, in_channel_num, input_size),
            dtypes.f16,
            ends=(split_num * VECTOR_SIZE - input_size,),
        )
        input = gapi_input("image", tinput, byte_packed=True, input_packed=True)
    else:
        input = input_tensor

    idx = 0

    for layer_configuration, kernel in zip(conv_layer_configurations, conv_kernels):

        return_at_stage = Conv1dStageName.EXPLODED_CONV_RES

        activation_function = layer_configuration.get(
            "conv_activation_function", "none"
        )

        print("return_at_stage: ", return_at_stage)

        output_tensor = ttl_conv1d(
            input=input,
            conv_kernel=kernel,
            in_channel_num=layer_configuration["in_channel_num"],
            out_channel_num=layer_configuration["out_channel_num"],
            padding=layer_configuration["conv_padding"],
            batch_num=layer_configuration["batch_num"],
            stride=layer_configuration["conv_stride"],
            return_at_stage=return_at_stage,
            activation_fnc=activation_function,
        )

        idx += 1

        output_tensor = ttl_maxpool1d(
            image=output_tensor,
            kernel_size=layer_configuration["pooling_kernel_size"],
            channel_num=layer_configuration["out_channel_num"],
            stride=layer_configuration["pooling_stride"],
            batch_num=layer_configuration["batch_num"],
            padding=layer_configuration["pooling_padding"],
            exploded_input=True,
            channel_stride=4,
        )

        input = output_tensor

    output_tensor = gapi.vector_pack(output_tensor)

    input = output_tensor

    for layer_configuration, weights_loc, bias in zip(
        fc_layer_configurations, fc_weights, fc_biases
    ):

        activation_function = layer_configuration.get("activation_function", "none")

        weights_loc = weights_loc.transpose(1, 0).copy()

        output_tensor = ttl_linear(
            input=input,
            weights=weights_loc,
            bias=bias,
            activation_fnc=activation_function,
        )

        idx += 1

        input = output_tensor

    return output_tensor
