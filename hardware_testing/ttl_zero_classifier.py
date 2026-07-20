from ttl.ops import conv2d as ttl_conv2d
from ttl.ops import linear as ttl_linear
from ttl.ops import maxpool2d as ttl_maxpool2d
from ttl.utils import gapi_input
from ttl import Layout, dtypes
from ttl import gapi
from ttl import GroqProgram

from typing import List, Dict, Optional, Any
import numpy as np

from ttl.constants import VECTOR_SIZE

from ttl.utils import clean_inner_dim
from ttl.utils import tile, untile

SLT_SIZE_X_2D = 4
SLT_SIZE_Y_2D = 4


def zero_classifier_model_to_ttl(
    conv_layer_configurations: List[Dict[str, int]],
    fc_layer_configurations: List[Dict[str, int]],
    weights: Dict[str, List[np.ndarray]],
    input_size: Optional[List[int]] = None,
    input_tensor: Optional[GroqProgram] = None,
) -> GroqProgram:

    batch_num = conv_layer_configurations[0]["batch_num"]

    conv_kernels = weights["conv_weights"]
    fc_weights = weights["fc_weights"]
    fc_biases = weights["fc_biases"]

    if input_tensor is None:

        assert input_size is not None, "input_size is required"

        tinput = Layout.create(
            (batch_num, 1, *input_size),
            dtypes.f16,
            axes=(3,),
        )
        input = gapi_input("image", tinput, byte_packed=True, input_packed=True)
    else:
        input = input_tensor

    print("input: ", input.out_tmemrefs[0])
    idx = 0
    for layer_configuration, kernel in zip(conv_layer_configurations, conv_kernels):

        activation_function = layer_configuration.get(
            "conv_activation_function", "none"
        )

        if idx == 0:
            input = tile(input, (SLT_SIZE_X_2D, SLT_SIZE_Y_2D))

        output_tensor = ttl_conv2d(
            input=input,
            conv_kernel=kernel,
            in_channel_num=layer_configuration["in_channel_num"],
            out_channel_num=layer_configuration["out_channel_num"],
            padding=layer_configuration["conv_padding"],
            batch_num=layer_configuration["batch_num"],
            stride=layer_configuration["conv_stride"],
            activation_fnc=activation_function,
        )

        # output_tensor = clean_inner_dim(output_tensor)

        output_tensor = ttl_maxpool2d(
            image=output_tensor,
            kernel_size=layer_configuration["pooling_kernel_size"],
            stride=layer_configuration["pooling_stride"],
            padding=layer_configuration["pooling_padding"],
        )

        if idx == len(conv_layer_configurations) - 1:
            output_tensor = untile(output_tensor)
            # output_tensor = clean_inner_dim(output_tensor)

        idx += 1

        # output_tensor = clean_inner_dim(output_tensor)

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

        input = output_tensor

    print("output tensor from zero classifier: ", output_tensor.out_tmemrefs[0])

    return output_tensor
