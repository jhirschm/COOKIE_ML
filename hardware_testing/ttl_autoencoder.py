from ttl.ops import conv2d as ttl_conv2d
from ttl.ops import (
    convtranspose2d as ttl_convtranspose2d,
)
from ttl.ops import maxpool2d as ttl_maxpool2d
from ttl.utils import gapi_input
from ttl import Layout, dtypes
from ttl import GroqProgram

from typing import List, Dict, Optional, Tuple
import numpy as np

from ttl.constants import VECTOR_SIZE

from ttl.utils import clean_inner_dim
from ttl.utils import tile, untile

SLT_SIZE_X_2D = 4
SLT_SIZE_Y_2D = 4


def autoencoder_model_to_ttl(
    layer_configurations_encoder: List[Dict[str, int]],
    layer_configurations_decoder: List[Dict[str, int]],
    kernels: Dict[str, List[np.ndarray]],
    input_size: Optional[List[int]] = None,
    input_tensor: Optional[GroqProgram] = None,
) -> GroqProgram:

    in_channel_num = layer_configurations_encoder[0]["in_channel_num"]
    batch_num = layer_configurations_encoder[0]["batch_num"]

    encoder_kernels = kernels["encoder_weights"]

    if input_tensor is None:

        assert input_size is not None, "input_size is required"

        tinput = Layout.create(
            (batch_num, in_channel_num, *input_size),
            dtypes.f16,
            axes=(3,),
        )

        input = gapi_input("image", tinput, byte_packed=True, input_packed=True)
    else:
        input = input_tensor

    idx = 0

    # input = clean_inner_dim(input)

    for layer_configuration, kernel in zip(
        layer_configurations_encoder, encoder_kernels
    ):

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

        # output_tensor = ttl_maxpool2d(
        #     image=output_tensor,
        #     kernel_size=layer_configuration["pooling_kernel_size"],
        #     stride=layer_configuration["pooling_stride"],
        #     padding=layer_configuration["pooling_padding"],
        # )

        # output_tensor = clean_inner_dim(output_tensor)

        if (
            idx == len(layer_configurations_encoder) - 1
            and len(layer_configurations_decoder) == 0
        ):
            output_tensor = untile(output_tensor)
            # output_tensor = clean_inner_dim(output_tensor)

        idx += 1

        input = output_tensor

    in_channel_num = layer_configurations_decoder[0]["in_channel_num"]
    batch_num = layer_configurations_decoder[0]["batch_num"]

    decoder_kernels = kernels["decoder_weights"]

    tinput = output_tensor

    idx = 0

    for layer_configuration, kernel in zip(
        layer_configurations_decoder, decoder_kernels
    ):

        activation_function = layer_configuration.get(
            "conv_activation_function", "none"
        )

        output_tensor = ttl_convtranspose2d(
            input=input,
            conv_kernel=kernel,
            in_channel_num=layer_configuration["in_channel_num"],
            out_channel_num=layer_configuration["out_channel_num"],
            padding=layer_configuration["conv_padding"],
            batch_num=layer_configuration["batch_num"],
            stride=layer_configuration["conv_stride"],
            activation_fnc=activation_function,
        )

        if idx == len(layer_configurations_decoder) - 1:
            output_tensor = untile(output_tensor)
            output_tensor = clean_inner_dim(output_tensor)

        idx += 1

        input = output_tensor

    print("autoencoder output_tensor: ", output_tensor.out_tmemrefs[0])

    return output_tensor
