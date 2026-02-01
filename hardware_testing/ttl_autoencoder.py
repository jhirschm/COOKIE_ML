from gstruct.ops import conv1d as ttl_conv1d, Conv1dStageName
from gstruct.ops import (
    convtranspose1d as ttl_convtranspose1d,
)
from gstruct.ops import maxpool1d as ttl_maxpool1d
from gstruct import TiledMemref, dtypes, GroqBuffer
from gstruct import gstruct
from gstruct import GroqMLIR

from typing import List, Dict, Optional
import numpy as np

from gstruct.constants import VECTOR_SIZE


def autoencoder_model_to_ttl(
    layer_configurations_encoder: List[Dict[str, int]],
    layer_configurations_decoder: List[Dict[str, int]],
    kernels: Dict[str, List[np.ndarray]],
    input_size: Optional[int] = None,
    input_tensor: Optional[GroqMLIR] = None,
) -> GroqMLIR:

    in_channel_num = layer_configurations_encoder[0]["in_channel_num"]
    batch_num = layer_configurations_encoder[0]["batch_num"]

    encoder_kernels = kernels["encoder_weights"]

    if input_tensor is None:
        split_num = (input_size + VECTOR_SIZE - 1) // VECTOR_SIZE

        tinput = TiledMemref(
            (batch_num, in_channel_num, input_size),
            dtypes.f16,
            ends=(split_num * 320 - input_size,),
        )
        input = GroqBuffer.input("image", tinput)
    else:
        input = input_tensor

    print("input.shape: ", input.out_tmemrefs[0])

    idx = 0

    for layer_configuration, kernel in zip(
        layer_configurations_encoder, encoder_kernels
    ):

        return_at_stage = Conv1dStageName.EXPLODED_CONV_RES

        activation_function = layer_configuration.get(
            "conv_activation_function", "none"
        )

        # print("return_at_stage: ", return_at_stage)

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

        # if idx == 1:
        #    print("??? output_tensor.shape: ", output_tensor[0].out_tmemrefs[0])
        #

        idx += 1

        # print("conv_unpacked.shape: ", output_tensor.out_tmemrefs[0])

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
        print("output_tensor.shape: ", output_tensor.out_tmemrefs[0])
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

        output_tensor = ttl_convtranspose1d(
            input=input,
            conv_kernel=kernel,
            in_channel_num=layer_configuration["in_channel_num"],
            out_channel_num=layer_configuration["out_channel_num"],
            padding=layer_configuration["conv_padding"],
            batch_num=layer_configuration["batch_num"],
            stride=layer_configuration["conv_stride"],
            activation_fnc=activation_function,
        )
        print("output_tensor.shape: ", output_tensor.out_tmemrefs[0])

        # Lout = 1 + (image_size - 1) * stride - 2 * padding_orig + kernel_size - 1

        idx += 1

        input = output_tensor

    return output_tensor
