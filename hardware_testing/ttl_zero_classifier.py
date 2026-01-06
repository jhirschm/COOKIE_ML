from gstruct.ops import conv1d as ttl_conv1d, Conv1dStageName, linear as ttl_linear
from gstruct.ops import maxpool1d as gstruct_maxpool1d
from gstruct import TiledMemref, dtypes, GroqBuffer
from gstruct import gstruct
from gstruct import GroqMLIR

from typing import List, Dict
import numpy as np

from gstruct.constants import VECTOR_SIZE


def zero_classifier_model_to_ttl(
    conv_layer_configurations: List[Dict[str, int]],
    fc_layer_configurations: List[Dict[str, int]],
    weights: Dict[str, List[np.ndarray]],
    input_size: int,
) -> GroqMLIR:

    in_channel_num = conv_layer_configurations[0]["in_channel_num"]
    batch_num = conv_layer_configurations[0]["batch_num"]

    conv_kernels = weights["conv_weights"]
    fc_weights = weights["fc_weights"]
    fc_biases = weights["fc_biases"]

    split_num = (input_size + VECTOR_SIZE - 1) // VECTOR_SIZE

    tinput = TiledMemref(
        (batch_num, in_channel_num, input_size),
        dtypes.f16,
        ends=(split_num * 320 - input_size,),
    )
    input_buffer = GroqBuffer.input("image", tinput)

    input = input_buffer
    print("input.shape: ", input.out_tmemrefs[0])

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

        # if idx == 1:
        #    print("??? output_tensor.shape: ", output_tensor[0].out_tmemrefs[0])

        idx += 1

        # print("conv_unpacked.shape: ", output_tensor.out_tmemrefs[0])

        output_tensor = gstruct_maxpool1d(
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

    print("after conv output_tensor.shape: ", output_tensor.out_tmemrefs[0])

    output_tensor = gstruct.vector_pack(output_tensor)
    print("after vector pack output_tensor.shape: ", output_tensor.out_tmemrefs[0])

    input = output_tensor

    for layer_configuration, weights_loc, bias in zip(
        fc_layer_configurations, fc_weights, fc_biases
    ):

        activation_function = layer_configuration.get("activation_function", "none")

        weights_loc = weights_loc.transpose(1, 0).copy()
        print("weight.shape: ", weights_loc.shape)

        print(bias)

        output_tensor = ttl_linear(
            input=input,
            weights=weights_loc,
            bias=bias,
            activation_fnc=activation_function,
        )

        print("after linear output_tensor.shape: ", output_tensor.out_tmemrefs[0])

        idx += 1

        input = output_tensor

    return output_tensor
