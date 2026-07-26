from ttl.ops import (
    activation,
)
from ttl import Layout, dtypes, GroqBuffer
from ttl import gapi

from ttl.utils import input_tensor as ttl_input_tensor
from ttl import GroqProgram

from ttl.tiled_tensor_language import vxm_ops


from typing import List, Dict, Any, Optional, Any
import numpy as np

from ttl.constants import VECTOR_SIZE, dtypes_to_np


def overall_model_to_ttl(
    layer_configurations_encoder: List[Dict[str, int]],
    layer_configurations_decoder: List[Dict[str, int]],
    autoencoder_kernels: Dict[str, List[np.ndarray]],
    conv_layer_configurations: List[Dict[str, int]],
    fc_layer_configurations: List[Dict[str, int]],
    zero_classifier_weights: Dict[str, List[np.ndarray]],
    lstm_layer_configurations: Dict[str, int],
    lstm_fc_layer_configurations: Dict[str, Any],
    lstm_classifier_weights: Dict[str, Any],
    input_size: Optional[List[int]] = None,
    input_tensor: Optional[GroqProgram] = None,
) -> GroqProgram:

    from ttl_autoencoder import autoencoder_model_to_ttl
    from ttl_zero_classifier import zero_classifier_model_to_ttl
    from ttl_lstm_pulse_num_classifier import lstm_pulse_num_classifier_model_to_ttl

    in_channel_num = conv_layer_configurations[0]["in_channel_num"]
    batch_num = conv_layer_configurations[0]["batch_num"]

    if input_tensor is None:

        assert input_size is not None, "input_size is required"

        tinput = Layout.create(
            (
                batch_num,
                in_channel_num,
                *input_size,
            ),
            dtypes.f16,
        )

        input_buffer = ttl_input_tensor(
            "image", tinput, byte_packed=True, input_packed=True
        )
    else:
        input_buffer = input_tensor

    output_tensor_autoencoder = autoencoder_model_to_ttl(
        layer_configurations_encoder,
        layer_configurations_decoder,
        autoencoder_kernels,
        input_tensor=input_buffer,
    )

    output_tensor_zero_classifier = zero_classifier_model_to_ttl(
        conv_layer_configurations,
        fc_layer_configurations,
        zero_classifier_weights,
        input_tensor=input_buffer,
    )

    probabilities = activation(output_tensor_zero_classifier, "sigmoid")

    # predictions = (probabilities > 0.5).float()
    probability_threshold = GroqBuffer.constant(
        value=np.full((VECTOR_SIZE,), 0.5, dtype=dtypes_to_np[probabilities.out_dtype])
    )

    predictions = gapi.vxm(
        vxm_ops.vxm_binary_cmp_gt, probabilities, probability_threshold
    )

    predictions = gapi.vxm(vxm_ops.vxm_unary_conv, predictions, conv_dtype=dtypes.f16)

    predictions = gapi.broadcast(predictions)

    output_tensor = gapi.vxm(
        vxm_ops.vxm_binary_mulsat,
        output_tensor_autoencoder,
        predictions,
    )

    output_tensor = gapi.reshape(
        output_tensor,
        output_tensor_autoencoder.out_tmemrefs[0],
    )

    output_tensor = gapi.reshape(
        output_tensor,
        output_tensor.out_tmemrefs[0].merge_axes(0, 2),
    )

    print("input tensor to lstm pulse num classifier: ", output_tensor.out_tmemrefs[0])

    output_tensor_lstm_pulse_num_classifier = lstm_pulse_num_classifier_model_to_ttl(
        lstm_layer_configurations,
        lstm_fc_layer_configurations,
        lstm_classifier_weights,
        input_tensor=output_tensor,
    )

    probabilities = activation(output_tensor_lstm_pulse_num_classifier, "sigmoid")

    predictions = gapi.vxm(
        vxm_ops.vxm_binary_cmp_gt, probabilities, probability_threshold
    )

    predictions = gapi.vxm(vxm_ops.vxm_unary_conv, predictions, conv_dtype=dtypes.f16)

    probabilities = gapi.reshape(
        probabilities,
        output_tensor_lstm_pulse_num_classifier.out_tmemrefs[0],
    )

    predictions = gapi.reshape(
        predictions,
        output_tensor_lstm_pulse_num_classifier.out_tmemrefs[0],
    )

    return probabilities, predictions
