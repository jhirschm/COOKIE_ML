from gstruct.ops import (
    conv1d as ttl_conv1d,
    Conv1dStageName,
    linear as ttl_linear,
    activation,
)
from gstruct.ops import maxpool1d as gstruct_maxpool1d
from gstruct import TiledMemref, dtypes, GroqBuffer
from gstruct import gstruct
from gstruct import GroqMLIR

from gstruct.tiled_tensor_language import vxm_ops


from typing import List, Dict
import numpy as np

from gstruct.constants import VECTOR_SIZE


def overall_model_to_ttl(
    layer_configurations_encoder: List[Dict[str, int]],
    layer_configurations_decoder: List[Dict[str, int]],
    autoencoder_kernels: Dict[str, List[np.ndarray]],
    conv_layer_configurations: List[Dict[str, int]],
    fc_layer_configurations: List[Dict[str, int]],
    zero_classifier_weights: Dict[str, List[np.ndarray]],
    input_size: int,
) -> GroqMLIR:

    from ttl_autoencoder import autoencoder_model_to_ttl
    from ttl_zero_classifier import zero_classifier_model_to_ttl

    output_tensor_autoencoder = autoencoder_model_to_ttl(
        layer_configurations_encoder,
        layer_configurations_decoder,
        autoencoder_kernels,
        input_size,
    )

    output_tensor_zero_classifier = zero_classifier_model_to_ttl(
        conv_layer_configurations,
        fc_layer_configurations,
        zero_classifier_weights,
        input_size,
    )

    print(output_tensor_autoencoder.out_tmemrefs[0])
    print(output_tensor_zero_classifier.out_tmemrefs[0])

    # tmp = np.zeros((VECTOR_SIZE,), dtype=np.float16)
    # tmp[0] = 1.0
    # output_tensor_zero_classifier = GroqBuffer.constant(value=tmp)

    probabilities = activation(output_tensor_zero_classifier, "sigmoid")

    # predictions = (probabilities > 0.5).float()
    probability_threshold = GroqBuffer.constant(
        value=np.full((VECTOR_SIZE,), 0.5, dtype=np.float16)
    )

    predictions = gstruct.vxm(
        vxm_ops.vxm_binary_cmp_gt, probabilities, probability_threshold
    )

    predictions = gstruct.vxm(
        vxm_ops.vxm_unary_conv, predictions, conv_dtype=dtypes.f16
    )

    predictions = gstruct.broadcast(predictions)

    output_tensor = gstruct.vxm(
        vxm_ops.vxm_binary_mulsat,
        output_tensor_autoencoder,
        predictions,
    )

    output_tensor = gstruct.reshape(
        output_tensor,
        output_tensor_autoencoder.out_tmemrefs[0],
    )

    return output_tensor
