from ttl.ops import (
    multi_layer_lstm as ttl_multi_layer_lstm,
    Directions,
    Activations,
    linear as ttl_linear,
    layer_norm as ttl_layer_norm,
)
from ttl.ops import gapi_input
from ttl import Layout, dtypes
from ttl import gapi
from ttl import GroqProgram


from typing import List, Dict, Any, Union, Optional
import numpy as np

from ttl.constants import VECTOR_SIZE


def ifgo_to_iofc(weights):
    # Convert weights from IFGO to IOFC format
    W_i, W_f, W_g, W_o = np.split(weights, 4, axis=0)

    return np.concatenate([W_i, W_o, W_f, W_g], axis=0)


def lstm_pulse_num_classifier_model_to_ttl(
    lstm_layer_configurations: Dict[str, int],
    fc_layer_configurations: Dict[str, Any],
    lstm_classifier_weights: Dict[str, Any],
    input_size: Optional[int] = None,
    input_tensor: Optional[GroqProgram] = None,
) -> GroqProgram:

    assert (
        lstm_layer_configurations["bidirectional"] == True
    ), "Only bidirectional LSTM is supported at this time"

    assert (
        len(lstm_classifier_weights["lstm_layers"])
        == lstm_layer_configurations["num_lstm_layers"]
    ), "Number of LSTM layers in the model does not match the number of LSTM layers in the configuration"

    batch_num = lstm_layer_configurations["batch_num"]
    seq_length = lstm_layer_configurations["seq_length"]

    if lstm_layer_configurations["bidirectional"]:
        hidden_size = lstm_layer_configurations["hidden_size"] * 2
    else:
        hidden_size = lstm_layer_configurations["hidden_size"]

    if input_tensor is None:
        split_num = (input_size + VECTOR_SIZE - 1) // VECTOR_SIZE

        tinput = Layout(
            (
                batch_num,
                seq_length,
                input_size,
            ),
            dtypes.f16,
            ends=(split_num * 320 - input_size,),
        )
        input_buffer = gapi_input("image", tinput, byte_packed=True, input_packed=True)
    else:
        input_buffer = input_tensor

    # converting torch-extracted weight tensors to TTL format
    lstm_weights_ttl = []
    for layer in lstm_classifier_weights["lstm_layers"]:

        lstm_weights_ttl.append(
            {
                "W": np.stack(
                    [
                        ifgo_to_iofc(layer["forward"]["weight_ih"]),
                        ifgo_to_iofc(layer["reverse"]["weight_ih"]),
                    ],
                    axis=0,
                ),
                "R": np.stack(
                    [
                        ifgo_to_iofc(layer["forward"]["weight_hh"]),
                        ifgo_to_iofc(layer["reverse"]["weight_hh"]),
                    ],
                    axis=0,
                ),
                "B": np.stack(
                    [
                        np.concatenate(
                            [
                                ifgo_to_iofc(layer["forward"]["bias_ih"]),
                                ifgo_to_iofc(layer["forward"]["bias_hh"]),
                            ],
                            axis=-1,
                        ),
                        np.concatenate(
                            [
                                ifgo_to_iofc(layer["reverse"]["bias_ih"]),
                                ifgo_to_iofc(layer["reverse"]["bias_hh"]),
                            ],
                            axis=-1,
                        ),
                    ],
                    axis=0,
                ),
                "activations": [Activations.Sigmoid, Activations.Tanh, Activations.Tanh]
                * 2,
                "direction": Directions.bidirectional,
            }
        )

    output_tensor, _, _ = ttl_multi_layer_lstm(
        input_buffer,
        lstm_weights_ttl,
        batch_first=True,
    )

    if lstm_layer_configurations["layer_norm"]:
        layer_norm_weights = lstm_classifier_weights["layer_norm_layers"][0]["weight"]
        layer_norm_bias = lstm_classifier_weights["layer_norm_layers"][0]["bias"]

        output_tensor = ttl_layer_norm(
            output_tensor,
            hidden_size,
            gamma_np=layer_norm_weights,
            beta_np=layer_norm_bias,
        )

    if (
        lstm_layer_configurations["ignore_fc_layers"]
        and lstm_layer_configurations["ignore_output_layer"]
    ):
        return output_tensor
    else:
        # output_tensor = output_tensor[:, -1, :]
        vector_shape = output_tensor.out_vector_shape

        assert len(vector_shape) == 3, "Output tensor must have 3 dimensions"

        output_tensor = gapi.subview(
            output_tensor,
            static_offsets=[0, vector_shape[1] - 1, 0],
            static_sizes=[vector_shape[0], 1, vector_shape[2]],
            static_strides=[1, 1, 1],
        )

        output_tensor = gapi.reshape(
            output_tensor,
            output_tensor.out_tmemrefs[0].squeeze(1),
        )

    if not lstm_layer_configurations["ignore_fc_layers"]:

        layer_num = len(fc_layer_configurations["fc_layers"])
        if fc_layer_configurations["layer_norm"]:
            # after each linear layer a normalization layer is applied
            layer_num = 2 * layer_num

        assert layer_num == len(
            lstm_classifier_weights["fc_layers"]
        ), "Number of FC layers in the model does not match the number of FC layers in the configuration"

        step = 2 if fc_layer_configurations["layer_norm"] else 1
        for layer_idx in range(0, layer_num, step):

            weights = lstm_classifier_weights["fc_layers"][layer_idx]["weight"]
            bias = lstm_classifier_weights["fc_layers"][layer_idx]["bias"]

            activation_function = fc_layer_configurations["activation_function"]

            output_tensor = ttl_linear(
                input=output_tensor,
                weights=weights.transpose(1, 0),
                bias=bias,
                activation_fnc=activation_function,
            )

            if fc_layer_configurations["layer_norm"]:

                weights = lstm_classifier_weights["fc_layers"][layer_idx + 1]["weight"]
                bias = lstm_classifier_weights["fc_layers"][layer_idx + 1]["bias"]

                output_tensor = ttl_layer_norm(
                    output_tensor,
                    weights.shape[0],
                    gamma_np=weights,
                    beta_np=bias,
                )

    if not lstm_layer_configurations["ignore_output_layer"]:
        weights = lstm_classifier_weights["output_layers"][0]["weight"]
        bias = lstm_classifier_weights["output_layers"][0]["bias"]

        output_tensor = ttl_linear(
            input=output_tensor,
            weights=weights.transpose(1, 0),
            bias=bias,
        )

    return output_tensor
