import torch.nn as nn
from typing import List, Any


def get_activation_function(activation_function_name):
    """Get the activation function based on the name."""
    if activation_function_name == "ReLU":
        return nn.ReLU()
    elif activation_function_name == "Sigmoid":
        return nn.Sigmoid()
    elif activation_function_name == "Tanh":
        return nn.Tanh()
    else:
        return None


def gen_autoencoder_torch_layers(
    encoder_layer_configurations: List[dict],
    decoder_layer_configurations: List[dict],
) -> tuple[List[List[Any]], List[List[Any]]]:
    """
    Generate PyTorch layers for autoencoder from layer configurations.

    Args:
        encoder_layer_configurations: List of dictionaries containing encoder layer configurations
        decoder_layer_configurations: List of dictionaries containing decoder layer configurations

    Returns:
        Tuple of (encoder_layers, decoder_layers) where each is a list of layer pairs [layer, activation]
    """
    # Torch Encoder layers
    encoder_layers = []
    for layer_conf in encoder_layer_configurations:

        activation_function = get_activation_function(
            layer_conf["conv_activation_function"]
        )

        sub_layers = []
        sub_layers.append(
            [
                nn.Conv1d(  # convolutional layer
                    layer_conf["in_channel_num"],
                    layer_conf["out_channel_num"],
                    kernel_size=layer_conf["conv_kernel_size"],
                    stride=layer_conf["conv_stride"],
                    padding=layer_conf["conv_padding"],
                    bias=False,
                ),
                (activation_function),  # activation function
            ]
        )

        sub_layers.append(
            [
                nn.MaxPool1d(  # pooling layer
                    kernel_size=layer_conf["pooling_kernel_size"],
                    stride=layer_conf["pooling_stride"],
                    padding=layer_conf["pooling_padding"],
                ),
                None,
            ]
        )

        encoder_layers.extend(sub_layers)

    # Torch Decoder layers
    decoder_layers = []
    for layer_conf in decoder_layer_configurations:

        activation_function = get_activation_function(
            layer_conf["conv_activation_function"]
        )

        decoder_layers.append(
            [
                nn.ConvTranspose1d(  # convolutional layer
                    layer_conf["in_channel_num"],
                    layer_conf["out_channel_num"],
                    kernel_size=layer_conf["conv_kernel_size"],
                    stride=layer_conf["conv_stride"],
                    padding=layer_conf["conv_padding"],
                    bias=False,
                ),
                (activation_function),  # activation function
            ]
        )

    return encoder_layers, decoder_layers
