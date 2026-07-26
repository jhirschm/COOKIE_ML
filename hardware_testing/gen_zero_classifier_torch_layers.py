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


def gen_zero_classifier_torch_layers(
    conv_layer_configurations: List[dict],
    fc_layer_configurations: List[dict],
) -> tuple[List[List[Any]], List[List[Any]]]:
    """
    Generate PyTorch layers for zero mask classifier from layer configurations.

    Args:
        conv_layer_configurations: List of dictionaries containing convolutional layer configurations
        fc_layer_configurations: List of dictionaries containing fully connected layer configurations

    Returns:
        Tuple of (conv_layers, fc_layers) where each is a list of layer pairs [layer, activation]
    """
    import torch

    # Torch Zero Mask Classifier layers
    zero_mask_classifier_conv_layers = []
    for layer_conf in conv_layer_configurations:

        activation_function = get_activation_function(
            layer_conf["conv_activation_function"]
        )

        sub_layers = []
        sub_layers.append(
            [
                nn.Conv2d(  # convolutional layer
                    layer_conf["in_channel_num"],
                    layer_conf["out_channel_num"],
                    kernel_size=layer_conf["conv_kernel_size"],
                    stride=layer_conf["conv_stride"],
                    padding=layer_conf["conv_padding"],
                    bias=True,
                ),
                (activation_function),  # activation function
            ]
        )

        sub_layers.append(
            [
                nn.MaxPool2d(  # pooling layer
                    kernel_size=layer_conf["pooling_kernel_size"],
                    stride=layer_conf["pooling_stride"],
                    padding=layer_conf["pooling_padding"],
                ),
                None,
            ]
        )

        zero_mask_classifier_conv_layers.extend(sub_layers)

    """
    # Calculate the output size after conv layers
    def get_conv_output_size(input_size, conv_layers):
        x = torch.randn(input_size)
        model = nn.Sequential(
            *[
                layer
                for layer_pair in conv_layers
                for layer in layer_pair
                if layer is not None
            ]
        )
        x = model(x)
        return x.shape


    output_size = get_conv_output_size((1, 16, 512), zero_mask_classifier_conv_layers)
    print(f"Output size after conv layers: {output_size}")
    """
    zero_mask_classifier_fc_layers = []
    for layer_conf in fc_layer_configurations:

        activation_function = get_activation_function(layer_conf["activation_function"])

        zero_mask_classifier_fc_layers.append(
            [
                nn.Linear(layer_conf["input_size"], layer_conf["output_size"]),
                (activation_function),
            ]
        )

    return zero_mask_classifier_conv_layers, zero_mask_classifier_fc_layers
