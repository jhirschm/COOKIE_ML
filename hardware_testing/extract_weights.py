import torch
import numpy as np
from typing import Dict, List, Any
import re


def extract_autoencoder_weights(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, List[np.ndarray]]:
    """
    Extract encoder/decoder weight and bias fields from a state_dict and return them as a dictionary.

    Args:
        state_dict: Dictionary containing model parameters (from model.state_dict())

    Returns:
        Dictionary with four keys:
        - "encoder_weights": List of numpy arrays (float16) containing encoder layer weights, ordered by layer index
        - "encoder_biases": List of numpy arrays (float16) containing encoder layer biases, ordered by layer index
        - "decoder_weights": List of numpy arrays (float16) containing decoder layer weights, ordered by layer index
        - "decoder_biases": List of numpy arrays (float16) containing decoder layer biases, ordered by layer index
    """
    encoder_weights = []
    encoder_biases = []
    decoder_weights = []
    decoder_biases = []
    # Filter for keys that start with "encoder." or "decoder." and end with ".weight" or ".bias"
    for key in sorted(state_dict.keys()):
        if key.startswith("encoder."):
            if key.endswith(".weight"):
                weight = state_dict[key].detach().cpu().numpy().astype(np.float16)
                encoder_weights.append(weight)
            elif key.endswith(".bias"):
                bias = state_dict[key].detach().cpu().numpy().astype(np.float16)
                encoder_biases.append(bias)
        if key.startswith("decoder."):
            if key.endswith(".weight"):
                weight = state_dict[key].detach().cpu().numpy().astype(np.float16)
                decoder_weights.append(weight)
            elif key.endswith(".bias"):
                bias = state_dict[key].detach().cpu().numpy().astype(np.float16)
                decoder_biases.append(bias)

    return {
        "encoder_weights": encoder_weights,
        "encoder_biases": encoder_biases,
        "decoder_weights": decoder_weights,
        "decoder_biases": decoder_biases,
    }


def extract_classifier_weights(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, List[np.ndarray]]:
    """
    Extract convolutional layer weights and biases and fully connected layer weights and biases from a state_dict.

    Args:
        state_dict: Dictionary containing model parameters (from model.state_dict())

    Returns:
        Dictionary with four keys:
        - "conv_weights": List of numpy arrays (float16) containing convolutional layer weights, ordered by layer index
        - "conv_biases": List of numpy arrays (float16) containing convolutional layer biases, ordered by layer index
        - "fc_weights": List of numpy arrays (float16) containing fully connected layer weights, ordered by layer index
        - "fc_biases": List of numpy arrays (float16) containing fully connected layer biases, ordered by layer index
    """
    conv_weights = []
    conv_biases = []
    fc_weights = []
    fc_biases = []

    # Filter for keys that contain conv-related patterns and end with ".weight" or ".bias"
    # This handles Conv1d, Conv2d, ConvTranspose1d, ConvTranspose2d layers
    for key in sorted(state_dict.keys()):
        key_lower = key.lower()
        if "conv" in key_lower or "convtranspose" in key_lower:
            if key.endswith(".weight"):
                weight = state_dict[key].detach().cpu().numpy().astype(np.float16)
                conv_weights.append(weight)
            elif key.endswith(".bias"):
                bias = state_dict[key].detach().cpu().numpy().astype(np.float16)
                conv_biases.append(bias)

    # Filter for keys that contain fc/linear patterns
    # Extract weights and biases separately
    for key in sorted(state_dict.keys()):
        key_lower = key.lower()
        if ("fc" in key_lower or "linear" in key_lower) and key.endswith(".weight"):
            weight = state_dict[key].detach().cpu().numpy().astype(np.float16)
            fc_weights.append(weight)
        elif ("fc" in key_lower or "linear" in key_lower) and key.endswith(".bias"):
            bias = state_dict[key].detach().cpu().numpy().astype(np.float16)
            fc_biases.append(bias)

    return {
        "conv_weights": conv_weights,
        "conv_biases": conv_biases,
        "fc_weights": fc_weights,
        "fc_biases": fc_biases,
    }


def extract_lstm_classifier_weights(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    """
    Extract LSTM and fully connected layer weights and biases from a state_dict, organized by layer.

    Args:
        state_dict: Dictionary containing model parameters (from model.state_dict())

    Returns:
        Dictionary with the following structure:
        - "lstm_layers": List of dictionaries, one per LSTM layer, each containing:
            - "layer_index": int, layer index (0, 1, 2, etc.)
            - "forward": dict with keys:
                - "weight_ih": input-to-hidden weight (float16 numpy array)
                - "weight_hh": hidden-to-hidden weight (float16 numpy array)
                - "bias_ih": input-to-hidden bias (float16 numpy array)
                - "bias_hh": hidden-to-hidden bias (float16 numpy array)
            - "reverse": dict with keys (same structure as forward, if bidirectional):
                - "weight_ih": input-to-hidden weight (float16 numpy array)
                - "weight_hh": hidden-to-hidden weight (float16 numpy array)
                - "bias_ih": input-to-hidden bias (float16 numpy array)
                - "bias_hh": hidden-to-hidden bias (float16 numpy array)
        - "fc_layers": List of dictionaries, one per FC layer, each containing:
            - "layer_index": int or str, layer index or name (e.g., 0, 3, "output")
            - "weight": weight tensor (float16 numpy array)
            - "bias": bias tensor (float16 numpy array)
    """
    # Dictionary to store LSTM layers by layer index
    lstm_layers_dict: Dict[int, Dict[str, Any]] = {}
    # Dictionary to store FC layers by layer index/name
    fc_layers_dict: Dict[str, Dict[str, Any]] = {}
    # Dictionary to store output layer by layer index/name
    output_layer_dict: Dict[str, Dict[str, Any]] = {}
    # Dictionary to store layer norm by layer index/name
    layer_norm_dict: Dict[str, Dict[str, Any]] = {}

    # Pattern to extract LSTM layer index: lstm.weight_ih_l0 -> 0
    # Use non-greedy match to correctly separate param_type from layer index
    lstm_pattern = re.compile(r"lstm\.(.+?)_l(\d+)(?:_reverse)?$")

    for key in sorted(state_dict.keys()):
        # Process LSTM layers
        if key.startswith("lstm."):
            match = lstm_pattern.match(key)
            if match:
                param_type = match.group(1)  # weight_ih, weight_hh, bias_ih, bias_hh
                layer_idx = int(match.group(2))  # 0, 1, 2, etc.
                is_reverse = "_reverse" in key

                # Initialize layer dict if not exists
                if layer_idx not in lstm_layers_dict:
                    lstm_layers_dict[layer_idx] = {
                        "layer_index": layer_idx,
                        "forward": {},
                        "reverse": {},
                    }

                # Convert to numpy float16
                tensor = state_dict[key].detach().cpu().numpy().astype(np.float16)

                # Store in appropriate direction
                direction = "reverse" if is_reverse else "forward"
                lstm_layers_dict[layer_idx][direction][param_type] = tensor

        # Process FC layers (fc_layers.0, fc_layers.3, output_layer)
        elif key.startswith("fc_layers."):
            # Pattern: fc_layers.0.weight -> layer "0"
            match = re.match(r"fc_layers\.(\d+)\.(weight|bias)$", key)
            if match:
                layer_idx = match.group(1)
                param_type = match.group(2)  # weight or bias

                # Initialize layer dict if not exists
                if layer_idx not in fc_layers_dict:
                    fc_layers_dict[layer_idx] = {
                        "layer_index": int(layer_idx),
                    }

                # Convert to numpy float16
                tensor = state_dict[key].detach().cpu().numpy().astype(np.float16)
                fc_layers_dict[layer_idx][param_type] = tensor

        elif key.startswith("output_layer"):
            # Pattern: output_layer.weight -> layer "output"
            match = re.match(r"output_layer\.(weight|bias)$", key)
            if match:
                layer_idx = "output"
                param_type = match.group(1)  # weight or bias

                # Initialize layer dict if not exists
                if layer_idx not in output_layer_dict:
                    output_layer_dict[layer_idx] = {
                        "layer_index": layer_idx,
                    }

                # Convert to numpy float16
                tensor = state_dict[key].detach().cpu().numpy().astype(np.float16)
                output_layer_dict[layer_idx][param_type] = tensor

        elif key.startswith("layer_norm"):
            # Pattern: layer_norm.weight -> layer "layer_norm"
            match = re.match(r"layer_norm\.(weight|bias)$", key)
            if match:
                layer_idx = "layer_norm"
                param_type = match.group(1)  # weight or bias

                # Initialize layer dict if not exists
                if layer_idx not in layer_norm_dict:
                    layer_norm_dict[layer_idx] = {
                        "layer_index": layer_idx,
                    }

                # Convert to numpy float16
                tensor = state_dict[key].detach().cpu().numpy().astype(np.float16)
                layer_norm_dict[layer_idx][param_type] = tensor

    # Convert dictionaries to lists, ordered by layer index
    lstm_layers = [lstm_layers_dict[idx] for idx in sorted(lstm_layers_dict.keys())]

    # For FC layers and output layer, handle mixed numeric and string indices
    fc_layers = []
    output_layers = []
    layer_norm_layers = []
    # First add numeric indices sorted numerically
    numeric_keys = [
        k for k in fc_layers_dict.keys() if isinstance(k, str) and k.isdigit()
    ]
    for key in sorted(numeric_keys, key=int):
        fc_layers.append(fc_layers_dict[key])
    # Then add non-numeric keys (like "output")
    for key in fc_layers_dict.keys():
        if not (isinstance(key, str) and key.isdigit()):
            fc_layers.append(fc_layers_dict[key])

    numeric_keys_output = [
        k for k in output_layer_dict.keys() if isinstance(k, str) and k.isdigit()
    ]
    for key in sorted(numeric_keys_output, key=int):
        output_layers.append(output_layer_dict[key])
    for key in output_layer_dict.keys():
        if not (isinstance(key, str) and key.isdigit()):
            output_layers.append(output_layer_dict[key])

    numeric_keys_layer_norm = [
        k for k in layer_norm_dict.keys() if isinstance(k, str) and k.isdigit()
    ]
    for key in sorted(numeric_keys_layer_norm, key=int):
        layer_norm_layers.append(layer_norm_dict[key])
    for key in layer_norm_dict.keys():
        if not (isinstance(key, str) and key.isdigit()):
            layer_norm_layers.append(layer_norm_dict[key])

    return {
        "lstm_layers": lstm_layers,
        "fc_layers": fc_layers,
        "output_layers": output_layers,
        "layer_norm_layers": layer_norm_layers,
    }
