import torch
import numpy as np
from typing import Dict, List


def extract_autoencoder_weights(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, List[np.ndarray]]:
    """
    Extract encoder.*.weight and decoder.*.weight fields from a state_dict and return them as a dictionary.

    Args:
        state_dict: Dictionary containing model parameters (from model.state_dict())

    Returns:
        Dictionary with two keys:
        - "encoder_weights": List of numpy arrays (float16) containing encoder layer weights, ordered by layer index
        - "decoder_weights": List of numpy arrays (float16) containing decoder layer weights, ordered by layer index
    """
    encoder_weights = []
    decoder_weights = []
    # Filter for keys that start with "encoder." and end with ".weight"
    for key in sorted(state_dict.keys()):
        if key.startswith("encoder.") and key.endswith(".weight"):
            weight = state_dict[key].detach().cpu().numpy().astype(np.float16)
            encoder_weights.append(weight)
        if key.startswith("decoder.") and key.endswith(".weight"):
            weight = state_dict[key].detach().cpu().numpy().astype(np.float16)
            decoder_weights.append(weight)

    return {"encoder_weights": encoder_weights, "decoder_weights": decoder_weights}


def extract_classifier_weights(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, List[np.ndarray]]:
    """
    Extract convolutional layer weights and fully connected layer weights and biases from a state_dict.

    Args:
        state_dict: Dictionary containing model parameters (from model.state_dict())

    Returns:
        Dictionary with three keys:
        - "conv_weights": List of numpy arrays (float16) containing convolutional layer weights, ordered by layer index
        - "fc_weights": List of numpy arrays (float16) containing fully connected layer weights, ordered by layer index
        - "fc_biases": List of numpy arrays (float16) containing fully connected layer biases, ordered by layer index
    """
    conv_weights = []
    fc_weights = []
    fc_biases = []

    # Filter for keys that contain conv-related patterns and end with ".weight"
    # This handles Conv1d, Conv2d, ConvTranspose1d, ConvTranspose2d layers
    for key in sorted(state_dict.keys()):
        key_lower = key.lower()
        if ("conv" in key_lower or "convtranspose" in key_lower) and key.endswith(
            ".weight"
        ):
            weight = state_dict[key].detach().cpu().numpy().astype(np.float16)
            conv_weights.append(weight)

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
        "fc_weights": fc_weights,
        "fc_biases": fc_biases,
    }

