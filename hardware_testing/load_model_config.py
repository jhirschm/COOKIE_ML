"""
Module for loading and processing model configuration from JSON files.
"""

import os
import json


def get_model_config(config_path=None):
    """
    Load and process model configuration from a JSON file.

    Args:
        config_path: Path to the model configuration JSON file.
                    If None, defaults to model_config.json in the same directory as this file.

    Returns:
        Tuple containing:
        - layer_configurations_encoder: List of encoder layer configurations
        - layer_configurations_decoder: List of decoder layer configurations
        - conv_layer_configurations_zero_mask_classifier: List of conv layer configurations for zero mask classifier
        - fc_layer_configurations_zero_mask_classifier: List of FC layer configurations for zero mask classifier
        - lstm_layer_configurations_lstm_pulseNum_classifier: Dict of LSTM layer configurations
        - fc_layer_configurations_lstm_pulseNum_classifier: Dict of FC layer configurations for LSTM classifier
        - input_size: Input size for the model
    """
    if config_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "model_config.json")

    with open(config_path, "r") as f:
        config = json.load(f)

    input_size = config["input_size"]
    batch_num = config["batch_num"]
    encoder_config = config["encoder"]
    decoder_config = config["decoder"]
    zero_mask_classifier_config = config["zero_mask_classifier"]
    lstm_pulseNum_classifier_config = config["lstm_pulseNum_classifier"]
    layer_configurations_encoder = []
    layer_configurations_decoder = []
    conv_layer_configurations_zero_mask_classifier = []
    fc_layer_configurations_zero_mask_classifier = []
    lstm_layer_configurations_lstm_pulseNum_classifier = (
        lstm_pulseNum_classifier_config["lstm_layer_configurations"][0]
    )
    fc_layer_configurations_lstm_pulseNum_classifier = lstm_pulseNum_classifier_config[
        "fc_layer_configurations"
    ][0]

    # Add batch_num to each layer configuration
    for layer_conf in encoder_config["layer_configurations"]:
        layer_conf_with_batch = layer_conf.copy()
        layer_conf_with_batch["batch_num"] = batch_num
        layer_configurations_encoder.append(layer_conf_with_batch)

    for layer_conf in decoder_config["layer_configurations"]:
        layer_conf_with_batch = layer_conf.copy()
        layer_conf_with_batch["batch_num"] = batch_num
        layer_configurations_decoder.append(layer_conf_with_batch)

    for layer_conf in zero_mask_classifier_config["conv_layer_configurations"]:
        layer_conf_with_batch = layer_conf.copy()
        layer_conf_with_batch["batch_num"] = batch_num
        conv_layer_configurations_zero_mask_classifier.append(layer_conf_with_batch)

    for layer_conf in zero_mask_classifier_config["fc_layer_configurations"]:
        layer_conf_with_batch = layer_conf.copy()
        layer_conf_with_batch["batch_num"] = batch_num
        fc_layer_configurations_zero_mask_classifier.append(layer_conf_with_batch)

    lstm_layer_configurations_lstm_pulseNum_classifier["batch_num"] = batch_num
    lstm_layer_configurations_lstm_pulseNum_classifier["layer_norm"] = (
        lstm_pulseNum_classifier_config.get("layer_norm", False)
    )
    fc_layer_configurations_lstm_pulseNum_classifier["batch_num"] = batch_num
    fc_layer_configurations_lstm_pulseNum_classifier["layer_norm"] = (
        lstm_pulseNum_classifier_config.get("layer_norm", False)
    )

    return (
        layer_configurations_encoder,
        layer_configurations_decoder,
        conv_layer_configurations_zero_mask_classifier,
        fc_layer_configurations_zero_mask_classifier,
        lstm_layer_configurations_lstm_pulseNum_classifier,
        fc_layer_configurations_lstm_pulseNum_classifier,
        input_size,
    )
