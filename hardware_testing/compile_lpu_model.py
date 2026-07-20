"""
Module for compiling convolution programs for Groq LPU (Language Processing Unit).

This module provides functions to compile encoder models using different compilation
methods (gAPI, gMLIR, or Compiler) for execution on Groq hardware accelerators.
"""

import numpy as np
from enum import Enum

from typing import Any, Union, List, Dict, Tuple


from ttl import GroqProgram, compile_with_compiler

import torch


class CompilerType(Enum):
    """Enumeration of available compiler types for Groq LPU compilation.

    Attributes:
        gAPI: Use the Groq API (gAPI) compiler for compilation.
        ttl: Use the TiledTensorLanguage to compile the program.
        Compiler: Use the standard Groq compiler for compilation.
    """

    Compiler = "Compiler"
    ttl = "tiled_tensor_language"


def compile_encoder_with_compiler(
    model: torch.nn.Module,
    image: torch.Tensor,
    program_name: str = "encoder",
) -> Union[dict[str, Union[str, Any]], Any]:
    """Compile an encoder model using the Groq Compiler.

    This function compiles a PyTorch encoder model for execution on Groq hardware
    using the standard Groq compiler. The compiled program is saved to the
    "encoderCompiler" output directory.

    Args:
        model: PyTorch neural network module representing the encoder to compile.
        image: Example input tensor used for shape inference during compilation.
            Should match the expected input shape of the model.
        program_name: Name of the compiled program. Defaults to "encoder".

    Returns:
        Dictionary containing compilation results with keys:
            - "iop_file": Path to the compiled IOP file
            - "output_dir": Directory where compiled files are saved
            - "program_name": Name of the compiled program
            Additional keys may be present depending on the compiler output.

    Raises:
        Exception: If compilation fails, the underlying exception is raised
            with error details.
    """
    # Set file names used below
    output_dir = "encoderCompiler"

    return compile_with_compiler(
        model, image, program_name, output_dir, gen_vis_data=False
    )


def compile_autoencoder_with_ttl(
    layer_configurations_encoder: List[Dict[str, int]],
    layer_configurations_decoder: List[Dict[str, int]],
    kernels: Dict[str, List[np.ndarray]],
    input_size: int,
    output_tensor_name: str = "autoencoder_result",
    program_name: str = "autoencoder",
) -> Union[dict[str, Union[str, Any]], Any]:

    from ttl_autoencoder import autoencoder_model_to_ttl

    output_dir = "./autoencoderTTL"

    output_tensor = autoencoder_model_to_ttl(
        layer_configurations_encoder,
        layer_configurations_decoder,
        kernels,
        input_size,
    )

    return compile_ttl_model(
        output_tensor, output_tensor_name, program_name, output_dir
    )


def compile_zero_classifier_with_ttl(
    conv_layer_configurations: List[Dict[str, int]],
    fc_layer_configurations: List[Dict[str, int]],
    weights: Dict[str, List[np.ndarray]],
    input_size: int,
    output_tensor_name: str = "classifier_result",
    program_name: str = "classifier",
) -> Union[dict[str, Union[str, Any]], Any]:

    from ttl_zero_classifier import zero_classifier_model_to_ttl

    output_dir = "./classifierTTL"

    output_tensor = zero_classifier_model_to_ttl(
        conv_layer_configurations,
        fc_layer_configurations,
        weights,
        input_size,
    )

    return compile_ttl_model(
        output_tensor, output_tensor_name, program_name, output_dir
    )


def compile_lstm_pulsenum_classifier_with_ttl(
    lstm_layer_configurations: Dict[str, int],
    fc_layer_configurations: Dict[str, int],
    lstm_classifier_weights: Dict[str, Any],
    input_size: int,
    output_tensor_name: str = "lstm_pulsenum_classifier_result",
    program_name: str = "lstm_pulsenum_classifier",
) -> Union[dict[str, Union[str, Any]], Any]:

    from ttl_lstm_pulse_num_classifier import lstm_pulse_num_classifier_model_to_ttl

    output_dir = "./LSTMPulseNumClassifierTTL"

    output_tensor = lstm_pulse_num_classifier_model_to_ttl(
        lstm_layer_configurations,
        fc_layer_configurations,
        lstm_classifier_weights,
        input_size,
    )

    return compile_ttl_model(
        output_tensor, output_tensor_name, program_name, output_dir
    )


def compile_pulsenum_classifier_workflow_with_ttl(
    layer_configurations_encoder: List[Dict[str, int]],
    layer_configurations_decoder: List[Dict[str, int]],
    autoencoder_kernels: Dict[str, List[np.ndarray]],
    conv_layer_configurations: List[Dict[str, int]],
    fc_layer_configurations: List[Dict[str, int]],
    zero_classifier_weights: Dict[str, List[np.ndarray]],
    lstm_layer_configurations_lstm_pulseNum_classifier: Dict[str, int],
    fc_layer_configurations_lstm_pulseNum_classifier: Dict[str, Any],
    lstm_classifier_weights: Dict[str, Any],
    input_size: int,
    output_tensor_name: Tuple[str, str] = (
        "probs",
        "preds",
    ),  # probabilities and predictions
    program_name: str = "lstm_pulsenum_classifier",
) -> Union[dict[str, Union[str, Any]], Any]:

    from ttl_overall_model import overall_model_to_ttl

    output_dir = "./overalModelTTL"

    output_tensors = overall_model_to_ttl(
        layer_configurations_encoder,
        layer_configurations_decoder,
        autoencoder_kernels,
        conv_layer_configurations,
        fc_layer_configurations,
        zero_classifier_weights,
        lstm_layer_configurations_lstm_pulseNum_classifier,
        fc_layer_configurations_lstm_pulseNum_classifier,
        lstm_classifier_weights,
        input_size,
    )

    return compile_ttl_model(
        output_tensors, output_tensor_name, program_name, output_dir
    )


def compile_ttl_model(
    model: Union[GroqProgram, Tuple[GroqProgram, GroqProgram]],
    output_tensor_name: Union[str, Tuple[str, ...]] = "model_result",
    program_name: str = "model",
    output_dir: str = "./modelTTL",
) -> Union[dict[str, Union[str, Any]], Any]:

    from ttl import ttl_to_iop
    from ttl.utils import gapi_output

    try:

        if isinstance(model, tuple):

            output_buffer = [
                gapi_output(
                    output_tensor_name[idx],
                    model[idx],
                    byte_packed=True,
                    output_packed=True,
                )
                for idx in range(len(model))
            ]

        else:
            output_buffer = [
                gapi_output(
                    output_tensor_name, model, byte_packed=True, output_packed=True
                )
            ]

        iop_file = ttl_to_iop(output_buffer, program_name, output_dir)

        compiled_program = {
            "iop_file": iop_file,
            "output_dir": output_dir,
            "program_name": program_name,
        }

    except Exception as e:
        print(f"Error message: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback

        traceback.print_exc()
        return None

    return compiled_program
