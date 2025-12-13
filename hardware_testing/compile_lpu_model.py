"""
Use to compile convolution programs for LPU
"""

import numpy as np
import groq.api as g

from typing import Any, Union

from groq_convolution.conv1d import VECTOR_SIZE


def compile_with_g_api(
    tsp_layers, input: np.ndarray
) -> Union[dict[str, Union[str, Any]], Any]:

    # with g.ProgramContext() as pc:

    input_mt = g.input_tensor(
        shape=input.shape,
        dtype=g.float16,
        name="image",
        layout="H1(W), -1, S2",
        split_sizes=VECTOR_SIZE,
    )

    predecessors = [None]
    time_loc = 0
    for layer_idx, tsp_layer in enumerate(tsp_layers):
        with g.ResourceScope(
            name=f"encoder_layer_{layer_idx}",
            is_buffered=True,
            time=time_loc,
            predecessors=predecessors,
        ) as encoder_layer_scope:

            result_mt = tsp_layer(input_mt, time=0)

        predecessors = [encoder_layer_scope]
        time_loc = None
        input_mt = result_mt

    result_mt.set_program_output()
    ourput_dir = "./convolution1D"
    program_name = "convolution1D"

    try:

        iop_file = g.compile(
            base_name=program_name,
            output_dir=ourput_dir,
            result_tensor=result_mt,
            gen_vis_data=True,
        )

    except Exception as e:
        print(f"Error message: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback

        traceback.print_exc()
        raise e

    g.write_visualizer_data("groqview_convolution1D")

    return {
        "iop_file": iop_file,
        "ourput_dir": ourput_dir,
        "program_name": program_name,
    }
