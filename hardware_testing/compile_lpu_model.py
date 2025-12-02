"""
Use to compile convolution programs for LPU
"""

import numpy as np
import groq.api as g

from typing import Any, Union


def compile_with_g_api(
    tsp_layers, input: np.ndarray
) -> Union[dict[str, Union[str, Any]], Any]:

    # TODO: check for float16
    input_mt = g.input_tensor(
        shape=input.shape,
        dtype=g.float16,
        name="image",
        layout="H1(W), -1, S2",
    )

    for tsp_layer in tsp_layers:
        result_mt = tsp_layer(input_mt, time=0)
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

    except:
        raise Exception("Failed to build convolution program!")

    g.write_visualizer_data("groqview_convolution1D")

    return {
        "iop_file": iop_file,
        "ourput_dir": ourput_dir,
        "program_name": program_name,
    }
