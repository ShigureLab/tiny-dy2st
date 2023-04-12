from typing import Callable, ParamSpec, TypeVar

import tiny_dy2st

Input = ParamSpec("Input")
Output = TypeVar("Output")

dyfunc2stfunc_map = {
    tiny_dy2st.dygraph.ops.create_tensor: tiny_dy2st.static.ops.create_variable,
    tiny_dy2st.dygraph.ops.add: tiny_dy2st.static.ops.add,
    tiny_dy2st.dygraph.ops.div: tiny_dy2st.static.ops.div,
    tiny_dy2st.dygraph.ops.mul: tiny_dy2st.static.ops.mul,
    tiny_dy2st.dygraph.ops.sub: tiny_dy2st.static.ops.sub,
}


def convert_call(func: Callable[Input, Output]) -> Callable[Input, Output]:
    if func in dyfunc2stfunc_map:
        return dyfunc2stfunc_map[func]
    return func
