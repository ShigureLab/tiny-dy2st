import ast
import inspect
import os
from typing import Callable, ParamSpec, TypeVar

import tiny_dy2st
from tiny_dy2st.dy2st.transformers.transformer import Dy2StTransformer
from tiny_dy2st.dygraph import Tensor, create_tensor, run_static_graph
from tiny_dy2st.static import GLOBAL_GRAPH, Variable, create_variable
from tiny_dy2st.static.ir import Node
from tiny_dy2st.utils import unique_name

TINY_DY2ST_DEBUG = os.getenv("TINY_DY2ST_DEBUG")

Input = ParamSpec("Input")
Output = TypeVar("Output")

CACHE: dict[str, tuple[Node, list[Variable], Variable]] = {}


def dy2st(dyfunc: Callable[Input, Output]) -> Callable[Input, Output]:
    def wrapper(*args: Tensor):
        # Step 1: 转写期
        dysrc = inspect.getsource(dyfunc)
        # 利用 Cache 避免重复组网
        if dysrc in CACHE:
            graph, args_vars, out_var = CACHE[dysrc]
            args_values = [arg.data for arg in args]
            inputs_dict = {
                arg_var.name: arg_value
                for arg_var, arg_value in zip(args_vars, args_values)
            }
        else:
            # 源码转写
            stsrc = dysrc2stsrc(dysrc)

            if TINY_DY2ST_DEBUG is not None and TINY_DY2ST_DEBUG.lower() in [
                "1",
                "true",
                "on",
            ]:
                print("Transformed source code:")
                print(stsrc)
            # 执行以获取静态函数
            func_name = dyfunc.__name__
            exec_globals = dict(globals())
            exec_globals.update(
                {
                    "tiny_dy2st": tiny_dy2st,
                    "__jst": tiny_dy2st.dy2st.runtime_convertor,
                }
            )
            exec(
                f"""
{stsrc}
stfunc = {func_name}
""",
                exec_globals,
            )
            stfunc = exec_globals["stfunc"]

            # Step 2: 组网期
            args_values = [arg.data for arg in args]
            # Tensor -> Variable
            args_vars = [
                create_variable(arg.name, arg.dtype, arg.shape) for arg in args
            ]

            # 构造输入
            inputs_dict = {
                arg_var.name: arg_value
                for arg_var, arg_value in zip(args_vars, args_values)
            }

            # 传入 Variable 进行组网
            out_var = stfunc(*args_vars)
            graph = GLOBAL_GRAPH[out_var.name]
            CACHE[dysrc] = graph, args_vars, out_var

        # Step 3: 执行期
        # 利用动态图 run_static_graph API 来运行静态图
        out = run_static_graph(graph, inputs_dict, [out_var.name])
        return create_tensor(
            unique_name(), out_var.dtype, out_var.shape, out[out_var.name]
        )

    return wrapper


def dysrc2stsrc(dysrc: str):
    if dysrc.startswith("@dy2st\n"):
        dysrc = dysrc[7:]
    transformer = Dy2StTransformer()
    dyast = ast.parse(dysrc)
    stast = transformer.visit(dyast)
    stsrc = ast.unparse(stast)
    return stsrc
