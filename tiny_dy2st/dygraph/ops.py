from __future__ import annotations

from typing import TYPE_CHECKING

from tiny_dy2st.base import DType, Value
from tiny_dy2st.dygraph.tensor import Tensor
from tiny_dy2st.utils import unique_name

if TYPE_CHECKING:
    from tiny_dy2st.static.ir import Node


def create_tensor(
    name: str, dtype: DType, shape: tuple[int, ...], data: Value
) -> Tensor:
    return Tensor(name, dtype, shape, data)


def add(x: Tensor, y: Tensor, name: str | None = None) -> Tensor:
    if name is None:
        name = unique_name()
    else:
        name = unique_name(name)
    return Tensor(name, x.dtype, x.shape, x.data + y.data)


def sub(x: Tensor, y: Tensor, name: str | None = None) -> Tensor:
    if name is None:
        name = unique_name()
    else:
        name = unique_name(name)
    return Tensor(name, x.dtype, x.shape, x.data - y.data)


def mul(x: Tensor, y: Tensor, name: str | None = None) -> Tensor:
    if name is None:
        name = unique_name()
    else:
        name = unique_name(name)
    return Tensor(name, x.dtype, x.shape, x.data * y.data)


def div(x: Tensor, y: Tensor, name: str | None = None) -> Tensor:
    if name is None:
        name = unique_name()
    else:
        name = unique_name(name)
    return Tensor(name, x.dtype, x.shape, x.data / y.data)


def run_static_graph(
    graph: Node, inputs_dict: dict[str, Value], outputs_list: list[str]
) -> dict[str, Value]:
    from tiny_dy2st.static import Executor

    executor = Executor()
    return executor.run(graph, inputs_dict, outputs_list)


magic_methods = {
    "__add__": add,
}

for method_name, method in magic_methods.items():
    setattr(Tensor, method_name, method)
