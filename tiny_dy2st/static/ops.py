from __future__ import annotations

from tiny_dy2st.base import DType
from tiny_dy2st.static.ir import OPType, emit_ir
from tiny_dy2st.static.variable import Variable
from tiny_dy2st.utils import unique_name


def create_variable(name: str, dtype: DType, shape: tuple[int, ...]):
    var = Variable(name, dtype, shape)
    emit_ir(OPType.PLACEHOLDER, [], name=var.name)
    return var


def add(x: Variable, y: Variable, name: str | None = None) -> Variable:
    if name is None:
        name = unique_name()
    else:
        name = unique_name(name)

    emit_ir(OPType.ADD, [x, y], name=name)
    return Variable(name, x.dtype, x.shape)


def sub(x: Variable, y: Variable, name: str | None = None) -> Variable:
    if name is None:
        name = unique_name()
    else:
        name = unique_name(name)

    emit_ir(OPType.SUB, [x, y], name=name)
    return Variable(name, x.dtype, x.shape)


def mul(x: Variable, y: Variable, name: str | None = None) -> Variable:
    if name is None:
        name = unique_name()
    else:
        name = unique_name(name)

    emit_ir(OPType.MUL, [x, y], name=name)
    return Variable(name, x.dtype, x.shape)


def div(x: Variable, y: Variable, name: str | None = None) -> Variable:
    if name is None:
        name = unique_name()
    else:
        name = unique_name(name)

    emit_ir(OPType.DIV, [x, y], name=name)
    return Variable(name, x.dtype, x.shape)


magic_methods = {
    "__add__": add,
}

for method_name, method in magic_methods.items():
    setattr(Variable, method_name, method)
