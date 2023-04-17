import tiny_dy2st
from tiny_dy2st.base import DType
from tiny_dy2st.dygraph import Tensor, create_tensor


def func(x: Tensor, y: Tensor) -> Tensor:
    z = x + y
    a = tiny_dy2st.dygraph.ops.mul(z, z)
    return a


x = create_tensor("x", DType.int32, (1,), 1)
y = create_tensor("y", DType.int32, (1,), 2)
print(func(x, y))
print(func(x, y))