# from tiny_dy2st.dygraph import Tensor, create_tensor
# from tiny_dy2st.base import DType

# x = create_tensor("x", DType.int32, (1, ), 1)
# y = create_tensor("y", DType.int32, (1, ), 2)

# print(x + y)

# from tiny_dy2st.static import Variable, create_variable
# from tiny_dy2st.base import DType
# from tiny_dy2st.static import Executor, GLOBAL_GRAPH

# x = create_variable("x", DType.int32, (1, ))
# y = create_variable("y", DType.int32, (1, ))

# z = x + y

# executor = Executor()
# # print(GLOBAL_GRAPH)
# out = executor.run(GLOBAL_GRAPH[z.name], {x.name: 1, y.name: 2}, [z.name])
# z_value = out[z.name]
# print(z_value)

import tiny_dy2st
from tiny_dy2st.base import DType
from tiny_dy2st.dy2st.dy2st import dy2st
from tiny_dy2st.dygraph import Tensor, create_tensor


@dy2st
def func(x: Tensor, y: Tensor) -> Tensor:
    z = x + y
    a = tiny_dy2st.dygraph.ops.mul(z, z)
    return a


x = create_tensor("x", DType.int32, (1,), 1)
y = create_tensor("y", DType.int32, (1,), 2)
print(func(x, y))
print(func(x, y))
