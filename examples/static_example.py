import tiny_dy2st
from tiny_dy2st.static import Variable, create_variable
from tiny_dy2st.base import DType
from tiny_dy2st.static import Executor, GLOBAL_GRAPH

x = create_variable("x", DType.int32, (1, ))
y = create_variable("y", DType.int32, (1, ))

def func(x: Variable, y: Variable) -> Variable:
    z = x + y
    a = tiny_dy2st.static.ops.mul(z, z)
    return a

z = func(x, y)

executor = Executor()
out = executor.run(GLOBAL_GRAPH[z.name], {x.name: 1, y.name: 2}, [z.name])
z_value = out[z.name]
print(z_value)
out = executor.run(GLOBAL_GRAPH[z.name], {x.name: 1, y.name: 2}, [z.name])
z_value = out[z.name]
print(z_value)