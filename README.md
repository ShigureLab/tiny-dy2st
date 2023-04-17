# Tiny-Dy2St

Tiny-Dy2St 是一个非常简单的 PaddlePaddle 动转静流程示例项目，它包含了以下三部分：

- 简陋的动态图模块 `dygraph`
- 简陋的静态图模块 `static`
- 简陋的动转静模块 `dy2st`

与 PaddlePaddle 不同的是，tiny-dy2st 没有动静统一的 API，所有 API 都是动态图专属/静态图专属的，因此这一部分也是需要进行转换的。

> **Note**
>
> 最低要求 Python 版本为 3.9

下面的是动态图示例代码：

```python
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
```

下面是静态图示例代码

```python
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
```

只需要在动态图上加一个 `@dy2st` 来进行装饰，就可以方便地实现动转静啦～

```python
import tiny_dy2st
from tiny_dy2st.dy2st import dy2st
from tiny_dy2st.base import DType
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
```

我们可以用在执行时添加环境变量 `TINY_DY2ST_DEBUG=ON` 来查看转换后的代码

```bash
PYTHONPATH=. TINY_DY2ST_DEBUG=ON python examples/dy2st_example.py
```
