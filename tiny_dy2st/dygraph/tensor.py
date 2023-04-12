from __future__ import annotations

from tiny_dy2st.base import BaseTensor, DType, Value


class Tensor(BaseTensor):
    def __init__(self, name: str, dtype: DType, shape: tuple[int, ...], data: Value):
        self.data = data
        super().__init__(name, dtype, shape)

    def __repr__(self):
        return f"Tensor({self.name} {self.dtype.name} {self.shape}) data: {self.data}"
