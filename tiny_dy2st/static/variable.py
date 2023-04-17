from __future__ import annotations

from tiny_dy2st.base import BaseTensor, DType


class Variable(BaseTensor):
    def __init__(self, name: str, dtype: DType, shape: tuple[int, ...]):
        super().__init__(name, dtype, shape)

    def __repr__(self):
        return f"Variable({self.name} {self.dtype.name} {self.shape})"
