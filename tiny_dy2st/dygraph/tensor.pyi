from __future__ import annotations

from tiny_dy2st.base import BaseTensor, DType, Value

class Tensor(BaseTensor):
    data: Value
    def __init__(
        self, name: str, dtype: DType, shape: tuple[int, ...], data: Value
    ) -> None: ...
    def __repr__(self) -> str: ...
    def __add__(self, other: Tensor) -> Tensor: ...
