from __future__ import annotations

from enum import Enum
from typing_extensions import TypeAlias


class DType(Enum):
    uint8 = 0
    int8 = 1
    int16 = 2
    int32 = 3
    int64 = 4

    float16 = 5
    bfloat16 = 6
    float32 = 7
    float64 = 8

    complex64 = 9
    complex128 = 10

    bool = 11


Value: TypeAlias = float


class BaseTensor:
    def __init__(self, name: str, dtype: DType, shape: tuple[int, ...]):
        self.name = name
        self.dtype = dtype
        self.shape = shape
