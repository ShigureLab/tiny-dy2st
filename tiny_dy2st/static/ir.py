from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tiny_dy2st.static import Variable

GLOBAL_GRAPH: dict[str, Node] = {}


class OPType(Enum):
    PLACEHOLDER = 0
    ADD = 1
    SUB = 2
    MUL = 3
    DIV = 4


class Node:
    def __init__(self, name: str, op: OPType, children: list[Node]) -> None:
        self.name = name
        self.op = op
        self.children = children


def emit_ir(op: OPType, args: list[Variable], name: str):
    children = [GLOBAL_GRAPH[var.name] for var in args]
    node = Node(name, op, children)
    GLOBAL_GRAPH[name] = node
