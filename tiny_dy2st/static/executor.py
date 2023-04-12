from __future__ import annotations

from typing import TYPE_CHECKING

from tiny_dy2st.static.ir import Node, OPType

if TYPE_CHECKING:
    from tiny_dy2st.base import Value


class Executor:
    def __init__(self):
        ...

    def run(
        self, graph: Node, inputs_dict: dict[str, Value], outputs_list: list[str]
    ) -> dict[str, Value]:
        scope: dict[str, Value] = {}
        self.bind_inputs(inputs_dict, scope)
        self._run_with_scope(graph, scope)
        return self.get_outputs(outputs_list, scope)

    def bind_inputs(self, inputs_dict: dict[str, Value], scope: dict[str, Value]):
        for name, value in inputs_dict.items():
            scope[name] = value

    def get_outputs(
        self, outputs_list: list[str], scope: dict[str, Value]
    ) -> dict[str, Value]:
        outputs: dict[str, Value] = {}
        for name in outputs_list:
            outputs[name] = scope[name]
        return outputs

    def _run_with_scope(self, node: Node, scope: dict[str, Value]):
        if node.op == OPType.ADD:
            for child in node.children:
                self._run_with_scope(child, scope)
            value = scope[node.children[0].name] + scope[node.children[1].name]
            scope[node.name] = value
        elif node.op == OPType.SUB:
            for child in node.children:
                self._run_with_scope(child, scope)
            value = scope[node.children[0].name] - scope[node.children[1].name]
            scope[node.name] = value
        elif node.op == OPType.MUL:
            for child in node.children:
                self._run_with_scope(child, scope)
            value = scope[node.children[0].name] * scope[node.children[1].name]
            scope[node.name] = value
        elif node.op == OPType.DIV:
            for child in node.children:
                self._run_with_scope(child, scope)
            value = scope[node.children[0].name] / scope[node.children[1].name]
            scope[node.name] = value
