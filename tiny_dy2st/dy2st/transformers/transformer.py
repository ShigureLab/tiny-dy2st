from __future__ import annotations

import ast


class Dy2StTransformer(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call):
        """
        func(x) -> __jst.convert_call(func)(x)
        """
        func = node.func
        args = node.args
        keywords = node.keywords

        return ast.Call(
            func=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="__jst", ctx=ast.Load()),
                    attr="convert_call",
                    ctx=ast.Load(),
                ),
                args=[func],
                keywords=[],
            ),
            args=args,
            keywords=keywords,
        )
