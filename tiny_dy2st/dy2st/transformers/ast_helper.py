import ast


def print_ast(node: ast.AST):
    print(ast.dump(node, indent=4))
