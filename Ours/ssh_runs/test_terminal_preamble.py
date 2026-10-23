"""The SSH display shim must accept notebook calls with several outputs."""

import ast

from export_notebooks import PREAMBLE


def test_display_accepts_multiple_values(capsys) -> None:
    tree = ast.parse(PREAMBLE)
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == "display"]
    assert len(functions) == 1
    module = ast.Module(body=functions, type_ignores=[])
    namespace = {}
    exec(compile(module, "<terminal-display>", "exec"), namespace)

    namespace["display"]("heading", object(), "caption")
    assert capsys.readouterr().out == "heading\ncaption\n"
