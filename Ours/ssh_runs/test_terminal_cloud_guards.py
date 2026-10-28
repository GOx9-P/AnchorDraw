"""Notebook-only imports must not be reachable on the SSH host."""

import ast

import pytest

from validate_ports import check_terminal_cloud_imports


@pytest.mark.parametrize("source", [
    "from google.colab import drive",
    "from kaggle_secrets import UserSecretsClient",
    "get_ipython().run_line_magic('pip', 'install x')",
    "input('Enter token: ')",
])
def test_reachable_notebook_api_is_rejected(source):
    with pytest.raises(RuntimeError):
        check_terminal_cloud_imports(ast.parse(source), "synthetic.py")


@pytest.mark.parametrize("source", [
    "try:\n    from kaggle_secrets import UserSecretsClient\nexcept Exception:\n    pass",
    "SAVE_EXPORT_TO_GOOGLE_DRIVE = False\nif SAVE_EXPORT_TO_GOOGLE_DRIVE:\n    from google.colab import drive",
])
def test_guarded_optional_cloud_api_is_allowed(source):
    check_terminal_cloud_imports(ast.parse(source), "synthetic.py")
