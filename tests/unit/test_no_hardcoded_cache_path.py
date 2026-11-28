"""Nenhum teste fixa o caminho do cache local de pesos e amostras.

Na CI, ``/root`` é ilegível e ``Path.is_file()`` levanta ``PermissionError`` no
Python 3.11, quebrando a coleta inteira. A raiz vem de ``MEDAI_CACHE`` via
``tests/support/cache_paths.py``; docstrings e comentários podem citá-la.
"""

from __future__ import annotations

import ast
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1]
LITERAL = "/root/medai-cache"


def _offending_strings(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and LITERAL in node.value
        and id(node) not in docstrings
    ]


def test_cache_root_only_comes_from_cache_paths() -> None:
    allowed = TESTS / "support" / "cache_paths.py"
    hits = {
        str(path.relative_to(TESTS)): lines
        for path in sorted(TESTS.rglob("*.py"))
        if path != allowed and path != Path(__file__).resolve()
        for lines in [_offending_strings(path)]
        if lines
    }
    assert not hits, f"use cache_paths.MEDAI_CACHE em vez de {LITERAL!r}: {hits}"
