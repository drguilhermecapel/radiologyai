"""Higiene do pacote — os portões de saída da Fase 1 (ROADMAP §8)."""

from __future__ import annotations

import subprocess
import sys

import pytest

import radiologyai


def test_version_is_declared():
    assert radiologyai.__version__


def test_selftest_passes():
    result = radiologyai.selftest()
    assert result["version"] == radiologyai.__version__


def test_import_does_not_pull_heavy_deps():
    """`import radiologyai` não deve trazer torch/numpy/SimpleITK."""
    code = (
        "import sys, radiologyai; "
        "heavy=[m for m in ('torch','numpy','SimpleITK','cv2') if m in sys.modules]; "
        "print(','.join(heavy))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "", f"import pesado no __init__: {out.stdout.strip()}"


def test_selftest_reports_optional_deps():
    optional = radiologyai.selftest()["optional"]
    assert isinstance(optional, dict)
    assert "torch" in optional


@pytest.mark.requirement("REQ-001")
def test_no_silent_import_fallbacks():
    """O anti-padrão do legado: `except ImportError: self.x = None`.

    A detecção é por AST, não por texto: docstrings que *descrevem* o
    anti-padrão (como as deste pacote) não são ocorrências dele.
    """
    import ast
    from pathlib import Path

    root = Path(radiologyai.__file__).parent
    offenders = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler) or node.type is None:
                continue
            names = (
                [node.type]
                if isinstance(node.type, ast.Name)
                else list(getattr(node.type, "elts", []))
            )
            if any(getattr(n, "id", None) in {"ImportError", "ModuleNotFoundError"} for n in names):
                offenders.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not offenders, f"import silencioso encontrado em: {offenders}"


@pytest.mark.requirement("REQ-002")
def test_no_random_in_non_evaluation_code():
    """np.random só é permitido no reamostrador de bootstrap, com seed explícita."""
    from pathlib import Path

    root = Path(radiologyai.__file__).parent
    allowed = {"evaluation/metrics.py"}
    offenders = []
    for p in root.rglob("*.py"):
        rel = p.relative_to(root).as_posix()
        if rel in allowed:
            continue
        text = p.read_text(encoding="utf-8")
        if "np.random" in text or "random.random" in text:
            offenders.append(rel)
    assert not offenders, f"aleatoriedade fora da avaliação: {offenders}"
