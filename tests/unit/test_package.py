"""Higiene do pacote — os portões de saída da Fase 1 (ROADMAP §8)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

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


def _guard():
    """Carrega scripts/check_honesty.py apontado para o repositório real.

    Os testes abaixo delegam ao guardião em vez de reimplementar as regras.
    Duas implementações da mesma regra divergem; uma diverge do código.
    """
    import importlib.util
    import sys
    from pathlib import Path

    script = Path(radiologyai.__file__).resolve().parents[2] / "scripts" / "check_honesty.py"
    spec = importlib.util.spec_from_file_location("check_honesty_pkg", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_honesty_pkg"] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.requirement("REQ-001")
def test_no_silent_import_fallbacks():
    """O anti-padrão do v1: `except ImportError: self.x = None`.

    Detecção por AST no guardião: docstrings que *descrevem* o anti-padrão
    (como as deste pacote) não são ocorrências dele.
    """
    offenders = _guard().rule_4_silent_imports()
    assert not offenders, [str(v) for v in offenders]


@pytest.mark.requirement("REQ-002")
def test_no_random_in_non_evaluation_code():
    """np.random só é permitido no reamostrador de bootstrap, com seed explícita.

    Detecção token-aware: `explain/gradcam.py` cita `np.random.normal` na
    docstring para documentar o que o v1 fazia — isso não é uso.
    """
    offenders = _guard().rule_2_randomness()
    assert not offenders, [str(v) for v in offenders]


class TestSupportedPythonBoundary:
    """A faixa declarada em pyproject (>=3.11,<3.14) precisa bater com selftest().

    Este teste existe porque o selftest exigia 3.11.x, depois <3.13, enquanto o Colab roda 3.13:
    a instalação passava e o selftest derrubava o notebook na célula seguinte.
    """

    @pytest.mark.parametrize("version", [(3, 11, 9), (3, 12, 4), (3, 13, 15)])
    def test_supported_versions_pass(self, monkeypatch, version):
        monkeypatch.setattr(sys, "version_info", version + (("final", 0)))
        assert radiologyai.selftest()["version"]

    @pytest.mark.parametrize("version", [(3, 10, 12), (3, 14, 0)])
    def test_unsupported_versions_fail_loud(self, monkeypatch, version):
        monkeypatch.setattr(sys, "version_info", version + (("final", 0)))
        with pytest.raises(RuntimeError, match="3.11 a 3.13"):
            radiologyai.selftest()

    def test_pyproject_range_matches_selftest(self):
        import tomllib

        root = Path(radiologyai.__file__).resolve().parents[2]
        spec = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        assert spec["project"]["requires-python"] == ">=3.11,<3.14"
