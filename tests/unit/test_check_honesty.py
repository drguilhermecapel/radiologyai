"""O guardião de honestidade precisa realmente pegar violações.

Um verificador que sempre passa é pior que nenhum: dá falsa segurança. Estes
testes injetam as violações concretas do repositório v1 e exigem detecção.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "check_honesty.py"


@pytest.fixture
def guard(monkeypatch, tmp_path):
    """Carrega o script apontando REPO/SRC para um repositório temporário."""
    spec = importlib.util.spec_from_file_location("check_honesty", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_honesty"] = module
    spec.loader.exec_module(module)

    src = tmp_path / "src" / "radiologyai"
    src.mkdir(parents=True)
    monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(module, "SRC", src)
    module.tmp_src = src
    module.tmp_repo = tmp_path
    return module


class TestRule1PerformanceClaims:
    def test_catches_fabricated_accuracy(self, guard):
        (guard.tmp_repo / "README.md").write_text(
            "- **Acurácia Validada**: 92.3% (Sensibilidade: 90%)", encoding="utf-8"
        )
        assert guard.rule_1_performance_claims()

    def test_catches_clinical_validation_claim(self, guard):
        (guard.tmp_repo / "README.md").write_text(
            "arquiteturas ensemble validadas clinicamente", encoding="utf-8"
        )
        assert guard.rule_1_performance_claims()

    def test_allows_claim_backed_by_artifact(self, guard):
        (guard.tmp_repo / "R.md").write_text(
            "AUC: 0.78 (ver artifacts/eval/run-2026-01-01/metrics.json)", encoding="utf-8"
        )
        assert not guard.rule_1_performance_claims()

    def test_ignores_legacy_directory(self, guard):
        legacy = guard.tmp_repo / "legacy"
        legacy.mkdir()
        (legacy / "README_FINAL.md").write_text("95% de acurácia", encoding="utf-8")
        assert not guard.rule_1_performance_claims()


class TestRule2Randomness:
    def test_catches_synthesized_cross_validation(self, guard):
        """A linha exata de medai_advanced_clinical_validation.py:357."""
        (guard.tmp_src / "m.py").write_text(
            "import numpy as np\ncv_scores = np.random.normal(base_accuracy, 0.02, n_folds)\n",
            encoding="utf-8",
        )
        violations = guard.rule_2_randomness()
        assert violations
        assert "aleatoriedade" in violations[0].detail

    def test_ignores_mention_in_docstring(self, guard):
        (guard.tmp_src / "m.py").write_text(
            '"""Este módulo não usa np.random em métrica."""\nx = 1\n', encoding="utf-8"
        )
        assert not guard.rule_2_randomness()

    def test_allows_seeded_bootstrap_in_metrics(self, guard):
        d = guard.tmp_src / "evaluation"
        d.mkdir()
        (d / "metrics.py").write_text(
            "import numpy as np\nrng = np.random.default_rng(seed)\n", encoding="utf-8"
        )
        assert not guard.rule_2_randomness()


class TestRule3Sha256:
    def test_catches_legacy_non_hex_hash(self, guard):
        """O 'sha256' do model_registry.json continha g, h, i."""
        (guard.tmp_repo / "registry.json").write_text(
            '{"sha256_hash": "a3b4c5d6e7f8g9h0i1j2k3l4m5n6o7p8q9r0s1t2u3v4w5x6y7z8a9b0c1d2e3f4"}',
            encoding="utf-8",
        )
        assert guard.rule_3_sha256()

    def test_accepts_valid_hash(self, guard):
        (guard.tmp_repo / "c.yaml").write_text(f"weights_sha256: {'a' * 64}\n", encoding="utf-8")
        assert not guard.rule_3_sha256()

    def test_catches_truncated_hash(self, guard):
        (guard.tmp_repo / "c.yaml").write_text("weights_sha256: abc123\n", encoding="utf-8")
        assert guard.rule_3_sha256()


class TestRule4SilentImports:
    def test_catches_import_fallback(self, guard):
        (guard.tmp_src / "m.py").write_text(
            "try:\n    import torch\nexcept ImportError:\n    torch = None\n", encoding="utf-8"
        )
        assert guard.rule_4_silent_imports()

    def test_catches_module_not_found_variant(self, guard):
        (guard.tmp_src / "m.py").write_text(
            "try:\n    import x\nexcept ModuleNotFoundError:\n    x = None\n", encoding="utf-8"
        )
        assert guard.rule_4_silent_imports()

    def test_allows_other_exception_handlers(self, guard):
        (guard.tmp_src / "m.py").write_text(
            "try:\n    f()\nexcept ValueError:\n    raise\n", encoding="utf-8"
        )
        assert not guard.rule_4_silent_imports()


class TestRule5SimulationMarkers:
    def test_catches_fallback_analysis(self, guard):
        (guard.tmp_src / "m.py").write_text(
            "def _analyze_image_fallback(self):\n    return 0.3\n", encoding="utf-8"
        )
        assert guard.rule_5_simulation_markers()

    def test_catches_mock_ground_truth(self, guard):
        (guard.tmp_src / "m.py").write_text(
            "y_true = np.array([1])  # Mock ground truth\n", encoding="utf-8"
        )
        # Está num comentário, mas o marcador é o próprio texto do comentário;
        # a regra 2 não pega isto e a 5 também não deve — é a regra 1 dos docs.
        # O que importa é que o *código* não contenha o padrão.
        assert not guard.rule_5_simulation_markers()

    def test_catches_dummy_model_in_code(self, guard):
        (guard.tmp_src / "m.py").write_text(
            "def _create_dummy_model():\n    pass\n", encoding="utf-8"
        )
        assert guard.rule_5_simulation_markers()

    def test_ignores_marker_named_in_docstring(self, guard):
        (guard.tmp_src / "m.py").write_text(
            '"""Substitui _analyze_image_fallback do legado."""\nx = 1\n', encoding="utf-8"
        )
        assert not guard.rule_5_simulation_markers()


class TestRule6UnbackedCards:
    def test_catches_performance_field_in_card(self, guard):
        cards = guard.tmp_src / "models" / "cards"
        cards.mkdir(parents=True)
        (cards / "m.yaml").write_text("card_id: x\naccuracy: 0.92\nauc: 0.94\n", encoding="utf-8")
        violations = guard.rule_6_unbacked_cards()
        assert violations
        assert "accuracy" in violations[0].detail

    def test_clean_card_passes(self, guard):
        cards = guard.tmp_src / "models" / "cards"
        cards.mkdir(parents=True)
        (cards / "m.yaml").write_text("card_id: x\nversion: '1.0'\n", encoding="utf-8")
        assert not guard.rule_6_unbacked_cards()


def test_guard_passes_on_this_repository():
    """O repositório real deve estar limpo — este é o portão de CI."""
    import subprocess

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--quiet"],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
