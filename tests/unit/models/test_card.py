"""Model card — a validação que teria pegado models/model_registry.json."""

from __future__ import annotations

import hashlib

import pytest
import yaml

from radiologyai.errors import ModelCardError, ModelNotFoundError, WeightsIntegrityError
from radiologyai.models.card import ModelCard
from radiologyai.models.registry import get_card, list_cards

VALID_SHA = "a" * 64


def make_card(**overrides) -> dict:
    base = {
        "card_id": "test-model",
        "display_name": "Modelo de Teste",
        "version": "1.0.0",
        "modality": "XR",
        "backend": "torch",
        "task": "multilabel-classification",
        "labels": ["Pneumonia", "Effusion"],
        "input_shape": [224, 224],
        "weights_uri": "file://weights.pt",
        "weights_sha256": VALID_SHA,
        "trained_on": ["PadChest"],
        "license": "CC-BY-4.0",
    }
    base.update(overrides)
    return base


@pytest.mark.requirement("REQ-020")
class TestSha256Validation:
    """O 'sha256' do legado continha g/h/i — não era hexadecimal."""

    def test_valid_hash_accepted(self):
        assert ModelCard.model_validate(make_card()).weights_sha256 == VALID_SHA

    def test_legacy_placeholder_rejected(self):
        legado = "a3b4c5d6e7f8g9h0i1j2k3l4m5n6o7p8q9r0s1t2u3v4w5x6y7z8a9b0c1d2e3f4"
        with pytest.raises(ValueError, match="hexadecimais"):
            ModelCard.model_validate(make_card(weights_sha256=legado))

    def test_wrong_length_rejected(self):
        with pytest.raises(ValueError, match="hexadecimais"):
            ModelCard.model_validate(make_card(weights_sha256="abc123"))

    def test_uppercase_rejected(self):
        with pytest.raises(ValueError, match="hexadecimais"):
            ModelCard.model_validate(make_card(weights_sha256="A" * 64))


class TestNoPerformanceFields:
    """Card não carrega métrica. Desempenho vive em artifacts/eval/."""

    @pytest.mark.parametrize("field", ["accuracy", "auc", "sensitivity", "specificity"])
    def test_performance_field_rejected(self, field):
        with pytest.raises(ValueError, match="[Ee]xtra"):
            ModelCard.model_validate(make_card(**{field: 0.92}))

    def test_no_evaluation_means_no_measured_performance(self):
        assert ModelCard.model_validate(make_card()).has_measured_performance is False

    def test_evaluation_run_marks_measured(self):
        card = ModelCard.model_validate(make_card(evaluation_runs=["run-2026-01-01"]))
        assert card.has_measured_performance is True


class TestWeightsIntegrity:
    def test_matching_hash_passes(self, tmp_path):
        blob = b"pesos falsos para teste"
        p = tmp_path / "w.pt"
        p.write_bytes(blob)
        card = ModelCard.model_validate(make_card(weights_sha256=hashlib.sha256(blob).hexdigest()))
        card.verify_weights(p)

    def test_mismatched_hash_raises(self, tmp_path):
        p = tmp_path / "w.pt"
        p.write_bytes(b"conteudo diferente")
        with pytest.raises(WeightsIntegrityError, match="integridade"):
            ModelCard.model_validate(make_card()).verify_weights(p)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(WeightsIntegrityError, match="não encontrado"):
            ModelCard.model_validate(make_card()).verify_weights(tmp_path / "x.pt")


class TestYamlLoading:
    def test_roundtrip(self, tmp_path):
        p = tmp_path / "c.yaml"
        p.write_text(yaml.safe_dump(make_card()), encoding="utf-8")
        assert ModelCard.from_yaml(p).card_id == "test-model"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ModelCardError, match="não encontrado"):
            ModelCard.from_yaml(tmp_path / "x.yaml")

    def test_invalid_card_names_path(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.safe_dump(make_card(weights_sha256="nope")), encoding="utf-8")
        with pytest.raises(ModelCardError, match="bad.yaml"):
            ModelCard.from_yaml(p)


class TestRegistry:
    def test_shipped_cards_are_valid(self):
        """Todo card distribuído carrega e valida — sha256 hex incluído."""
        for card in list_cards():
            assert len(card.weights_sha256) == 64

    def test_every_claimed_run_has_an_artifact_in_this_checkout(self):
        """Um card só alega desempenho apontando para artefato presente no repositório.

        Antes da primeira medição real este teste exigia evaluation_runs vazio.
        Agora existe um artefato: o invariante passa a ser que cada run listado
        tenha o seu artifacts/eval/<run>/metrics.json versionado.
        """
        from pathlib import Path

        root = Path(__file__).resolve().parents[3]
        for card in list_cards():
            for run_id in card.evaluation_runs:
                artifact = root / "artifacts" / "eval" / run_id / "metrics.json"
                assert artifact.is_file(), f"{card.card_id} referencia {run_id} sem artefato"

    def test_baseline_card_now_has_measured_performance(self):
        card = get_card("xrv-densenet121-pc")
        assert card.has_measured_performance
        assert "xrv-densenet121-pc__20260912T202549Z" in card.evaluation_runs

    def test_baseline_card_is_padchest_not_all(self):
        """A linha de base usa pesos PadChest, não '-all' (que viu o NIH).

        Avaliar '-all' no NIH seria in-distribution — uma fabricação sutil.
        """
        card = get_card("xrv-densenet121-pc")
        assert card.trained_on == ("PadChest",)
        assert "NIH" not in " ".join(card.trained_on)

    def test_get_unknown_card_raises(self):
        with pytest.raises(ModelNotFoundError, match="não encontrado"):
            get_card("inexistente")
