"""O executor de avaliação decide vazamento por chave de linhagem (REQ-184)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from radiologyai.data.manifest import Manifest, ManifestRow
from radiologyai.errors import EvaluationError
from radiologyai.evaluation.regression import check_baseline_regression
from radiologyai.evaluation.runner import (
    EvaluationConfig,
    check_leakage,
    dataset_key,
    run_evaluation,
)
from radiologyai.models.card import ModelCard

LABELS = ("Pneumonia", "Effusion")


class FakeBackend:
    labels = LABELS

    def describe(self):
        return {"backend": "fake"}


def card(trained_on: tuple[str, ...]) -> ModelCard:
    return ModelCard.model_validate(
        {
            "card_id": "fake-card",
            "display_name": "fake",
            "version": "1",
            "modality": "XR",
            "backend": "torch",
            "task": "multilabel-classification",
            "labels": list(LABELS),
            "input_shape": [1, 224, 224],
            "weights_uri": "file:///x.pt",
            "weights_sha256": "a" * 64,
            "trained_on": list(trained_on),
            "license": "Apache-2.0",
        }
    )


def manifest(name: str, n: int = 80) -> Manifest:
    rows = [
        ManifestRow(
            image_id=f"i{i}",
            patient_id=str(i // 2),
            labels={"Pneumonia": int(i % 3 == 0), "Effusion": int(i % 2 == 0)},
        )
        for i in range(n)
    ]
    return Manifest(name=name, split="teste", label_names=LABELS, rows=rows)


class TestLineageKeys:
    @pytest.mark.requirement("REQ-184")
    @pytest.mark.parametrize("evaluated", ["siim-acr-pneumothorax", "rsna-pneumonia"])
    def test_nih_derived_dataset_is_in_distribution_for_nih_model(self, evaluated):
        """Por substring, estes saíam 'externo' (são anotações de imagens do NIH)."""
        nih = card(("NIH ChestX-ray14",))
        assert check_leakage(nih, manifest(evaluated)) == "in-distribution"

    @pytest.mark.requirement("REQ-184")
    def test_exact_aliases_and_keys(self):
        assert dataset_key("NIH ChestX-ray14") == "nih-cxr14"
        assert dataset_key("nih-cxr14") == "nih-cxr14"
        assert dataset_key("  PadChest ") == "padchest"

    @pytest.mark.requirement("REQ-184")
    @pytest.mark.parametrize("name", ["NIH", "nih chest", "ChestX-ray14 (NIH)", "MeuDataset"])
    def test_unknown_names_are_errors_not_external(self, name):
        with pytest.raises(EvaluationError, match="chave de linhagem"):
            check_leakage(card((name,)), manifest("NIH ChestX-ray14"))
        with pytest.raises(EvaluationError, match="chave de linhagem"):
            check_leakage(card(("PadChest",)), manifest(name))

    @pytest.mark.requirement("REQ-184")
    def test_explicit_key_wins_over_manifest_name(self):
        nih = card(("NIH ChestX-ray14",))
        m = manifest("nome livre qualquer")
        assert check_leakage(nih, m, evaluated_key="siim-acr-pneumothorax") == "in-distribution"

    @pytest.mark.requirement("REQ-184")
    def test_possible_overlap_is_not_external_in_artifact(self, tmp_path):
        rng = np.random.default_rng(0)
        m = manifest("breast-lesions-usg")
        run_dir = run_evaluation(
            card=card(("busi",)),
            backend=FakeBackend(),
            manifest=m,
            y_score=rng.uniform(0, 1, (len(m), len(LABELS))),
            output_dir=tmp_path,
            config=EvaluationConfig(n_bootstrap=30),
        )
        data = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        assert data["dataset"]["leakage"] == "possible"
        assert data["dataset"]["leakage_status"] == "possivel"
        assert data["dataset"]["external_to_training_data"] is False
        assert data["dataset"]["lineage_key"] == "breast-lesions-usg"
        assert "NÃO é validação externa" in data["limitations"][0]
        regression = check_baseline_regression(data, data, require_external_dataset=True)
        assert not regression.passed
