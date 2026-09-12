"""Executor de avaliação — proveniência, detecção de vazamento e determinismo."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from radiologyai.data.manifest import Manifest, ManifestRow
from radiologyai.evaluation.runner import (
    EvaluationConfig,
    check_leakage,
    run_evaluation,
)
from radiologyai.models.card import ModelCard

MODEL_LABELS = ("Pneumonia", "Effusion", "Cardiomegaly", "Lung Opacity")
MANIFEST_LABELS = ("Pneumonia", "Effusion", "Cardiomegaly")


class FakeBackend:
    """Backend determinístico para testar o executor sem torch."""

    def __init__(self, labels: tuple[str, ...] = MODEL_LABELS) -> None:
        self._labels = labels

    @property
    def labels(self) -> tuple[str, ...]:
        return self._labels

    def predict(self, image):  # noqa: ANN001, ANN201, ARG002
        return np.zeros(len(self._labels), dtype=np.float32)

    def predict_batch(self, images):  # noqa: ANN001, ANN201
        return np.stack([self.predict(i) for i in images])

    def describe(self) -> dict[str, Any]:
        return {"backend": "fake", "weights_name": "fake", "trained_on": []}


def make_card(trained_on: tuple[str, ...] = ("PadChest",)) -> ModelCard:
    return ModelCard.model_validate(
        {
            "card_id": "fake-model",
            "display_name": "Fake",
            "version": "1.0.0",
            "modality": "XR",
            "backend": "torch",
            "task": "multilabel-classification",
            "labels": list(MODEL_LABELS),
            "input_shape": [224, 224],
            "weights_uri": "file://w.pt",
            "weights_sha256": "a" * 64,
            "trained_on": list(trained_on),
            "license": "MIT",
        }
    )


def make_manifest(n: int = 80, name: str = "NIH ChestX-ray14") -> Manifest:
    rows = [
        ManifestRow(
            image_id=f"i{i:04d}.png",
            patient_id=str(i // 2),
            labels={
                "Pneumonia": int(i % 3 == 0),
                "Effusion": int(i % 2 == 0),
                "Cardiomegaly": int(i < 5),  # suporte insuficiente de propósito
            },
            view_position="PA" if i % 2 else "AP",
            patient_sex="M" if i % 2 else "F",
            patient_age_years=25.0 + (i % 60),
        )
        for i in range(n)
    ]
    return Manifest(name=name, split="test", label_names=MANIFEST_LABELS, rows=rows)


def make_scores(manifest: Manifest, seed: int = 3) -> np.ndarray:
    """Escores correlacionados com a verdade, para produzir AUROC não-degenerado."""
    rng = np.random.default_rng(seed)
    truth = np.array(manifest.label_matrix())
    scores = rng.uniform(0, 0.5, size=(len(manifest), len(MODEL_LABELS))).astype(np.float32)
    for i, name in enumerate(MANIFEST_LABELS):
        scores[:, i] += 0.4 * truth[:, list(MANIFEST_LABELS).index(name)]
    return np.clip(scores, 0, 1)


class TestLeakageDetection:
    def test_padchest_model_on_nih_is_external(self):
        assert check_leakage(make_card(("PadChest",)), make_manifest()) == "externo"

    def test_nih_trained_model_on_nih_is_in_distribution(self):
        """Avaliar pesos '-all' (que viram o NIH) no NIH não é validação externa."""
        card = make_card(("NIH ChestX-ray14", "PadChest", "CheXpert"))
        assert check_leakage(card, make_manifest()) == "in-distribution"

    def test_model_card_always_declares_training_data(self):
        """trained_on é obrigatório: a checagem de vazamento nunca fica sem base."""
        with pytest.raises(ValueError, match="at least 1 item"):
            make_card(())

    def test_different_dataset_is_external(self):
        assert check_leakage(make_card(("PadChest",)), make_manifest(name="CheXpert")) == "externo"


class TestRunEvaluation:
    @pytest.fixture
    def artifact(self, tmp_path):
        m = make_manifest()
        run_dir = run_evaluation(
            card=make_card(),
            backend=FakeBackend(),
            manifest=m,
            y_score=make_scores(m),
            output_dir=tmp_path,
            config=EvaluationConfig(n_bootstrap=100),
            dataset_limitations=("Limitação de teste.",),
        )
        return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")), run_dir

    def test_writes_metrics_and_manifest(self, artifact):
        _, run_dir = artifact
        assert (run_dir / "metrics.json").is_file()
        assert (run_dir / "manifest_used.csv").is_file()

    @pytest.mark.requirement("REQ-050")
    def test_records_full_provenance(self, artifact):
        """git_sha, weights_sha256, manifest_sha256, seed e versões de biblioteca."""
        m, _ = artifact
        assert m["environment"]["git_sha"]
        assert m["environment"]["libraries"]
        assert len(m["model"]["weights_sha256"]) == 64
        assert len(m["dataset"]["manifest_sha256"]) == 64
        assert m["config"]["seed"] == 20260101

    def test_reports_ci_for_each_label(self, artifact):
        m, _ = artifact
        for entry in m["per_label"].values():
            lo, hi = entry["auroc_ci95"]
            assert lo <= entry["auroc"] <= hi
            assert "bootstrap" in entry["ci_method"]

    def test_low_support_label_is_declared_not_dropped(self, artifact):
        """Cardiomegaly tem 5 positivos: não é avaliado, mas é declarado."""
        m, _ = artifact
        not_evaluated = {x["label"] for x in m["not_evaluated"]}
        assert "Cardiomegaly" in not_evaluated
        assert "Cardiomegaly" not in m["per_label"]

    def test_model_label_absent_from_dataset_is_declared(self, artifact):
        """'Lung Opacity' existe no modelo e não no NIH — declarado, não descartado."""
        m, _ = artifact
        reasons = {x["label"]: x["reason"] for x in m["not_evaluated"]}
        assert "Lung Opacity" in reasons
        assert "sem rótulo correspondente" in reasons["Lung Opacity"]

    def test_subgroups_reported(self, artifact):
        m, _ = artifact
        assert set(m["subgroups"]) == {"sex", "age_band", "view_position"}

    def test_limitations_include_no_clinical_validation(self, artifact):
        m, _ = artifact
        joined = " ".join(m["limitations"])
        assert "Limitação de teste." in joined
        assert "NÃO constitui validação clínica" in joined
        assert "NÃO são probabilidade de doença" in joined

    def test_external_flag_set(self, artifact):
        m, _ = artifact
        assert m["dataset"]["external_to_training_data"] is True

    @pytest.mark.requirement("REQ-050")
    def test_same_seed_is_reproducible(self, tmp_path):
        m = make_manifest()
        scores = make_scores(m)
        results = []
        for i in range(2):
            run_dir = run_evaluation(
                card=make_card(),
                backend=FakeBackend(),
                manifest=m,
                y_score=scores,
                output_dir=tmp_path / str(i),
                config=EvaluationConfig(n_bootstrap=100),
            )
            data = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            for key in ("run_id", "timestamp_utc"):
                data.pop(key)
            results.append(data)
        assert results[0]["per_label"] == results[1]["per_label"]


class TestInDistributionWarning:
    def test_leakage_warning_is_first_limitation(self, tmp_path):
        m = make_manifest()
        run_dir = run_evaluation(
            card=make_card(("NIH ChestX-ray14",)),
            backend=FakeBackend(),
            manifest=m,
            y_score=make_scores(m),
            output_dir=tmp_path,
            config=EvaluationConfig(n_bootstrap=50),
        )
        data = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        assert data["dataset"]["external_to_training_data"] is False
        assert "ATENÇÃO" in data["limitations"][0]
        assert "NÃO é validação externa" in data["limitations"][0]


class TestLoadPngCanonical:
    """Normalização pelo fundo de escala, como o torchxrayvision treina."""

    def test_uint8_is_divided_by_255_not_stretched(self, tmp_path):
        import numpy as np
        from PIL import Image

        from radiologyai.evaluation import load_png

        arr = np.full((8, 8), 51, dtype=np.uint8)
        arr[0, 0] = 204
        Image.fromarray(arr).save(tmp_path / "a.png")
        out = load_png(tmp_path / "a.png")
        assert out[1, 1] == pytest.approx(51 / 255)
        assert out[0, 0] == pytest.approx(204 / 255)
        assert out.max() < 1.0, "min-max por imagem esticaria o máximo para 1.0"

    def test_uint16_is_divided_by_65535(self, tmp_path):
        import numpy as np
        from PIL import Image

        from radiologyai.evaluation import load_png

        arr = np.full((4, 4), 6553, dtype=np.uint16)
        Image.fromarray(arr).save(tmp_path / "b.png")
        assert load_png(tmp_path / "b.png")[0, 0] == pytest.approx(6553 / 65535, abs=1e-6)

    def test_output_in_unit_range_float32(self, tmp_path):
        import numpy as np
        from PIL import Image

        from radiologyai.evaluation import load_png

        Image.fromarray(np.arange(64, dtype=np.uint8).reshape(8, 8)).save(tmp_path / "c.png")
        out = load_png(tmp_path / "c.png")
        assert out.dtype == np.float32
        assert 0.0 <= out.min() <= out.max() <= 1.0
