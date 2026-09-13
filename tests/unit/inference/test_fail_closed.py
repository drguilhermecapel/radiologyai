"""O motor de inferência falha fechada — nunca fabrica predição.

Contraparte direta de `legacy/src/medai_inference_system.py`, que devolvia
`_create_dummy_model()` sem pesos e caía numa heurística de OpenCV.
"""

from __future__ import annotations

import hashlib

import pytest

from radiologyai.errors import (
    BackendUnavailableError,
    ModelNotFoundError,
    OutOfScopeError,
    WeightsIntegrityError,
)
from radiologyai.inference.engine import InferenceEngine, sha256_file
from radiologyai.models.card import ModelCard

BLOB = b"pesos sinteticos para teste de integridade"


def card_for(sha: str, backend: str = "torch") -> ModelCard:
    return ModelCard.model_validate(
        {
            "card_id": "xr-test",
            "display_name": "XR Teste",
            "version": "0.1.0",
            "modality": "XR",
            "backend": backend,
            "task": "multilabel-classification",
            "labels": ["Pneumonia"],
            "input_shape": [224, 224],
            "weights_uri": "file://w.pt",
            "weights_sha256": sha,
            "trained_on": ["PadChest"],
            "license": "CC-BY-4.0",
        }
    )


@pytest.fixture
def weights(tmp_path):
    p = tmp_path / "w.pt"
    p.write_bytes(BLOB)
    return p


@pytest.mark.requirement("REQ-030")
class TestFailClosed:
    def test_missing_weights_raises(self, tmp_path):
        with pytest.raises(ModelNotFoundError, match="não encontrados"):
            InferenceEngine(card_for("a" * 64), tmp_path / "ausente.pt")

    def test_missing_weights_message_forbids_substitute(self, tmp_path):
        with pytest.raises(ModelNotFoundError, match="Nenhum modelo substituto"):
            InferenceEngine(card_for("a" * 64), tmp_path / "ausente.pt")

    def test_hash_mismatch_raises_at_construction(self, weights):
        with pytest.raises(WeightsIntegrityError, match="integridade"):
            InferenceEngine(card_for("b" * 64), weights)

    def test_hash_mismatch_blocks_inference(self, weights):
        with pytest.raises(WeightsIntegrityError, match="Nenhuma inferência"):
            InferenceEngine(card_for("b" * 64), weights)

    def test_valid_hash_constructs(self, weights):
        engine = InferenceEngine(card_for(hashlib.sha256(BLOB).hexdigest()), weights)
        assert engine.card.card_id == "xr-test"

    def test_sha256_file_matches_hashlib(self, weights):
        assert sha256_file(weights) == hashlib.sha256(BLOB).hexdigest()


@pytest.mark.requirement("REQ-031")
class TestScopeGateRunsBeforeModel:
    def test_out_of_scope_input_rejected(self, weights, lateral_dataset, write_dicom):
        engine = InferenceEngine(card_for(hashlib.sha256(BLOB).hexdigest()), weights)
        with pytest.raises(OutOfScopeError):
            engine.predict(write_dicom(lateral_dataset))

    def test_validate_study_needs_no_pixels(self, weights, cr_dataset, write_dicom):
        engine = InferenceEngine(card_for(hashlib.sha256(BLOB).hexdigest()), weights)
        assert engine.validate_study(write_dicom(cr_dataset)).accepted

    def test_in_scope_input_stops_without_fabricating(self, weights, cr_dataset, write_dicom):
        """Entrada válida atravessa o gate e para — sem inventar resultado.

        Qualquer que seja o motivo (backend ausente ou ainda não implementado),
        o resultado é sempre uma exceção. Nunca um escore.
        """
        engine = InferenceEngine(card_for(hashlib.sha256(BLOB).hexdigest()), weights)
        with pytest.raises(BackendUnavailableError):
            engine.predict(write_dicom(cr_dataset))

    @pytest.mark.parametrize(("backend", "package"), [("torch", "torch"), ("onnx", "onnxruntime")])
    def test_missing_backend_package_names_the_remedy(
        self, weights, cr_dataset, write_dicom, backend, package
    ):
        """Quando o pacote do backend falta, a mensagem diz exatamente o que instalar."""
        pytest.importorskip  # noqa: B018 - marcador de intenção
        import importlib.util

        if importlib.util.find_spec(package) is not None:
            pytest.skip(f"{package} está instalado; este teste cobre o caso ausente")

        engine = InferenceEngine(
            card_for(hashlib.sha256(BLOB).hexdigest(), backend=backend), weights
        )
        with pytest.raises(BackendUnavailableError) as exc:
            engine.predict(write_dicom(cr_dataset))
        message = str(exc.value)
        assert package in message
        assert "radiologyai[ml]" in message
        assert "Nenhum caminho alternativo" in message
