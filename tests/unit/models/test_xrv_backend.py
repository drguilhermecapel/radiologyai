"""Backend torchxrayvision — pesos reais, sem o mapeamento perigoso do legado.

Estes testes são pulados quando torch/torchxrayvision não estão instalados
(extra ``[ml]``). Quando estão, baixam ~28 MB de pesos na primeira execução.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

HAS_ML = all(importlib.util.find_spec(m) is not None for m in ("torch", "torchxrayvision"))
pytestmark = pytest.mark.skipif(not HAS_ML, reason="requer o extra [ml]")

WEIGHTS = "densenet121-res224-pc"


@pytest.fixture(scope="module")
def backend():
    from radiologyai.models.backends.xrv import TorchXRayVisionBackend

    return TorchXRayVisionBackend(WEIGHTS)


@pytest.fixture
def image() -> np.ndarray:
    """Rampa determinística em [0, 1] — nunca aleatória."""
    return np.linspace(0, 1, 512 * 512, dtype=np.float32).reshape(512, 512)


class TestLabelIntegrity:
    def test_trained_positions_match_hardcoded_order(self, backend):
        """Se o torchxrayvision mudar a ORDEM, isto DEVE quebrar, não degradar.

        Uma divergência silenciosa entre versões trocaria as patologias entre si:
        um pneumotórax passaria a ser reportado no lugar de outro achado.
        """
        from radiologyai.modalities.xr.labels import XRV_PATHOLOGIES

        for i, name in enumerate(backend.labels):
            if i in backend.untrained_indices:
                continue
            assert name == XRV_PATHOLOGIES[i], f"ordem divergiu no índice {i}"

    @pytest.mark.requirement("REQ-021")
    def test_untrained_heads_are_marked_not_silently_named(self, backend):
        """Os pesos PadChest não treinaram 3 das 18 saídas.

        Se usássemos ``default_pathologies`` como se fosse a lista do modelo,
        reportaríamos AUROC de 'Lung Lesion' a partir de uma cabeça não treinada.
        """
        assert backend.untrained_indices == (14, 16, 17)
        for i in backend.untrained_indices:
            assert backend.labels[i].startswith("__untrained_")

    def test_trained_labels_exclude_untrained(self, backend):
        assert len(backend.trained_labels) == 15
        assert all(not n.startswith("__untrained_") for n in backend.trained_labels)

    def test_all_fourteen_nih_labels_are_trained_in_pc_weights(self, backend):
        """Os 14 rótulos do NIH têm cabeça treinada nos pesos PadChest."""
        from radiologyai.modalities.xr.labels import NIH_CXR14_LABELS

        for label in NIH_CXR14_LABELS:
            assert label in backend.trained_labels, f"{label} sem cabeça treinada"

    def test_all_nih_labels_resolvable(self, backend):
        from radiologyai.modalities.xr.labels import xrv_to_nih_indices

        for label, idx in xrv_to_nih_indices().items():
            assert backend.labels[idx] == label

    def test_eighteen_outputs(self, backend):
        assert len(backend.labels) == 18


class TestNoDangerousCollapsing:
    """O legado mapeava Pneumothorax->pneumonia e Cardiomegaly->normal."""

    def test_pneumothorax_reported_natively(self, backend, image):
        scores = dict(zip(backend.labels, backend.predict(image), strict=True))
        assert "Pneumothorax" in scores
        assert scores["Pneumothorax"] != scores["Pneumonia"]

    def test_cardiomegaly_reported_natively(self, backend):
        assert "Cardiomegaly" in backend.labels

    def test_no_normal_category_invented(self, backend):
        """O sistema nunca afirma normalidade (REG-01 §4)."""
        assert not any(label.lower() in ("normal", "no finding") for label in backend.labels)

    def test_backend_exposes_no_pathology_mapping(self, backend):
        assert not hasattr(backend, "pathology_mapping")
        assert not hasattr(backend, "clinical_thresholds")


class TestInference:
    def test_output_shape_and_range(self, backend, image):
        out = backend.predict(image)
        assert out.shape == (18,)
        assert 0.0 <= out.min() <= out.max() <= 1.0

    def test_deterministic(self, backend, image):
        assert np.array_equal(backend.predict(image), backend.predict(image))

    def test_batch_matches_single(self, backend, image):
        batch = backend.predict_batch([image, image])
        assert batch.shape == (2, 18)
        assert np.allclose(batch[0], backend.predict(image), atol=1e-5)

    def test_rejects_non_2d(self, backend):
        with pytest.raises(ValueError, match="2D"):
            backend.predict(np.zeros((3, 64, 64), dtype=np.float32))

    def test_declares_training_data(self, backend):
        """Necessário para a detecção de vazamento."""
        assert backend.trained_on == ("PadChest",)

    def test_describe_records_versions(self, backend):
        d = backend.describe()
        assert d["weights_name"] == WEIGHTS
        assert d["torch_version"]
        assert d["trained_on"] == ["PadChest"]


class TestWeightsIntegrity:
    def test_shipped_card_matches_real_weights(self):
        """O sha256 do card confere com o arquivo publicado de verdade."""
        from radiologyai.models import get_card
        from radiologyai.models.backends.xrv import weights_cache_path

        card = get_card("xrv-densenet121-pc")
        card.verify_weights(weights_cache_path(WEIGHTS))

    def test_build_backend_verifies_integrity(self):
        from radiologyai.models import get_card
        from radiologyai.models.backends import build_backend

        assert build_backend(get_card("xrv-densenet121-pc")).labels

    def test_all_weights_not_shipped_as_card(self):
        """Não distribuímos card para '-all': avaliá-lo no NIH seria in-distribution."""
        from radiologyai.models import list_cards

        assert not any(c.card_id.endswith("-all") for c in list_cards())


class TestDevicePlacement:
    """A entrada vai para o dispositivo do modelo — por construção.

    Primeira execução em GPU no Colab: ``to('cuda')`` movia os pesos e a
    entrada ficava na CPU (``Input type (torch.FloatTensor) and weight type
    (torch.cuda.FloatTensor) should be the same``). Sem GPU no CI, o que se
    pode verificar é que o dispositivo é lido dos parâmetros e que toda
    entrada passa por ``.to(self.device)`` antes do forward.
    """

    def test_device_is_read_from_parameters(self, backend):

        assert backend.device == next(backend._model.parameters()).device
        assert backend.device.type == "cpu"

    def test_no_separate_device_state(self, backend):
        assert not hasattr(backend, "_device")

    def test_to_returns_self_and_keeps_consistency(self, backend):
        assert backend.to("cpu") is backend
        assert backend.device.type == "cpu"

    @pytest.mark.parametrize("method", ["predict", "predict_batch"])
    def test_forward_receives_tensor_on_model_device(self, backend, image, method, monkeypatch):
        """Espião no forward: o tensor recebido está no mesmo dispositivo dos pesos."""
        seen = {}
        real_model = backend._model

        class Spy:
            def __call__(self, x):
                seen["device"] = x.device
                return real_model(x)

            def parameters(self):
                return real_model.parameters()

        monkeypatch.setattr(backend, "_model", Spy())
        getattr(backend, method)(image if method == "predict" else [image, image])
        assert seen["device"] == next(real_model.parameters()).device

    def test_gradcam_input_on_model_device(self, backend, image):
        """A entrada do Grad-CAM é movida antes de requires_grad_, e segue folha."""
        explanation = backend.gradcam(image, "Cardiomegaly")
        assert not explanation.is_null
