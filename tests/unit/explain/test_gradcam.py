"""Grad-CAM — o teste que o código do v1 reprovaria.

O `GradCAMExplainer` do legado nunca referenciava `self.model`: produzia
`cv2.Canny` borrado com ruído gaussiano. Um mapa assim é indiferente ao modelo,
logo permanece não-nulo mesmo quando o modelo não tem gradiente algum.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from radiologyai.errors import InferenceError
from radiologyai.explain.gradcam import GradCAM, overlay_heatmap

HAS_TORCH = importlib.util.find_spec("torch") is not None
torch_only = pytest.mark.skipif(not HAS_TORCH, reason="requer torch (extra [ml])")


@pytest.fixture
def tiny_cnn():
    """CNN mínima com uma camada convolucional nomeada."""
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(4, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = torch.relu(self.conv(x))
            return self.fc(self.pool(h).flatten(1))

    model = Net()
    model.eval()
    return model


@pytest.fixture
def input_tensor():
    import torch

    torch.manual_seed(0)
    return torch.linspace(0, 1, 16 * 16).reshape(1, 1, 16, 16).requires_grad_(True)


@torch_only
class TestGradCAMIsRealNotSimulated:
    @pytest.mark.requirement("REQ-060")
    def test_null_gradients_yield_null_map(self, tiny_cnn, input_tensor):
        """Modelo sem gradiente na camada-alvo DEVE produzir mapa nulo.

        Esta é a propriedade que distingue Grad-CAM real de saliência simulada.
        O código do v1 devolveria um mapa de bordas não-nulo aqui.
        """
        import torch

        with torch.no_grad():
            tiny_cnn.fc.weight.zero_()
            tiny_cnn.fc.bias.zero_()

        cam = GradCAM(tiny_cnn, tiny_cnn.conv, "conv")
        explanation = cam.explain(input_tensor, target_index=0, target_label="X")

        assert explanation.is_null, "gradiente nulo produziu mapa não-nulo — é simulação"
        assert explanation.heatmap.max() == 0.0

    def test_nonzero_gradients_yield_nonnull_map(self, tiny_cnn, input_tensor):
        cam = GradCAM(tiny_cnn, tiny_cnn.conv, "conv")
        assert not cam.explain(input_tensor, 0, "X").is_null

    def test_map_depends_on_the_model(self, tiny_cnn, input_tensor):
        """Trocar os pesos DEVE trocar o mapa. Saliência simulada não mudaria."""
        import torch

        cam = GradCAM(tiny_cnn, tiny_cnn.conv, "conv")
        before = cam.explain(input_tensor, 0, "X").heatmap.copy()

        with torch.no_grad():
            tiny_cnn.conv.weight.mul_(-3.0)
        after = cam.explain(input_tensor, 0, "X").heatmap

        assert not np.allclose(before, after), "mapa indiferente ao modelo"

    def test_different_targets_give_different_maps(self, tiny_cnn, input_tensor):
        cam = GradCAM(tiny_cnn, tiny_cnn.conv, "conv")
        a = cam.explain(input_tensor, 0, "A").heatmap
        b = cam.explain(input_tensor, 2, "B").heatmap
        assert not np.allclose(a, b)

    def test_no_randomness_in_output(self, tiny_cnn, input_tensor):
        """O legado somava np.random.normal ao mapa."""
        cam = GradCAM(tiny_cnn, tiny_cnn.conv, "conv")
        assert np.array_equal(
            cam.explain(input_tensor, 0, "X").heatmap,
            cam.explain(input_tensor, 0, "X").heatmap,
        )


@torch_only
class TestGradCAMProvenance:
    def test_records_layer_and_score(self, tiny_cnn, input_tensor):
        cam = GradCAM(tiny_cnn, tiny_cnn.conv, "conv")
        d = cam.explain(input_tensor, 1, "Pneumonia").describe()
        assert d["layer_name"] == "conv"
        assert d["target_label"] == "Pneumonia"
        assert d["method"] == "grad-cam"

    def test_wrong_layer_fails_loud(self, tiny_cnn, input_tensor):
        """Camada fora do forward → erro, nunca um mapa substituto."""
        from torch import nn

        orphan = nn.Conv2d(1, 2, 1)
        cam = GradCAM(tiny_cnn, orphan, "orphan")
        with pytest.raises(InferenceError, match="hooks não capturaram"):
            cam.explain(input_tensor, 0, "X")

    def test_hooks_removed_after_use(self, tiny_cnn, input_tensor):
        cam = GradCAM(tiny_cnn, tiny_cnn.conv, "conv")
        cam.explain(input_tensor, 0, "X")
        assert len(tiny_cnn.conv._forward_hooks) == 0
        assert len(tiny_cnn.conv._backward_hooks) == 0

    def test_hooks_removed_even_on_error(self, tiny_cnn, input_tensor):
        from torch import nn

        cam = GradCAM(tiny_cnn, nn.Conv2d(1, 2, 1), "orphan")
        with pytest.raises(InferenceError):
            cam.explain(input_tensor, 0, "X")
        assert len(tiny_cnn.conv._forward_hooks) == 0


class TestOverlay:
    def test_output_is_rgb_in_unit_range(self):
        image = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64)
        heatmap = np.linspace(0, 1, 8 * 8, dtype=np.float32).reshape(8, 8)
        out = overlay_heatmap(image, heatmap)
        assert out.shape == (64, 64, 3)
        assert 0.0 <= out.min() <= out.max() <= 1.0

    def test_resizes_heatmap_to_image(self):
        image = np.zeros((32, 48), dtype=np.float32)
        out = overlay_heatmap(image, np.ones((4, 4), dtype=np.float32))
        assert out.shape == (32, 48, 3)

    def test_rejects_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            overlay_heatmap(np.zeros((8, 8), np.float32), np.zeros((4, 4), np.float32), alpha=2.0)

    def test_rejects_non_2d_image(self):
        with pytest.raises(ValueError, match="2D"):
            overlay_heatmap(np.zeros((3, 8, 8), np.float32), np.zeros((4, 4), np.float32))


@torch_only
class TestBackendIntegration:
    def test_real_model_gradcam(self):
        from radiologyai.models.backends.xrv import TorchXRayVisionBackend

        backend = TorchXRayVisionBackend("densenet121-res224-pc")
        image = np.linspace(0, 1, 256 * 256, dtype=np.float32).reshape(256, 256)
        explanation = backend.gradcam(image, "Cardiomegaly")
        assert not explanation.is_null
        assert explanation.layer_name == "features.denseblock4"

    def test_untrained_head_refuses_explanation(self):
        """Explicar uma saída sem significado produziria um mapa enganoso."""
        from radiologyai.models.backends.xrv import TorchXRayVisionBackend

        backend = TorchXRayVisionBackend("densenet121-res224-pc")
        image = np.zeros((256, 256), dtype=np.float32)
        with pytest.raises(ValueError, match="não treinada"):
            backend.gradcam(image, "__untrained_14")

    def test_unknown_label_refused(self):
        from radiologyai.models.backends.xrv import TorchXRayVisionBackend

        backend = TorchXRayVisionBackend("densenet121-res224-pc")
        with pytest.raises(ValueError, match="não é saída"):
            backend.gradcam(np.zeros((256, 256), dtype=np.float32), "Inexistente")
