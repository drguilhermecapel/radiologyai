"""Código vendorizado do mammoscreen/mammo-crop x código ORIGINAL do cartão (REQ-142).

``mg_original_reference.json`` foi gerado UMA vez com o ``modeling.py`` original
(Hugging Face, revisões fixadas) + albumentations 2.0.8 + OpenCV, sobre imagens
sintéticas determinísticas (ver ``synthetic`` abaixo, idêntica à do gerador).
Os testes de pré-processamento não precisam de pesos; os de saída usam os
pesos reais e pulam sem eles.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from typing import Any

import numpy as np
import pytest
from mg_support import REFERENCE, WEIGHTS

HAVE_BACKEND = all(importlib.util.find_spec(m) for m in ("torch", "timm", "cv2", "safetensors"))
HAVE_WEIGHTS = (WEIGHTS / "mammoscreen" / "model.safetensors").exists() and (
    WEIGHTS / "mammo-crop" / "model.safetensors"
).exists()
needs_backend = pytest.mark.skipif(not HAVE_BACKEND, reason="torch/timm/cv2 ausentes")
needs_weights = pytest.mark.skipif(
    not (HAVE_BACKEND and HAVE_WEIGHTS and REFERENCE.exists()), reason="pesos reais ausentes"
)


def synthetic(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    img = 40 + 120 * (xx / w) + 60 * np.sin(yy / 17.0) ** 2
    img[(yy - h * 0.55) ** 2 + (xx - w * 0.3) ** 2 < (min(h, w) * 0.12) ** 2] = 250
    img[:, int(w * 0.8) :] = 0
    return np.clip(img, 0, 255).astype(np.uint8)


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _cases() -> list[dict[str, Any]]:
    data: dict[str, Any] = json.loads(REFERENCE.read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = data["cases"]
    return cases


@needs_backend
@pytest.mark.skipif(not REFERENCE.exists(), reason="referência do original ausente")
@pytest.mark.requirement("REQ-142")
def test_mg_vendor_preprocess_bit_identical_to_original() -> None:
    from radiologyai.analysis.vendor.mammoscreen.model import (
        CROP_INPUT_SIZE,
        ENSEMBLE_IMAGE_SIZES,
        ENSEMBLE_PAD_TO_ASPECT,
        longest_max_size_pad,
        pad_to_aspect_ratio,
        resize_u8,
    )

    for case in _cases():
        img = synthetic(*case["shape"])
        assert _sha(img) == case["image_sha256"]
        assert _sha(resize_u8(img, CROP_INPUT_SIZE, CROP_INPUT_SIZE)) == case["crop_pre_sha256"]
        for i, ((h, w), pad) in enumerate(
            zip(ENSEMBLE_IMAGE_SIZES, ENSEMBLE_PAD_TO_ASPECT, strict=True)
        ):
            pre = (
                resize_u8(pad_to_aspect_ratio(img, h / w), h, w)
                if pad
                else longest_max_size_pad(img, h, w)
            )
            assert list(pre[None].shape) == case["net_pre_shape"][i]
            assert _sha(pre) == case[f"net{i}_pre_sha256"], f"net{i}"


@pytest.mark.requirement("REQ-142")
def test_mg_vendor_pad_to_aspect_ratio_logic() -> None:
    if not HAVE_BACKEND:
        pytest.skip("torch/timm/cv2 ausentes")
    from radiologyai.analysis.vendor.mammoscreen.model import pad_to_aspect_ratio

    img = np.ones((100, 20), dtype=np.uint8)
    out = pad_to_aspect_ratio(img, 2.0)  # h/w = 5 > 2 => largura 50, 15 + 15
    assert out.shape == (100, 50)
    assert out[:, :15].sum() == 0
    assert out[:, 35:].sum() == 0
    assert out[:, 15:35].all()
    wide = np.ones((20, 100), dtype=np.uint8)
    assert pad_to_aspect_ratio(wide, 2.0) is wide


@needs_backend
@pytest.mark.requirement("REQ-142")
def test_mg_vendor_gem_matches_original_formula() -> None:
    import torch

    from radiologyai.analysis.vendor.mammoscreen.model import GeM

    x = torch.linspace(-1.0, 3.0, 2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    got = GeM(p=3.0)(x)
    want = x.clamp(min=1e-6).pow(3).mean(dim=(2, 3)).pow(1.0 / 3)
    torch.testing.assert_close(got, want)


@needs_weights
@pytest.mark.slow
@pytest.mark.requirement("REQ-142")
def test_mg_vendor_outputs_match_original_with_real_weights() -> None:
    import torch
    from safetensors.torch import load_file

    from radiologyai.analysis.vendor.mammoscreen.model import MammoCropNet, MammoEnsemble

    torch.set_num_threads(4)
    crop = MammoCropNet()
    crop.load_state_dict(load_file(str(WEIGHTS / "mammo-crop" / "model.safetensors")), strict=True)
    crop.eval()
    ens = MammoEnsemble()
    ens.load_state_dict(load_file(str(WEIGHTS / "mammoscreen" / "model.safetensors")), strict=True)
    ens.eval()
    for case in _cases():
        img = synthetic(*case["shape"])
        assert list(crop.crop_box(img)) == case["crop_box"]
        cancer, density, _ = ens.breast([img, img[:, ::-1].copy()])
        assert cancer == pytest.approx(case["cancer"], abs=1e-5)
        np.testing.assert_allclose(density, case["density"], atol=1e-5)


@pytest.mark.requirement("REQ-142")
def test_mg_backend_missing_names_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    from radiologyai.analysis.adapters import mg_common
    from radiologyai.errors import BackendUnavailableError

    real = importlib.util.find_spec

    def fake(name: str, *args: Any, **kwargs: Any) -> Any:
        return None if name == "cv2" else real(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake)
    with pytest.raises(BackendUnavailableError, match="mammography"):
        mg_common.require_mg_backend()
