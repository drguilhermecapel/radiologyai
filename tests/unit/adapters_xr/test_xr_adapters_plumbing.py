"""Encanamento dos adaptadores de RX sem pesos reais (REQ-121..REQ-127).

Sem torch: as funções que executam rede (probabilidade de incidência, erro de
reconstrução, segmentação) são substituídas no teste por funções que devolvem
números ou máscaras GEOMÉTRICAS fixos; confere-se a decisão do portão, a
serialização em ``extras["gate"]``, o ICT com geometria e as abstenções. Com
torch: módulos mínimos com semente fixa no lugar das redes (DenseNet com
``features``/``classifier``, Dinov2 reduzido, CLIP com ``encode_*``) para
exercitar ``load``/``logits``/``cam``/``explain``. Nenhum desses números tem
significado clínico; nada sai de ``tests/``.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from radiologyai.analysis.adapters import xr_anatomy, xr_gates, xr_xrv
from radiologyai.analysis.adapters.xr_anatomy import PSPNET_TARGETS, ChestXDetAnatomyAdapter
from radiologyai.analysis.adapters.xr_chexzero import ChexzeroAdapter
from radiologyai.analysis.adapters.xr_gates import XrvAutoencoderGateAdapter, XrvViewGateAdapter
from radiologyai.analysis.adapters.xr_raddino import (
    RADDINO_ENCODER_ID,
    RadDinoLinearProbeAdapter,
    raddino_canonical,
)
from radiologyai.analysis.base import AdapterOutput, ModelInputs, ResolvedWeights
from radiologyai.analysis.manifest import ModelManifest, load_manifest
from radiologyai.analysis.registry import PACKAGE_MANIFEST_DIR
from radiologyai.analysis.types import AbstentionReason, ImageData, ImageKind, ImageMeta, Modality
from radiologyai.catalog.schema import ExamType
from radiologyai.errors import InferenceError, ManifestError, UnsupportedInputError

GATES = ("xrv-view-xinario", "xrv-ae-ood")


def _have(*mods: str) -> bool:
    return all(importlib.util.find_spec(m) is not None for m in mods)


def _manifest(model_id: str) -> ModelManifest:
    manifest, _ = load_manifest(PACKAGE_MANIFEST_DIR / f"{model_id}.yaml")
    return manifest


def _cxr(view: str | None = "PA", size: int = 512, tag: str = "") -> ImageData:
    pixels = np.tile(np.linspace(0, 4095, size, dtype=np.float32), (size, 1))
    meta = ImageMeta(
        source_format="dicom",
        content_sha256=hashlib.sha256(f"{view}{size}{tag}".encode()).hexdigest(),
        view_position=view,
        patient_age_years=50.0,
    )
    return ImageData(
        pixels=pixels,
        kind=ImageKind.PROJECTION,
        modality=Modality.DX,
        meta=meta,
        value_range=(0.0, 4095.0),
    )


def _passing_gates(image: ImageData) -> dict[str, AdapterOutput]:
    def gate(name: str, reason: AbstentionReason) -> AdapterOutput:
        return AdapterOutput(
            extras={
                "gate": {
                    "image_sha256": image.sha256,
                    "name": name,
                    "passed": True,
                    "reason": reason.value,
                    "value": 0.9,
                    "threshold": 0.5,
                }
            }
        )

    return {
        GATES[0]: gate("view_frontal", AbstentionReason.VIEW),
        GATES[1]: gate("ae_reconstruction_mae", AbstentionReason.OOD),
    }


def _inputs(image: ImageData, gates: bool = True) -> ModelInputs:
    return ModelInputs(
        images=(image,), exam_type=ExamType.CXR, upstream=_passing_gates(image) if gates else {}
    )


# --------------------------------------------------------------------------- #
# Portões (REQ-122) — decisão e serialização
# --------------------------------------------------------------------------- #
@pytest.mark.requirement("REQ-122")
def test_xr_ood_gate_requires_measured_threshold() -> None:
    manifest = _manifest("xrv-ae-ood")
    for bad in (None, 0.0, float("nan"), -1.0):
        cfg = {k: v for k, v in manifest.config.items() if k != "ood_mae_max"}
        if bad is not None:
            cfg["ood_mae_max"] = bad
        with pytest.raises(ManifestError, match="ood_mae_max"):
            XrvAutoencoderGateAdapter(manifest.model_copy(update={"config": cfg}))
    assert XrvAutoencoderGateAdapter(manifest).threshold == pytest.approx(0.029835)


@pytest.mark.requirement("REQ-122")
@pytest.mark.parametrize(("probs", "passed"), [((0.8, 0.2), True), ((0.3, 0.7), False)])
def test_xr_view_gate_decision_is_argmax(
    monkeypatch: pytest.MonkeyPatch, probs: tuple[float, float], passed: bool
) -> None:
    adapter = XrvViewGateAdapter(_manifest("xrv-view-xinario"))
    image = _cxr()
    monkeypatch.setattr(adapter, "_batch", lambda _inputs, size: ("lote", size))
    monkeypatch.setattr(xr_gates, "view_probabilities", lambda _m, _b: np.array([probs]))
    gate = adapter.predict(_inputs(image, gates=False)).extras["gate"]
    assert gate["passed"] is passed
    assert gate["image_sha256"] == image.sha256
    assert gate["reason"] == AbstentionReason.VIEW.value
    assert (gate["detail"] is None) is passed
    adapter.unload()


@pytest.mark.requirement("REQ-122")
@pytest.mark.parametrize(("mae", "passed"), [(0.01, True), (0.2, False)])
def test_xr_ood_gate_decision_uses_manifest_threshold(
    monkeypatch: pytest.MonkeyPatch, mae: float, passed: bool
) -> None:
    adapter = XrvAutoencoderGateAdapter(_manifest("xrv-ae-ood"))
    monkeypatch.setattr(adapter, "_batch", lambda _inputs, size: ("lote", size))
    monkeypatch.setattr(xr_gates, "ae_reconstruction_mae", lambda _m, _b: np.array([mae]))
    out = adapter.predict(_inputs(_cxr(), gates=False))
    assert out.extras["gate"]["passed"] is passed
    assert out.extras["gate"]["threshold"] == adapter.threshold
    assert out.extras["ae_mae"] == mae


# --------------------------------------------------------------------------- #
# Anatomia / ICT (REQ-124) — máscaras geométricas no lugar da rede
# --------------------------------------------------------------------------- #
def _anatomy_masks() -> np.ndarray:
    masks = np.zeros((len(PSPNET_TARGETS), 512, 512), dtype=bool)
    masks[PSPNET_TARGETS.index("Right Lung"), 80:400, 60:240] = True
    masks[PSPNET_TARGETS.index("Left Lung"), 80:420, 272:452] = True
    masks[PSPNET_TARGETS.index("Heart"), 240:380, 200:320] = True
    return masks


@pytest.mark.requirement("REQ-124")
@pytest.mark.parametrize("view", ["PA", "AP"])
def test_xr_anatomy_ctr_geometry_and_pa_only_threshold(
    monkeypatch: pytest.MonkeyPatch, view: str
) -> None:
    manifest = _manifest("xrv-pspnet-chestxdet")
    with pytest.raises(ManifestError, match="depends_on"):
        ChestXDetAnatomyAdapter(manifest.model_copy(update={"depends_on": ()}))
    adapter = ChestXDetAnatomyAdapter(manifest)
    image = _cxr(view=view, size=600)
    assert adapter.applicability(_inputs(image)).applicable
    assert AbstentionReason.UPSTREAM in adapter.applicability(_inputs(image, False)).reasons
    monkeypatch.setattr(xr_anatomy, "xrv_input", lambda _img, size: np.zeros((size, size)))
    monkeypatch.setattr(adapter, "segment", lambda _plane: _anatomy_masks())
    out = adapter.predict(_inputs(image))
    meas = {m.key: m for m in out.measurements}
    ctr = meas["cardiothoracic_ratio"]
    assert ctr.value == pytest.approx(120 / (452 - 60))
    assert [p.label for p in ctr.geometry] == ["heart", "thorax"]
    (x0, _), (x1, _) = ctr.geometry[0].xy
    assert x1 - x0 == pytest.approx(119 * 600 / 512, abs=1.0)  # linha na geometria de origem
    pa = meas["cardiothoracic_ratio_pa"]
    if view == "PA":
        assert pa.value == ctr.value
    else:
        assert pa.value is None
        assert pa.abstention is AbstentionReason.VIEW
    heart_map, lungs_map = out.labelmaps
    assert heart_map.array.shape == (600, 600)
    assert set(np.unique(lungs_map.array)) == {0, 1, 2}
    adapter.unload()


@pytest.mark.requirement("REQ-124")
def test_xr_anatomy_empty_heart_abstains_everywhere(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = ChestXDetAnatomyAdapter(_manifest("xrv-pspnet-chestxdet"))
    masks = _anatomy_masks()
    masks[PSPNET_TARGETS.index("Heart")] = False
    monkeypatch.setattr(xr_anatomy, "xrv_input", lambda _img, size: np.zeros((size, size)))
    monkeypatch.setattr(adapter, "segment", lambda _plane: masks)
    out = adapter.predict(_inputs(_cxr()))
    for m in out.measurements:
        assert m.value is None
        assert m.abstention is AbstentionReason.QUALITY


# --------------------------------------------------------------------------- #
# RAD-DINO e CheXzero — contratos sem torch (REQ-125, REQ-126)
# --------------------------------------------------------------------------- #
@pytest.mark.requirement("REQ-125")
def test_xr_raddino_manifest_contract_and_gates() -> None:
    manifest = _manifest("xr-probe-tb")
    with pytest.raises(ManifestError, match="exatamente um"):
        RadDinoLinearProbeAdapter(manifest.model_copy(update={"labels": ()}))
    with pytest.raises(ManifestError, match="pesos devem ser"):
        RadDinoLinearProbeAdapter(manifest.model_copy(update={"weights": manifest.weights[:1]}))
    with pytest.raises(ManifestError, match="square"):
        RadDinoLinearProbeAdapter(
            manifest.model_copy(update={"config": {**manifest.config, "square": "fit"}})
        )
    with pytest.raises(ManifestError, match="depends_on"):
        RadDinoLinearProbeAdapter(manifest.model_copy(update={"depends_on": ()}))
    adapter = RadDinoLinearProbeAdapter(manifest)
    image = _cxr()
    assert adapter.applicability(_inputs(image)).applicable
    assert AbstentionReason.UPSTREAM in adapter.applicability(_inputs(image, False)).reasons
    with pytest.raises(InferenceError, match="não carregado"):
        adapter.predict(_inputs(image))
    adapter.unload()


@pytest.mark.requirement("REQ-125")
@pytest.mark.skipif(not _have("PIL"), reason="Pillow ausente")
def test_xr_raddino_canonical_crop_pad_and_range() -> None:
    wide = _cxr(size=64)
    wide = ImageData(
        pixels=np.tile(np.linspace(0, 4095, 96, dtype=np.float32), (64, 1)),
        kind=wide.kind,
        modality=wide.modality,
        meta=wide.meta,
        value_range=wide.value_range,
    )
    crop = raddino_canonical(wide, "crop")
    pad = raddino_canonical(wide, "pad")
    assert crop.shape == pad.shape == (518, 518)
    assert pad[0].max() == 0
    assert crop[0].max() > 0
    with pytest.raises(ValueError, match="square"):
        raddino_canonical(wide, "fit")
    bad = ImageData(
        pixels=wide.pixels * 2.0,
        kind=wide.kind,
        modality=wide.modality,
        meta=wide.meta,
        value_range=wide.value_range,
    )
    with pytest.raises(UnsupportedInputError, match="faixa"):
        raddino_canonical(bad, "crop")
    nan = ImageData(
        pixels=np.full((8, 8), np.nan, dtype=np.float32),
        kind=wide.kind,
        modality=wide.modality,
        meta=wide.meta,
        value_range=wide.value_range,
    )
    with pytest.raises(UnsupportedInputError, match="não finitos"):
        raddino_canonical(nan, "crop")


@pytest.mark.requirement("REQ-126")
def test_xr_chexzero_manifest_contract_and_gates() -> None:
    manifest = _manifest("chexzero")
    with pytest.raises(ManifestError, match="zero_shot"):
        ChexzeroAdapter(manifest.model_copy(update={"task": "classification"}))
    with pytest.raises(ManifestError, match="templates"):
        ChexzeroAdapter(
            manifest.model_copy(update={"config": {**manifest.config, "templates": ["{}"]}})
        )
    adapter = ChexzeroAdapter(manifest)
    assert adapter.templates == ("{}", "no {}")
    image = _cxr()
    assert adapter.applicability(_inputs(image)).applicable
    assert AbstentionReason.UPSTREAM in adapter.applicability(_inputs(image, False)).reasons
    adapter.unload()


# --------------------------------------------------------------------------- #
# Com torch: módulos mínimos (REQ-121, REQ-123, REQ-125, REQ-126)
# --------------------------------------------------------------------------- #
needs_torch = pytest.mark.skipif(not _have("torch"), reason="torch ausente (extra ml)")


def _tiny_densenet(n_out: int) -> Any:
    import torch

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = torch.nn.Conv2d(1, 4, kernel_size=32, stride=32)
            self.classifier = torch.nn.Linear(4, n_out)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            f = torch.nn.functional.relu(self.features(x))
            return self.classifier(f.mean(dim=(2, 3)))

    torch.manual_seed(0)
    return Tiny().eval()


@needs_torch
@pytest.mark.requirement("REQ-123")
def test_xr_classifier_logits_cam_and_explain_with_tiny_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    adapter = xr_xrv.XrvClassifierAdapter(_manifest("xrv-densenet121-res224-all"))
    image = _cxr(size=300)
    with pytest.raises(InferenceError, match="não carregado"):
        adapter.logits(torch.zeros(1, 1, 224, 224))
    adapter._model = _tiny_densenet(3)
    labels = [spec.key for spec in adapter.manifest.labels[:2]]
    adapter._heads = {labels[0]: 0, labels[1]: 2}
    monkeypatch.setattr(
        xr_xrv,
        "xrv_input",
        lambda _img, size: np.linspace(-1, 1, size * size, dtype=np.float32).reshape(size, size),
    )
    out = adapter.predict(_inputs(image))
    assert [s.label for s in out.scores] == labels
    batch = adapter._tensor(_inputs(image))
    with torch.inference_mode():
        expected = adapter._model(batch)[0]
    assert out.scores[1].value == pytest.approx(float(expected[2]), abs=1e-6)
    maps = adapter.explain(_inputs(image), [labels[0], "nao_declarado"])
    assert len(maps) == 1
    assert maps[0].heatmap.shape == (7, 7)
    assert float(maps[0].heatmap.max()) <= 1.0
    adapter.arch = "resnet"
    assert adapter.explain(_inputs(image), labels) == []
    with pytest.raises(InferenceError, match="DenseNet"):
        adapter.cam(batch, 0)
    adapter.unload()


@needs_torch
@pytest.mark.requirement("REQ-122")
def test_xr_gate_tensor_functions_with_tiny_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    from radiologyai.analysis.adapters.xr_common import XRV_SCALE

    class TwoWay(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor([1.0, -1.0]))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.mean(dim=(1, 2, 3))[:, None] * self.w[None]

    class HalfAE(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"out": x * 0.5 + self.p}

    batch = torch.full((1, 1, 8, 8), 100.0)
    probs = xr_gates.view_probabilities(TwoWay(), batch)
    assert probs.shape == (1, 2)
    assert probs[0, 0] > probs[0, 1]
    mae = xr_gates.ae_reconstruction_mae(HalfAE(), batch)
    assert mae[0] == pytest.approx(50.0 / (2 * XRV_SCALE))

    gate = XrvViewGateAdapter(_manifest("xrv-view-xinario"))
    with pytest.raises(InferenceError, match="não carregado"):
        gate._batch(_inputs(_cxr()), 224)
    gate._model = TwoWay()
    monkeypatch.setattr(xr_gates, "xrv_input", lambda _img, size: np.ones((size, size), np.float32))
    assert tuple(gate._batch(_inputs(_cxr()), 224).shape) == (1, 1, 224, 224)
    assert gate.predict(_inputs(_cxr())).extras["gate"]["passed"] is True


@needs_torch
@pytest.mark.requirement("REQ-126")
def test_xr_chexzero_load_and_probabilities_with_tiny_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not _have("open_clip", "torchvision", "PIL"):
        pytest.skip("open_clip/torchvision/Pillow ausentes")
    import torch

    from radiologyai.analysis.adapters import xr_chexzero

    class TinyClip(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            torch.manual_seed(0)
            self.text = torch.nn.Embedding(49408, 8)
            self.image = torch.nn.Linear(3, 8)

        def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
            return self.text(tokens).mean(dim=1)

        def encode_image(self, x: torch.Tensor) -> torch.Tensor:
            return self.image(x.mean(dim=(2, 3)))

    monkeypatch.setattr(xr_chexzero, "build_chexzero_clip", lambda _state: TinyClip())
    adapter = ChexzeroAdapter(_manifest("chexzero"))
    image = _cxr(size=256)
    with pytest.raises(InferenceError, match="não carregado"):
        adapter.probabilities(image)
    files, hashes = {}, {}
    for w in adapter.manifest.weights:
        path = tmp_path / f"{w.key}.pt"
        torch.save({"x": torch.zeros(1)}, str(path))
        files[w.key] = path
        hashes[w.key] = hashlib.sha256(path.read_bytes()).hexdigest()
    adapter.load(ResolvedWeights(files=files, sha256=hashes), "cpu")
    assert len(adapter._models) == len(adapter.manifest.weights)
    x = xr_chexzero.chexzero_image_tensor(image)
    assert tuple(x.shape) == (1, 3, 224, 224)
    out = adapter.predict(_inputs(image))
    assert len(out.scores) == len(adapter.manifest.labels)
    assert all(0.0 < s.value < 1.0 and s.kind == "softmax" for s in out.scores)
    assert out.extras["n_checkpoints"] == len(adapter.manifest.weights)


@needs_torch
@pytest.mark.requirement("REQ-125")
def test_xr_raddino_build_load_and_predict_with_tiny_dinov2(tmp_path: Path) -> None:
    if not _have("transformers", "safetensors", "PIL", "sklearn"):
        pytest.skip("transformers/safetensors/Pillow ausentes")
    import torch
    from transformers import Dinov2Config, Dinov2Model

    from radiologyai.analysis.adapters import xr_raddino
    from radiologyai.analysis.probes.artifact import ProbeArtifact, save_probe

    cfg = Dinov2Config(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        patch_size=74,
        image_size=518,
    )
    torch.manual_seed(0)
    model = Dinov2Model(cfg)
    model_dir = tmp_path / "dino"
    model.save_pretrained(str(model_dir))
    config_bytes = (model_dir / "config.json").read_bytes()
    weight_bytes = (model_dir / "model.safetensors").read_bytes()
    built = xr_raddino.build_raddino(config_bytes, weight_bytes)
    deeper = Dinov2Model(
        Dinov2Config(**{**cfg.to_dict(), "num_hidden_layers": 2}),
    )
    deeper_dir = tmp_path / "dino2"
    deeper.save_pretrained(str(deeper_dir))
    with pytest.raises(ManifestError, match="incompatíveis"):
        xr_raddino.build_raddino(config_bytes, (deeper_dir / "model.safetensors").read_bytes())

    manifest = _manifest("xr-probe-tb")
    adapter = RadDinoLinearProbeAdapter(manifest)
    label = manifest.labels[0].key
    coef = np.zeros(16)
    coef[0] = 1.0
    probe = ProbeArtifact(
        label=label,
        encoder_id=RADDINO_ENCODER_ID,
        embedding_dim=16,
        mean=np.zeros(16),
        scale=np.ones(16),
        coef=coef,
        intercept=0.5,
        training_data_sha256="b" * 64,
        meta={"square": adapter.square},
    )

    def weights_for(p: ProbeArtifact) -> ResolvedWeights:
        tensors, meta = save_probe(p, tmp_path / f"probe-{p.label}-{p.meta.get('square')}")
        files = {
            "encoder_config": model_dir / "config.json",
            "encoder_weights": model_dir / "model.safetensors",
            "probe_tensors": tensors,
            "probe_meta": meta,
        }
        return ResolvedWeights(
            files=files,
            sha256={k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in files.items()},
        )

    for bad, match in (
        (ProbeArtifact(**{**probe.__dict__, "label": "outro"}), "artefato é da sonda"),
        (ProbeArtifact(**{**probe.__dict__, "encoder_id": "x@y"}), "ajustada sobre"),
        (ProbeArtifact(**{**probe.__dict__, "meta": {"square": "pad"}}), "enquadramento"),
    ):
        with pytest.raises(ManifestError, match=match):
            adapter.load(weights_for(bad), "cpu")
    adapter.load(weights_for(probe), "cpu")
    image = _cxr(size=300)
    out = adapter.predict(_inputs(image))
    batch = xr_raddino.raddino_tensor(raddino_canonical(image, adapter.square))
    emb = xr_raddino.raddino_embeddings(built, batch)
    assert out.scores[0].value == pytest.approx(0.5 + float(emb[0, 0]), abs=1e-4)
    assert out.extras["encoder"] == RADDINO_ENCODER_ID
