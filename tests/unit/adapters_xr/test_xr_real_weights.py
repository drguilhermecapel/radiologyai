"""Adaptadores de RX com PESOS REAIS (pulados sem pesos/dados; rode sob o lock pesado).

Variáveis: ``MEDAI_WEIGHTS_DIR`` (padrão ``/root/medai-cache/weights/registry``) e os
dados de amostra em ``/root/medai-cache/datasets``.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from radiologyai.analysis.base import ModelInputs
from radiologyai.analysis.imaging import load_dicom, load_raster
from radiologyai.analysis.types import AbstentionReason, ImageData, Modality
from radiologyai.catalog.schema import ExamType

WEIGHTS = Path(os.environ.get("MEDAI_WEIGHTS_DIR", "/root/medai-cache/weights/registry"))
XRV_CACHE = Path("/root/medai-cache/weights/xrv")
DATA = Path("/root/medai-cache/datasets")
SAMPLES = DATA / "xrv-samples"
HAVE = (
    importlib.util.find_spec("torchxrayvision") is not None
    and (WEIGHTS / "xrv-densenet121-res224-pc").is_dir()
    and SAMPLES.is_dir()
)
pytestmark = [
    pytest.mark.skipif(not HAVE, reason="pesos xrv/amostras ausentes"),
    pytest.mark.slow,
]
CLASSIFIERS = [
    "xrv-densenet121-res224-all",
    "xrv-densenet121-res224-pc",
    "xrv-densenet121-res224-chex",
    "xrv-densenet121-res224-mimic_ch",
    "xrv-densenet121-res224-mimic_nb",
    "xrv-densenet121-res224-nih",
    "xrv-densenet121-res224-rsna",
    "xrv-resnet50-res512-all",
]


@pytest.fixture(scope="module")
def registry() -> Any:
    import torch

    from radiologyai.analysis.registry import Registry
    from radiologyai.config.settings import Settings

    torch.set_num_threads(4)
    settings = Settings(weights_dir=WEIGHTS, device="cpu", max_loaded_models=4)
    return Registry(weights_dir=WEIGHTS, settings=settings)


def _png(name: str, view: str = "PA") -> ImageData:
    return load_raster(
        SAMPLES / name,
        modality=Modality.DX,
        view_position=view,
        patient_age_years=50.0,
        body_part="CHEST",
    )


def _dicoms() -> list[Path]:
    return sorted((DATA / "samples-cxr-dicom").glob("*/*.dcm")) + sorted(SAMPLES.glob("*.dcm"))


def _library_input(path: Path, size: int) -> Any:
    """Pré-processamento do PRÓPRIO xrv (load_image/read_xray_dcm + crop + resize)."""
    import torch
    import torchvision
    import torchxrayvision as xrv

    img = xrv.utils.load_image(str(path))
    tf = torchvision.transforms.Compose(
        [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(size)]
    )
    return torch.from_numpy(tf(img))[None]


@pytest.mark.parametrize("model_id", CLASSIFIERS)
@pytest.mark.requirement("REQ-121")
def test_logits_equal_library_output_before_op_norm(registry: Any, model_id: str) -> None:
    import torch
    import torchxrayvision as xrv

    from radiologyai.analysis.adapters.xr_common import xrv_input

    adapter = registry.load(model_id)
    key = adapter.manifest.config["library_weights"]
    lib = xrv.models.get_model(key, cache_dir=str(XRV_CACHE))
    lib.op_threshs = None
    lib.apply_sigmoid = False
    size = adapter._size
    paths = [SAMPLES / "00000001_000.png", SAMPLES / "00027426_000.png", *_dicoms()]
    for path in paths:
        image = load_dicom(path) if path.suffix == ".dcm" else _png(path.name)
        ours = adapter.logits(torch.from_numpy(xrv_input(image, size))[None, None])[0]
        with torch.inference_mode():
            ref = lib(_library_input(path, size))[0]
        diff = float((ours - ref).abs().max())
        assert diff < 1e-4, f"{model_id} {path.name}: diferença {diff}"
    registry.unload_all()


@pytest.mark.requirement("REQ-121")
def test_untrained_heads_are_never_emitted(registry: Any) -> None:
    adapter = registry.load("xrv-densenet121-res224-pc")
    out = adapter.predict(ModelInputs(images=(_png("00000001_000.png"),), exam_type=ExamType.CXR))
    labels = {s.label for s in out.scores}
    assert "lung_lesion" not in labels  # cabeça sem treino no conjunto PadChest
    assert "enlarged_cardiomediastinum" not in labels
    assert all(s.kind == "logit" for s in out.scores)
    assert len(labels) == 15


def _spearman(a: Any, b: Any) -> float:
    ra = np.argsort(np.argsort(np.asarray(a).ravel()))
    rb = np.argsort(np.argsort(np.asarray(b).ravel()))
    return float(np.corrcoef(ra, rb)[0, 1])


@pytest.mark.requirement("REQ-123")
def test_cam_is_exact_and_fails_model_randomisation(registry: Any) -> None:
    import copy

    import torch

    from radiologyai.analysis.adapters.xr_common import xrv_input

    adapter = registry.load("xrv-densenet121-res224-all")
    image = _png("00000001_000.png")
    batch = torch.from_numpy(xrv_input(image, 224))[None, None]
    head = adapter._heads["cardiomegaly"]
    cam = adapter.cam(batch, head)
    # CAM exato: média espacial + viés == logit da cabeça
    logit = float(adapter.logits(batch)[0, head])
    bias = float(adapter._model.classifier.bias[head].detach())
    assert float(cam.mean()) + bias == pytest.approx(logit, abs=1e-3)
    # teste de sanidade por randomização (Adebayo et al. 2018): pesos aleatórios => mapa diferente
    original = adapter._model
    try:
        randomized = copy.deepcopy(original)
        gen = torch.Generator().manual_seed(0)
        with torch.no_grad():
            w = randomized.classifier.weight
            w.copy_(torch.randn(w.shape, generator=gen) * w.std())
        adapter._model = randomized
        cam_rand = adapter.cam(batch, head)
        for name, module in randomized.features.named_children():
            if name in ("denseblock4", "norm5"):
                for p in module.parameters():
                    p.data.copy_(torch.randn(p.shape, generator=gen) * (p.std() + 1e-3))
        cam_rand_deep = adapter.cam(batch, head)
    finally:
        adapter._model = original
    assert abs(_spearman(cam, cam_rand)) < 0.5
    assert abs(_spearman(cam, cam_rand_deep)) < 0.5
    maps = adapter.explain(
        ModelInputs(images=(image,), exam_type=ExamType.CXR), ["cardiomegaly", "nao_declarado"]
    )
    assert [m.label for m in maps] == ["cardiomegaly"]
    assert maps[0].heatmap.min() >= 0.0
    assert maps[0].heatmap.max() <= 1.0
    assert maps[0].placement.type == "bbox"


def _openi_laterals(n: int) -> list[Path]:
    import pandas as pd

    index = DATA / "openi" / "openi_index.csv"
    if not index.is_file():
        return []
    d = pd.read_csv(index).sort_values("imageid")
    d = d[d["View Position"].isin(["LL", "LATERAL"])].head(n)
    return [DATA / "openi" / "png" / f"{i}.png" for i in d["imageid"]]


@pytest.mark.requirement("REQ-122")
def test_view_gate_on_real_frontal_and_lateral(registry: Any) -> None:
    adapter = registry.load("xrv-view-xinario")
    frontal = adapter.predict(
        ModelInputs(images=(_png("00000001_000.png"),), exam_type=ExamType.CXR)
    )
    assert frontal.extras["gate"]["passed"] is True
    laterals = _openi_laterals(5)
    if not laterals:
        pytest.skip("Open-i ausente")
    for path in laterals:
        # cabeçalho declarado PA; o modelo vê uma lateral: o portão reprova
        image = load_raster(path, modality=Modality.DX, view_position="PA", patient_age_years=50.0)
        out = adapter.predict(ModelInputs(images=(image,), exam_type=ExamType.CXR))
        assert out.extras["gate"]["passed"] is False, path.name
        assert out.extras["gate"]["reason"] == AbstentionReason.VIEW.value


@pytest.mark.skipif(not (WEIGHTS / "xrv-ae-ood").is_dir(), reason="portão OOD sem pesos")
@pytest.mark.requirement("REQ-122")
def test_ood_gate_accepts_chest_and_rejects_noise(registry: Any) -> None:
    from radiologyai.analysis.manifest import load_manifest
    from radiologyai.analysis.registry import PACKAGE_MANIFEST_DIR
    from radiologyai.analysis.types import ImageKind, ImageMeta

    manifest, _ = load_manifest(PACKAGE_MANIFEST_DIR / "xrv-ae-ood.yaml")
    if manifest.status != "available":
        pytest.skip("limiar do portão OOD ainda não medido")
    adapter = registry.load("xrv-ae-ood")
    chest = adapter.predict(ModelInputs(images=(_png("00000001_000.png"),), exam_type=ExamType.CXR))
    assert chest.extras["gate"]["passed"] is True
    pixels = (np.arange(512 * 512) % 251).astype(np.float32).reshape(512, 512)
    meta = ImageMeta(source_format="raster", content_sha256="0" * 64, view_position="PA")
    noise = ImageData(pixels=pixels, kind=ImageKind.PROJECTION, modality=Modality.DX, meta=meta)
    out = adapter.predict(ModelInputs(images=(noise,), exam_type=ExamType.CXR))
    assert out.extras["gate"]["passed"] is False
    assert out.extras["gate"]["reason"] == AbstentionReason.OOD.value


@pytest.mark.requirement("REQ-124")
def test_pspnet_ctr_on_real_pa_and_ap(registry: Any) -> None:
    adapter = registry.load("xrv-pspnet-chestxdet")
    image = _png("00000001_000.png", view="PA")
    out = adapter.predict(ModelInputs(images=(image,), exam_type=ExamType.CXR))
    by_key = {m.key: m for m in out.measurements}
    ctr = by_key["cardiothoracic_ratio"]
    assert ctr.value is not None
    assert 0.3 < ctr.value < 0.75
    assert [g.label for g in ctr.geometry] == ["heart", "thorax"]
    h, w = image.plane_shape
    for g in ctr.geometry:
        for x, y in g.xy:
            assert 0 <= x < w
            assert 0 <= y < h
    assert by_key["cardiothoracic_ratio_pa"].value == ctr.value
    assert {lm.name for lm in out.labelmaps} == {"heart", "lungs"}
    assert out.labelmaps[0].array.shape == (h, w)
    ap = adapter.predict(
        ModelInputs(images=(_png("00000001_000.png", view="AP"),), exam_type=ExamType.CXR)
    )
    ap_by_key = {m.key: m for m in ap.measurements}
    assert ap_by_key["cardiothoracic_ratio"].value == ctr.value
    assert ap_by_key["cardiothoracic_ratio_pa"].value is None
    assert ap_by_key["cardiothoracic_ratio_pa"].abstention is AbstentionReason.VIEW


@pytest.mark.skipif(
    importlib.util.find_spec("open_clip") is None or not (WEIGHTS / "chexzero").is_dir(),
    reason="open_clip/pesos CheXzero ausentes",
)
@pytest.mark.requirement("REQ-126")
def test_chexzero_reproduces_reference_implementation(registry: Any) -> None:
    ref_path = Path("/root/medai-cache/spikes/zeroshot/out/chexzero_00000001_000.png.json")
    if not ref_path.is_file():
        pytest.skip("saída da implementação de referência ausente")
    ref = json.loads(ref_path.read_text())["ensemble_prob"]
    adapter = registry.load("chexzero")
    out = adapter.predict(ModelInputs(images=(_png("00000001_000.png"),), exam_type=ExamType.CXR))
    ours = {s.label: s.value for s in out.scores}
    pairs = {
        "atelectasis": "Atelectasis",
        "cardiomegaly": "Cardiomegaly",
        "pleural_effusion": "Pleural Effusion",
        "pneumothorax": "Pneumothorax",
        "nodule": "Nodule",
    }
    for key, name in pairs.items():
        assert ours[key] == pytest.approx(ref[name], abs=2e-4), key
    assert all(s.kind == "softmax" for s in out.scores)
    registry.unload_all()


@pytest.mark.requirement("REQ-122")
def test_full_pipeline_on_real_dicom(registry: Any, tmp_path: Path) -> None:
    from radiologyai.analysis.artifacts import ArtifactStore
    from radiologyai.analysis.pipeline import analyze
    from radiologyai.analysis.types import ExamInputs
    from radiologyai.catalog.catalog import Catalog

    path = next(SAMPLES.glob("*.dcm"))
    image = load_dicom(path, patient_age_years=50.0)
    catalog = Catalog([], None)  # sem condições catalogadas: tudo nível D (modo pesquisa)
    result = analyze(
        ExamInputs(images=(image,), exam_type=ExamType.CXR),
        registry=registry,
        catalog=catalog,
        store=ArtifactStore(tmp_path / "store"),
        model_ids=["xrv-densenet121-res224-pc", "xrv-pspnet-chestxdet"],
    )
    runs = {r.model_id: r for r in result.runs}
    assert runs["xrv-view-xinario"].status == "completed"
    assert runs["xrv-densenet121-res224-pc"].status in ("completed", "abstained")
    if runs["xrv-densenet121-res224-pc"].status == "completed":
        findings = [f for f in result.findings if f.label == "pneumothorax"]
        assert findings
        assert findings[0].raw_score is not None
        assert findings[0].score_kind == "logit"


@pytest.mark.skipif(
    not (WEIGHTS / "xr-probe-tb" / "probe.safetensors").is_file(), reason="sonda de TB ausente"
)
@pytest.mark.requirement("REQ-125")
def test_tb_probe_adapter_reproduces_saved_held_out_score(registry: Any) -> None:
    import pandas as pd

    preds = Path("/root/medai-cache/eval-preds/xr-probe-tb/tb-montgomery/tuberculosis.parquet")
    if not preds.is_file():
        pytest.skip("predições da sonda ausentes")
    row = pd.read_parquet(preds).iloc[0]
    path = DATA / "tb-montgomery" / "raw" / "CXR_png" / f"{row.image_id}.png"
    image = load_raster(path, modality=Modality.DX, view_position="PA", patient_age_years=40.0)
    assert image.sha256 == row.image_sha256
    adapter = registry.load("xr-probe-tb")
    out = adapter.predict(ModelInputs(images=(image,), exam_type=ExamType.CXR))
    (score,) = out.scores
    assert score.kind == "logit"
    assert score.value == pytest.approx(float(row.y_score), abs=1e-3)
    registry.unload_all()
