"""Encanamento dos adaptadores de mamografia sem pesos reais (REQ-142, REQ-143).

Duas camadas:

* sem torch — o modelo é substituído por um objeto Python que devolve números
  FIXOS e conhecidos; o teste confere roteamento por mama, recortes a montante,
  avisos e abstenções. Esses números não têm significado clínico e nunca saem
  de ``tests/``;
* com torch — o ``create_model`` do timm é trocado por uma convolução mínima
  inicializada com semente fixa, para exercitar ``load`` (safetensors com
  sha256 conferido, ``strict=True``), o pré-processamento por rede e a média do
  ensemble sem baixar os pesos de verdade.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from mg_support import gradient, mg_image

from radiologyai.analysis.adapters.mg_mammoscreen import (
    CROP_MODEL_ID,
    DENSITY_CLASSES,
    MammoCropAdapter,
    MammoscreenAdapter,
    apply_crop,
)
from radiologyai.analysis.adapters.mg_monai_density import MonaiBreastDensityAdapter
from radiologyai.analysis.base import AdapterOutput, ModelInputs, ResolvedWeights
from radiologyai.analysis.manifest import ModelManifest, load_manifest
from radiologyai.analysis.types import AbstentionReason
from radiologyai.catalog.schema import ExamType
from radiologyai.errors import InferenceError, IntegrityCheckError, ManifestError

MANIFESTS = Path(__file__).resolve().parents[3] / "src/radiologyai/analysis/manifests"


def _have(*mods: str) -> bool:
    return all(importlib.util.find_spec(m) is not None for m in mods)


def _manifest(model_id: str) -> ModelManifest:
    manifest, _ = load_manifest(MANIFESTS / f"{model_id}.yaml")
    return manifest


def _four_views() -> tuple[Any, ...]:
    return (
        mg_image(gradient(), laterality="L", view="CC", tag="lcc"),
        mg_image(gradient(), laterality="L", view="MLO", tag="lmlo"),
        mg_image(gradient(), laterality="R", view="CC", tag="rcc"),
    )


class _FixedCrop:
    """Caixa fixa: metade central da imagem (geometria conhecida)."""

    def crop_box(self, img: np.ndarray) -> tuple[int, int, int, int]:
        h, w = img.shape
        return w // 4, h // 4, w // 2, h // 2


class _FixedEnsemble:
    """Devolve números fixos e registra as formas recebidas (só encanamento)."""

    def __init__(self) -> None:
        self.calls: list[list[tuple[int, ...]]] = []

    def breast(self, views: list[np.ndarray]) -> tuple[float, list[float], list[float]]:
        self.calls.append([v.shape for v in views])
        return 0.25, [0.1, 0.2, 0.3, 0.4], [0.2, 0.25, 0.3]


def _crop_upstream(adapter: MammoCropAdapter, inputs: ModelInputs) -> dict[str, AdapterOutput]:
    return {CROP_MODEL_ID: adapter.predict(inputs)}


@pytest.mark.requirement("REQ-142")
def test_mg_apply_crop_slices_like_model_card() -> None:
    img = np.arange(100, dtype=np.uint8).reshape(10, 10)
    assert apply_crop(img, (2, 3, 4, 5)).shape == (5, 4)
    assert apply_crop(img, (-2, -1, 4, 5)).shape == (4, 2)
    with pytest.raises(InferenceError, match="mama não localizada"):
        apply_crop(img, (9, 9, 1, 1))


@pytest.mark.requirement("REQ-143")
def test_mg_crop_adapter_boxes_by_role_and_sha() -> None:
    adapter = MammoCropAdapter(_manifest("mammo-crop"))
    inputs = ModelInputs(images=_four_views(), exam_type=ExamType.MG)
    with pytest.raises(InferenceError, match="não carregado"):
        adapter.boxes(inputs)
    assert adapter.applicability(inputs).applicable
    adapter._model = _FixedCrop()
    out = adapter.predict(inputs)
    by_role = out.extras["crop_boxes_by_role"]
    assert set(by_role) == {"L-CC", "L-MLO", "R-CC"}
    assert by_role["L-CC"] == [65, 75, 130, 150]
    assert set(out.extras["crop_boxes"]) == {img.sha256 for img in inputs.images}
    assert not out.scores
    assert not out.class_scores
    adapter.unload()
    assert adapter._model is None
    assert not adapter.loaded


@pytest.mark.requirement("REQ-143")
def test_mg_mammoscreen_manifest_contract_errors() -> None:
    manifest = _manifest("mammoscreen")
    with pytest.raises(ManifestError, match="depends_on"):
        MammoscreenAdapter(manifest.model_copy(update={"depends_on": ()}))
    with pytest.raises(ManifestError, match="rótulos declarados"):
        MammoscreenAdapter(manifest.model_copy(update={"labels": manifest.labels[:2]}))
    bad_density = tuple(
        spec.model_copy(update={"classes": ("1", "2", "3", "4")})
        if spec.key.startswith("density")
        else spec
        for spec in manifest.labels
    )
    with pytest.raises(ManifestError, match="A, B, C, D"):
        MammoscreenAdapter(manifest.model_copy(update={"labels": bad_density}))


@pytest.mark.requirement("REQ-143")
def test_mg_mammoscreen_requires_upstream_boxes_and_scores_per_breast() -> None:
    crop = MammoCropAdapter(_manifest("mammo-crop"))
    crop._model = _FixedCrop()
    adapter = MammoscreenAdapter(_manifest("mammoscreen"))
    images = _four_views()
    bare = ModelInputs(images=images, exam_type=ExamType.MG)
    qa = adapter.applicability(bare)
    assert not qa.applicable
    assert qa.reasons == (AbstentionReason.UPSTREAM,)
    assert any(c.name == f"upstream:{CROP_MODEL_ID}" and not c.passed for c in qa.checks)

    wrong_exam = ModelInputs(images=images, exam_type=ExamType.CXR)
    assert AbstentionReason.EXAM_TYPE in adapter.applicability(wrong_exam).reasons

    inputs = ModelInputs(images=images, exam_type=ExamType.MG, upstream=_crop_upstream(crop, bare))
    assert adapter.applicability(inputs).applicable
    with pytest.raises(InferenceError, match="não carregado"):
        adapter.predict(inputs)

    fake = _FixedEnsemble()
    adapter._model = fake
    out = adapter.predict(inputs)
    scores = {s.label: s for s in out.scores}
    assert set(scores) == {"malignancy_left", "malignancy_right"}
    assert scores["malignancy_left"].kind == "sigmoid"
    assert out.class_scores["density_right"] == dict(
        zip(DENSITY_CLASSES, [0.1, 0.2, 0.3, 0.4], strict=True)
    )
    assert out.extras["views"] == {"left": ["CC", "MLO"], "right": ["CC"]}
    assert out.extras["per_net_cancer"]["left"] == [0.2, 0.25, 0.3]
    # cada incidência chega recortada pela caixa a montante (metade central)
    assert fake.calls == [[(150, 130), (150, 130)], [(150, 130)]]
    assert any("mama right: só a incidência CC" in w for w in out.warnings)

    # a caixa some entre a aplicabilidade e a predição: erro explícito, não palpite
    stale = ModelInputs(
        images=images,
        exam_type=ExamType.MG,
        upstream={CROP_MODEL_ID: AdapterOutput(extras={"crop_boxes": {}})},
    )
    with pytest.raises(InferenceError, match="caixa de recorte ausente"):
        adapter.predict(stale)
    adapter.unload()
    assert adapter._model is None


@pytest.mark.requirement("REQ-143")
def test_mg_monai_density_contract_and_guards() -> None:
    manifest = _manifest("monai-breast-density")
    spec = manifest.label("density")
    with pytest.raises(ManifestError, match="sigmoid"):
        MonaiBreastDensityAdapter(
            manifest.model_copy(
                update={"labels": (spec.model_copy(update={"score_kind": "softmax"}),)}
            )
        )
    adapter = MonaiBreastDensityAdapter(manifest)
    one = ModelInputs(images=(mg_image(gradient(400, 340), view="MLO"),), exam_type=ExamType.MG)
    assert adapter.applicability(one).applicable
    if _have("torch"):  # predict importa torch antes de conferir o modelo
        with pytest.raises(InferenceError, match="não carregado"):
            adapter.predict(one)
    adapter.unload()
    assert not adapter.loaded


@pytest.mark.skipif(not _have("safetensors"), reason="safetensors ausente (extra foundation)")
def test_mg_read_verified_rejects_swapped_file(tmp_path: Path) -> None:
    from radiologyai.analysis.adapters.mg_common import load_safetensors_verified

    blob = tmp_path / "model.safetensors"
    blob.write_bytes(b"conteudo trocado")
    weights = ResolvedWeights(files={"model": blob}, sha256={"model": "0" * 64})
    with pytest.raises(IntegrityCheckError):
        load_safetensors_verified(weights, "model")


# --------------------------------------------------------------------------- #
# Com torch: arquitetura mínima no lugar do timm (sem pesos reais)
# --------------------------------------------------------------------------- #
needs_mg_backend = pytest.mark.skipif(
    not _have("torch", "timm", "cv2", "safetensors", "torchvision"),
    reason="torch/timm/cv2/safetensors ausentes",
)


def _tiny_backbone(name: str, **kwargs: Any) -> Any:
    import torch

    channels = {"mobilenetv3_small_100": 1024, "tf_efficientnetv2_s": 1280}[name]
    assert kwargs["in_chans"] == 1
    assert kwargs["pretrained"] is False
    return torch.nn.Conv2d(1, channels, kernel_size=32, stride=32)


def _weights(tmp_path: Path, name: str, module: Any) -> ResolvedWeights:
    from safetensors.torch import save_file

    path = tmp_path / f"{name}.safetensors"
    save_file({k: v.contiguous() for k, v in module.state_dict().items()}, str(path))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return ResolvedWeights(files={"model": path}, sha256={"model": digest})


@needs_mg_backend
@pytest.mark.requirement("REQ-142")
def test_mg_vendor_ensemble_and_adapters_load_with_tiny_backbone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    from radiologyai.analysis.vendor.mammoscreen import model as vendor

    monkeypatch.setattr(vendor, "create_model", _tiny_backbone)
    torch.manual_seed(0)
    crop_net = vendor.MammoCropNet().eval()
    ensemble = vendor.MammoEnsemble().eval()
    assert len(ensemble.nets) == 3
    with pytest.raises(ValueError, match="ao menos uma"):
        ensemble.breast([])

    crop = MammoCropAdapter(_manifest("mammo-crop"))
    crop.load(_weights(tmp_path, "crop", crop_net), "cpu")
    screen = MammoscreenAdapter(_manifest("mammoscreen"))
    screen.load(_weights(tmp_path, "screen", ensemble), "cpu")
    assert crop.loaded
    assert screen.loaded

    images = _four_views()
    bare = ModelInputs(images=images, exam_type=ExamType.MG)
    boxes = crop.boxes(bare)
    for x, y, w, h in boxes.values():
        assert 0 <= x <= 260
        assert 0 <= y <= 300
        assert w >= 0
        assert h >= 0
    # caixa conhecida a jusante (a da rede não treinada pode ser degenerada)
    fixed = {img.sha256: [20, 30, 200, 240] for img in images}
    inputs = ModelInputs(
        images=images,
        exam_type=ExamType.MG,
        upstream={CROP_MODEL_ID: AdapterOutput(extras={"crop_boxes": fixed})},
    )
    out = screen.predict(inputs)
    for side in ("left", "right"):
        probs = out.class_scores[f"density_{side}"]
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-5)
        per_net = out.extras["per_net_cancer"][side]
        cancer = next(s.value for s in out.scores if s.label == f"malignancy_{side}")
        assert cancer == pytest.approx(sum(per_net) / 3, abs=1e-6)

    # pesos com outra arquitetura: strict=True recusa
    other = torch.nn.Linear(2, 2)
    with pytest.raises(RuntimeError):
        MammoCropAdapter(_manifest("mammo-crop")).load(_weights(tmp_path, "x", other), "cpu")


@needs_mg_backend
@pytest.mark.requirement("REQ-143")
def test_mg_monai_density_tiny_inception_path(tmp_path: Path) -> None:
    import io

    import torch
    import torchvision

    from radiologyai.analysis.adapters import mg_monai_density as mod

    with pytest.raises(InferenceError, match="constante"):
        mod.monai_input(np.zeros((40, 30), dtype=np.uint8))
    x = mod.monai_input(np.arange(1200, dtype=np.uint8).reshape(40, 30))
    assert tuple(x.shape) == (1, 3, 299, 299)
    assert float(x.min()) >= 0.0
    assert float(x.max()) <= 1.0

    torch.manual_seed(0)
    net = torchvision.models.inception_v3(
        weights=None, aux_logits=True, transform_input=True, init_weights=False
    )
    net.fc = torch.nn.Linear(2048, 4)
    state = {f"features.{k}": v for k, v in net.state_dict().items()}
    buf = io.BytesIO()
    torch.save(state, buf)
    path = tmp_path / "model.pt"
    path.write_bytes(buf.getvalue())
    weights = ResolvedWeights(
        files={"model": path}, sha256={"model": hashlib.sha256(buf.getvalue()).hexdigest()}
    )
    adapter = MonaiBreastDensityAdapter(_manifest("monai-breast-density"))
    adapter.load(weights, "cpu")
    one = ModelInputs(images=(mg_image(gradient(), view="CC"),), exam_type=ExamType.MG)
    out = adapter.predict(one)
    probs = out.class_scores["density"]
    assert list(probs) == list(DENSITY_CLASSES)
    assert all(0.0 < p < 1.0 for p in probs.values())  # sigmoide por classe
    assert out.extras["role"] == "L-CC"
    two = ModelInputs(images=_four_views()[:2], exam_type=ExamType.MG)
    with pytest.raises(InferenceError, match="exatamente uma"):
        adapter.predict(two)
