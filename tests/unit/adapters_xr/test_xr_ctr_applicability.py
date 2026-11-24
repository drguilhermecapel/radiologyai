"""Geometria do índice cardiotorácico e portões de aplicabilidade dos adaptadores de RX."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from radiologyai.analysis.adapters.xr_anatomy import cardiothoracic_ratio
from radiologyai.analysis.adapters.xr_xrv import XrvClassifierAdapter
from radiologyai.analysis.base import AdapterOutput, ModelInputs
from radiologyai.analysis.manifest import load_manifest
from radiologyai.analysis.registry import PACKAGE_MANIFEST_DIR
from radiologyai.analysis.types import AbstentionReason, ImageData, ImageKind, ImageMeta, Modality
from radiologyai.catalog.schema import ExamType

MANIFEST = PACKAGE_MANIFEST_DIR / "xrv-densenet121-res224-pc.yaml"
GATES = ("xrv-view-xinario", "xrv-ae-ood")


def _rect(shape: tuple[int, int], y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def _frontal_masks(heart_x: tuple[int, int] = (100, 160)) -> tuple[np.ndarray, ...]:
    shape = (256, 256)
    right = _rect(shape, 40, 200, 30, 120)  # pulmão direito à esquerda da imagem
    left = _rect(shape, 40, 210, 136, 226)
    heart = _rect(shape, 120, 190, heart_x[0], heart_x[1])
    return heart, left, right


@pytest.mark.requirement("REQ-124")
def test_ctr_value_and_line_geometry() -> None:
    heart, left, right = _frontal_masks()
    result = cardiothoracic_ratio(heart, left, right)
    assert result.reason is None
    assert result.heart_width_px == 60
    assert result.thorax_width_px == 226 - 30
    assert result.value == pytest.approx(60 / 196)
    (hx0, hy0), (hx1, hy1) = result.heart_line  # type: ignore[misc]
    assert (hx0, hx1) == (100.0, 159.0)
    assert hy0 == hy1
    (tx0, ty0), (tx1, ty1) = result.thorax_line  # type: ignore[misc]
    assert (tx0, tx1) == (30.0, 225.0)
    assert ty0 == ty1
    assert 40 <= ty0 < 200


@pytest.mark.requirement("REQ-124")
def test_ctr_uses_widest_row_with_both_lungs() -> None:
    heart, left, right = _frontal_masks()
    left[180:210, 136:240] = True  # base do pulmão esquerdo mais larga, mas só ele nessa faixa
    right[150:200, 20:120] = True  # faixa em que os dois pulmões coexistem e o tórax é mais largo
    result = cardiothoracic_ratio(heart, left, right)
    assert result.value is not None
    assert result.thorax_width_px == 239 - 20 + 1
    assert result.thorax_line is not None
    assert 180 <= result.thorax_line[0][1] < 200


@pytest.mark.requirement("REQ-124")
def test_ctr_abstains_on_empty_heart() -> None:
    _, left, right = _frontal_masks()
    result = cardiothoracic_ratio(np.zeros_like(left), left, right)
    assert result.value is None
    assert result.reason is AbstentionReason.QUALITY


@pytest.mark.requirement("REQ-124")
def test_ctr_abstains_on_missing_lung() -> None:
    heart, left, _ = _frontal_masks()
    result = cardiothoracic_ratio(heart, left, np.zeros_like(left))
    assert result.value is None
    assert result.reason is AbstentionReason.QUALITY


@pytest.mark.requirement("REQ-124")
def test_ctr_abstains_when_chest_is_cropped() -> None:
    heart, left, right = _frontal_masks()
    left[60:120, 226:256] = True  # pulmão encosta na borda direita da imagem
    result = cardiothoracic_ratio(heart, left, right)
    assert result.value is None
    assert result.reason is AbstentionReason.GEOMETRY
    assert "cortado" in result.detail


@pytest.mark.requirement("REQ-124")
def test_ctr_abstains_on_lateral_like_geometry() -> None:
    # Numa lateral (ou com lados trocados) o "pulmão direito" não fica à esquerda.
    heart, left, right = _frontal_masks()
    result = cardiothoracic_ratio(heart, right, left)
    assert result.value is None
    assert result.reason is AbstentionReason.GEOMETRY


@pytest.mark.requirement("REQ-124")
def test_ctr_abstains_without_overlapping_rows() -> None:
    shape = (256, 256)
    right = _rect(shape, 10, 100, 30, 120)
    left = _rect(shape, 150, 240, 136, 226)
    heart = _rect(shape, 120, 190, 100, 160)
    result = cardiothoracic_ratio(heart, left, right)
    assert result.value is None
    assert result.reason is AbstentionReason.GEOMETRY


# --------------------------------------------------------------------------- #
# Aplicabilidade do classificador
# --------------------------------------------------------------------------- #
def _cxr(view: str | None = "PA", age: float | None = 50.0, size: int = 512) -> ImageData:
    meta = ImageMeta(
        source_format="dicom",
        content_sha256=hashlib.sha256(f"{view}{age}{size}".encode()).hexdigest(),
        view_position=view,
        patient_age_years=age,
    )
    return ImageData(
        pixels=np.zeros((size, size), dtype=np.float32),
        kind=ImageKind.PROJECTION,
        modality=Modality.DX,
        meta=meta,
        value_range=(0.0, 4095.0),
    )


def _gate(image: ImageData, name: str, passed: bool, reason: AbstentionReason) -> AdapterOutput:
    return AdapterOutput(
        extras={
            "gate": {
                "image_sha256": image.sha256,
                "name": name,
                "passed": passed,
                "reason": reason.value,
                "value": 0.9,
                "threshold": 0.5,
            }
        }
    )


def _adapter() -> XrvClassifierAdapter:
    manifest, _ = load_manifest(Path(MANIFEST))
    return XrvClassifierAdapter(manifest)


def _inputs(image: ImageData, *, view_ok: bool = True, ood_ok: bool = True) -> ModelInputs:
    upstream = {
        GATES[0]: _gate(image, "view_frontal", view_ok, AbstentionReason.VIEW),
        GATES[1]: _gate(image, "ae_reconstruction_mae", ood_ok, AbstentionReason.OOD),
    }
    return ModelInputs(images=(image,), exam_type=ExamType.CXR, upstream=upstream)


@pytest.mark.requirement("REQ-122")
def test_frontal_adult_with_passing_gates_is_applicable() -> None:
    report = _adapter().applicability(_inputs(_cxr()))
    assert report.applicable, report


@pytest.mark.requirement("REQ-122")
@pytest.mark.parametrize(
    ("image", "reason"),
    [
        (_cxr(view="LATERAL"), AbstentionReason.VIEW),
        (_cxr(view=None), AbstentionReason.VIEW),
        (_cxr(age=9.0), AbstentionReason.POPULATION),
        (_cxr(age=None), AbstentionReason.POPULATION),
        (_cxr(size=128), AbstentionReason.QUALITY),
    ],
)
def test_static_gate_rejects(image: ImageData, reason: AbstentionReason) -> None:
    report = _adapter().applicability(_inputs(image))
    assert not report.applicable
    assert reason in report.reasons


@pytest.mark.requirement("REQ-122")
def test_view_model_overrides_wrong_header() -> None:
    report = _adapter().applicability(_inputs(_cxr(), view_ok=False))
    assert not report.applicable
    assert report.reasons == (AbstentionReason.VIEW,)


@pytest.mark.requirement("REQ-122")
def test_ood_gate_rejects() -> None:
    report = _adapter().applicability(_inputs(_cxr(), ood_ok=False))
    assert not report.applicable
    assert report.reasons == (AbstentionReason.OOD,)


@pytest.mark.requirement("REQ-122")
def test_gate_decision_for_another_image_counts_as_missing() -> None:
    image, other = _cxr(), _cxr(age=60.0)
    inputs = _inputs(other)
    inputs = ModelInputs(images=(image,), exam_type=ExamType.CXR, upstream=inputs.upstream)
    report = _adapter().applicability(inputs)
    assert not report.applicable
    assert report.reasons == (AbstentionReason.UPSTREAM,)


@pytest.mark.requirement("REQ-122")
def test_wrong_modality_and_exam_type_are_rejected() -> None:
    image = _cxr()
    ct = ImageData(
        pixels=image.pixels,
        kind=ImageKind.PROJECTION,
        modality=Modality.CT,
        meta=image.meta,
        value_range=image.value_range,
    )
    report = _adapter().applicability(_inputs(ct))
    assert AbstentionReason.MODALITY in report.reasons
    inputs = ModelInputs(images=(image,), exam_type=ExamType.MSK_XR)
    assert AbstentionReason.EXAM_TYPE in _adapter().applicability(inputs).reasons
