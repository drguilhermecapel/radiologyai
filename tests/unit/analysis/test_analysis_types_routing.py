"""Invariantes dos tipos, repositório de artefatos e roteamento / tipo de exame."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from analysis_support import manifest_dict, projection

from radiologyai.analysis.artifacts import ArtifactStore
from radiologyai.analysis.routing import Route, RoutingTable, infer_exam_type, role_of
from radiologyai.analysis.types import (
    AbstentionReason,
    AnalysisResult,
    ArtifactRef,
    Finding,
    ImageData,
    ImageKind,
    Measurement,
    Modality,
    QAReport,
)
from radiologyai.calibration.decision import OperatingPoint
from radiologyai.catalog.schema import CodedConcept, EvidenceLevel, ExamType
from radiologyai.errors import ArtifactStoreError, ConfigurationError

CODE = CodedConcept(scheme="99MEDAI", value="x", meaning="x")


def _finding(**overrides: Any) -> Finding:
    data: dict[str, Any] = {
        "condition_id": "cxr.x",
        "label": "x",
        "code": CODE,
        "kind": "binary",
        "raw_score": 0.4,
        "score_kind": "sigmoid",
        "decision": "score_only",
        "evidence_level": EvidenceLevel.D,
        "visibility": "research",
        "provenance_idx": 0,
    }
    data.update(overrides)
    return Finding.model_validate(data)


def test_import_is_light() -> None:
    code = (
        "import sys, radiologyai.analysis.types, radiologyai.analysis.pipeline, "
        "radiologyai.analysis.registry, radiologyai.analysis.imaging; "
        "print(','.join(m for m in ('torch','numpy','SimpleITK','cv2','pydicom','yaml') "
        "if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""


@pytest.mark.requirement("REQ-101")
def test_probability_requires_calibration_id() -> None:
    with pytest.raises(ValueError, match="calibration_id"):
        _finding(probability=0.3)
    assert _finding(probability=0.3, calibration_id="cal").probability == 0.3


@pytest.mark.requirement("REQ-101")
def test_decision_requires_operating_point() -> None:
    with pytest.raises(ValueError, match="ponto de operação"):
        _finding(decision="positive")
    point = OperatingPoint(name="p", role="balanced", threshold=0.5, on="sigmoid", source="s")
    assert _finding(decision="negative", operating_points=(point,)).decision == "negative"


@pytest.mark.requirement("REQ-102")
def test_level_d_cannot_be_main() -> None:
    with pytest.raises(ValueError, match="pesquisa"):
        _finding(visibility="main")
    with pytest.raises(ValueError, match="visibilidade"):
        _finding(evidence_level=EvidenceLevel.A, visibility="research")


@pytest.mark.requirement("REQ-103")
def test_abstention_shape() -> None:
    with pytest.raises(ValueError, match="sem escore"):
        _finding(raw_score=None)
    with pytest.raises(ValueError, match="abstention_reason"):
        _finding(decision="abstained")
    ok = _finding(raw_score=None, decision="abstained", abstention_reason=AbstentionReason.OOD)
    assert ok.abstention_reason is AbstentionReason.OOD
    with pytest.raises(ValueError, match="abstention_reason"):
        Measurement(name="m", code=CODE, value=None, unit="1", method="x", provenance_idx=0)
    with pytest.raises(ValueError, match="motivo"):
        QAReport(applicable=False)


def test_result_checks_cross_references() -> None:
    base: dict[str, Any] = {
        "analysis_id": "00000000-0000-0000-0000-000000000000",
        "created_at": "2026-11-24T00:00:00Z",
        "exam_type": "cxr",
        "input_sha256": ("a" * 64,),
        "status": "completed",
    }
    with pytest.raises(ValueError, match="provenance_idx"):
        AnalysisResult.model_validate({**base, "findings": [_finding().model_dump()]})
    with pytest.raises(ValueError, match="research_use_only"):
        AnalysisResult.model_validate({**base, "intended_use": "diagnostic"})


def test_image_data_contract() -> None:
    with pytest.raises(TypeError, match="float32"):
        ImageData(
            pixels=np.zeros((4, 4), dtype=np.float64),  # type: ignore[arg-type]
            kind=ImageKind.PROJECTION,
            modality=Modality.DX,
            meta=projection().meta,
        )
    with pytest.raises(ValueError, match="dimensões"):
        ImageData(
            pixels=np.zeros((2, 4, 4), dtype=np.float32),
            kind=ImageKind.PROJECTION,
            modality=Modality.DX,
            meta=projection().meta,
        )
    with pytest.raises(ValueError, match="spacing"):
        ImageData(
            pixels=np.zeros((4, 4), dtype=np.float32),
            kind=ImageKind.PROJECTION,
            modality=Modality.DX,
            meta=projection().meta,
            spacing_mm=(0.0, 1.0),
        )
    cine = ImageData(
        pixels=np.zeros((3, 4, 5, 3), dtype=np.float32),
        kind=ImageKind.CINE,
        modality=Modality.US,
        meta=projection().meta,
        color=True,
    )
    assert cine.plane_shape == (4, 5)


# --------------------------------------------------------------------------- #
# Artefatos
# --------------------------------------------------------------------------- #
@pytest.mark.requirement("REQ-107")
def test_store_is_content_addressed_and_verified(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "s")
    ref = store.put_bytes(b"conteudo", "application/json")
    assert ref == store.put_bytes(b"conteudo", "application/json")
    assert store.read_bytes(ref) == b"conteudo"
    assert store.read_bytes(ref.sha256) == b"conteudo"
    store.path(ref).write_bytes(b"adulterado")
    with pytest.raises(ArtifactStoreError, match="corrompido"):
        store.read_bytes(ref)


@pytest.mark.requirement("REQ-107")
def test_store_refuses_traversal(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "s")
    ref = store.put_bytes(b"x", "application/json")
    evil = ArtifactRef(sha256=ref.sha256, media_type="application/json", relpath="../../etc/x.json")
    with pytest.raises(ArtifactStoreError):
        store.path(evil)
    with pytest.raises(ArtifactStoreError):
        store.path("../" + "a" * 62)


def test_labelmap_roundtrip(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "s")
    mask = np.zeros((3, 4, 5), dtype=np.int64)
    mask[1, 2, 3] = 300
    ref = store.put_labelmap(mask)
    back = store.read_labelmap(ref)
    assert back.dtype == np.uint16
    assert np.array_equal(back, mask)
    with pytest.raises(ValueError, match="inteiro"):
        store.put_labelmap(np.zeros((2, 2), dtype=np.float32))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Tipo de exame e papéis
# --------------------------------------------------------------------------- #
@pytest.mark.requirement("REQ-104")
@pytest.mark.parametrize(
    ("modality", "text", "expected"),
    [
        (Modality.DX, "CHEST", ExamType.CXR),
        (Modality.CR, "HAND", ExamType.MSK_XR),
        (Modality.DX, "CHEST HAND", ExamType.UNKNOWN),
        (Modality.DX, None, ExamType.UNKNOWN),
        (Modality.MG, None, ExamType.MG),
        (Modality.CT, "HEAD", ExamType.CT_HEAD),
        (Modality.CT, "CHEST", ExamType.CT_CHEST),
        (Modality.CT, "ABDOMEN", ExamType.CT_ABDOMEN),
        (Modality.CT, None, ExamType.UNKNOWN),
        (Modality.MR, "BRAIN", ExamType.MR_BRAIN),
        (Modality.US, "BREAST", ExamType.US_BREAST),
        (Modality.US, "ECHO", ExamType.ECHO),
        (Modality.US, None, ExamType.UNKNOWN),
        (Modality.OT, "CHEST", ExamType.UNKNOWN),
    ],
)
def test_exam_type_inference_is_conservative(
    modality: Modality, text: str | None, expected: ExamType
) -> None:
    image = projection(modality=modality, body_part=text)
    assert infer_exam_type([image]) == expected


@pytest.mark.requirement("REQ-104")
def test_mixed_modalities_are_unknown() -> None:
    images = [projection(modality=Modality.DX), projection(modality=Modality.CT)]
    assert infer_exam_type(images) is ExamType.UNKNOWN
    assert infer_exam_type([]) is ExamType.UNKNOWN


def test_roles() -> None:
    mg = projection(modality=Modality.MG, laterality="R", view_position="MLO")
    assert role_of(mg) == "R-MLO"
    assert role_of(projection(modality=Modality.MG)) is None
    assert role_of(projection(modality=Modality.MR, sequence_role="FLAIR")) == "flair"
    assert role_of(projection(modality=Modality.US, view_position="A4C")) == "a4c"


def test_routing_validation_rejects_unknown_and_inapplicable(make_registry: Any) -> None:
    registry = make_registry(manifest_dict("fake-bright", "Brightness"))
    RoutingTable(exam_types={ExamType.CXR: Route(main=("fake-bright",))}).validate_against(registry)
    with pytest.raises(ConfigurationError, match="desconhecido"):
        RoutingTable(exam_types={ExamType.CXR: Route(main=("nope",))}).validate_against(registry)
    with pytest.raises(ConfigurationError, match="não declara"):
        RoutingTable(exam_types={ExamType.MG: Route(main=("fake-bright",))}).validate_against(
            registry
        )


def test_package_routing_file_loads(settings: Any) -> None:
    from radiologyai.analysis.registry import Registry
    from radiologyai.analysis.routing import load_routing

    table = load_routing(Registry(settings=settings))
    assert isinstance(table, RoutingTable)
