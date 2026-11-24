from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from medai.core.types import (
    AbstentionReason,
    Finding,
    ImageData,
    ImageKind,
    ImageMeta,
    Measurement,
    Modality,
    Primitive,
    QAReport,
)

CODE = {"scheme": "99MEDAI", "value": "x", "meaning": "x"}
POINT = {"name": "p", "role": "balanced", "threshold": 0.5, "on": "sigmoid", "source": "test"}


def finding(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "condition_id": "cxr.x",
        "label": "x",
        "code": CODE,
        "kind": "binary",
        "raw_score": 0.7,
        "score_kind": "sigmoid",
        "operating_points": [POINT],
        "decision": "positive",
        "evidence_level": "A",
        "visibility": "main",
        "provenance_idx": 0,
    }
    base.update(overrides)
    return base


def test_valid_finding() -> None:
    assert Finding.model_validate(finding()).decision == "positive"


@pytest.mark.parametrize(
    "bad",
    [
        {"probability": 0.8},  # probability without a calibration id
        {"calibration_id": "cal-1"},  # calibration id without a probability
        {"operating_points": []},  # a call without a sourced operating point
        {"decision": "abstained"},  # abstention without a reason
        {"abstention_reason": AbstentionReason.OOD},  # reason without abstention
        {"raw_score": None},  # no score but not an abstention
        {"evidence_level": "D"},  # level D shown in the main view
        {"kind": "categorical"},  # categorical without classes
        {"kind": "derived"},  # derived without a measurement
    ],
)
def test_finding_invariants(bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Finding.model_validate(finding(**bad))


def test_level_d_is_allowed_in_research_view() -> None:
    f = Finding.model_validate(finding(evidence_level="D", visibility="research"))
    assert f.visibility == "research"


def test_abstained_finding_has_no_score() -> None:
    f = Finding.model_validate(
        finding(
            raw_score=None,
            decision="abstained",
            abstention_reason="out_of_distribution",
            operating_points=[],
        )
    )
    assert f.raw_score is None


def test_measurement_needs_value_or_reason() -> None:
    with pytest.raises(ValidationError):
        Measurement(name="m", code=CODE, value=None, unit="1", method="t", provenance_idx=0)  # type: ignore[arg-type]


def test_qa_reject_requires_reason() -> None:
    with pytest.raises(ValidationError):
        QAReport(applicable=False)
    assert QAReport.reject(AbstentionReason.VIEW).reasons == [AbstentionReason.VIEW]


def test_primitive_arity() -> None:
    with pytest.raises(ValidationError):
        Primitive(type="line", xy=[(0.0, 0.0)])
    assert Primitive(type="bbox", xy=[(0.0, 0.0), (1.0, 1.0)]).type == "bbox"


def _meta() -> ImageMeta:
    return ImageMeta(source_format="raster", content_sha256="0" * 64)


def test_image_data_rejects_wrong_dtype_and_shape() -> None:
    with pytest.raises(TypeError):
        ImageData(
            pixels=np.zeros((4, 4), dtype=np.uint8),
            kind=ImageKind.PROJECTION,
            modality=Modality.DX,
            meta=_meta(),
        )  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="spatial dims"):
        ImageData(
            pixels=np.zeros((4, 4), dtype=np.float32),
            kind=ImageKind.VOLUME,
            modality=Modality.CT,
            meta=_meta(),
        )
    with pytest.raises(ValueError, match="trailing dimension"):
        ImageData(
            pixels=np.zeros((4, 4, 2), dtype=np.float32),
            kind=ImageKind.PROJECTION,
            modality=Modality.US,
            meta=_meta(),
            color=True,
        )
    colour = ImageData(
        pixels=np.zeros((2, 4, 4, 3), dtype=np.float32),
        kind=ImageKind.CINE,
        modality=Modality.US,
        meta=_meta(),
        color=True,
    )
    assert colour.sha256 == "0" * 64
