"""Invariantes dos tipos de resultado: entradas incoerentes falham ALTO (REQ-100).

Cada caso monta um objeto que violaria o contrato (medida sem valor e sem
motivo, relatório não aplicável sem motivo, execução falha sem explicação,
imagem com geometria inválida) e confere que a validação recusa.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pytest

from radiologyai.analysis.types import (
    AbstentionReason,
    CodedConcept,
    ImageData,
    ImageKind,
    ImageMeta,
    Measurement,
    ModelRun,
    Primitive,
    QAReport,
)

CODE = CodedConcept(scheme="SCT", value="0", meaning="teste")


def _image(**kw: Any) -> ImageData:
    pixels = kw.pop("pixels", np.zeros((4, 4), np.float32))
    meta = ImageMeta(
        source_format="raster", content_sha256=hashlib.sha256(pixels.tobytes()).hexdigest()
    )
    base: dict[str, Any] = {
        "pixels": pixels,
        "kind": ImageKind.PROJECTION,
        "modality": "DX",
        "meta": meta,
        "value_range": (0.0, 255.0),
    }
    base.update(kw)
    return ImageData(**base)


@pytest.mark.requirement("REQ-100")
@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"pixels": np.zeros((2, 4, 4), np.float32)}, "dimensões espaciais"),
        ({"pixels": np.zeros((4, 4, 4), np.float32), "color": True}, "última dimensão"),
        ({"spacing_mm": (1.0,)}, "2 ou 3 entradas"),
        ({"spacing_mm": (1.0, -1.0)}, "positivo e finito"),
        ({"affine_lps": np.eye(3)}, "4x4"),
        ({"value_range": (10.0, 10.0)}, "value_range inválido"),
    ],
)
def test_analysis_image_data_rejects_incoherent_geometry(kw: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _image(**kw)


@pytest.mark.requirement("REQ-100")
def test_analysis_image_meta_rejects_bad_sha() -> None:
    with pytest.raises(ValueError, match="64 caracteres"):
        ImageMeta(source_format="raster", content_sha256="ABC")


@pytest.mark.requirement("REQ-100")
@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"type": "point", "xy": ((1.0, 1.0), (2.0, 2.0))}, "exatamente 1"),
        ({"type": "bbox", "xy": ((1.0, 1.0),)}, "exatamente 2"),
        ({"type": "polygon", "xy": ((0.0, 0.0), (1.0, 1.0))}, "ao menos 3"),
        ({"type": "line", "xy": ((0.0, float("nan")), (1.0, 1.0))}, "finitas"),
    ],
)
def test_analysis_primitive_arity(kw: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Primitive(**kw)


@pytest.mark.requirement("REQ-100")
def test_analysis_measurement_value_xor_reason() -> None:
    common: dict[str, Any] = {"name": "x", "code": CODE, "unit": "mL", "method": "m"}
    with pytest.raises(ValueError, match="exige abstention_reason"):
        Measurement(value=None, provenance_idx=0, **common)
    with pytest.raises(ValueError, match="não pode ter abstention_reason"):
        Measurement(
            value=1.0, abstention_reason=AbstentionReason.QUALITY, provenance_idx=0, **common
        )
    with pytest.raises(ValueError, match="finito"):
        Measurement(value=float("inf"), provenance_idx=0, **common)
    ok = Measurement(
        value=None, abstention_reason=AbstentionReason.GEOMETRY, provenance_idx=0, **common
    )
    assert ok.value is None


@pytest.mark.requirement("REQ-100")
def test_analysis_qa_report_and_model_run_shape() -> None:
    with pytest.raises(ValueError, match="ao menos um motivo"):
        QAReport(applicable=False)
    with pytest.raises(ValueError, match="não pode ter motivos"):
        QAReport(applicable=True, reasons=(AbstentionReason.INPUT,))
    with pytest.raises(ValueError, match="explicar o motivo"):
        ModelRun(model_id="m", status="failed")
    with pytest.raises(ValueError, match="provenance_idx"):
        ModelRun(model_id="m", status="completed")
    assert ModelRun(model_id="m", status="planned", error="sem pesos abertos").error
