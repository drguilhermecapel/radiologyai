"""Plugin de modalidade MG: escopo de entrada (REQ-144)."""

from __future__ import annotations

import numpy as np
import pytest

from radiologyai.errors import UnsupportedInputError
from radiologyai.io.metadata import StudyMetadata
from radiologyai.modalities.mg import MGPlugin
from radiologyai.modalities.mg.plugin import BREAST_TOMOSYNTHESIS_STORAGE

FFDM = "1.2.840.10008.5.1.4.1.1.1.2"


def _meta(**kw: object) -> StudyMetadata:
    base: dict[str, object] = {
        "modality": "MG",
        "sop_class_uid": FFDM,
        "view_position": None,
        "rows": 2294,
        "columns": 1914,
        "patient_age_years": 56.0,
    }
    base.update(kw)
    return StudyMetadata.model_validate(base)


@pytest.mark.requirement("REQ-144")
def test_mg_plugin_accepts_ffdm() -> None:
    plugin = MGPlugin()
    assert plugin.implemented
    assert plugin.validate(_meta()).accepted
    assert plugin.validate(_meta(view_position="MLO")).accepted


@pytest.mark.parametrize(
    ("kw", "fragment"),
    [
        ({"modality": "DX"}, "não é MG"),
        ({"sop_class_uid": BREAST_TOMOSYNTHESIS_STORAGE}, "tomossíntese"),
        ({"number_of_frames": 60}, "multiframe"),
        ({"view_position": "XCCL"}, "fora do escopo"),
        ({"rows": 128}, "abaixo do mínimo"),
        ({"patient_age_years": 15.0}, "pediátrica"),
    ],
)
@pytest.mark.requirement("REQ-144")
def test_mg_plugin_rejects_out_of_scope(kw: dict[str, object], fragment: str) -> None:
    report = MGPlugin().validate(_meta(**kw))
    assert not report.accepted
    assert any(fragment in r for r in report.reasons)


@pytest.mark.requirement("REQ-144")
def test_mg_plugin_preprocess_does_not_normalize() -> None:
    arr = np.arange(12, dtype=np.float64).reshape(3, 4) * 100
    out = MGPlugin().preprocess(arr, _meta())
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, arr.astype(np.float32))
    with pytest.raises(UnsupportedInputError):
        MGPlugin().preprocess(np.zeros((2, 3, 4)), _meta())
    with pytest.raises(UnsupportedInputError):
        MGPlugin().preprocess(np.array([[np.nan, 1.0]]), _meta())
