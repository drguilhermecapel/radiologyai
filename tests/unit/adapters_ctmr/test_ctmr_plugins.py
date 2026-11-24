"""Plugins de modalidade TC e RM: escopo antes do modelo e pré-processamento sem normalização."""

from __future__ import annotations

import numpy as np
import pytest

from radiologyai.errors import UnsupportedInputError
from radiologyai.io.metadata import StudyMetadata
from radiologyai.modalities.ct.plugin import CT_IMAGE_STORAGE, CTPlugin
from radiologyai.modalities.mr.plugin import MR_IMAGE_STORAGE, MRPlugin
from radiologyai.modalities.registry import resolve


def _meta(**kw) -> StudyMetadata:
    base = {
        "modality": "CT",
        "sop_class_uid": CT_IMAGE_STORAGE,
        "rows": 512,
        "columns": 512,
        "pixel_spacing": (0.7, 0.7),
        "slice_thickness": 1.0,
        "patient_age_years": 60.0,
    }
    base.update(kw)
    return StudyMetadata(**base)


@pytest.mark.requirement("REQ-137")
def test_registered_and_implemented():
    assert isinstance(resolve("CT"), CTPlugin)
    assert isinstance(resolve("mr"), MRPlugin)
    assert CTPlugin.implemented
    assert MRPlugin.implemented
    assert CTPlugin.dimensionality == MRPlugin.dimensionality == "3d"


@pytest.mark.requirement("REQ-137")
def test_ct_accepts_in_scope_series():
    report = CTPlugin().validate(_meta())
    assert report.accepted, report.reasons


@pytest.mark.requirement("REQ-137")
@pytest.mark.parametrize(
    ("override", "fragment"),
    [
        ({"modality": "MR"}, "não é CT"),
        ({"sop_class_uid": "1.2.840.10008.5.1.4.1.1.2.1"}, "SOP Class"),  # Enhanced CT
        ({"number_of_frames": 120}, "multiframe"),
        ({"pixel_spacing": None}, "PixelSpacing"),
        ({"rows": 32}, "abaixo do mínimo"),
        ({"patient_age_years": 9.0}, "pediátrica"),
    ],
)
def test_ct_rejects_out_of_scope(override, fragment):
    report = CTPlugin().validate(_meta(**override))
    assert not report.accepted
    assert any(fragment in r for r in report.reasons)


@pytest.mark.requirement("REQ-137")
def test_ct_missing_age_is_not_rejected_here():
    assert CTPlugin().validate(_meta(patient_age_years=None)).accepted


@pytest.mark.requirement("REQ-137")
def test_mr_scope():
    plugin = MRPlugin()
    assert plugin.validate(_meta(modality="MR", sop_class_uid=MR_IMAGE_STORAGE)).accepted
    report = plugin.validate(_meta(modality="MR"))  # SOP de TC numa série de RM
    assert not report.accepted


@pytest.mark.requirement("REQ-137")
def test_ct_preprocess_keeps_hu():
    vol = np.full((4, 8, 8), -1000.0, dtype=np.float64)
    vol[1:3, 2:6, 2:6] = 40.0
    out = CTPlugin().preprocess(vol, _meta())
    assert out.dtype == np.float32
    assert out.shape == (4, 8, 8)
    assert float(out.min()) == -1000.0
    assert float(out.max()) == 40.0


@pytest.mark.requirement("REQ-137")
@pytest.mark.parametrize(
    "vol",
    [
        np.zeros((8, 8), dtype=np.float32),  # 2D
        np.full((1, 8, 8), -1000.0, dtype=np.float32),  # um corte
        np.full((3, 8, 8), np.nan, dtype=np.float32),
        np.full((3, 8, 8), 1500.0, dtype=np.float32),  # sem ar: não parece HU
    ],
)
def test_ct_preprocess_rejects(vol):
    with pytest.raises(UnsupportedInputError):
        CTPlugin().preprocess(vol, _meta())


@pytest.mark.requirement("REQ-137")
def test_mr_preprocess():
    vol = np.arange(2 * 4 * 4, dtype=np.float64).reshape(2, 4, 4)
    out = MRPlugin().preprocess(vol, _meta(modality="MR"))
    assert out.dtype == np.float32
    assert np.array_equal(out, vol.astype(np.float32))
    with pytest.raises(UnsupportedInputError, match="constante"):
        MRPlugin().preprocess(np.ones((2, 4, 4)), _meta(modality="MR"))
