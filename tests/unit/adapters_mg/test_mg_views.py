"""Incidência e lateralidade de mamografia a partir do cabeçalho (REQ-141)."""

from __future__ import annotations

import numpy as np
import pytest
from mg_support import gradient, mg_image
from pydicom.dataset import Dataset

from radiologyai.analysis.base import ModelInputs
from radiologyai.analysis.routing import role_of
from radiologyai.catalog.schema import ExamType
from radiologyai.modalities.mg.loader import with_dataset_details
from radiologyai.modalities.mg.views import (
    laterality_from_orientation,
    parse_laterality,
    parse_view,
)


def _code(value: str, meaning: str, scheme: str = "SCT") -> Dataset:
    item = Dataset()
    item.CodeValue = value
    item.CodingSchemeDesignator = scheme
    item.CodeMeaning = meaning
    return item


@pytest.mark.parametrize(
    ("view_position", "codes", "expected"),
    [
        ("CC", None, "CC"),
        (" mlo ", None, "MLO"),
        ("", [_code("399368009", "medio-lateral oblique")], "MLO"),
        (None, [_code("399162004", "cranio-caudal")], "CC"),
        (
            None,
            [{"CodeValue": "x", "CodingSchemeDesignator": "99X", "CodeMeaning": "Cranio-Caudal"}],
            "CC",
        ),
        (None, [_code("399260004", "medio-lateral")], "ML"),
        (None, [_code("123", "algo", scheme="99X")], None),
        (None, None, None),
    ],
)
@pytest.mark.requirement("REQ-141")
def test_mg_parse_view(view_position: object, codes: object, expected: str | None) -> None:
    assert parse_view(view_position, codes) == expected  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("orientation", "expected"),
    [
        (["P", "L"], "R"),  # CC direita (CMMD D1-0563)
        (["P", "FL"], "R"),  # MLO direita
        (["A", "FR"], "L"),  # MLO esquerda (CMMD D2-0424)
        (["A", "R"], "L"),
        ("A\\FR", "L"),
        (["A", "F"], None),  # sem componente lateral: ambíguo
        (["A", "LR"], None),
        (["A"], None),
        (None, None),
    ],
)
@pytest.mark.requirement("REQ-141")
def test_mg_laterality_from_orientation(orientation: object, expected: str | None) -> None:
    assert laterality_from_orientation(orientation) == expected


@pytest.mark.requirement("REQ-141")
def test_mg_parse_laterality_precedence() -> None:
    assert parse_laterality("R", "L", ["A", "FR"]) == ("R", "ImageLaterality")
    assert parse_laterality(None, "l", None) == ("L", "Laterality")
    assert parse_laterality("", None, ["P", "FL"]) == ("R", "PatientOrientation")
    assert parse_laterality("B", None, ["P", "FL"]) == (None, "ImageLaterality")
    assert parse_laterality(None, None, None) == (None, None)


@pytest.mark.requirement("REQ-141")
def test_mg_with_dataset_details_fills_missing_from_header() -> None:
    image = mg_image(gradient(), laterality=None, view=None)
    ds = Dataset()
    ds.PatientOrientation = ["P", "FL"]
    ds.ViewCodeSequence = [_code("399368009", "medio-lateral oblique")]
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsStored = 8
    ds.PixelRepresentation = 0
    ds.WindowCenter = 128
    ds.WindowWidth = 256
    ds.VOILUTFunction = "SIGMOID"
    out = with_dataset_details(image, ds)
    assert out.meta.laterality == "R"
    assert out.meta.view_position == "MLO"
    assert out.extra["laterality_source"] == "PatientOrientation"
    assert out.extra["dicom_voi"]["function"] == "SIGMOID"
    assert role_of(out) == "R-MLO"


@pytest.mark.requirement("REQ-141")
def test_mg_with_dataset_details_keeps_explicit_values() -> None:
    image = mg_image(gradient(), laterality="L", view="CC")
    ds = Dataset()
    ds.PatientOrientation = ["P", "FL"]  # contradiz; o valor explícito prevalece
    ds.PhotometricInterpretation = "MONOCHROME2"
    out = with_dataset_details(image, ds)
    assert (out.meta.laterality, out.meta.view_position) == ("L", "CC")
    assert "laterality_source" not in out.extra


@pytest.mark.requirement("REQ-141")
def test_mg_breast_views_use_only_own_side() -> None:
    from radiologyai.analysis.adapters.mg_common import breast_views, role_inputs

    images = (
        mg_image(gradient(), laterality="R", view="MLO"),
        mg_image(gradient(), laterality="L", view="CC"),
        mg_image(gradient(), laterality=None, view="CC"),  # sem lado: ignorada
        mg_image(gradient(), laterality="R", view="CC"),
        mg_image(np.zeros((300, 260)), laterality="L", view="XCCL"),  # fora de CC/MLO
    )
    selected = role_inputs(ModelInputs(images=images, exam_type=ExamType.MG))
    assert set(selected.roles) == {"L-CC", "R-CC", "R-MLO"}
    groups = breast_views(selected)
    assert [v for v, _ in groups["R"]] == ["CC", "MLO"]
    assert [v for v, _ in groups["L"]] == ["CC"]
    for lat, views in groups.items():
        for _, idx in views:
            assert selected.images[idx].meta.laterality == lat
