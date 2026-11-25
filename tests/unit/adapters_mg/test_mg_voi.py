"""VOI completa e conversão 8 bits fiel ao original; portão de aplicabilidade MG (REQ-140)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from mg_support import gradient, mg_image, voi
from pydicom.dataset import Dataset
from pydicom.pixels.processing import apply_voi_lut

from radiologyai.analysis.adapters.mg_common import mg_applicability, to_uint8
from radiologyai.analysis.base import ModelInputs
from radiologyai.analysis.manifest import load_manifest
from radiologyai.analysis.types import AbstentionReason
from radiologyai.catalog.schema import ExamType
from radiologyai.errors import UnsupportedInputError
from radiologyai.modalities.mg.voi import apply_voi, voi_from_dataset

MANIFESTS = "src/radiologyai/analysis/manifests"


def _dataset(params: dict[str, Any]) -> Dataset:
    ds = Dataset()
    ds.PhotometricInterpretation = params["photometric"]
    ds.BitsStored = params["bits_stored"]
    ds.PixelRepresentation = 0
    ds.RescaleSlope = 1
    ds.RescaleIntercept = 0
    if params["window_center"]:
        ds.WindowCenter = params["window_center"][0]
        ds.WindowWidth = params["window_width"][0]
    if params["function"]:
        ds.VOILUTFunction = params["function"]
    if params["lut"]:
        item = Dataset()
        item.LUTDescriptor = params["lut"]["descriptor"]
        item.LUTData = params["lut"]["data"]
        ds.VOILUTSequence = [item]
    return ds


def _original(raw: np.ndarray, ds: Dataset) -> np.ndarray:
    """``load_image_from_dicom`` do ``ianpan/mammoscreen`` (sobre o pixel_array cru)."""
    arr = apply_voi_lut(raw, ds)
    if ds.PhotometricInterpretation == "MONOCHROME1":
        arr = arr.max() - arr
    arr = arr - arr.min()
    arr = arr / arr.max()
    return (arr * 255).astype("uint8")


LUT = {"descriptor": [4096, 0, 8], "data": [int(v) for v in (np.arange(4096) // 16) ** 1]}
CASES = {
    "sigmoid": voi(center=1800.0, width=1200.0, function="SIGMOID"),
    "linear": voi(center=2000.0, width=1500.0),
    "linear_exact": voi(center=2000.0, width=1500.0, function="LINEAR_EXACT"),
    "no_voi": voi(),
    "lut": voi(lut=LUT, center=10.0, width=20.0),  # LUT tem precedência sobre a janela
    "mono1_sigmoid": voi(
        photometric="MONOCHROME1", center=1800.0, width=1200.0, function="SIGMOID"
    ),
}


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.requirement("REQ-140")
def test_mg_to_uint8_matches_original_algorithm(name: str) -> None:
    params = CASES[name]
    raw = gradient().astype(np.uint16)
    lo, hi = 0.0, 4095.0
    stored = raw.astype(np.float32)
    # o carregador comum entrega MONOCHROME2: inverte MONOCHROME1 pela faixa representável
    pixels = (lo + hi) - stored if params["photometric"] == "MONOCHROME1" else stored
    got = to_uint8(mg_image(pixels, voi_params=params, value_range=(lo, hi)))
    want = _original(raw, _dataset(params))
    np.testing.assert_array_equal(got, want)


@pytest.mark.requirement("REQ-140")
def test_mg_sigmoid_differs_from_linear_window() -> None:
    raw = gradient().astype(np.float32)
    sig = to_uint8(mg_image(raw, voi_params=CASES["sigmoid"]))
    lin = to_uint8(mg_image(raw, voi_params=voi(center=1800.0, width=1200.0)))
    assert not np.array_equal(sig, lin)


@pytest.mark.requirement("REQ-140")
def test_mg_voi_from_dataset_round_trip() -> None:
    ds = _dataset(CASES["lut"])
    params = voi_from_dataset(ds)
    assert params["lut"]["descriptor"] == [4096, 0, 8]
    assert params["window_center"] == [10.0]
    assert params["function"] is None
    raw = gradient().astype(np.int64)
    np.testing.assert_allclose(apply_voi(raw, params), apply_voi_lut(raw, ds))


@pytest.mark.requirement("REQ-140")
def test_mg_unknown_voi_function_rejected() -> None:
    with pytest.raises(UnsupportedInputError, match="VOILUTFunction"):
        apply_voi(gradient(), voi(center=1.0, width=2.0, function="LOG"))


@pytest.mark.requirement("REQ-140")
def test_mg_dicom_without_voi_is_refused() -> None:
    image = mg_image(gradient())
    bare = type(image)(
        pixels=image.pixels,
        kind=image.kind,
        modality=image.modality,
        meta=image.meta,
        value_range=image.value_range,
    )
    with pytest.raises(UnsupportedInputError, match="load_mammogram"):
        to_uint8(bare)
    manifest, _ = load_manifest(_manifest("mammoscreen"))
    qa = mg_applicability(manifest, ModelInputs(images=(bare,), exam_type=ExamType.MG))
    assert not qa.applicable
    assert AbstentionReason.INPUT in qa.reasons


@pytest.mark.requirement("REQ-140")
def test_mg_raster_8bit_passes_through_and_16bit_refused() -> None:
    pixels = np.round(gradient(top=255))
    img = mg_image(pixels, source_format="raster", value_range=(0.0, 255.0))
    np.testing.assert_array_equal(to_uint8(img), pixels.astype(np.uint8))
    with pytest.raises(UnsupportedInputError, match="8 bits"):
        to_uint8(mg_image(gradient(), source_format="raster", value_range=(0.0, 65535.0)))


@pytest.mark.requirement("REQ-140")
def test_mg_constant_image_refused() -> None:
    with pytest.raises(UnsupportedInputError, match="constante"):
        to_uint8(mg_image(np.full((300, 300), 7.0)))


def _manifest(model_id: str) -> Any:
    from pathlib import Path

    return Path(__file__).resolve().parents[3] / MANIFESTS / f"{model_id}.yaml"


@pytest.mark.requirement("REQ-140")
def test_mg_applicability_accepts_partial_views_and_rejects_processing() -> None:
    manifest, _ = load_manifest(_manifest("mammoscreen"))
    one = mg_image(gradient(), laterality="R", view="MLO")
    qa = mg_applicability(manifest, ModelInputs(images=(one,), exam_type=ExamType.MG))
    assert qa.applicable, qa
    raw = mg_image(gradient(), intent="FOR PROCESSING")
    qa = mg_applicability(manifest, ModelInputs(images=(raw,), exam_type=ExamType.MG))
    assert AbstentionReason.ACQUISITION in qa.reasons
    young = mg_image(gradient(), age=None)
    qa = mg_applicability(manifest, ModelInputs(images=(young,), exam_type=ExamType.MG))
    assert AbstentionReason.POPULATION in qa.reasons
    nolat = mg_image(gradient(), laterality=None)
    qa = mg_applicability(manifest, ModelInputs(images=(nolat,), exam_type=ExamType.MG))
    assert qa.reasons == (AbstentionReason.MISSING_INPUT,)
    qa = mg_applicability(manifest, ModelInputs(images=(one,), exam_type=ExamType.CXR))
    assert AbstentionReason.EXAM_TYPE in qa.reasons
