"""Pré-processamento de RX para os modelos TorchXRayVision (sem pesos)."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from radiologyai.analysis.adapters.xr_common import (
    center_crop_box,
    crop_and_resize,
    xrv_input,
    xrv_normalize,
)
from radiologyai.analysis.imaging import load_dicom
from radiologyai.analysis.types import ImageData, ImageKind, ImageMeta, Modality
from radiologyai.errors import UnsupportedInputError

HAS_SKIMAGE = importlib.util.find_spec("skimage") is not None
DX_SOP = "1.2.840.10008.5.1.4.1.1.1.1"


def _image(pixels: np.ndarray, value_range: tuple[float, float], color: bool = False) -> ImageData:
    meta = ImageMeta(source_format="raster", content_sha256=hashlib.sha256(b"x").hexdigest())
    return ImageData(
        pixels=pixels.astype(np.float32),
        kind=ImageKind.PROJECTION,
        modality=Modality.DX,
        meta=meta,
        value_range=value_range,
        color=color,
    )


def _dicom(path: Path, stored: np.ndarray, photometric: str, bits: int) -> Path:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = DX_SOP
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = DX_SOP
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "DX"
    ds.ViewPosition = "PA"
    ds.Rows, ds.Columns = stored.shape
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = 16
    ds.BitsStored = bits
    ds.HighBit = bits - 1
    ds.PixelRepresentation = 0
    ds.PhotometricInterpretation = photometric
    ds.PixelData = stored.astype(np.uint16).tobytes()
    ds.save_as(path, enforce_file_format=True)
    return path


@pytest.mark.requirement("REQ-120")
def test_16bit_values_scale_without_wraparound() -> None:
    # Valores > 255 num arquivo de 16 bits: um cast para uint8 faria wraparound.
    pixels = np.array([[0, 255, 256, 4095, 65535]], dtype=np.float32)
    out = xrv_normalize(_image(pixels, (0.0, 65535.0)))
    expected = (2 * pixels / 65535.0 - 1) * 1024
    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-3)
    assert np.all(np.diff(out[0]) > 0), "a ordem dos valores deve ser preservada"
    assert out.min() == -1024.0
    assert out.max() == 1024.0


@pytest.mark.requirement("REQ-120")
def test_scaling_uses_representable_range_not_pixel_max() -> None:
    # Imagem 12 bits cujo máximo observado é 1000: escala pela faixa (4095), não pelo máximo.
    pixels = np.array([[0.0, 500.0, 1000.0]], dtype=np.float32)
    out = xrv_normalize(_image(pixels, (0.0, 4095.0)))
    assert out[0, 2] == pytest.approx((2 * 1000 / 4095 - 1) * 1024, abs=1e-3)
    assert out.max() < 0, "um escalonamento pelo máximo dos pixels levaria o pico a +1024"


@pytest.mark.requirement("REQ-120")
def test_signed_value_range_is_honoured() -> None:
    pixels = np.array([[-2048.0, 0.0, 2047.0]], dtype=np.float32)
    out = xrv_normalize(_image(pixels, (-2048.0, 2047.0)))
    assert out[0, 0] == pytest.approx(-1024.0)
    assert out[0, 2] == pytest.approx(1024.0)


@pytest.mark.requirement("REQ-120")
def test_pixels_outside_representable_range_are_refused() -> None:
    pixels = np.array([[0.0, 300.0]], dtype=np.float32)
    with pytest.raises(UnsupportedInputError, match="fora da faixa representável"):
        xrv_normalize(_image(pixels, (0.0, 255.0)))


@pytest.mark.requirement("REQ-120")
def test_non_finite_pixels_are_refused() -> None:
    pixels = np.array([[0.0, np.nan]], dtype=np.float32)
    with pytest.raises(UnsupportedInputError, match="não finitos"):
        xrv_normalize(_image(pixels, (0.0, 255.0)))


@pytest.mark.requirement("REQ-120")
def test_color_raster_uses_channel_mean() -> None:
    rgb = np.zeros((2, 2, 3), dtype=np.float32)
    rgb[..., 0] = 255.0
    out = xrv_normalize(_image(rgb, (0.0, 255.0), color=True))
    np.testing.assert_allclose(out, (2 * 85.0 / 255.0 - 1) * 1024, atol=1e-3)


@pytest.mark.requirement("REQ-120")
def test_monochrome1_dicom_is_inverted_before_scaling(tmp_path: Path) -> None:
    stored = np.array([[0, 1023], [512, 100]], dtype=np.uint16)
    mono1 = load_dicom(_dicom(tmp_path / "m1.dcm", stored, "MONOCHROME1", 10))
    mono2 = load_dicom(_dicom(tmp_path / "m2.dcm", 1023 - stored, "MONOCHROME2", 10))
    assert mono1.value_range == (0.0, 1023.0)
    a, b = xrv_normalize(mono1), xrv_normalize(mono2)
    np.testing.assert_allclose(a, b, atol=1e-3)
    # stored 0 em MONOCHROME1 é o mais claro => +1024
    assert a[0, 0] == pytest.approx(1024.0)
    assert a[0, 1] == pytest.approx(-1024.0)


@pytest.mark.requirement("REQ-120")
def test_voi_window_is_not_baked_into_pixels(tmp_path: Path) -> None:
    stored = np.array([[0, 4095], [2048, 1000]], dtype=np.uint16)
    path = _dicom(tmp_path / "w.dcm", stored, "MONOCHROME2", 12)
    import pydicom

    ds = pydicom.dcmread(path)
    ds.WindowCenter, ds.WindowWidth = 1000, 200
    ds.save_as(path)
    image = load_dicom(path)
    assert image.window == (1000.0, 200.0)
    out = xrv_normalize(image)
    assert out[0, 0] == pytest.approx(-1024.0)
    assert out[0, 1] == pytest.approx(1024.0)
    assert out[1, 1] == pytest.approx((2 * 1000 / 4095 - 1) * 1024, abs=1e-2)


def test_center_crop_box_matches_xrv_arithmetic() -> None:
    assert center_crop_box(10, 6) == (2, 0, 6)
    assert center_crop_box(6, 11) == (0, 2, 6)  # 11 // 2 - 6 // 2
    assert center_crop_box(7, 7) == (0, 0, 7)


@pytest.mark.skipif(not HAS_SKIMAGE, reason="scikit-image ausente (extra ml)")
@pytest.mark.requirement("REQ-120")
def test_crop_and_resize_shape_and_range() -> None:
    plane = np.linspace(-1024, 1024, 300 * 400, dtype=np.float32).reshape(300, 400)
    out = crop_and_resize(plane, 224)
    assert out.shape == (224, 224)
    assert out.dtype == np.float32
    assert -1024.0 <= out.min() <= out.max() <= 1024.0


@pytest.mark.skipif(not HAS_SKIMAGE, reason="scikit-image ausente (extra ml)")
def test_xrv_input_refuses_volume() -> None:
    meta = ImageMeta(source_format="nifti", content_sha256=hashlib.sha256(b"v").hexdigest())
    volume = ImageData(
        pixels=np.zeros((2, 4, 4), dtype=np.float32),
        kind=ImageKind.VOLUME,
        modality=Modality.CT,
        meta=meta,
        value_range=(-1024.0, 3071.0),
    )
    with pytest.raises(UnsupportedInputError, match="projeção 2D"):
        xrv_input(volume, 224)
