"""Construção de ImageData: DICOM, série → volume, NIfTI, raster e cine.

Séries sintéticas são montadas com pydicom em memória (rampas determinísticas,
nunca pixels de paciente). Os testes sobre amostras reais em
``/root/medai-cache/datasets`` pulam quando o cache não existe.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
import pytest
from cache_paths import MEDAI_CACHE
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from radiologyai.analysis.imaging import (
    load_cine,
    load_dicom,
    load_nifti,
    load_path,
    load_raster,
    load_series,
)
from radiologyai.analysis.routing import infer_exam_type, role_of
from radiologyai.analysis.types import ImageKind, Modality
from radiologyai.catalog.schema import ExamType
from radiologyai.errors import UnsupportedInputError
from radiologyai.io.series import list_series, load_series_volume

US_MULTIFRAME = "1.2.840.10008.5.1.4.1.1.3.1"
DX_SOP = "1.2.840.10008.5.1.4.1.1.1.1"
CACHE = MEDAI_CACHE / "datasets"


def _ds(rows: int, cols: int, sop_class: str, **attrs: Any) -> Dataset:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = sop_class
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = attrs.pop("StudyInstanceUID", generate_uid())
    ds.SeriesInstanceUID = attrs.pop("SeriesInstanceUID", generate_uid())
    ds.Rows, ds.Columns = rows, cols
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.PhotometricInterpretation = "MONOCHROME2"
    for key, value in attrs.items():
        setattr(ds, key, value)
    return ds


def _write(ds: Dataset, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(path, enforce_file_format=True)
    return path


def _ct_series(
    directory: Path,
    z_positions: list[float],
    *,
    series_uid: str | None = None,
    intercepts: list[float] | None = None,
    shuffle_names: bool = True,
) -> str:
    series_uid = series_uid or generate_uid()
    study_uid = generate_uid()
    n = len(z_positions)
    for i, z in enumerate(z_positions):
        ds = _ds(
            4,
            5,
            CTImageStorage,
            SeriesInstanceUID=series_uid,
            StudyInstanceUID=study_uid,
            Modality="CT",
            BodyPartExamined="HEAD",
            ImageOrientationPatient=[1, 0, 0, 0, 1, 0],
            ImagePositionPatient=[-10.0, -20.0, z],
            PixelSpacing=[0.5, 0.7],
            SliceThickness=2.5,
            RescaleSlope=1,
            RescaleIntercept=(intercepts[i] if intercepts else -1024),
            InstanceNumber=n - i,  # ordem de instância propositalmente invertida
        )
        pixels = np.full((4, 5), 1000 + int(z * 10), dtype=np.uint16)
        ds.PixelData = pixels.tobytes()
        name = f"{(n - i):03d}.dcm" if shuffle_names else f"{i:03d}.dcm"
        _write(ds, directory / name)
    return series_uid


# --------------------------------------------------------------------------- #
# Séries
# --------------------------------------------------------------------------- #
@pytest.mark.requirement("REQ-106")
def test_series_is_sorted_by_position_with_geometry(tmp_path: Path) -> None:
    zs = [30.0, 0.0, 10.0, 20.0]
    _ct_series(tmp_path / "s", zs)
    image = load_series(tmp_path / "s")
    assert image.kind is ImageKind.VOLUME
    assert image.modality is Modality.CT
    assert image.pixels.shape == (4, 4, 5)
    # HU = 1000 + 10 z - 1024, em ordem crescente de z
    assert [float(image.pixels[k, 0, 0]) for k in range(4)] == [-24.0, 76.0, 176.0, 276.0]
    assert image.spacing_mm == (10.0, 0.5, 0.7)
    assert image.affine_lps is not None
    np.testing.assert_allclose(image.affine_lps[:3, 3], [-10.0, -20.0, 0.0])
    np.testing.assert_allclose(np.diag(image.affine_lps)[:3], [0.7, 0.5, 10.0])
    assert image.meta.source_format == "dicom_series"
    assert len(image.meta.sop_uids) == 4
    assert image.meta.n_frames == 4
    assert infer_exam_type([image]) is ExamType.CT_HEAD
    assert image.value_range == (-1024.0, 4095.0 - 1024.0)


def test_series_rescale_may_vary_per_slice(tmp_path: Path) -> None:
    _ct_series(tmp_path / "s", [0.0, 5.0], intercepts=[-1024.0, -1000.0])
    image = load_series(tmp_path / "s")
    assert float(image.pixels[0, 0, 0]) == pytest.approx(1000 - 1024)
    assert float(image.pixels[1, 0, 0]) == pytest.approx(1050 - 1000)


@pytest.mark.requirement("REQ-106")
def test_irregular_spacing_is_refused(tmp_path: Path) -> None:
    _ct_series(tmp_path / "s", [0.0, 1.0, 2.0, 5.0])
    with pytest.raises(UnsupportedInputError, match="irregular"):
        load_series(tmp_path / "s")


@pytest.mark.requirement("REQ-106")
def test_duplicate_positions_are_refused(tmp_path: Path) -> None:
    _ct_series(tmp_path / "s", [0.0, 1.0, 1.0])
    with pytest.raises(UnsupportedInputError, match="duplicadas"):
        load_series(tmp_path / "s")


@pytest.mark.requirement("REQ-106")
def test_multiple_series_require_a_choice(tmp_path: Path) -> None:
    first = _ct_series(tmp_path / "s" / "a", [0.0, 1.0])
    second = _ct_series(tmp_path / "s" / "b", [0.0, 2.0])
    assert set(list_series(tmp_path / "s")) == {first, second}
    with pytest.raises(UnsupportedInputError, match="séries"):
        load_series(tmp_path / "s")
    chosen = load_series(tmp_path / "s", series_uid=second)
    assert chosen.spacing_mm is not None
    assert chosen.spacing_mm[0] == pytest.approx(2.0)


def test_inconsistent_orientation_is_refused(tmp_path: Path) -> None:
    _ct_series(tmp_path / "s", [0.0, 1.0, 2.0])
    path = sorted((tmp_path / "s").glob("*.dcm"))[0]
    ds = pydicom.dcmread(path)
    ds.ImageOrientationPatient = [0, 1, 0, 0, 0, -1]
    ds.save_as(path)
    with pytest.raises(UnsupportedInputError, match="orientação"):
        load_series_volume(tmp_path / "s")


def test_single_slice_is_not_a_volume(tmp_path: Path) -> None:
    _ct_series(tmp_path / "s", [0.0])
    with pytest.raises(UnsupportedInputError, match="único"):
        load_series(tmp_path / "s")


# --------------------------------------------------------------------------- #
# Arquivo DICOM único
# --------------------------------------------------------------------------- #
@pytest.mark.requirement("REQ-106")
def test_monochrome1_is_inverted_by_representable_range(tmp_path: Path) -> None:
    ds = _ds(
        2,
        3,
        DX_SOP,
        Modality="DX",
        PhotometricInterpretation="MONOCHROME1",
        BodyPartExamined="CHEST",
        ViewPosition="PA",
        PatientAge="045Y",
        PatientSex="F",
        PatientName="NAO^DEVE^APARECER",
        ImagerPixelSpacing=[0.15, 0.15],
        WindowCenter=[2000, 1000],
        WindowWidth=[4000, 2000],
    )
    stored = np.array([[0, 100, 4095], [10, 20, 30]], dtype=np.uint16)
    ds.PixelData = stored.tobytes()
    image = load_dicom(_write(ds, tmp_path / "dx.dcm"))
    assert image.kind is ImageKind.PROJECTION
    assert image.value_range == (0.0, 4095.0)
    np.testing.assert_array_equal(image.pixels, (4095.0 - stored).astype(np.float32))
    assert image.window == (2000.0, 4000.0)  # VOI guardada, não aplicada
    assert image.spacing_mm == (0.15, 0.15)
    assert image.meta.pixel_spacing_source == "ImagerPixelSpacing"
    assert image.meta.photometric_interpretation == "MONOCHROME1"
    assert image.meta.patient_age_years == 45
    assert "NAO^DEVE" not in image.meta.model_dump_json()
    assert infer_exam_type([image]) is ExamType.CXR


def test_mammography_view_from_view_code_sequence(tmp_path: Path) -> None:
    ds = _ds(2, 2, DX_SOP, Modality="MG", ImageLaterality="L", BodyPartExamined="BREAST")
    code = Dataset()
    code.CodeValue = "399368009"
    code.CodingSchemeDesignator = "SCT"
    code.CodeMeaning = "medio-lateral oblique"
    ds.ViewCodeSequence = [code]
    ds.PixelData = np.zeros((2, 2), dtype=np.uint16).tobytes()
    image = load_dicom(_write(ds, tmp_path / "mg.dcm"))
    assert image.meta.view_position == "MLO"
    assert role_of(image) == "L-MLO"


def test_ct_slice_is_in_hounsfield_units(tmp_path: Path) -> None:
    ds = _ds(2, 2, CTImageStorage, Modality="CT", RescaleSlope=2, RescaleIntercept=-1024)
    ds.PixelData = np.array([[0, 512], [1024, 2047]], dtype=np.uint16).tobytes()
    image = load_dicom(_write(ds, tmp_path / "ct.dcm"))
    np.testing.assert_array_equal(image.pixels, [[-1024, 0], [1024, 3070]])


def test_modality_lut_sequence_is_applied(tmp_path: Path) -> None:
    ds = _ds(1, 3, DX_SOP, Modality="DX")
    ds.BitsStored = 2
    ds.HighBit = 1
    item = Dataset()
    item.LUTDescriptor = [4, 0, 16]
    item.LUTData = np.array([10, 20, 30, 40], dtype=np.uint16).tobytes()
    item.ModalityLUTType = "US"
    ds.ModalityLUTSequence = [item]
    ds.PixelData = np.array([[0, 1, 3]], dtype=np.uint16).tobytes()
    image = load_dicom(_write(ds, tmp_path / "lut.dcm"))
    np.testing.assert_array_equal(image.pixels, [[10, 20, 40]])
    assert image.value_range == (10.0, 40.0)


def test_unsupported_modality_and_multiframe_ct(tmp_path: Path) -> None:
    ds = _ds(2, 2, DX_SOP, Modality="PT")
    ds.PixelData = np.zeros((2, 2), dtype=np.uint16).tobytes()
    with pytest.raises(UnsupportedInputError, match="modalidade"):
        load_dicom(_write(ds, tmp_path / "pt.dcm"))
    ds = _ds(2, 2, CTImageStorage, Modality="CT", NumberOfFrames=3)
    ds.PixelData = np.zeros((3, 2, 2), dtype=np.uint16).tobytes()
    with pytest.raises(UnsupportedInputError, match="multiframe"):
        load_dicom(_write(ds, tmp_path / "mf.dcm"))


def test_us_multiframe_is_cine_with_region_spacing(tmp_path: Path) -> None:
    ds = _ds(
        4,
        6,
        US_MULTIFRAME,
        Modality="US",
        NumberOfFrames=3,
        FrameTime=33.3,
        StudyDescription="ECHO TTE",
    )
    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 8, 8, 7
    region = Dataset()
    region.PhysicalUnitsXDirection = 3
    region.PhysicalUnitsYDirection = 3
    region.PhysicalDeltaX = 0.03
    region.PhysicalDeltaY = 0.02
    ds.SequenceOfUltrasoundRegions = [region]
    frames = np.stack([np.full((4, 6), v, dtype=np.uint8) for v in (10, 20, 30)])
    ds.PixelData = frames.tobytes()
    image = load_cine(_write(ds, tmp_path / "cine.dcm"))
    assert image.kind is ImageKind.CINE
    assert image.pixels.shape == (3, 4, 6)
    assert image.meta.frame_time_ms == pytest.approx(33.3)
    assert image.spacing_mm == pytest.approx((0.2, 0.3))
    assert image.meta.pixel_spacing_source == "UltrasoundRegions"
    assert infer_exam_type([image]) is ExamType.ECHO


def test_single_frame_is_not_cine(tmp_path: Path) -> None:
    ds = _ds(2, 2, DX_SOP, Modality="US")
    ds.PixelData = np.zeros((2, 2), dtype=np.uint16).tobytes()
    with pytest.raises(UnsupportedInputError, match="cine"):
        load_cine(_write(ds, tmp_path / "us.dcm"))


# --------------------------------------------------------------------------- #
# NIfTI e raster
# --------------------------------------------------------------------------- #
needs_nibabel = pytest.mark.skipif(
    importlib.util.find_spec("nibabel") is None, reason="nibabel ausente (extra imaging)"
)
needs_pillow = pytest.mark.skipif(
    importlib.util.find_spec("PIL") is None, reason="Pillow ausente (extra imaging)"
)


@needs_nibabel
def test_nifti_volume_is_transposed_and_converted_to_lps(tmp_path: Path) -> None:
    import nibabel as nib

    data = np.arange(2 * 3 * 4, dtype=np.int16).reshape(2, 3, 4)  # (i, j, k)
    affine = np.diag([0.8, 0.9, 2.0, 1.0])
    affine[:3, 3] = [5.0, 6.0, 7.0]
    path = tmp_path / "vol.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), str(path))
    image = load_nifti(path, modality=Modality.CT)
    assert image.pixels.shape == (4, 3, 2)
    assert float(image.pixels[3, 2, 1]) == float(data[1, 2, 3])
    assert image.spacing_mm == pytest.approx((2.0, 0.9, 0.8))
    assert image.affine_lps is not None
    np.testing.assert_allclose(image.affine_lps[:3, 3], [-5.0, -6.0, 7.0])
    assert image.value_range == (-32768.0, 32767.0)
    with pytest.raises(UnsupportedInputError, match="modality"):
        load_path(path)


@needs_nibabel
def test_nifti_cine_sequence(tmp_path: Path) -> None:
    import nibabel as nib

    data = np.zeros((6, 5, 1, 3), dtype=np.uint8)
    for t in range(3):
        data[..., 0, t] = 10 * t
    img = nib.Nifti1Image(data, np.diag([0.3, 0.3, 1.0, 1.0]))
    img.header.set_zooms((0.3, 0.3, 1.0, 0.04))
    img.header.set_xyzt_units("mm", "sec")
    path = tmp_path / "seq.nii.gz"
    nib.save(img, str(path))
    image = load_cine(path)
    assert image.kind is ImageKind.CINE
    assert image.pixels.shape == (3, 5, 6)
    assert [float(image.pixels[t].mean()) for t in range(3)] == [0.0, 10.0, 20.0]
    assert image.meta.frame_time_ms == pytest.approx(40.0)


@needs_pillow
def test_raster_png_is_ot_and_grayscale(tmp_path: Path) -> None:
    from PIL import Image

    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    path = tmp_path / "x.png"
    Image.fromarray(rgb).save(path)
    image = load_raster(path)
    assert image.modality is Modality.OT
    assert image.pixels.shape == (4, 5)
    assert image.value_range == (0.0, 255.0)
    assert infer_exam_type([image]) is ExamType.UNKNOWN
    colored = load_raster(path, keep_color=True, modality=Modality.US)
    assert colored.color
    assert colored.pixels.shape == (4, 5, 3)
    tagged = load_raster(path, modality=Modality.DX, body_part="CHEST", view_position="PA")
    assert infer_exam_type([tagged]) is ExamType.CXR


@needs_pillow
def test_raster_16bit_keeps_range(tmp_path: Path) -> None:
    from PIL import Image

    arr = np.array([[0, 1000], [40000, 65535]], dtype=np.uint16)
    path = tmp_path / "x16.png"
    Image.fromarray(arr).save(path)
    image = load_raster(path, modality=Modality.DX)
    assert image.value_range == (0.0, 65535.0)
    assert float(image.pixels.max()) == 65535.0


# --------------------------------------------------------------------------- #
# Amostras reais (pulam sem o cache local)
# --------------------------------------------------------------------------- #
LIDC = CACHE / "samples-ct" / "tcia" / "LIDC-IDRI-0078"
CAMUS = CACHE / "camus-sample" / "patient0001" / "patient0001_2CH_half_sequence.nii.gz"
CXR_DICOM = CACHE / "samples-cxr-dicom"


@pytest.mark.skipif(not LIDC.is_dir(), reason="amostra LIDC ausente do cache local")
def test_real_lidc_series_loads_as_hu_volume() -> None:
    image = load_series(LIDC)
    assert image.kind is ImageKind.VOLUME
    assert image.pixels.shape[0] > 10
    assert image.spacing_mm is not None
    assert all(s > 0 for s in image.spacing_mm)
    assert float(np.percentile(image.pixels, 1)) < -500  # ar em HU
    assert float(image.pixels.max()) > 200  # osso em HU


@pytest.mark.skipif(not CAMUS.is_file(), reason="amostra CAMUS ausente do cache local")
@needs_nibabel
def test_real_camus_sequence_loads_as_cine() -> None:
    image = load_cine(CAMUS)
    assert image.kind is ImageKind.CINE
    assert image.pixels.shape[0] > 5


@pytest.mark.skipif(not CXR_DICOM.is_dir(), reason="amostras CXR DICOM ausentes")
def test_real_cxr_dicoms_load_as_projection() -> None:
    files = sorted(CXR_DICOM.rglob("*.dcm"))
    assert files
    for path in files:
        image = load_dicom(path)
        assert image.kind is ImageKind.PROJECTION
        assert image.modality in (Modality.CR, Modality.DX)
        lo, hi = image.value_range
        assert lo <= float(image.pixels.min())
        assert float(image.pixels.max()) <= hi


CMMD = CACHE / "cmmd-sample"


@pytest.mark.skipif(not CMMD.is_dir(), reason="amostra CMMD ausente do cache local")
def test_real_cmmd_mammograms_have_roles() -> None:
    roles = {role_of(load_dicom(p)) for p in sorted(CMMD.rglob("*.dcm"))}
    assert None not in roles
    assert roles <= {"L-CC", "R-CC", "L-MLO", "R-MLO"}
