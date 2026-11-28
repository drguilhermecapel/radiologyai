"""Plugin US (quadro e cine), QA de imagem e detecção de anotações gravadas."""

from __future__ import annotations

import numpy as np
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from us_support import have, sector_frame, us_image

from radiologyai.analysis.types import AbstentionReason, ImageKind
from radiologyai.errors import UnsupportedInputError
from radiologyai.io.metadata import StudyMetadata
from radiologyai.modalities.us import USPlugin
from radiologyai.modalities.us.qa import exam_qa, overlay_mask


def meta(**kw: object) -> StudyMetadata:
    base: dict[str, object] = {"modality": "US", "rows": 480, "columns": 640}
    base.update(kw)
    return StudyMetadata.model_validate(base)


@pytest.mark.requirement("REQ-149")
def test_plugin_is_implemented_cine() -> None:
    p = USPlugin()
    assert p.implemented is True
    assert p.dimensionality == "cine"


@pytest.mark.parametrize(
    ("kw", "needle"),
    [
        ({"modality": "CT"}, "não é US"),
        ({"rows": 64}, "abaixo do mínimo"),
        ({"number_of_frames": 5000}, "quadros"),
        ({"patient_age_years": 9.0}, "pediátrica"),
    ],
)
@pytest.mark.requirement("REQ-149")
def test_plugin_rejects_out_of_scope(kw: dict[str, object], needle: str) -> None:
    report = USPlugin().validate(meta(**kw))
    assert not report.accepted
    assert any(needle in r for r in report.reasons)


@pytest.mark.requirement("REQ-149")
def test_plugin_accepts_cine_and_still() -> None:
    assert USPlugin().validate(meta(number_of_frames=60)).accepted
    assert USPlugin().validate(meta()).accepted


@pytest.mark.requirement("REQ-149")
def test_preprocess_shapes_and_finiteness() -> None:
    p = USPlugin()
    assert p.preprocess(np.zeros((8, 8)), meta()).dtype == np.float32
    assert p.preprocess(np.zeros((3, 8, 8)), meta()).shape == (3, 8, 8)
    assert p.preprocess(np.zeros((8, 8, 3)), meta()).shape == (8, 8, 3)
    with pytest.raises(UnsupportedInputError):
        p.preprocess(np.zeros(8), meta())
    bad = np.zeros((8, 8))
    bad[0, 0] = np.nan
    with pytest.raises(UnsupportedInputError):
        p.preprocess(bad, meta())


@pytest.mark.skipif(not have("scipy"), reason="scipy ausente (extra eval)")
@pytest.mark.requirement("REQ-149")
def test_overlay_detects_text_outside_sector_and_color_marks() -> None:
    gray = sector_frame()
    rgb = np.repeat(gray[..., None], 3, axis=2)
    rgb[2:8, 2:40] = 255.0  # "texto" fora do setor
    rgb[100:104, 118:122] = (255.0, 255.0, 0.0)  # marca amarela dentro do setor
    report = overlay_mask(us_image(rgb, color=True), 0)
    assert report.mask[4, 20]
    assert report.mask[101, 120]
    assert not report.mask[150, 120]
    assert report.color_fraction > 0
    assert report.outside_sector_fraction > 0
    assert report.sector_bbox is not None


@pytest.mark.skipif(not have("scipy"), reason="scipy ausente (extra eval)")
@pytest.mark.requirement("REQ-149")
def test_exam_qa_reports_cine_spacing_and_rejects_heavy_overlay() -> None:
    clip = np.stack([sector_frame()] * 4)
    ok = exam_qa(us_image(clip, kind=ImageKind.CINE, frame_time_ms=20.0))
    assert ok.applicable
    names = {c.name: c for c in ok.checks}
    assert names["n_frames"].value == 4.0
    assert names["frame_time_ms"].value == 20.0
    no_sp = exam_qa(us_image(sector_frame(), spacing=None), require_spacing=True)
    assert not no_sp.applicable
    assert AbstentionReason.GEOMETRY in no_sp.reasons
    rgb = np.repeat(sector_frame()[..., None], 3, axis=2)
    rgb[:, :, 0] = 250.0  # Doppler/cor dominando o quadro
    heavy = exam_qa(us_image(rgb, color=True))
    assert not heavy.applicable
    assert AbstentionReason.QUALITY in heavy.reasons


def _us_multiframe(path: object) -> None:
    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.3.1"
    fm.MediaStorageSOPInstanceUID = generate_uid()
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = fm
    ds.SOPClassUID = fm.MediaStorageSOPClassUID
    ds.SOPInstanceUID = fm.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "US"
    ds.Rows, ds.Columns = 32, 40
    ds.NumberOfFrames = 3
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.FrameTime = "25"
    region = Dataset()
    region.PhysicalUnitsXDirection = 3
    region.PhysicalUnitsYDirection = 3
    region.PhysicalDeltaX = 0.02
    region.PhysicalDeltaY = 0.03
    ds.SequenceOfUltrasoundRegions = [region]
    ds.PixelData = (np.arange(3 * 32 * 40) % 251).astype(np.uint8).tobytes()
    ds.save_as(path, enforce_file_format=True)


@pytest.mark.requirement("REQ-149")
def test_dicom_multiframe_us_loads_as_cine_with_region_spacing(tmp_path: object) -> None:
    from pathlib import Path

    from radiologyai.analysis.imaging import load_dicom

    p = Path(str(tmp_path)) / "clip.dcm"
    _us_multiframe(p)
    image = load_dicom(p)
    assert image.kind is ImageKind.CINE
    assert image.pixels.shape == (3, 32, 40)
    assert image.spacing_mm == pytest.approx((0.3, 0.2))
    assert image.meta.pixel_spacing_source == "UltrasoundRegions"
    assert image.meta.frame_time_ms == 25.0
