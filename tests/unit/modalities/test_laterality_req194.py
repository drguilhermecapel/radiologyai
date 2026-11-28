"""REQ-194 — lateralidade em TC, US e eco: só do cabeçalho DICOM, nunca do pixel.

Fixtures sintéticas (nenhum pixel de paciente). Padrão de REQ-043 (RX) e REQ-141 (mamografia).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from radiologyai.io.metadata import StudyMetadata, extract_metadata, resolve_laterality
from radiologyai.modalities.ct.plugin import CTPlugin
from radiologyai.modalities.us.plugin import USPlugin

US_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.6.1"


def _meta(**kw: str | None) -> StudyMetadata:
    return StudyMetadata(modality=kw.pop("modality", None) or "US", **kw)  # type: ignore[arg-type]


def _us_dataset(**tags: str) -> Dataset:
    ds = Dataset()
    ds.Modality = "US"
    ds.SOPClassUID = US_SOP_CLASS
    ds.Rows = 480
    ds.Columns = 640
    ds.PatientAge = "050Y"
    for name, value in tags.items():
        setattr(ds, name, value)
    return ds


@pytest.mark.requirement("REQ-194")
class TestResolveLaterality:
    def test_absent_is_unknown_not_guessed(self):
        side = resolve_laterality(_meta())
        assert side.value is None
        assert side.status == "unknown"
        assert side.label == "lateralidade desconhecida"
        assert not side.is_problem

    @pytest.mark.parametrize(
        ("tags", "value", "source"),
        [
            ({"laterality": "R"}, "R", "Laterality"),
            ({"image_laterality": "L"}, "L", "ImageLaterality"),
            ({"image_laterality": "L", "laterality": "L"}, "L", "ImageLaterality"),
            ({"frame_laterality": "R", "image_laterality": "R"}, "R", "FrameLaterality"),
            ({"image_laterality": "B"}, "B", "ImageLaterality"),
        ],
    )
    def test_agreeing_tags_give_side_and_source(self, tags, value, source):
        side = resolve_laterality(_meta(**tags))
        assert (side.value, side.status, side.source) == (value, "known", source)

    @pytest.mark.parametrize(
        "tags",
        [
            {"laterality": "L", "image_laterality": "R"},
            {"frame_laterality": "L", "laterality": "R"},
            {"image_laterality": "U", "laterality": "L"},
            {"frame_laterality": "L\\R"},
        ],
    )
    def test_conflict_is_unknown_with_explicit_detail(self, tags):
        side = resolve_laterality(_meta(**tags))
        assert side.value is None
        assert side.status == "conflict"
        assert side.is_problem
        assert side.label == "lateralidade desconhecida"
        assert "conflito" in (side.detail or "")

    @pytest.mark.parametrize(
        "tags", [{"laterality": "B"}, {"laterality": "U"}, {"image_laterality": "X"}]
    )
    def test_invalid_code_is_unknown(self, tags):
        side = resolve_laterality(_meta(**tags))
        assert side.value is None
        assert side.status == "invalid"
        assert side.is_problem

    def test_unpaired_has_no_side(self):
        side = resolve_laterality(_meta(image_laterality="U"))
        assert side.value is None
        assert side.status == "unpaired"
        assert not side.is_problem


@pytest.mark.requirement("REQ-194")
class TestFrameLateralityExtraction:
    @staticmethod
    def _anatomy(value: str) -> Dataset:
        item = Dataset()
        item.FrameLaterality = value
        group = Dataset()
        group.FrameAnatomySequence = [item]
        return group

    def test_shared_functional_group(self):
        ds = _us_dataset()
        ds.SharedFunctionalGroupsSequence = [self._anatomy("L")]
        meta = extract_metadata(ds)
        assert meta.frame_laterality == "L"
        assert resolve_laterality(meta).value == "L"

    def test_per_frame_values_that_differ_are_a_conflict(self):
        ds = _us_dataset()
        ds.PerFrameFunctionalGroupsSequence = [self._anatomy("L"), self._anatomy("R")]
        meta = extract_metadata(ds)
        assert meta.frame_laterality == "L\\R"
        assert resolve_laterality(meta).status == "conflict"


@pytest.mark.requirement("REQ-194")
class TestUSLaterality:
    def test_side_from_dicom_header(self):
        meta = extract_metadata(_us_dataset(Laterality="R", BodyPartExamined="BREAST"))
        assert USPlugin().validate(meta).accepted
        assert USPlugin.laterality(meta).value == "R"
        assert USPlugin.laterality(meta).label == "direita"

    def test_absent_side_accepted_and_reported_unknown(self):
        meta = extract_metadata(_us_dataset(BodyPartExamined="BREAST"))
        assert USPlugin().validate(meta).accepted
        assert USPlugin.laterality(meta).label == "lateralidade desconhecida"

    def test_side_not_taken_from_file_name_or_description(self):
        """Nome de arquivo ou descrição com 'LEFT' não decidem o lado."""
        ds = _us_dataset(SeriesDescription="BREAST LEFT", BodyPartExamined="BREAST")
        meta = extract_metadata(ds)
        assert USPlugin.laterality(meta).value is None

    def test_conflicting_tags_rejected_explicitly(self):
        meta = extract_metadata(_us_dataset(Laterality="L", ImageLaterality="R"))
        report = USPlugin().validate(meta)
        assert not report.accepted
        assert any("lateralidade DICOM não confiável" in r for r in report.reasons)
        assert any("lado desconhecido" in r for r in report.reasons)

    def test_invalid_code_rejected(self):
        report = USPlugin().validate(extract_metadata(_us_dataset(ImageLaterality="X")))
        assert not report.accepted

    def test_preprocess_keeps_left_right_order(self):
        frame = np.zeros((128, 160), dtype=np.float32)
        frame[:, :80] = 1.0
        out = USPlugin().preprocess(frame, _meta())
        assert float(out[:, :80].mean()) == 1.0
        assert float(out[:, 80:].mean()) == 0.0

    def test_echo_unpaired_or_absent_never_gets_a_side(self):
        """Eco: coração é estrutura ímpar; sem tag ou ``U`` => nenhum lado na saída."""
        for meta in (
            extract_metadata(_us_dataset(BodyPartExamined="HEART")),
            extract_metadata(_us_dataset(BodyPartExamined="HEART", ImageLaterality="U")),
        ):
            assert USPlugin().validate(meta).accepted
            assert USPlugin.laterality(meta).value is None


@pytest.mark.requirement("REQ-194")
class TestCTLaterality:
    def test_series_laterality_from_header(self, ct_dataset):
        ct_dataset.Laterality = "L"
        meta = extract_metadata(ct_dataset)
        assert CTPlugin().validate(meta).accepted
        assert CTPlugin.laterality(meta).value == "L"

    def test_absent_is_unknown_and_accepted(self, ct_dataset):
        meta = extract_metadata(ct_dataset)
        assert CTPlugin().validate(meta).accepted
        assert CTPlugin.laterality(meta).label == "lateralidade desconhecida"

    def test_conflict_rejected(self, ct_dataset):
        ct_dataset.Laterality = "R"
        ct_dataset.ImageLaterality = "L"
        report = CTPlugin().validate(extract_metadata(ct_dataset))
        assert not report.accepted
        assert any("lateralidade DICOM não confiável" in r for r in report.reasons)


def _write_ct_series(directory: Path, row_dir: tuple[float, float, float]) -> None:
    """4 cortes 16x16; um marcador brilhante nas colunas da DIREITA da matriz."""
    directory.mkdir()
    study, series = generate_uid(), generate_uid()
    for k in range(4):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = Dataset()
        ds.file_meta = meta
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID, ds.SeriesInstanceUID = study, series
        ds.Modality = "CT"
        ds.Rows = ds.Columns = 16
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.RescaleSlope, ds.RescaleIntercept = 1.0, -1024.0
        ds.PixelSpacing = [1.0, 1.0]
        ds.SliceThickness = 2.0
        ds.ImageOrientationPatient = [*row_dir, 0.0, 1.0, 0.0]
        ds.ImagePositionPatient = [0.0, 0.0, 2.0 * k]
        ds.InstanceNumber = k + 1
        px = np.zeros((16, 16), dtype=np.int16)
        px[6:10, 12:15] = 2000  # colunas altas = fim da linha da matriz
        ds.PixelData = px.tobytes()
        ds.save_as(directory / f"{k}.dcm", enforce_file_format=True)


@pytest.mark.requirement("REQ-194")
class TestCTSideIsGeometric:
    """O lado de uma estrutura de TC vem de ImageOrientationPatient, não da matriz."""

    @pytest.mark.parametrize(
        ("row_dir", "patient_left"),
        [((1.0, 0.0, 0.0), True), ((-1.0, 0.0, 0.0), False)],
    )
    def test_marker_side_follows_orientation_into_model_input(
        self, tmp_path, row_dir, patient_left
    ):
        nib = pytest.importorskip("nibabel")
        from radiologyai.analysis.adapters.ct_totalseg import write_nifti
        from radiologyai.analysis.imaging import load_series

        _write_ct_series(tmp_path / "s", row_dir)
        image = load_series(tmp_path / "s")
        write_nifti(image, tmp_path / "x.nii.gz")
        img = nib.load(str(tmp_path / "x.nii.gz"))
        data = np.asarray(img.dataobj)
        ijk = np.argwhere(data > 0).mean(axis=0)
        x_ras = float((img.affine @ np.array([*ijk, 1.0]))[0])
        # RAS: x negativo = esquerda do paciente; o mesmo pixel muda de lado com a orientação
        assert (x_ras < 0) is patient_left


@pytest.mark.requirement("REQ-194")
def test_image_meta_never_picks_a_side_when_tags_conflict() -> None:
    """A análise (image_meta_from_dataset) usa a mesma resolução: conflito => lado desconhecido."""
    from pydicom.dataset import Dataset

    from radiologyai.analysis.imaging import image_meta_from_dataset

    def meta(**tags: str) -> object:
        ds = Dataset()
        ds.Modality = "US"
        for k, v in tags.items():
            setattr(ds, k, v)
        return image_meta_from_dataset(ds, content_sha256="0" * 64).laterality

    assert meta(ImageLaterality="L", Laterality="R") is None
    assert meta(ImageLaterality="L") == "L"
    assert meta() is None


@pytest.mark.requirement("REQ-194")
@pytest.mark.requirement("REQ-141")
class TestMGLaterality:
    """Mamografia: conflito entre tags nunca vira lado (achado de mama no lado errado = H-06)."""

    def test_parse_laterality_conflict_gives_no_side(self) -> None:
        from radiologyai.modalities.mg.views import parse_laterality

        assert parse_laterality("L", "R", ["P", "L"]) == (None, "conflito")
        assert parse_laterality("L", "B") == (None, "conflito")
        assert parse_laterality("L", "L") == ("L", "ImageLaterality")
        assert parse_laterality(None, "R") == ("R", "Laterality")
        # PatientOrientation só sem nenhuma tag de lado
        assert parse_laterality(None, None, ["A", "R"]) == ("L", "PatientOrientation")
        assert parse_laterality("R", None, ["A", "R"]) == ("R", "ImageLaterality")

    def test_mg_gate_rejects_conflicting_tags(self) -> None:
        from radiologyai.io.metadata import extract_metadata
        from radiologyai.modalities.mg.plugin import MGPlugin

        ds = Dataset()
        ds.Modality = "MG"
        ds.ImageLaterality = "L"
        ds.Laterality = "R"
        report = MGPlugin().validate(extract_metadata(ds))
        assert not report.accepted
        assert any("lateralidade" in r for r in report.reasons)
