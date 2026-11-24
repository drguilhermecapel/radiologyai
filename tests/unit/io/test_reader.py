"""Testes do leitor DICOM — cobrem os defeitos concretos do sistema legado."""

from __future__ import annotations

import numpy as np
import pytest

from radiologyai.errors import InvalidDICOMError, MissingTagError
from radiologyai.io.reader import read_dicom, to_pixel_array


class TestReadDicom:
    def test_reads_valid_file(self, cr_dataset, write_dicom):
        ds = read_dicom(write_dicom(cr_dataset))
        assert ds.Modality == "CR"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(InvalidDICOMError, match="não encontrado"):
            read_dicom(tmp_path / "inexistente.dcm")

    def test_non_dicom_raises(self, tmp_path):
        p = tmp_path / "lixo.dcm"
        p.write_bytes(b"isto nao e dicom" * 100)
        with pytest.raises(InvalidDICOMError):
            read_dicom(p)

    def test_stop_before_pixels_skips_pixeldata(self, cr_dataset, write_dicom):
        ds = read_dicom(write_dicom(cr_dataset), stop_before_pixels=True)
        assert "PixelData" not in ds


@pytest.mark.requirement("REQ-005")
class TestModalityLUT:
    """RescaleSlope/Intercept devem produzir HU, sem quantização (HONEST_STATUS §10)."""

    def test_ct_returns_hounsfield_units(self, ct_dataset):
        arr = to_pixel_array(ct_dataset, scale="modality")
        # Armazenado 0..2047, intercept -1024 => HU -1024..1023
        assert arr.min() == pytest.approx(-1024.0)
        assert arr.max() == pytest.approx(1023.0)

    def test_output_is_float32_not_uint8(self, ct_dataset):
        """O defeito do legado: (arr * 255).astype(np.uint8) destruía a faixa HU."""
        arr = to_pixel_array(ct_dataset, scale="modality")
        assert arr.dtype == np.float32

    def test_hu_range_spans_air_to_bone(self, ct_dataset):
        arr = to_pixel_array(ct_dataset, scale="modality")
        assert arr.min() < -900, "ar (~-1000 HU) deve ser representável"
        assert arr.max() > 900, "osso denso deve ser representável"

    def test_stored_scale_ignores_rescale(self, ct_dataset):
        arr = to_pixel_array(ct_dataset, scale="stored")
        assert arr.min() == pytest.approx(0.0)


@pytest.mark.requirement("REQ-006")
class TestMonochrome1:
    """MONOCHROME1 precisa ser invertido, senão a imagem fica em negativo (H-05)."""

    def test_monochrome1_is_inverted(self, monochrome1_dataset):
        raw = to_pixel_array(monochrome1_dataset, scale="stored")
        converted = to_pixel_array(monochrome1_dataset, scale="modality")
        # A rampa cresce da esquerda para a direita nos valores brutos;
        # após inversão deve decrescer.
        assert raw[0, 0] < raw[0, -1]
        assert converted[0, 0] > converted[0, -1]

    def test_monochrome2_is_not_inverted(self, cr_dataset):
        raw = to_pixel_array(cr_dataset, scale="stored")
        converted = to_pixel_array(cr_dataset, scale="modality")
        assert np.allclose(raw, converted)

    def test_inversion_preserves_dynamic_range(self, monochrome1_dataset):
        raw = to_pixel_array(monochrome1_dataset, scale="stored")
        converted = to_pixel_array(monochrome1_dataset, scale="modality")
        assert converted.max() - converted.min() == pytest.approx(raw.max() - raw.min())


class TestVOI:
    def test_multivalue_window_uses_first_pair(self, windowed_dataset):
        """MultiValue de WindowCenter/Width não pode quebrar a leitura."""
        arr = to_pixel_array(windowed_dataset, scale="voi")
        assert arr.dtype == np.float32
        assert 0.0 <= arr.min() <= arr.max() <= 1.0

    def test_no_window_falls_back_to_minmax(self, cr_dataset):
        arr = to_pixel_array(cr_dataset, scale="voi")
        assert arr.min() == pytest.approx(0.0)
        assert arr.max() == pytest.approx(1.0)

    def test_missing_pixeldata_raises(self, cr_dataset, write_dicom):
        ds = read_dicom(write_dicom(cr_dataset), stop_before_pixels=True)
        with pytest.raises(MissingTagError, match="PixelData"):
            to_pixel_array(ds)
