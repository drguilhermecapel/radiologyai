"""Fixtures de teste: datasets DICOM sintéticos construídos em memória.

Nenhum pixel de paciente entra no repositório. Estas fixtures são construídas
com pydicom a cada execução, cobrindo exatamente os casos que o sistema legado
tratava errado: MONOCHROME1, RescaleSlope/Intercept, janela multivalorada,
tags ausentes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

CR_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.1"


def _base_dataset(rows: int, cols: int, sop_class: str) -> Dataset:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = generate_uid()

    ds = Dataset()
    ds.file_meta = meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.SOPClassUID = sop_class
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()

    ds.Rows = rows
    ds.Columns = cols
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    return ds


def _gradient(rows: int, cols: int, lo: int, hi: int) -> np.ndarray:
    """Rampa horizontal determinística — nunca aleatória."""
    row = np.linspace(lo, hi, cols, dtype=np.int16)
    return np.tile(row, (rows, 1))


@pytest.fixture
def cr_dataset() -> Dataset:
    """Radiografia frontal de adulto, dentro do escopo declarado."""
    ds = _base_dataset(512, 512, CR_SOP_CLASS)
    ds.Modality = "CR"
    ds.ViewPosition = "PA"
    ds.PatientSex = "M"
    ds.PatientAge = "045Y"
    ds.PatientID = "PACIENTE-12345"
    ds.PatientName = "TESTE^SINTETICO"
    ds.PatientBirthDate = "19800115"
    ds.AccessionNumber = "ACC-999"
    ds.InstitutionName = "Hospital de Teste"
    ds.StudyDate = "20260315"
    ds.StudyTime = "141530"
    ds.BodyPartExamined = "CHEST"
    ds.Manufacturer = "FabricanteTeste"
    ds.PixelData = _gradient(512, 512, 0, 4095).tobytes()
    return ds


@pytest.fixture
def ct_dataset() -> Dataset:
    """TC com RescaleSlope/Intercept — o caso que o legado quantizava para uint8."""
    ds = _base_dataset(64, 64, CTImageStorage)
    ds.Modality = "CT"
    ds.PatientSex = "F"
    ds.PatientAge = "060Y"
    ds.PatientID = "PACIENTE-CT-1"
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = -1024.0
    ds.SliceThickness = 1.25
    ds.PixelSpacing = [0.7, 0.7]
    # Valores armazenados 0..2047 => HU -1024..+1023, cobrindo ar, água e osso.
    ds.PixelData = _gradient(64, 64, 0, 2047).tobytes()
    return ds


@pytest.fixture
def monochrome1_dataset() -> Dataset:
    """MONOCHROME1 — mínimo é branco. O legado nunca invertia (perigo H-05)."""
    ds = _base_dataset(32, 32, CR_SOP_CLASS)
    ds.Modality = "DX"
    ds.ViewPosition = "AP"
    ds.PatientAge = "030Y"
    ds.PhotometricInterpretation = "MONOCHROME1"
    ds.PixelData = _gradient(32, 32, 0, 1000).tobytes()
    return ds


@pytest.fixture
def lateral_dataset() -> Dataset:
    """Incidência lateral — fora do uso pretendido (REG-01 §3)."""
    ds = _base_dataset(512, 512, CR_SOP_CLASS)
    ds.Modality = "CR"
    ds.ViewPosition = "LATERAL"
    ds.PatientAge = "050Y"
    ds.PixelData = _gradient(512, 512, 0, 4095).tobytes()
    return ds


@pytest.fixture
def pediatric_dataset() -> Dataset:
    """Paciente de 8 anos — população pediátrica está fora do escopo."""
    ds = _base_dataset(512, 512, CR_SOP_CLASS)
    ds.Modality = "CR"
    ds.ViewPosition = "PA"
    ds.PatientAge = "008Y"
    ds.PixelData = _gradient(512, 512, 0, 4095).tobytes()
    return ds


@pytest.fixture
def windowed_dataset() -> Dataset:
    """Janela multivalorada no cabeçalho — o legado não lidava com MultiValue."""
    ds = _base_dataset(32, 32, CR_SOP_CLASS)
    ds.Modality = "CR"
    ds.ViewPosition = "PA"
    ds.PatientAge = "040Y"
    ds.WindowCenter = [500, 1000]
    ds.WindowWidth = [1000, 2000]
    ds.PixelData = _gradient(32, 32, 0, 1000).tobytes()
    return ds


@pytest.fixture
def write_dicom(tmp_path: Path):
    """Grava um dataset em disco e devolve o caminho."""

    def _write(ds: Dataset, name: str = "study.dcm") -> Path:
        path = tmp_path / name
        pydicom.dcmwrite(str(path), ds, write_like_original=False)
        return path

    return _write
