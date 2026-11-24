"""Metadados de exame, tipados.

O sistema legado devolvia ``dict[str, str]`` de ``extract_metadata()``, o que
tornava impossível validar em tempo de tipo se uma modalidade estava presente
ou se a idade era numérica. Aqui o contrato é explícito.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

_AGE_RE = re.compile(r"^(\d{1,3})([DWMY])$")


def parse_dicom_age(value: str | None) -> float | None:
    """Converte Patient Age no formato DICOM AS (ex.: ``045Y``, ``018M``) em anos.

    Devolve ``None`` quando ausente ou irreconhecível — nunca levanta, porque
    idade ausente é comum e o gate de escopo trata isso separadamente.
    """
    if not value:
        return None
    m = _AGE_RE.match(value.strip().upper())
    if not m:
        return None
    n, unit = int(m.group(1)), m.group(2)
    return {"Y": n, "M": n / 12.0, "W": n / 52.0, "D": n / 365.25}[unit]


class StudyMetadata(BaseModel):
    """Metadados relevantes de um exame, extraídos do cabeçalho DICOM.

    Nenhum campo aqui carrega identificador direto de paciente. Nome, ID
    original, data de nascimento e afins são removidos por
    :func:`radiologyai.io.deident.deidentify` antes deste objeto existir.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    modality: str = Field(description="DICOM (0008,0060) Modality, ex.: 'DX', 'CT'")
    sop_class_uid: str | None = None
    view_position: str | None = Field(
        default=None, description="DICOM (0018,5101), ex.: 'PA', 'AP', 'LATERAL'"
    )
    patient_sex: str | None = None
    patient_age_years: float | None = None
    body_part: str | None = None
    photometric_interpretation: str | None = None
    manufacturer: str | None = None
    rows: int | None = None
    columns: int | None = None
    pixel_spacing: tuple[float, float] | None = None
    slice_thickness: float | None = None
    number_of_frames: int = 1
    pseudo_patient_id: str | None = Field(
        default=None, description="Pseudo-ID determinístico (HMAC), quando anonimizado"
    )

    @property
    def is_multiframe(self) -> bool:
        return self.number_of_frames > 1


def _get(ds: Any, name: str, default: object = None) -> Any:
    value = getattr(ds, name, default)
    return default if value in ("", None) else value


def extract_metadata(ds: Any) -> StudyMetadata:
    """Extrai :class:`StudyMetadata` de um ``pydicom.Dataset``."""
    spacing = _get(ds, "PixelSpacing")
    pixel_spacing: tuple[float, float] | None = None
    if spacing is not None:
        try:
            pixel_spacing = (float(spacing[0]), float(spacing[1]))
        except (TypeError, ValueError, IndexError):
            pixel_spacing = None

    thickness = _get(ds, "SliceThickness")
    try:
        slice_thickness = float(thickness) if thickness is not None else None
    except (TypeError, ValueError):
        slice_thickness = None

    try:
        n_frames = int(_get(ds, "NumberOfFrames", 1) or 1)
    except (TypeError, ValueError):
        n_frames = 1

    return StudyMetadata(
        modality=str(_get(ds, "Modality", "") or ""),
        sop_class_uid=(str(v) if (v := _get(ds, "SOPClassUID")) else None),
        view_position=(str(v).upper() if (v := _get(ds, "ViewPosition")) else None),
        patient_sex=(str(v).upper() if (v := _get(ds, "PatientSex")) else None),
        patient_age_years=parse_dicom_age(str(v) if (v := _get(ds, "PatientAge")) else None),
        body_part=(str(v).upper() if (v := _get(ds, "BodyPartExamined")) else None),
        photometric_interpretation=(
            str(v).upper() if (v := _get(ds, "PhotometricInterpretation")) else None
        ),
        manufacturer=(str(v) if (v := _get(ds, "Manufacturer")) else None),
        rows=(int(v) if (v := _get(ds, "Rows")) else None),
        columns=(int(v) if (v := _get(ds, "Columns")) else None),
        pixel_spacing=pixel_spacing,
        slice_thickness=slice_thickness,
        number_of_frames=n_frames,
    )
