"""Metadados de exame, tipados.

O sistema legado devolvia ``dict[str, str]`` de ``extract_metadata()``, o que
tornava impossível validar em tempo de tipo se uma modalidade estava presente
ou se a idade era numérica. Aqui o contrato é explícito.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

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
    laterality: str | None = Field(default=None, description="DICOM (0020,0060) Laterality")
    image_laterality: str | None = Field(
        default=None, description="DICOM (0020,0062) ImageLaterality, ex.: 'L', 'R', 'U', 'B'"
    )
    frame_laterality: str | None = Field(
        default=None,
        description=(
            "DICOM (0020,9072) FrameLaterality da FrameAnatomySequence (objetos multiframe "
            "enhanced); valores distintos entre quadros ficam unidos por '\\'"
        ),
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
        laterality=(str(v).strip().upper() if (v := _get(ds, "Laterality")) else None),
        image_laterality=(str(v).strip().upper() if (v := _get(ds, "ImageLaterality")) else None),
        frame_laterality=_frame_laterality(ds),
    )


def _frame_laterality(ds: Any) -> str | None:
    """``FrameLaterality`` dos grupos funcionais (compartilhado e por quadro), sem inferir nada."""
    values: list[str] = []
    groups = list(_get(ds, "SharedFunctionalGroupsSequence", None) or ())
    groups += list(_get(ds, "PerFrameFunctionalGroupsSequence", None) or ())
    for group in groups:
        for anatomy in getattr(group, "FrameAnatomySequence", None) or ():
            value = getattr(anatomy, "FrameLaterality", None)
            if value not in ("", None):
                text = str(value).strip().upper()
                if text not in values:
                    values.append(text)
    return "\\".join(sorted(values)) if values else None


# --------------------------------------------------------------------- lateralidade (REQ-194)

#: códigos válidos por tag (PS3.3): ``Laterality`` só R/L; as outras admitem U (ímpar) e B (ambos)
_LATERALITY_CODES: dict[str, frozenset[str]] = {
    "FrameLaterality": frozenset({"L", "R", "U", "B"}),
    "ImageLaterality": frozenset({"L", "R", "U", "B"}),
    "Laterality": frozenset({"L", "R"}),
}

LateralityStatus = Literal["known", "unknown", "unpaired", "conflict", "invalid"]

_SIDE_LABEL = {"L": "esquerda", "R": "direita", "B": "bilateral"}


@dataclass(frozen=True)
class LateralityResolution:
    """Lado de um objeto DICOM decidido **só** pelas tags; nunca pelo pixel nem pelo nome.

    ``value`` é ``None`` sempre que o cabeçalho não decide o lado sem ambiguidade (ausente,
    ``U``, código inválido ou tags em conflito); ``status`` diz por quê e ``detail`` traz as
    tags lidas, para o aviso explícito.
    """

    value: Literal["L", "R", "B"] | None
    status: LateralityStatus
    source: str | None = None
    detail: str | None = None

    @property
    def label(self) -> str:
        """Texto para a saída: o lado, ou "lateralidade desconhecida" — nunca um palpite."""
        if self.value is not None:
            return _SIDE_LABEL[self.value]
        if self.status == "unpaired":
            return "estrutura ímpar (sem lado)"
        return "lateralidade desconhecida"

    @property
    def is_problem(self) -> bool:
        """Código inválido ou conflito entre tags: o cabeçalho se contradiz."""
        return self.status in ("conflict", "invalid")


def resolve_laterality(metadata: StudyMetadata) -> LateralityResolution:
    """Resolve o lado a partir de ``FrameLaterality``, ``ImageLaterality`` e ``Laterality``.

    Regras (REQ-194, mesmas de REQ-043 para conflito):

    * só tags DICOM contam; nenhuma tag => ``unknown`` (``value=None``);
    * código fora do conjunto válido da tag => ``invalid`` (``value=None``);
    * tags presentes com lados diferentes (inclui ``U`` contra ``L``/``R``, e quadros com
      ``FrameLaterality`` distintos) => ``conflict`` (``value=None``);
    * todas ``U`` => ``unpaired`` (``value=None``);
    * caso contrário o lado comum, com a tag mais específica como ``source``
      (``FrameLaterality`` > ``ImageLaterality`` > ``Laterality``).
    """
    read = {
        "FrameLaterality": metadata.frame_laterality,
        "ImageLaterality": metadata.image_laterality,
        "Laterality": metadata.laterality,
    }
    present = {tag: value for tag, value in read.items() if value}
    if not present:
        return LateralityResolution(
            value=None, status="unknown", detail="nenhuma tag de lateralidade"
        )
    detail = "; ".join(f"{tag}={value!r}" for tag, value in present.items())
    sides: dict[str, list[str]] = {}
    for tag, raw in present.items():
        codes = [c.strip() for c in raw.split("\\") if c.strip()]
        bad = [c for c in codes if c not in _LATERALITY_CODES[tag]]
        if bad or not codes:
            return LateralityResolution(
                value=None, status="invalid", detail=f"código inválido em {tag}: {raw!r} ({detail})"
            )
        for code in codes:
            sides.setdefault(code, []).append(tag)
    if len(sides) > 1:
        return LateralityResolution(
            value=None, status="conflict", detail=f"tags de lateralidade em conflito: {detail}"
        )
    (code,) = sides
    if code == "U":
        return LateralityResolution(value=None, status="unpaired", detail=detail)
    side: Literal["L", "R", "B"] = "L" if code == "L" else "R" if code == "R" else "B"
    return LateralityResolution(value=side, status="known", source=sides[code][0], detail=detail)
