"""Tipos de domínio compartilhados por adaptadores, pipeline, API e laudo.

CONTRATO. Adaptadores, pipeline, API, geradores de laudo e frontend são
construídos sobre estes tipos; mudanças passam pelo responsável pelos contratos.

Regras codificadas aqui
-----------------------
* Três números nunca se confundem: ``raw_score`` (o que o modelo emitiu),
  ``op_normalized`` (reescala em que 0,5 é o ponto de operação) e
  ``probability`` (calibrada; ``None`` sem artefato de calibração ajustado).
* Um achado só é "positivo"/"negativo" em relação a um ponto de operação
  explícito e com origem. Sem ele a decisão é ``"score_only"``.
* Todo resultado carrega proveniência (modelo, versão, hashes dos pesos,
  licença) e o nível de evidência da condição, que decide se ele aparece na
  visão principal (níveis A-C) ou só no modo pesquisa (nível D).
* ``import`` deste módulo não traz numpy: ``ImageData`` só o importa ao validar.

Reutiliza :class:`~radiologyai.catalog.schema.EvidenceLevel`,
:class:`~radiologyai.catalog.schema.CodedConcept`,
:class:`~radiologyai.catalog.schema.ExamType` e
:class:`~radiologyai.calibration.decision.OperatingPoint` — não os duplica.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from radiologyai.calibration.decision import OperatingPoint, OperatingScale
from radiologyai.catalog.schema import (
    CodedConcept,
    EvidenceLevel,
    ExamType,
    Visibility,
    visibility_for,
)

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    FloatArray = npt.NDArray[np.float32]
    Float64Array = npt.NDArray[np.float64]

__all__ = [
    "DISCLAIMER_PT",
    "PRIVATE_SCHEME",
    "SCHEMA_VERSION",
    "AbstentionReason",
    "AnalysisResult",
    "ArtifactRef",
    "CodedConcept",
    "EvidenceLevel",
    "ExamInputs",
    "ExamType",
    "Explanation",
    "Finding",
    "FindingDecision",
    "ImageData",
    "ImageKind",
    "ImageMeta",
    "Measurement",
    "Modality",
    "ModelProvenance",
    "ModelRun",
    "OperatingPoint",
    "OperatingScale",
    "Primitive",
    "QACheck",
    "QAReport",
    "ScoreKind",
    "Segmentation",
    "Visibility",
]

SCHEMA_VERSION: Literal["1"] = "1"
PRIVATE_SCHEME = "99MEDAI"
DISCLAIMER_PT: Literal[
    "Achado sugerido para confirmação médica. Software de pesquisa; não é diagnóstico."
] = "Achado sugerido para confirmação médica. Software de pesquisa; não é diagnóstico."

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


# --------------------------------------------------------------------------- #
# Enumerações
# --------------------------------------------------------------------------- #
class Modality(StrEnum):
    """Códigos DICOM (0008,0060) tratados. ``OT`` cobre raster sem cabeçalho."""

    CR = "CR"
    DX = "DX"
    CT = "CT"
    MR = "MR"
    MG = "MG"
    US = "US"
    OT = "OT"


class ImageKind(StrEnum):
    """Dimensionalidade física de uma imagem."""

    PROJECTION = "projection"  # imagem 2D única (radiografia, mamografia, quadro de US)
    VOLUME = "volume"  # pilha 3D com geometria (TC, RM)
    CINE = "cine"  # 2D + tempo (eco / clipe de ultrassom)


class AbstentionReason(StrEnum):
    """Por que o sistema recusou produzir um número. Sempre explícito."""

    MODALITY = "not_applicable_modality"
    EXAM_TYPE = "not_applicable_exam_type"
    VIEW = "not_applicable_view"
    BODY_PART = "not_applicable_body_part"
    QUALITY = "low_image_quality"
    OOD = "out_of_distribution"
    BAND = "low_confidence_band"
    GEOMETRY = "missing_geometry"
    ACQUISITION = "acquisition_precondition_not_met"
    INPUT = "unsupported_input"
    MISSING_INPUT = "required_input_missing"
    POPULATION = "unvalidated_population"
    UPSTREAM = "upstream_model_unavailable"


ScoreKind = Literal["logit", "sigmoid", "softmax", "measurement", "similarity"]
FindingDecision = Literal[
    "positive",  # no ponto de inclusão (rule-in) ou além dele
    "negative",  # aquém do ponto de exclusão (rule-out)
    "indeterminate",  # entre exclusão e inclusão (banda de baixa confiança)
    "category",  # saída categórica; ver ``category``
    "score_only",  # não há ponto de operação com origem; escore sem veredito
    "abstained",  # o sistema recusou; ver ``abstention_reason``
]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())


def _check_sha256(value: str, name: str) -> str:
    if not _SHA256_RE.match(value):
        raise ValueError(
            f"{name} deve ter 64 caracteres hexadecimais minúsculos; recebido {value!r}"
        )
    return value


# --------------------------------------------------------------------------- #
# Imagens
# --------------------------------------------------------------------------- #
class ImageMeta(_Frozen):
    """Metadados de aquisição usados em roteamento, portões e proveniência.

    NÃO contém identificador direto de paciente. ``patient_pseudo_id`` é um
    pseudônimo produzido na ingestão (HMAC, ver :mod:`radiologyai.io.deident`).
    """

    source_format: Literal["dicom", "dicom_series", "nifti", "raster", "video"]
    content_sha256: str
    study_uid: str | None = None
    series_uid: str | None = None
    sop_uids: tuple[str, ...] = ()
    sop_class_uid: str | None = None
    patient_pseudo_id: str | None = None
    patient_sex: Literal["F", "M", "O"] | None = None
    patient_age_years: float | None = None

    body_part: str | None = None
    view_position: str | None = None  # PA / AP / LATERAL / CC / MLO / A4C ...
    laterality: Literal["L", "R", "B"] | None = None
    study_description: str | None = None
    series_description: str | None = None
    sequence_role: str | None = None  # RM: t1 / t1c / t2 / flair / dwi / adc ...
    presentation_intent: str | None = None  # FOR PRESENTATION / FOR PROCESSING
    photometric_interpretation: str | None = None  # como gravado, antes da conversão
    bits_stored: int | None = None
    rows: int | None = None
    columns: int | None = None
    n_frames: int = 1
    frame_time_ms: float | None = None
    pixel_spacing_source: Literal[
        "PixelSpacing",
        "ImagerPixelSpacing",
        "UltrasoundRegions",
        "nifti",
        "none",
    ] = "none"

    # TC / RM: fatos de aquisição que condicionam biomarcadores quantitativos
    slice_thickness_mm: float | None = None
    kvp: float | None = None
    convolution_kernel: str | None = None
    contrast_agent: str | None = None  # ContrastBolusAgent como gravado ("" = tag vazia)
    contrast_phase: Literal["none", "arterial", "portal_venous", "delayed", "unknown"] | None = None

    manufacturer: str | None = None
    manufacturer_model: str | None = None
    burned_in_annotation: bool | None = None

    @field_validator("content_sha256")
    @classmethod
    def _sha(cls, v: str) -> str:
        return _check_sha256(v, "content_sha256")


@dataclass(frozen=True, slots=True)
class ImageData:
    """Imagem decodificada com significado físico anexado.

    Convenções de ``pixels``
    ------------------------
    * dtype float32; polaridade sempre MONOCHROME2 (maior = mais claro).
    * PROJECTION: ``(H, W)`` ou ``(H, W, 3)`` quando ``color``.
    * VOLUME: ``(Z, Y, X)``; em TC os valores são Unidades Hounsfield.
    * CINE: ``(T, H, W)`` ou ``(T, H, W, 3)`` quando ``color``.
    * Valores lineares após a Modality LUT. Nenhuma janela de exibição é
      aplicada; ``window`` guarda a VOI (centro, largura) do arquivo, se houver.
    * ``spacing_mm``: ``(linha, coluna)`` ou ``(z, y, x)``; ``None`` impede
      qualquer medida em mm.
    * ``affine_lps``: 4x4 que leva o índice ``(x=coluna, y=linha, z=corte, 1)``
      a coordenadas do paciente em LPS (convenção DICOM/ITK).
    * ``value_range``: faixa representável pelo formato, nunca ``pixels.max()``.
    """

    pixels: FloatArray
    kind: ImageKind
    modality: Modality
    meta: ImageMeta
    spacing_mm: tuple[float, ...] | None = None
    affine_lps: Float64Array | None = None
    window: tuple[float, float] | None = None
    value_range: tuple[float, float] = (0.0, 255.0)
    color: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        import numpy as np

        if not isinstance(self.pixels, np.ndarray) or self.pixels.dtype != np.float32:
            dtype = getattr(self.pixels, "dtype", type(self.pixels).__name__)
            raise TypeError(f"ImageData.pixels deve ser ndarray float32; recebido {dtype}")
        expected = {ImageKind.PROJECTION: 2, ImageKind.VOLUME: 3, ImageKind.CINE: 3}[self.kind]
        ndim = self.pixels.ndim - (1 if self.color else 0)
        if ndim != expected:
            raise ValueError(
                f"imagem {self.kind.value} exige {expected} dimensões espaciais"
                f"{' + cor' if self.color else ''}; recebido shape {self.pixels.shape}"
            )
        if self.color and self.pixels.shape[-1] != 3:
            raise ValueError("imagens coloridas exigem última dimensão igual a 3")
        if self.spacing_mm is not None:
            if len(self.spacing_mm) not in (2, 3):
                raise ValueError("spacing_mm deve ter 2 ou 3 entradas")
            if not all(math.isfinite(s) and s > 0 for s in self.spacing_mm):
                raise ValueError(
                    f"spacing_mm deve ser positivo e finito; recebido {self.spacing_mm}"
                )
        if self.affine_lps is not None and self.affine_lps.shape != (4, 4):
            raise ValueError("affine_lps deve ser 4x4")
        lo, hi = self.value_range
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise ValueError(f"value_range inválido: {self.value_range}")

    @property
    def sha256(self) -> str:
        """sha256 do conteúdo de origem (arquivo, ou série ordenada)."""
        return self.meta.content_sha256

    @property
    def plane_shape(self) -> tuple[int, int]:
        """(altura, largura) do plano da imagem, independentemente de cor e tempo."""
        shape = self.pixels.shape[:-1] if self.color else self.pixels.shape
        return int(shape[-2]), int(shape[-1])


# --------------------------------------------------------------------------- #
# Blocos do resultado (serializáveis em JSON; persistidos e enviados ao cliente)
# --------------------------------------------------------------------------- #
class ModelProvenance(_Frozen):
    """O que é preciso para saber exatamente qual modelo produziu um item."""

    model_id: str
    model_version: str
    weights_sha256: dict[str, str]
    manifest_sha256: str
    code_version: str
    license_code: str
    license_weights: str
    commercial_use: Literal["yes", "no", "unclear"]
    runtime: dict[str, str] = Field(default_factory=dict)

    @field_validator("manifest_sha256")
    @classmethod
    def _manifest_sha(cls, v: str) -> str:
        return _check_sha256(v, "manifest_sha256")

    @field_validator("weights_sha256")
    @classmethod
    def _weights_sha(cls, v: dict[str, str]) -> dict[str, str]:
        for key, digest in v.items():
            _check_sha256(digest, f"weights_sha256[{key!r}]")
        return v


class Primitive(_Frozen):
    """Geometria em coordenadas de pixel de uma imagem (x = coluna, y = linha)."""

    type: Literal["point", "line", "bbox", "polygon"]
    xy: tuple[tuple[float, float], ...]
    frame: int = 0  # índice do corte (volume) ou do quadro (cine)
    label: str | None = None

    @model_validator(mode="after")
    def _arity(self) -> Self:
        need = {"point": 1, "line": 2, "bbox": 2}.get(self.type)
        if need is not None and len(self.xy) != need:
            raise ValueError(f"{self.type} exige exatamente {need} ponto(s)")
        if self.type == "polygon" and len(self.xy) < 3:
            raise ValueError("polígono exige ao menos 3 pontos")
        if not all(math.isfinite(c) for pt in self.xy for c in pt):
            raise ValueError("coordenadas devem ser finitas")
        return self


class ArtifactRef(_Frozen):
    """Ponteiro para um arquivo do repositório endereçado por conteúdo."""

    sha256: str
    media_type: str
    relpath: str
    shape: tuple[int, ...] | None = None

    @field_validator("sha256")
    @classmethod
    def _sha(cls, v: str) -> str:
        return _check_sha256(v, "sha256")


class Finding(_Frozen):
    """Um achado sugerido para confirmação médica — nunca um diagnóstico."""

    condition_id: str  # chave no catálogo de condições
    label: str  # chave de rótulo do próprio modelo (manifesto)
    code: CodedConcept
    kind: Literal["binary", "categorical", "derived"]

    raw_score: float | None  # None somente quando o sistema se absteve
    score_kind: ScoreKind
    op_normalized: float | None = None  # 0,5 == ponto de operação primário
    probability: float | None = None  # calibrada; None sem calibração ajustada
    calibration_id: str | None = None
    class_scores: dict[str, float] | None = None  # saídas categóricas
    category: str | None = None

    operating_points: tuple[OperatingPoint, ...] = ()
    decision: FindingDecision
    abstention_reason: AbstentionReason | None = None
    # Por que o achado ficou só com escore (ex.: decisão suspensa, REQ-193); só com score_only.
    decision_note: str | None = None

    evidence_level: EvidenceLevel
    visibility: Visibility
    measurement_ref: str | None = None  # Measurement.name para achados derivados
    explanation_ids: tuple[str, ...] = ()
    image_idx: int = 0
    provenance_idx: int

    @model_validator(mode="after")
    def _invariants(self) -> Self:
        if (self.probability is None) != (self.calibration_id is None):
            raise ValueError("probability e calibration_id andam juntos (ambos ou nenhum)")
        if self.probability is not None and not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability deve estar em [0, 1]")
        if self.decision in ("positive", "negative", "indeterminate") and not self.operating_points:
            raise ValueError(f"decisão {self.decision!r} exige ponto de operação com origem")
        if self.decision == "abstained" and self.abstention_reason is None:
            raise ValueError("achado com abstenção exige abstention_reason")
        if self.decision != "abstained" and self.abstention_reason is not None:
            raise ValueError("abstention_reason só é válido com decision='abstained'")
        if self.decision_note is not None and self.decision != "score_only":
            raise ValueError("decision_note só é válido com decision='score_only'")
        if self.raw_score is None and self.decision != "abstained":
            raise ValueError("achado sem escore só pode ser abstenção")
        if self.raw_score is not None and not math.isfinite(self.raw_score):
            raise ValueError("raw_score deve ser finito")
        if self.kind == "categorical" and self.decision != "abstained":
            if self.class_scores is None or self.category is None:
                raise ValueError("achado categórico exige class_scores e category")
            if self.category not in self.class_scores:
                raise ValueError("category deve ser uma das classes de class_scores")
        if self.decision == "category" and self.kind != "categorical":
            raise ValueError("decision='category' só vale para achado categórico")
        if self.kind == "derived" and self.measurement_ref is None:
            raise ValueError("achado derivado deve referenciar uma medida")
        if self.visibility != visibility_for(self.evidence_level):
            raise ValueError(
                f"nível {self.evidence_level.value} exige visibilidade "
                f"{visibility_for(self.evidence_level)!r}; nível D só no modo pesquisa"
            )
        return self


class Segmentation(_Frozen):
    """Mapa de rótulos inteiro na geometria da imagem de origem."""

    id: str
    name: str
    labels: dict[int, CodedConcept]
    label_keys: dict[int, str]  # índice -> chave de rótulo do modelo
    mask: ArtifactRef
    image_idx: int = 0
    provenance_idx: int


class Measurement(_Frozen):
    """Medida quantitativa: valor OU motivo de abstenção, nunca um valor inventado."""

    name: str  # chave estável, ex.: "cardiothoracic_ratio"
    code: CodedConcept
    value: float | None
    unit: str  # UCUM quando existir ("mL", "mm", "[hnsf'U]", "1" para razões, "%")
    method: str
    derived_from: tuple[str, ...] = ()  # ids de segmentação
    geometry: tuple[Primitive, ...] = ()
    preconditions: dict[str, str] = Field(default_factory=dict)  # fatos de aquisição usados
    abstention_reason: AbstentionReason | None = None
    note: str | None = None
    image_idx: int = 0
    provenance_idx: int

    @model_validator(mode="after")
    def _value_or_reason(self) -> Self:
        if self.value is None and self.abstention_reason is None:
            raise ValueError("medida sem valor exige abstention_reason")
        if self.value is not None and self.abstention_reason is not None:
            raise ValueError("medida com valor não pode ter abstention_reason")
        if self.value is not None and not math.isfinite(self.value):
            raise ValueError("valor de medida deve ser finito")
        return self


class Explanation(_Frozen):
    """Mapa de explicação calculado pelo modelo real sobre a entrada real."""

    id: str
    method: Literal["cam", "gradcam", "attention", "occlusion"]
    target_label: str
    map: ArtifactRef  # PNG 8 bits, um canal
    placement: Primitive  # bbox, em pixels da imagem de origem, coberta pelo mapa
    image_idx: int = 0
    provenance_idx: int

    @model_validator(mode="after")
    def _bbox(self) -> Self:
        if self.placement.type != "bbox":
            raise ValueError("placement de explicação deve ser bbox")
        return self


class QACheck(_Frozen):
    """Uma verificação de aplicabilidade/qualidade com o valor observado."""

    name: str
    passed: bool
    value: float | str | None = None
    threshold: float | str | None = None
    detail: str | None = None


class QAReport(_Frozen):
    """Resultado do portão de aplicabilidade de um modelo para uma entrada."""

    applicable: bool
    checks: tuple[QACheck, ...] = ()
    reasons: tuple[AbstentionReason, ...] = ()

    @model_validator(mode="after")
    def _reasons_iff_not_applicable(self) -> Self:
        if not self.applicable and not self.reasons:
            raise ValueError("relatório não aplicável deve declarar ao menos um motivo")
        if self.applicable and self.reasons:
            raise ValueError("relatório aplicável não pode ter motivos de abstenção")
        return self

    @classmethod
    def ok(cls, checks: tuple[QACheck, ...] | list[QACheck] = ()) -> QAReport:
        return cls(applicable=True, checks=tuple(checks))

    @classmethod
    def reject(
        cls, *reasons: AbstentionReason, checks: tuple[QACheck, ...] | list[QACheck] = ()
    ) -> QAReport:
        return cls(applicable=False, checks=tuple(checks), reasons=tuple(reasons))


class ModelRun(_Frozen):
    """O que aconteceu com um modelo numa análise (rodou, absteve-se, falhou...)."""

    model_id: str
    status: Literal["completed", "abstained", "unavailable", "planned", "failed"]
    qa: QAReport | None = None
    error: str | None = None
    seconds: float | None = None
    provenance_idx: int | None = None

    @model_validator(mode="after")
    def _shape(self) -> Self:
        if self.status in ("unavailable", "planned", "failed") and not self.error:
            raise ValueError(f"execução com status {self.status!r} deve explicar o motivo (error)")
        if self.status == "completed" and self.provenance_idx is None:
            raise ValueError("execução concluída exige provenance_idx")
        return self


class AnalysisResult(_Frozen):
    """Resultado imutável de uma análise. Persistido em JSON; hasheado na assinatura."""

    schema_version: Literal["1"] = SCHEMA_VERSION
    analysis_id: UUID
    created_at: datetime
    exam_type: ExamType
    input_sha256: tuple[str, ...]
    status: Literal["completed", "partial", "abstained", "failed"]
    intended_use: Literal["research_use_only"] = "research_use_only"
    disclaimer: Literal[
        "Achado sugerido para confirmação médica. Software de pesquisa; não é diagnóstico."
    ] = DISCLAIMER_PT
    research_mode: bool = False

    provenance: tuple[ModelProvenance, ...] = ()
    runs: tuple[ModelRun, ...] = ()
    findings: tuple[Finding, ...] = ()
    segmentations: tuple[Segmentation, ...] = ()
    measurements: tuple[Measurement, ...] = ()
    explanations: tuple[Explanation, ...] = ()
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _references(self) -> Self:
        n = len(self.provenance)
        items: list[Finding | Segmentation | Measurement | Explanation] = [
            *self.findings,
            *self.segmentations,
            *self.measurements,
            *self.explanations,
        ]
        for item in items:
            if not 0 <= item.provenance_idx < n:
                raise ValueError(f"provenance_idx {item.provenance_idx} fora do intervalo (n={n})")
            if not 0 <= item.image_idx < max(1, len(self.input_sha256)):
                raise ValueError(f"image_idx {item.image_idx} fora do intervalo")
        for run in self.runs:
            if run.provenance_idx is not None and not 0 <= run.provenance_idx < n:
                raise ValueError(
                    f"provenance_idx {run.provenance_idx} da execução fora do intervalo"
                )
        explanation_ids = {e.id for e in self.explanations}
        measurement_names = {m.name for m in self.measurements}
        for finding in self.findings:
            missing = set(finding.explanation_ids) - explanation_ids
            if missing:
                raise ValueError(f"achado referencia explicações inexistentes: {sorted(missing)}")
            if finding.measurement_ref is not None and finding.measurement_ref not in (
                measurement_names
            ):
                raise ValueError(f"achado referencia medida inexistente: {finding.measurement_ref}")
        return self

    def main_findings(self) -> list[Finding]:
        """Achados elegíveis para a visão principal e para o laudo (níveis A-C)."""
        return [f for f in self.findings if f.visibility == "main"]

    def research_findings(self) -> list[Finding]:
        """Achados exploratórios (nível D): só no modo pesquisa, nunca no laudo sozinhos."""
        return [f for f in self.findings if f.visibility == "research"]


@dataclass(frozen=True, slots=True)
class ExamInputs:
    """Tudo o que se sabe de um exame no momento da análise."""

    images: tuple[ImageData, ...]
    exam_type: ExamType
    prompts: tuple[Primitive, ...] = ()
