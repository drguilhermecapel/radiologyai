"""Domain types shared by every layer of MedAI.

This module is a FROZEN CONTRACT: adapters, the pipeline, the API, the report
generators and the frontend types are all built against it. Change it only
through the lead (see CLAUDE.md, "Frozen contracts").

Design rules encoded here
-------------------------
* Three different numbers are never conflated:
  ``raw_score`` (what the model emitted), ``op_normalized`` (score rescaled so
  that 0.5 is the operating point) and ``probability`` (calibrated; ``None``
  unless a fitted calibration artifact exists).
* A finding is only "positive"/"negative" relative to an explicit, sourced
  operating point. Without one, the decision is ``"score_only"``.
* Every result carries provenance (model id, version, weight hashes) and the
  evidence level of the condition, which decides whether it is shown in the
  main view (levels A-C) or only in research mode (level D).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Literal, Self
from uuid import UUID

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION: Literal["1"] = "1"

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.integer]


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
class Modality(StrEnum):
    """DICOM modality codes we handle. ``OT`` covers raster uploads without tags."""

    CR = "CR"
    DX = "DX"
    CT = "CT"
    MR = "MR"
    MG = "MG"
    US = "US"
    OT = "OT"


class ImageKind(StrEnum):
    PROJECTION = "projection"  # single 2D image (radiograph, mammogram, US still)
    VOLUME = "volume"  # 3D stack with geometry (CT, MR)
    CINE = "cine"  # 2D + time (echo / ultrasound clip)


class ExamType(StrEnum):
    """Routing key. Decided from DICOM tags or by a human, never guessed silently."""

    CXR = "cxr"
    MSK_XR = "msk_xr"
    CT_HEAD = "ct_head"
    CT_CHEST = "ct_chest"
    CT_ABDOMEN = "ct_abdomen"
    CT_OTHER = "ct_other"
    MR_BRAIN = "mr_brain"
    MR_BODY = "mr_body"
    MG = "mg"
    US_BREAST = "us_breast"
    US_ABDOMEN = "us_abdomen"
    US_OTHER = "us_other"
    ECHO = "echo"
    UNKNOWN = "unknown"


class EvidenceLevel(StrEnum):
    """Evidence level of a (condition, model) pair. Computed from evidence files."""

    A = "A"  # externally validated by our own measurement
    B = "B"  # peer-reviewed published validation and/or guideline threshold
    C = "C"  # internally validated by our own measurement
    D = "D"  # exploratory (zero-shot or no per-condition validation)


class AbstentionReason(StrEnum):
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
OperatingScale = Literal["logit", "sigmoid", "softmax", "calibrated", "measurement", "similarity"]
Decision = Literal[
    "positive",  # at or beyond the rule-in operating point
    "negative",  # below the rule-out operating point
    "indeterminate",  # between rule-out and rule-in points (low-confidence band)
    "category",  # categorical output; see ``category``
    "score_only",  # no sourced operating point exists; score shown without a call
    "abstained",  # the system declined; see ``abstention_reason``
]
Visibility = Literal["main", "research"]


# --------------------------------------------------------------------------- #
# Image containers
# --------------------------------------------------------------------------- #
class ImageMeta(BaseModel):
    """Acquisition metadata needed for routing, gating and provenance.

    Contains NO direct patient identifiers. ``patient_pseudo_id`` is a salted
    pseudonym produced at ingest.
    """

    model_config = ConfigDict(frozen=True)

    source_format: Literal["dicom", "nifti", "raster", "video"]
    content_sha256: str
    study_uid: str | None = None
    series_uid: str | None = None
    sop_uids: tuple[str, ...] = ()
    patient_pseudo_id: str | None = None
    patient_sex: Literal["F", "M", "O"] | None = None
    patient_age_years: float | None = None

    body_part: str | None = None
    view_position: str | None = None  # PA / AP / LATERAL / CC / MLO ...
    laterality: Literal["L", "R", "B"] | None = None
    study_description: str | None = None
    series_description: str | None = None
    sequence_role: str | None = None  # MR: t1 / t1c / t2 / flair / dwi / adc ...
    presentation_intent: str | None = None  # FOR PRESENTATION / FOR PROCESSING
    photometric_interpretation: str | None = None  # as stored, before conversion
    bits_stored: int | None = None
    rows: int | None = None
    columns: int | None = None
    n_frames: int = 1
    frame_time_ms: float | None = None
    pixel_spacing_source: Literal["PixelSpacing", "ImagerPixelSpacing", "nifti", "none"] = "none"

    # CT / MR acquisition facts used to gate quantitative biomarkers
    slice_thickness_mm: float | None = None
    kvp: float | None = None
    convolution_kernel: str | None = None
    contrast_agent: str | None = None  # ContrastBolusAgent as recorded ("" = tag present, empty)
    contrast_phase: Literal["none", "arterial", "portal_venous", "delayed", "unknown"] | None = None

    manufacturer: str | None = None
    manufacturer_model: str | None = None
    burned_in_annotation: bool | None = None


@dataclass(frozen=True, slots=True)
class ImageData:
    """A decoded image with physical meaning attached.

    ``pixels`` conventions
    ----------------------
    * dtype float32, polarity always MONOCHROME2 (higher = brighter).
    * PROJECTION: ``(H, W)`` or ``(H, W, 3)`` when ``color``.
    * VOLUME: ``(Z, Y, X)``; CT values are Hounsfield units.
    * CINE: ``(T, H, W)`` or ``(T, H, W, 3)`` when ``color``.
    * Values are linear stored values after the Modality LUT. No display window
      is baked in; ``window`` carries the VOI centre/width when the file has one.
    """

    pixels: FloatArray
    kind: ImageKind
    modality: Modality
    meta: ImageMeta
    spacing_mm: tuple[float, ...] | None = (
        None  # (row, col) or (z, y, x); None => no mm measurements
    )
    affine_lps: npt.NDArray[np.float64] | None = None  # 4x4 voxel->patient (LPS) for volumes
    window: tuple[float, float] | None = None  # VOI (centre, width)
    value_range: tuple[float, float] = (0.0, 255.0)  # representable range, never pixels.max()
    color: bool = False

    def __post_init__(self) -> None:
        if self.pixels.dtype != np.float32:
            raise TypeError(f"ImageData.pixels must be float32, got {self.pixels.dtype}")
        expected = {ImageKind.PROJECTION: 2, ImageKind.VOLUME: 3, ImageKind.CINE: 3}[self.kind]
        ndim = self.pixels.ndim - (1 if self.color else 0)
        if ndim != expected:
            raise ValueError(
                f"{self.kind.value} image expects {expected} spatial dims"
                f"{' + colour' if self.color else ''}, got shape {self.pixels.shape}"
            )
        if self.color and self.pixels.shape[-1] != 3:
            raise ValueError("colour images must have a trailing dimension of 3")
        if self.spacing_mm is not None and len(self.spacing_mm) not in (2, 3):
            raise ValueError("spacing_mm must have 2 or 3 entries")
        if self.affine_lps is not None and self.affine_lps.shape != (4, 4):
            raise ValueError("affine_lps must be 4x4")

    @property
    def sha256(self) -> str:
        return self.meta.content_sha256


# --------------------------------------------------------------------------- #
# Result building blocks (JSON-serialisable; persisted and sent to the client)
# --------------------------------------------------------------------------- #
class CodedConcept(BaseModel):
    """A coded term. Only ``pydicom.sr.codedict`` codes or the private scheme 99MEDAI."""

    model_config = ConfigDict(frozen=True)

    scheme: str
    value: str
    meaning: str


class ModelProvenance(BaseModel):
    model_config = ConfigDict(frozen=True, protected_namespaces=())

    model_id: str
    model_version: str
    weights_sha256: dict[str, str]
    manifest_sha256: str
    code_version: str
    license_weights: str
    commercial_use: Literal["yes", "no", "unclear"]
    runtime: dict[str, str] = Field(default_factory=dict)


class OperatingPoint(BaseModel):
    """A decision threshold with its source. ``role`` gives it clinical meaning."""

    model_config = ConfigDict(frozen=True)

    name: str
    role: Literal["rule_out", "rule_in", "balanced", "guideline"]
    threshold: float
    on: OperatingScale
    direction: Literal["ge", "le"] = "ge"  # positive when score >= threshold ("ge") or <= ("le")
    source: str  # human-readable origin (library default, protocol fit, guideline DOI)
    evidence_ref: str | None = None  # path#key into evidence/ when measured by us
    sensitivity: float | None = None  # measured at this point, if known
    specificity: float | None = None


class Primitive(BaseModel):
    """Geometry in pixel coordinates of one image (x = column, y = row)."""

    type: Literal["point", "line", "bbox", "polygon"]
    xy: list[tuple[float, float]]
    frame: int = 0  # slice or time index for volumes / cines
    label: str | None = None

    @model_validator(mode="after")
    def _check_arity(self) -> Self:
        need = {"point": 1, "line": 2, "bbox": 2}.get(self.type)
        if need is not None and len(self.xy) != need:
            raise ValueError(f"{self.type} needs exactly {need} points")
        if self.type == "polygon" and len(self.xy) < 3:
            raise ValueError("polygon needs at least 3 points")
        return self


class ArtifactRef(BaseModel):
    """Pointer to a content-addressed file in the artifact store."""

    model_config = ConfigDict(frozen=True)

    sha256: str
    media_type: str
    relpath: str
    shape: tuple[int, ...] | None = None


class Finding(BaseModel):
    """One suggested finding for physician confirmation."""

    condition_id: str  # key in the condition catalog
    label: str  # the model's own label key
    code: CodedConcept
    kind: Literal["binary", "categorical", "derived"]

    raw_score: float | None  # None only when the system abstained (nothing was computed)
    score_kind: ScoreKind
    op_normalized: float | None = None  # 0.5 == primary operating point
    probability: float | None = None  # calibrated; None without a fitted calibration
    calibration_id: str | None = None
    class_scores: dict[str, float] | None = None  # categorical outputs
    category: str | None = None

    operating_points: list[OperatingPoint] = Field(default_factory=list)
    decision: Decision
    abstention_reason: AbstentionReason | None = None

    evidence_level: EvidenceLevel
    visibility: Visibility
    measurement_ref: str | None = None  # Measurement.name for derived findings
    explanation_ids: list[str] = Field(default_factory=list)
    image_idx: int = 0
    provenance_idx: int

    @model_validator(mode="after")
    def _invariants(self) -> Self:
        if (self.probability is None) != (self.calibration_id is None):
            raise ValueError("probability and calibration_id must be set together")
        if self.decision in ("positive", "negative", "indeterminate") and not self.operating_points:
            raise ValueError(f"decision {self.decision!r} requires a sourced operating point")
        if self.decision == "abstained" and self.abstention_reason is None:
            raise ValueError("abstained findings need an abstention_reason")
        if self.decision != "abstained" and self.abstention_reason is not None:
            raise ValueError("abstention_reason is only valid with decision='abstained'")
        if self.raw_score is None and self.decision != "abstained":
            raise ValueError("a finding without a score must be an abstention")
        if self.kind == "categorical" and (self.class_scores is None or self.category is None):
            raise ValueError("categorical findings need class_scores and category")
        if self.kind == "derived" and self.measurement_ref is None:
            raise ValueError("derived findings must reference a measurement")
        if self.evidence_level is EvidenceLevel.D and self.visibility != "research":
            raise ValueError("level D findings are research-mode only")
        return self


class Segmentation(BaseModel):
    id: str
    name: str
    labels: dict[int, CodedConcept]
    label_keys: dict[int, str]  # index -> model label key
    mask: ArtifactRef  # integer label map in the source image geometry
    image_idx: int = 0
    provenance_idx: int


class Measurement(BaseModel):
    name: str  # stable key, e.g. "cardiothoracic_ratio"
    code: CodedConcept
    value: float | None
    unit: str  # UCUM where one exists ("mL", "mm", "[hnsf'U]", "1" for ratios, "%")
    method: str
    derived_from: list[str] = Field(default_factory=list)  # segmentation ids
    geometry: list[Primitive] = Field(default_factory=list)
    preconditions: dict[str, str] = Field(default_factory=dict)  # acquisition facts relied on
    abstention_reason: AbstentionReason | None = None
    note: str | None = None
    image_idx: int = 0
    provenance_idx: int

    @model_validator(mode="after")
    def _value_or_reason(self) -> Self:
        if self.value is None and self.abstention_reason is None:
            raise ValueError("a measurement without a value needs an abstention_reason")
        return self


class Explanation(BaseModel):
    id: str
    method: Literal["cam", "gradcam", "attention", "occlusion"]
    target_label: str
    map: ArtifactRef  # 8-bit single-channel PNG
    placement: Primitive  # bbox in source-image pixels where the map applies
    image_idx: int = 0
    provenance_idx: int


class QACheck(BaseModel):
    name: str
    passed: bool
    value: float | str | None = None
    threshold: float | str | None = None
    detail: str | None = None


class QAReport(BaseModel):
    applicable: bool
    checks: list[QACheck] = Field(default_factory=list)
    reasons: list[AbstentionReason] = Field(default_factory=list)

    @model_validator(mode="after")
    def _reasons_iff_not_applicable(self) -> Self:
        if not self.applicable and not self.reasons:
            raise ValueError("a non-applicable QA report must state at least one reason")
        return self

    @classmethod
    def ok(cls, checks: list[QACheck] | None = None) -> QAReport:
        return cls(applicable=True, checks=checks or [])

    @classmethod
    def reject(cls, *reasons: AbstentionReason, checks: list[QACheck] | None = None) -> QAReport:
        return cls(applicable=False, checks=checks or [], reasons=list(reasons))


class ModelRun(BaseModel):
    """What happened to one model during an analysis (ran, abstained, failed)."""

    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    status: Literal["completed", "abstained", "unavailable", "failed"]
    qa: QAReport | None = None
    error: str | None = None
    seconds: float | None = None
    provenance_idx: int | None = None


class AnalysisResult(BaseModel):
    """Immutable outcome of one analysis. Persisted as JSON; hashed for sign-off."""

    schema_version: Literal["1"] = SCHEMA_VERSION
    analysis_id: UUID
    created_at: datetime
    exam_type: ExamType
    input_sha256: list[str]
    status: Literal["completed", "partial", "abstained", "failed"]
    intended_use: Literal["research_use_only"] = "research_use_only"

    provenance: list[ModelProvenance] = Field(default_factory=list)
    runs: list[ModelRun] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    segmentations: list[Segmentation] = Field(default_factory=list)
    measurements: list[Measurement] = Field(default_factory=list)
    explanations: list[Explanation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _provenance_indices(self) -> Self:
        n = len(self.provenance)
        items: list[Finding | Segmentation | Measurement | Explanation] = [
            *self.findings,
            *self.segmentations,
            *self.measurements,
            *self.explanations,
        ]
        for item in items:
            if not 0 <= item.provenance_idx < n:
                raise ValueError(f"provenance_idx {item.provenance_idx} out of range (n={n})")
        return self

    def main_findings(self) -> list[Finding]:
        """Findings eligible for the main view and for the report."""
        return [f for f in self.findings if f.visibility == "main"]

    def research_findings(self) -> list[Finding]:
        """Exploratory findings: research mode only, never auto-inserted in a report."""
        return [f for f in self.findings if f.visibility == "research"]


@dataclass(frozen=True, slots=True)
class ExamInputs:
    """Everything known about one exam at analysis time."""

    images: tuple[ImageData, ...]
    exam_type: ExamType
    prompts: tuple[Primitive, ...] = field(default_factory=tuple)
