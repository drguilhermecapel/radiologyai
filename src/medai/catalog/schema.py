"""Condition catalog and evidence schema.

FROZEN CONTRACT.

The catalog answers one question for the whole product: *for this condition and
this model, what is the strength of the evidence?* The answer (levels A-D) is
never typed by hand: :func:`compute_level` derives it from evidence records, and
``levels.json`` is a generated file that CI regenerates and compares.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from medai.core.types import CodedConcept, EvidenceLevel, ExamType, Visibility

EvidenceKind = Literal[
    "measured_external",  # our measurement; model not trained on this dataset
    "measured_internal",  # our measurement; patient-level hold-out of the training dataset
    "published",  # peer-reviewed validation by third parties, quote verified at source
    "guideline_threshold",  # guideline / landmark-study threshold applied to a validated measurement
]
Leakage = Literal["none", "possible", "known"]
ReferenceStandard = Literal[
    "radiologist_adjudicated",
    "radiologist_single_read",
    "report_nlp",
    "histopathology",
    "expert_segmentation",
    "clinical_reference_measurement",
    "other",
]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


# --------------------------------------------------------------------------- #
# Pre-registered protocol
# --------------------------------------------------------------------------- #
class Protocol(_Strict):
    """Acceptance criteria fixed BEFORE any measurement is run (pre-registration)."""

    protocol_id: str
    frozen_on: str  # ISO date of the commit that froze it

    # eligibility of a measured record
    min_positive: int = 30
    min_negative: int = 30
    # discrimination bar for levels A and C
    auroc_min: float = 0.80
    auroc_ci_low_min: float = 0.75
    # detection-style endpoints evaluated at a fixed, pre-specified threshold
    sens_spec_ci_low_min: float = 0.80
    # agreement bar for continuous measurements (e.g. ejection fraction)
    icc_min: float = 0.75
    # statistics
    ci_level: float = 0.95
    bootstrap_iterations: int = 2000
    bootstrap_cluster: Literal["patient"] = "patient"
    random_seed: int = 20261124
    # operating points fitted on the calibration half, reported on the test half
    calibration_fraction: float = 0.5
    rule_out_sensitivity_target: float = 0.90
    rule_in_specificity_target: float = 0.90
    # bar for third-party published evidence (level B)
    published_auroc_min: float = 0.80
    published_sens_spec_min: float = 0.80
    published_min_positive: int = 30
    segmentation_dice_min: float = 0.85
    subgroups: list[str] = Field(default_factory=lambda: ["sex", "age_band", "view"])


# --------------------------------------------------------------------------- #
# Catalog entries
# --------------------------------------------------------------------------- #
class ConditionSource(_Strict):
    model_id: str
    label: str


class Condition(_Strict):
    id: str  # "<group>.<slug>", e.g. "cxr.pneumothorax"
    name_pt: str
    name_en: str
    group: Literal["cxr", "msk_xr", "ct", "mr", "mg", "us", "echo"]
    exam_types: list[ExamType]
    system: str  # organ system / clinical area, for grouping in the UI
    code: CodedConcept
    icd10: str | None = None
    description_pt: str | None = None
    sources: list[ConditionSource] = Field(default_factory=list)


class Metric(_Strict):
    value: float
    ci_low: float | None = None
    ci_high: float | None = None


class EvidenceSource(_Strict):
    """Where a published number comes from. ``quote`` must literally contain the value."""

    title: str
    identifier: str | None = None  # DOI / PMID / guideline id
    url: str
    quote: str
    peer_reviewed: bool
    verified_at_source: bool = False  # set only by the citation-check tool


class MeasurementProvenance(_Strict):
    git_sha: str
    protocol_id: str
    dataset_id: str
    dataset_sha256: str  # hash of the dataset manifest (file list + hashes)
    weights_sha256: dict[str, str]
    command: str
    created_at: datetime


class EvidenceRecord(_Strict):
    id: str
    condition_id: str
    model_id: str
    label: str
    kind: EvidenceKind
    endpoint: Literal["discrimination", "detection", "agreement", "segmentation", "threshold"] = (
        "discrimination"
    )
    dataset: str | None = None
    population: str | None = None
    reference_standard: ReferenceStandard | None = None
    n_total: int | None = None
    n_positive: int | None = None
    metrics: dict[str, Metric] = Field(default_factory=dict)  # auroc, auprc, sensitivity, dice ...
    leakage: Leakage = "none"
    leakage_note: str | None = None
    passes_protocol: bool | None = None  # decided by the evaluator / ingestion tool, not by hand
    failure_reasons: list[str] = Field(default_factory=list)
    source: EvidenceSource | None = None  # required for published / guideline records
    provenance: MeasurementProvenance | None = None  # required for measured records
    caveats: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _shape(self) -> Self:
        measured = self.kind in ("measured_external", "measured_internal")
        if measured and self.provenance is None:
            raise ValueError(
                "measured evidence needs provenance (git sha, dataset and weight hashes)"
            )
        if measured and (self.n_total is None or self.n_positive is None):
            raise ValueError("measured evidence needs n_total and n_positive")
        if not measured and self.source is None:
            raise ValueError("published / guideline evidence needs a source with a verified quote")
        if self.passes_protocol is False and not self.failure_reasons:
            raise ValueError("a failing record must state failure_reasons")
        return self


class LevelEntry(_Strict):
    condition_id: str
    model_id: str
    label: str
    level: EvidenceLevel
    visibility: Visibility
    basis: list[str]  # evidence record ids that justify the level
    refuted_by_measurement: bool = False
    headline: str | None = None  # short human-readable summary generated from the records


class LevelsFile(_Strict):
    """Generated file ``levels.json``. Never edited by hand."""

    protocol_id: str
    evidence_digest: str  # sha256 over the sorted evidence records used
    entries: list[LevelEntry]


# --------------------------------------------------------------------------- #
# The level rule
# --------------------------------------------------------------------------- #
def compute_level(records: list[EvidenceRecord]) -> tuple[EvidenceLevel, list[str], bool]:
    """Derive the evidence level of one (condition, model, label) from its records.

    Returns ``(level, basis_record_ids, refuted_by_measurement)``.

    Rules (pre-registered):

    * **A** - at least one ``measured_external`` record with ``leakage == "none"``
      passes the protocol.
    * **refuted** - clean external measurements exist and *all* of them fail:
      the condition drops to **D** regardless of published evidence. Our own
      failed measurement outranks a third-party claim.
    * **B** - otherwise, a ``published`` or ``guideline_threshold`` record that was
      verified at source and passes the published bar.
    * **C** - otherwise, a passing ``measured_internal`` record, or a passing
      external measurement whose leakage is "possible"/"known".
    * **D** - everything else.
    """
    clean_external = [r for r in records if r.kind == "measured_external" and r.leakage == "none"]
    passing_clean = [r for r in clean_external if r.passes_protocol is True]
    if passing_clean:
        return EvidenceLevel.A, [r.id for r in passing_clean], False

    decided_clean = [r for r in clean_external if r.passes_protocol is not None]
    if decided_clean and all(r.passes_protocol is False for r in decided_clean):
        return EvidenceLevel.D, [r.id for r in decided_clean], True

    published = [
        r
        for r in records
        if r.kind in ("published", "guideline_threshold")
        and r.passes_protocol is True
        and r.source is not None
        and r.source.verified_at_source
    ]
    if published:
        return EvidenceLevel.B, [r.id for r in published], False

    internal = [
        r
        for r in records
        if r.passes_protocol is True
        and (
            r.kind == "measured_internal" or (r.kind == "measured_external" and r.leakage != "none")
        )
    ]
    if internal:
        return EvidenceLevel.C, [r.id for r in internal], False

    return EvidenceLevel.D, [], False


def visibility_for(level: EvidenceLevel) -> Visibility:
    return "research" if level is EvidenceLevel.D else "main"
