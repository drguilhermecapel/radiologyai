"""Modelos pydantic de requisição e resposta da API v1 (todos congelados, ``extra="forbid"``).

O ``AnalysisResult`` do pipeline é devolvido como está (tipo de
:mod:`radiologyai.analysis.types`), apenas com os achados de nível D
omitidos quando o modo pesquisa não foi pedido.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from radiologyai.analysis.types import AnalysisResult, Primitive
from radiologyai.catalog.schema import CodedConcept, ExamType
from radiologyai.reporting.content import ReportContent

__all__ = [
    "AnalysisCreateIn",
    "AnalysisInput",
    "AnalysisOut",
    "AnalysisSummary",
    "AuditEventOut",
    "AuditListOut",
    "AuditVerifyOut",
    "CatalogConditionOut",
    "CatalogEntryOut",
    "CatalogOut",
    "DicomwebImportIn",
    "DicomwebSearchOut",
    "ErrorOut",
    "ExamTypeIn",
    "ExportOut",
    "HealthOut",
    "HealthzOut",
    "LoginIn",
    "LoginOut",
    "MetricsOut",
    "ModelCardOut",
    "ReadyzOut",
    "ReportIntegrity",
    "ReportOut",
    "ReviewIn",
    "ReviewOut",
    "SeriesMetaOut",
    "SeriesOut",
    "SignIn",
    "StatusOut",
    "StudyListItem",
    "StudyListOut",
    "StudyOut",
    "UserCreateIn",
    "UserOut",
    "UserUpdateIn",
]


class _Out(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())


class _In(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())


class ErrorOut(_Out):
    detail: str | dict[str, Any] | list[Any]
    error: str | None = None


# --------------------------------------------------------------------------- #
# Saúde
# --------------------------------------------------------------------------- #
class HealthzOut(_Out):
    status: Literal["ok"]


class ReadyzOut(_Out):
    ready: bool
    database_revision: str | None
    expected_revision: str
    worker: Literal["running", "stopped", "disabled"]
    store_writable: bool
    deident_key_configured: bool
    detail: tuple[str, ...] = ()


class HealthOut(_Out):
    status: Literal["ok"]
    version: str
    disclaimer: str
    subsystems: dict[str, bool]


class StatusOut(_Out):
    version: str
    modalities: list[dict[str, Any]]
    clinically_validated: Literal[False]
    regulatory_status: str
    disclaimer: str


class MetricsOut(_Out):
    """Métricas MEDIDAS lidas de ``artifacts/eval``; ``measured=false`` quando não há."""

    measured: bool
    reason: str | None = None
    how_to_produce: str | None = None
    run_id: str | None = None
    model: str | None = None
    dataset: str | None = None
    external_to_training_data: bool | None = None
    macro_auroc: float | None = None
    per_label: dict[str, Any] | None = None
    not_evaluated: list[Any] | None = None
    subgroups: dict[str, Any] | None = None
    limitations: list[str] | None = None
    provenance: dict[str, Any] | None = None


# --------------------------------------------------------------------------- #
# Autenticação e usuários
# --------------------------------------------------------------------------- #
Role = Literal["admin", "physician", "technician", "viewer"]


class LoginIn(_In):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=1024)


class UserOut(_Out):
    id: str
    username: str
    full_name: str
    role: Role
    professional_id: str | None
    active: bool
    created_at: datetime
    last_login_at: datetime | None
    permissions: tuple[str, ...]


class LoginOut(_Out):
    token: str
    token_type: Literal["bearer"] = "bearer"
    expires_at: datetime
    user: UserOut


class UserCreateIn(_In):
    username: str = Field(min_length=3, max_length=64, pattern=r"^[A-Za-z0-9._@-]+$")
    full_name: str = Field(min_length=1, max_length=200)
    role: Role
    password: str = Field(min_length=1, max_length=1024)
    professional_id: str | None = Field(default=None, max_length=64)


class UserUpdateIn(_In):
    full_name: str | None = Field(default=None, min_length=1, max_length=200)
    role: Role | None = None
    active: bool | None = None
    password: str | None = Field(default=None, min_length=1, max_length=1024)
    professional_id: str | None = Field(default=None, max_length=64)


# --------------------------------------------------------------------------- #
# Estudos, séries, quadros
# --------------------------------------------------------------------------- #
class SeriesOut(_Out):
    id: str
    position: int
    series_uid: str | None
    modality: str
    source_format: str
    kind: Literal["projection", "volume", "cine"]
    description: str | None
    body_part: str | None
    view_position: str | None
    laterality: str | None
    rows: int
    columns: int
    n_frames: int
    n_instances: int
    content_sha256: str
    human_overrides: dict[str, Any]


class SeriesMetaOut(_Out):
    """Metadados para o visualizador; ver ``GET /series/{id}/frames/{i}`` para os pixels."""

    series: SeriesOut
    image_meta: dict[str, Any]
    spacing_mm: tuple[float, ...] | None
    window: tuple[float, float] | None
    value_range: tuple[float, float]
    color: bool
    affine_lps: list[list[float]] | None
    frame_dtype: Literal["uint16", "int16", "uint8"]
    rescale_slope: float
    rescale_intercept: float
    default_max_dim: int
    max_dim_cap: int


class AnalysisSummary(_Out):
    id: str
    state: Literal["queued", "running", "completed", "failed"]
    result_status: str | None
    requested_at: datetime
    finished_at: datetime | None
    include_research: bool
    n_main_findings: int
    n_research_findings: int


class StudyOut(_Out):
    id: str
    study_uid: str | None
    patient_pseudo_id: str | None
    modalities: tuple[str, ...]
    description: str | None
    exam_type: ExamType
    exam_type_source: Literal["metadata", "human"]
    exam_type_set_at: datetime | None
    source: Literal["upload", "dicomweb"]
    created_at: datetime
    warnings: tuple[str, ...]
    series: tuple[SeriesOut, ...]
    analyses: tuple[AnalysisSummary, ...]


ReviewStatus = Literal["no_analysis", "pending_review", "in_review", "signed"]


class StudyListItem(_Out):
    id: str
    modalities: tuple[str, ...]
    description: str | None
    exam_type: ExamType
    exam_type_source: Literal["metadata", "human"]
    created_at: datetime
    n_series: int
    latest_analysis: AnalysisSummary | None
    review_status: ReviewStatus


class StudyListOut(_Out):
    items: tuple[StudyListItem, ...]
    total: int
    limit: int
    offset: int


class ExamTypeIn(_In):
    exam_type: ExamType


# --------------------------------------------------------------------------- #
# Análises e revisões
# --------------------------------------------------------------------------- #
class AnalysisCreateIn(_In):
    model_ids: tuple[str, ...] | None = Field(
        default=None, description="Modelos explícitos (ignora o roteamento); dependências entram."
    )
    include_research: bool = Field(
        default=False,
        description="Roda também modelos exploratórios (exige papel com modo pesquisa).",
    )
    prompts: tuple[Primitive, ...] = Field(
        default=(), description="Prompts do usuário para modelos interativos (caixa/ponto)."
    )


class AnalysisInput(_Out):
    image_idx: int
    series_id: str
    content_sha256: str


class ReviewIn(_In):
    finding_idx: int = Field(ge=0)
    action: Literal["accept", "reject", "amend"]
    amended_decision: Literal["positive", "negative", "indeterminate"] | None = None
    amended_text: str | None = Field(default=None, max_length=4000)
    comment: str | None = Field(default=None, max_length=4000)
    result_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", description="Hash do resultado que o revisor viu."
    )


class ReviewOut(_Out):
    id: str
    analysis_id: str
    finding_idx: int
    action: Literal["accept", "reject", "amend"]
    amended_decision: str | None
    amended_text: str | None
    comment: str | None
    reviewer_id: str
    reviewer_username: str
    result_sha256: str
    created_at: datetime


class AnalysisOut(_Out):
    id: str
    study_id: str
    state: Literal["queued", "running", "completed", "failed"]
    exam_type: ExamType
    include_research: bool
    model_ids: tuple[str, ...] | None
    requested_by: str
    requested_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    error: str | None
    code_version: str
    inputs: tuple[AnalysisInput, ...]
    result_sha256: str | None
    result_status: str | None
    research_view: bool = Field(description="True quando achados de nível D estão incluídos.")
    hidden_research_findings: int
    finding_indices: tuple[int, ...] | None = Field(
        description="Para cada achado exibido em result.findings, o índice no resultado completo "
        "(use-o em ReviewIn.finding_idx)."
    )
    result: AnalysisResult | None
    reviews: tuple[ReviewOut, ...]


# --------------------------------------------------------------------------- #
# Laudos
# --------------------------------------------------------------------------- #
class SignIn(_In):
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    password: str = Field(
        min_length=1, max_length=1024, description="Reautenticação do signatário."
    )


class ReportIntegrity(_Out):
    content_matches_hash: bool
    result_matches_hash: bool
    signature_valid: bool | None


class ReportOut(_Out):
    id: str
    analysis_id: str
    created_by: str
    created_at: datetime
    report_sha256: str
    result_sha256: str
    signed: bool
    signed_by: str | None
    signed_by_name: str | None
    signed_at: datetime | None
    signature_sha256: str | None
    integrity: ReportIntegrity
    content: ReportContent
    downloads: dict[str, str]


class ExportOut(_Out):
    stored_sop_instance_uids: tuple[str, ...]
    destination: str


# --------------------------------------------------------------------------- #
# Modelos e catálogo
# --------------------------------------------------------------------------- #
class WeightOut(_Out):
    key: str
    filename: str
    sha256: str
    bytes: int
    state: Literal["ok", "missing", "mismatch", "unpinned", "unverified"] = Field(
        description="ok/mismatch só com verify_weights=true; 'unverified' = presente, hash não "
        "recalculado nesta consulta (a carga do modelo sempre confere)."
    )


class LabelOut(_Out):
    key: str
    source_label: str
    condition_id: str
    kind: str
    score_kind: str
    code: CodedConcept
    evidence_level: Literal["A", "B", "C", "D"]
    visibility: Literal["main", "research"]
    in_catalog: bool


class ModelCardOut(_Out):
    id: str
    name: str
    version: str
    status: Literal["available", "planned", "retired"]
    planned_reason: str | None
    task: str
    route: str
    summary: str
    intended_use: Literal["research_use_only"]
    capabilities: tuple[str, ...]
    exam_types: tuple[ExamType, ...]
    modalities: tuple[str, ...]
    license_code: str
    license_weights: str
    commercial_use: Literal["yes", "no", "unclear"]
    training_data_terms: str | None
    training_data: tuple[str, ...]
    depends_on: tuple[str, ...]
    weights: tuple[WeightOut, ...]
    labels: tuple[LabelOut, ...]
    limitations: tuple[str, ...]
    references: tuple[dict[str, str | None], ...]
    manifest_sha256: str


class CatalogEntryOut(_Out):
    model_id: str
    label: str
    level: Literal["A", "B", "C", "D"]
    visibility: Literal["main", "research"]
    headline: str | None
    basis: tuple[str, ...]


class CatalogConditionOut(_Out):
    id: str
    name_pt: str
    name_en: str
    group: str
    exam_types: tuple[ExamType, ...]
    system: str
    code: CodedConcept
    entries: tuple[CatalogEntryOut, ...]
    best_level: Literal["A", "B", "C", "D"]
    research_only: bool


class UncataloguedLabelOut(_Out):
    model_id: str
    label: str
    condition_id: str
    level: Literal["D"] = "D"
    visibility: Literal["research"] = "research"


class CatalogOut(_Out):
    protocol_id: str | None
    level_meanings: dict[str, str]
    conditions: tuple[CatalogConditionOut, ...]
    uncatalogued: tuple[UncataloguedLabelOut, ...]


# --------------------------------------------------------------------------- #
# Auditoria e DICOMweb
# --------------------------------------------------------------------------- #
class AuditEventOut(_Out):
    seq: int
    timestamp_utc: str
    event: str
    actor_id: str | None
    actor_username: str | None
    subject_type: str | None
    subject_id: str | None
    payload: dict[str, Any]
    prev_hash: str
    record_hash: str


class AuditListOut(_Out):
    items: tuple[AuditEventOut, ...]
    next_after_seq: int | None


class AuditVerifyOut(_Out):
    ok: bool
    events: int
    last_hash: str
    broken_at_seq: int | None
    detail: str | None


class DicomwebSearchOut(_Out):
    items: tuple[dict[str, Any], ...]


class DicomwebImportIn(_In):
    study_instance_uid: str = Field(pattern=r"^[0-9.]{1,64}$")
    exam_type: ExamType | None = None
