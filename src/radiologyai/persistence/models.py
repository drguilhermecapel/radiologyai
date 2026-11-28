"""Tabelas ORM da plataforma (SQLAlchemy 2, estilo declarativo tipado).

Regras de dados codificadas aqui:

* Nenhuma coluna guarda identificador direto de paciente. Estudos e séries
  guardam UIDs já pseudonimizados e o pseudônimo HMAC do paciente.
* ``analyses.result_json`` é escrito uma única vez (o worker grava o resultado
  completo e seu ``result_sha256``); reexecutar é criar outra linha.
* ``audit_events`` é encadeada por hash; ``prev_hash`` é único, o que impede
  duas cadeias concorrentes a partir do mesmo elo.
* Senhas só como hash argon2; tokens de sessão só como sha256.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator

__all__ = [
    "Analysis",
    "Artifact",
    "AuditEvent",
    "AuthSession",
    "Base",
    "FindingRow",
    "Instance",
    "Report",
    "Review",
    "Series",
    "Study",
    "User",
    "new_id",
    "utcnow",
]

NAMING = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


def new_id() -> str:
    """Identificador opaco (UUID4 em texto)."""
    return str(uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(UTC)


class UTCDateTime(TypeDecorator[datetime]):
    """Datas sempre com fuso UTC, inclusive no SQLite (que descarta o fuso)."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        del dialect
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("datas persistidas devem ter fuso horário (UTC)")
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        del dialect
        if value is None:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    username: Mapped[str] = mapped_column(String(64), unique=True)
    full_name: Mapped[str] = mapped_column(String(200))
    role: Mapped[str] = mapped_column(String(16))
    password_hash: Mapped[str] = mapped_column(String(255))
    professional_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    last_login_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)


class AuthSession(Base):
    __tablename__ = "auth_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    token_sha256: Mapped[str] = mapped_column(String(64), unique=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime())
    revoked_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    last_seen_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)

    user: Mapped[User] = relationship()


class Artifact(Base):
    """Índice dos arquivos no repositório endereçado por conteúdo."""

    __tablename__ = "artifacts"

    sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    media_type: Mapped[str] = mapped_column(String(64))
    relpath: Mapped[str] = mapped_column(String(128))
    size_bytes: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)


class Study(Base):
    __tablename__ = "studies"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    study_uid: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    patient_pseudo_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    modalities: Mapped[str] = mapped_column(String(64))
    description: Mapped[str | None] = mapped_column(String(256), nullable=True)
    exam_type: Mapped[str] = mapped_column(String(32), index=True)
    exam_type_source: Mapped[str] = mapped_column(String(16))  # metadata | human
    exam_type_set_by: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    exam_type_set_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    source: Mapped[str] = mapped_column(String(16))  # upload | dicomweb
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)
    warnings: Mapped[list[str]] = mapped_column(JSON, default=list)

    series: Mapped[list[Series]] = relationship(
        back_populates="study", order_by="Series.position", cascade="all, delete-orphan"
    )


class Series(Base):
    __tablename__ = "series"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    study_id: Mapped[str] = mapped_column(ForeignKey("studies.id"), index=True)
    position: Mapped[int] = mapped_column(Integer)
    series_uid: Mapped[str | None] = mapped_column(String(64), nullable=True)
    modality: Mapped[str] = mapped_column(String(8))
    source_format: Mapped[str] = mapped_column(String(16))  # dicom | dicom_series | nifti | raster
    kind: Mapped[str] = mapped_column(String(16))  # projection | volume | cine
    description: Mapped[str | None] = mapped_column(String(256), nullable=True)
    body_part: Mapped[str | None] = mapped_column(String(64), nullable=True)
    view_position: Mapped[str | None] = mapped_column(String(32), nullable=True)
    laterality: Mapped[str | None] = mapped_column(String(1), nullable=True)
    rows: Mapped[int] = mapped_column(Integer)
    columns: Mapped[int] = mapped_column(Integer)
    n_frames: Mapped[int] = mapped_column(Integer)
    n_instances: Mapped[int] = mapped_column(Integer)
    content_sha256: Mapped[str] = mapped_column(String(64))
    meta: Mapped[dict[str, Any]] = mapped_column(JSON)  # ImageMeta + geometria
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)

    study: Mapped[Study] = relationship(back_populates="series")
    instances: Mapped[list[Instance]] = relationship(
        back_populates="series", order_by="Instance.ordinal", cascade="all, delete-orphan"
    )


class Instance(Base):
    __tablename__ = "instances"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    series_id: Mapped[str] = mapped_column(ForeignKey("series.id"), index=True)
    ordinal: Mapped[int] = mapped_column(Integer)
    sop_uid: Mapped[str | None] = mapped_column(String(64), nullable=True)
    artifact_sha256: Mapped[str] = mapped_column(ForeignKey("artifacts.sha256"))
    filename: Mapped[str] = mapped_column(String(64))  # nome com sufixo correto p/ leitores

    series: Mapped[Series] = relationship(back_populates="instances")


class Analysis(Base):
    __tablename__ = "analyses"
    __table_args__ = (Index("ix_analyses_state_requested", "state", "requested_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    study_id: Mapped[str] = mapped_column(ForeignKey("studies.id"), index=True)
    state: Mapped[str] = mapped_column(String(16))  # queued | running | completed | failed
    exam_type: Mapped[str] = mapped_column(String(32))
    include_research: Mapped[bool] = mapped_column(Boolean, default=False)
    model_ids: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    prompts: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    inputs: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    requested_by: Mapped[str] = mapped_column(ForeignKey("users.id"))
    requested_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    worker_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    code_version: Mapped[str] = mapped_column(String(32))
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    result_status: Mapped[str | None] = mapped_column(String(16), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    study: Mapped[Study] = relationship()


class FindingRow(Base):
    """Achado desnormalizado do ``result_json`` para filtros e lista de trabalho."""

    __tablename__ = "findings"
    __table_args__ = (UniqueConstraint("analysis_id", "idx"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[str] = mapped_column(ForeignKey("analyses.id"), index=True)
    idx: Mapped[int] = mapped_column(Integer)
    model_id: Mapped[str] = mapped_column(String(64))
    condition_id: Mapped[str] = mapped_column(String(128), index=True)
    label: Mapped[str] = mapped_column(String(128))
    kind: Mapped[str] = mapped_column(String(16))
    decision: Mapped[str] = mapped_column(String(16))
    evidence_level: Mapped[str] = mapped_column(String(1))
    visibility: Mapped[str] = mapped_column(String(8))
    raw_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    op_normalized: Mapped[float | None] = mapped_column(Float, nullable=True)
    probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    abstention_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    image_idx: Mapped[int] = mapped_column(Integer)


class Review(Base):
    """Decisão do médico sobre um achado. Somente inserção; a última vale."""

    __tablename__ = "reviews"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    analysis_id: Mapped[str] = mapped_column(ForeignKey("analyses.id"), index=True)
    finding_idx: Mapped[int] = mapped_column(Integer)
    reviewer_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    action: Mapped[str] = mapped_column(String(8))  # accept | reject | amend
    amended_decision: Mapped[str | None] = mapped_column(String(16), nullable=True)
    amended_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_sha256: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)


class Report(Base):
    """Instantâneo imutável do laudo preliminar; a assinatura amarra os dois hashes."""

    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    analysis_id: Mapped[str] = mapped_column(ForeignKey("analyses.id"), index=True)
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    content_json: Mapped[str] = mapped_column(Text)
    report_sha256: Mapped[str] = mapped_column(String(64))
    result_sha256: Mapped[str] = mapped_column(String(64))
    signed_by: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    signed_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    signature_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)


class AuditEvent(Base):
    """Evento de auditoria encadeado por hash (somente inserção)."""

    __tablename__ = "audit_events"

    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    timestamp_utc: Mapped[str] = mapped_column(String(40))
    event: Mapped[str] = mapped_column(String(64), index=True)
    actor_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    actor_username: Mapped[str | None] = mapped_column(String(64), nullable=True)
    subject_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    subject_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    payload_json: Mapped[str] = mapped_column(Text)
    prev_hash: Mapped[str] = mapped_column(String(64), unique=True)
    record_hash: Mapped[str] = mapped_column(String(64), unique=True)
