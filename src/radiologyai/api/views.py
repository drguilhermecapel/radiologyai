"""Conversão de linhas ORM em modelos de resposta (sem lógica de negócio)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from radiologyai.api.schemas import (
    AnalysisInput,
    AnalysisOut,
    AnalysisSummary,
    ReviewOut,
    SeriesOut,
    StudyOut,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from radiologyai.analysis.types import AnalysisResult
    from radiologyai.persistence.models import Analysis, Review, Series, Study

__all__ = [
    "analysis_out",
    "analysis_summary",
    "filter_research",
    "review_out",
    "series_out",
    "study_out",
]


def series_out(s: Series) -> SeriesOut:
    return SeriesOut(
        id=s.id,
        position=s.position,
        series_uid=s.series_uid,
        modality=s.modality,
        source_format=s.source_format,
        kind=s.kind,  # type: ignore[arg-type]
        description=s.description,
        body_part=s.body_part,
        view_position=s.view_position,
        laterality=s.laterality,
        rows=s.rows,
        columns=s.columns,
        n_frames=s.n_frames,
        n_instances=s.n_instances,
        content_sha256=s.content_sha256,
        human_overrides=dict(s.meta.get("human_overrides") or {}),
    )


def _counts(session: Session, analysis_id: str) -> tuple[int, int]:
    from sqlalchemy import func, select

    from radiologyai.persistence.models import FindingRow

    rows = session.execute(
        select(FindingRow.visibility, func.count())
        .where(FindingRow.analysis_id == analysis_id)
        .group_by(FindingRow.visibility)
    ).all()
    by = {str(v): int(n) for v, n in rows}
    return by.get("main", 0), by.get("research", 0)


def analysis_summary(session: Session, a: Analysis) -> AnalysisSummary:
    main, research = _counts(session, a.id)
    return AnalysisSummary(
        id=a.id,
        state=a.state,  # type: ignore[arg-type]
        result_status=a.result_status,
        requested_at=a.requested_at,
        finished_at=a.finished_at,
        include_research=a.include_research,
        n_main_findings=main,
        n_research_findings=research,
    )


def study_out(session: Session, study: Study) -> StudyOut:
    from sqlalchemy import select

    from radiologyai.persistence.models import Analysis

    analyses = session.execute(
        select(Analysis).where(Analysis.study_id == study.id).order_by(Analysis.requested_at.desc())
    ).scalars()
    return StudyOut(
        id=study.id,
        study_uid=study.study_uid,
        patient_pseudo_id=study.patient_pseudo_id,
        modalities=tuple(m for m in study.modalities.split(",") if m),
        description=study.description,
        exam_type=study.exam_type,  # type: ignore[arg-type]
        exam_type_source=study.exam_type_source,  # type: ignore[arg-type]
        exam_type_set_at=study.exam_type_set_at,
        source=study.source,  # type: ignore[arg-type]
        created_at=study.created_at,
        warnings=tuple(study.warnings or ()),
        series=tuple(series_out(s) for s in study.series),
        analyses=tuple(analysis_summary(session, a) for a in analyses),
    )


def review_out(r: Review, username: str) -> ReviewOut:
    return ReviewOut(
        id=r.id,
        analysis_id=r.analysis_id,
        finding_idx=r.finding_idx,
        action=r.action,  # type: ignore[arg-type]
        amended_decision=r.amended_decision,
        amended_text=r.amended_text,
        comment=r.comment,
        reviewer_id=r.reviewer_id,
        reviewer_username=username,
        result_sha256=r.result_sha256,
        created_at=r.created_at,
    )


def filter_research(result: AnalysisResult) -> tuple[AnalysisResult, int]:
    """Remove achados de nível D (e explicações só deles); devolve o resultado e quantos saíram.

    A ordem é preservada; ``AnalysisOut.finding_indices`` dá, para cada achado
    exibido, o índice no resultado COMPLETO — é esse índice que a revisão usa.
    """
    kept = [f for f in result.findings if f.visibility == "main"]
    hidden = len(result.findings) - len(kept)
    referenced = {eid for f in kept for eid in f.explanation_ids}
    data: dict[str, Any] = result.model_dump()
    data["findings"] = [f.model_dump() for f in kept]
    data["explanations"] = [e.model_dump() for e in result.explanations if e.id in referenced]
    from radiologyai.analysis.types import AnalysisResult as Result

    return Result.model_validate(data), hidden


def analysis_out(
    session: Session, a: Analysis, *, research_view: bool, result: AnalysisResult | None
) -> AnalysisOut:
    from sqlalchemy import select

    from radiologyai.persistence.models import Review, User

    hidden = 0
    shown = result
    indices: tuple[int, ...] | None = None
    if result is not None:
        indices = tuple(
            i for i, f in enumerate(result.findings) if research_view or f.visibility == "main"
        )
        if not research_view:
            shown, hidden = filter_research(result)
    reviews = session.execute(
        select(Review, User.username)
        .join(User, User.id == Review.reviewer_id)
        .where(Review.analysis_id == a.id)
        .order_by(Review.created_at)
    ).all()
    return AnalysisOut(
        id=a.id,
        study_id=a.study_id,
        state=a.state,  # type: ignore[arg-type]
        exam_type=a.exam_type,  # type: ignore[arg-type]
        include_research=a.include_research,
        model_ids=tuple(a.model_ids) if a.model_ids else None,
        requested_by=a.requested_by,
        requested_at=a.requested_at,
        started_at=a.started_at,
        finished_at=a.finished_at,
        error=a.error,
        code_version=a.code_version,
        inputs=tuple(AnalysisInput.model_validate(i) for i in a.inputs),
        result_sha256=a.result_sha256,
        result_status=a.result_status,
        research_view=research_view,
        hidden_research_findings=hidden,
        finding_indices=indices,
        result=shown,
        reviews=tuple(review_out(r, u) for r, u in reviews),
    )
