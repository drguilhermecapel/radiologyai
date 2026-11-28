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

    from radiologyai.analysis.evidence_policy import HoldLookup
    from radiologyai.analysis.types import AnalysisResult
    from radiologyai.persistence.models import Analysis, Review, Series, Study

__all__ = [
    "analysis_out",
    "analysis_summary",
    "apply_holds",
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


def study_out(session: Session, study: Study, *, redact: bool = False) -> StudyOut:
    """Estudo com séries e resumos das análises; ``redact`` = modo silencioso (REQ-195)."""
    from sqlalchemy import select

    from radiologyai.api.shadow import redact_summary
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
        analyses=tuple(
            redact_summary(analysis_summary(session, a)) if redact else analysis_summary(session, a)
            for a in analyses
        ),
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
    """Remove a saída exploratória; devolve o resultado e quantos ACHADOS saíram.

    Saem os achados de nível D, as explicações só deles e as segmentações e
    medidas sem achado A-C ligado (:mod:`radiologyai.reporting.exposure`) —
    máscara ou volume de modelo nível D nunca aparece na visão padrão.

    A ordem é preservada; ``AnalysisOut.finding_indices`` e
    ``AnalysisOut.segmentation_indices`` dão, para cada item exibido, o índice
    no resultado COMPLETO — é esse índice que a revisão e os quadros usam.
    """
    from radiologyai.reporting.exposure import main_exposure

    kept = [f for f in result.findings if f.visibility == "main"]
    hidden = len(result.findings) - len(kept)
    referenced = {eid for f in kept for eid in f.explanation_ids}
    exposure = main_exposure(result)
    data: dict[str, Any] = result.model_dump()
    data["findings"] = [f.model_dump() for f in kept]
    data["explanations"] = [e.model_dump() for e in result.explanations if e.id in referenced]
    data["segmentations"] = [
        s.model_dump() for i, s in enumerate(result.segmentations) if i in exposure.segmentations
    ]
    data["measurements"] = [
        m.model_dump() for i, m in enumerate(result.measurements) if i in exposure.measurements
    ]
    from radiologyai.analysis.types import AnalysisResult as Result

    return Result.model_validate(data), hidden


def apply_holds(result: AnalysisResult, held: HoldLookup) -> AnalysisResult:
    """Reaplica as suspensões de decisão vigentes (REQ-193) a um resultado GRAVADO.

    Achado de trio suspenso — inclusive de análise anterior à suspensão — é servido como
    ``score_only``, sem ponto de operação, com o motivo em ``decision_note`` e um aviso no
    resultado. A ordem e os índices dos achados não mudam (a revisão usa o índice).
    """
    from radiologyai.analysis.evidence_policy import apply_hold

    findings = list(result.findings)
    warnings: list[str] = []
    for idx, finding in enumerate(findings):
        if finding.decision in ("score_only", "abstained"):
            continue
        model_id = result.provenance[finding.provenance_idx].model_id
        hold = held(finding.condition_id, model_id, finding.label)
        if hold is None:
            continue
        findings[idx] = apply_hold(finding, hold)
        warnings.append(
            f"achado {idx} ({finding.condition_id}, {model_id}:{finding.label}): decisão "
            f"gravada ({finding.decision}) retirada — {findings[idx].decision_note}"
        )
    if not warnings:
        return result
    return result.model_copy(
        update={"findings": tuple(findings), "warnings": (*result.warnings, *warnings)}
    )


def analysis_out(
    session: Session,
    a: Analysis,
    *,
    research_view: bool,
    result: AnalysisResult | None,
    held: HoldLookup | None = None,
) -> AnalysisOut:
    """Análise para a API; ``held`` reaplica as suspensões de decisão vigentes (REQ-193)."""
    from sqlalchemy import select

    from radiologyai.persistence.models import Review, User
    from radiologyai.reporting.exposure import main_exposure

    hidden = 0
    hidden_artifacts = 0
    shown = result
    indices: tuple[int, ...] | None = None
    seg_indices: tuple[int, ...] | None = None
    research_segs: tuple[int, ...] = ()
    research_measures: tuple[str, ...] = ()
    if result is not None and held is not None:
        result = shown = apply_holds(result, held)
    if result is not None:
        exposure = main_exposure(result)
        indices = tuple(
            i for i, f in enumerate(result.findings) if research_view or f.visibility == "main"
        )
        seg_indices = tuple(
            i
            for i in range(len(result.segmentations))
            if research_view or i in exposure.segmentations
        )
        hidden_artifacts = (
            len(result.segmentations)
            - len(exposure.segmentations)
            + len(result.measurements)
            - len(exposure.measurements)
        )
        if research_view:
            research_segs = tuple(
                i for i in range(len(result.segmentations)) if i not in exposure.segmentations
            )
            research_measures = tuple(
                m.name for i, m in enumerate(result.measurements) if i not in exposure.measurements
            )
        else:
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
        segmentation_indices=seg_indices,
        hidden_research_artifacts=0 if research_view else hidden_artifacts,
        research_segmentation_indices=research_segs,
        research_measurement_names=research_measures,
        result=shown,
        reviews=tuple(review_out(r, u) for r, u in reviews),
    )
