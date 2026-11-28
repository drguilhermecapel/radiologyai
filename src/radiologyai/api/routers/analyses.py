"""Análises (fila), revisões por achado e quadros de sobreposição de segmentação."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from sqlalchemy import select

from radiologyai.api.deps import Principal, get_platform, perm, require
from radiologyai.api.routers.studies import _ERRORS, FRAME_HEADERS, frame_headers
from radiologyai.api.schemas import AnalysisCreateIn, AnalysisOut, ReviewIn, ReviewOut
from radiologyai.api.shadow import ai_visible, audit_redaction, block_ai, redact_analysis
from radiologyai.api.views import analysis_out, review_out
from radiologyai.persistence.models import Analysis, Review, Study
from radiologyai.persistence.security import Permission
from radiologyai.worker.runtime import Platform

router = APIRouter(tags=["analyses"])


@router.post(
    "/studies/{study_id}/analyses",
    response_model=AnalysisOut,
    status_code=status.HTTP_202_ACCEPTED,
    responses=_ERRORS,
    openapi_extra=perm(Permission.ANALYSES_CREATE),
)
def create_analysis(
    principal: Annotated[Principal, Depends(require(Permission.ANALYSES_CREATE))],
    platform: Annotated[Platform, Depends(get_platform)],
    study_id: str,
    request: Request,
    body: AnalysisCreateIn | None = None,
) -> AnalysisOut:
    """Enfileira uma análise (202). Cada pedido cria uma análise nova; nada é sobrescrito.

    ``include_research`` exige papel com modo pesquisa; pedir explicitamente em
    ``model_ids`` um modelo exploratório (ou que dependa de um) também exige, e
    grava a análise como de modo pesquisa. ``prompts`` alimentam
    modelos interativos (ex.: caixa para MedSAM) e são do usuário, nunca inventados.
    """
    from radiologyai.worker.jobs import enqueue_analysis, research_models

    body = body or AnalysisCreateIn()
    if body.include_research and not principal.can(Permission.RESEARCH_MODE):
        raise HTTPException(status_code=403, detail="modo pesquisa não permitido para o seu papel")
    if body.model_ids and not principal.can(Permission.RESEARCH_MODE):
        exploratory = research_models(platform, body.model_ids)
        if exploratory:
            raise HTTPException(
                status_code=403,
                detail="modelo(s) exploratório(s) exigem modo pesquisa: " + ", ".join(exploratory),
            )
    row = enqueue_analysis(
        platform,
        study_id,
        user_id=principal.user_id,
        username=principal.username,
        model_ids=body.model_ids,
        include_research=body.include_research,
        prompts=[p.model_dump(mode="json") for p in body.prompts],
    )
    worker = getattr(request.app.state, "worker", None)
    if worker is not None:
        worker.notify()
    with platform.db.session() as session:
        analysis = session.get(Analysis, row.id)
        assert analysis is not None
        out = analysis_out(session, analysis, research_view=False, result=None)
    return out if ai_visible(platform, principal) else redact_analysis(out)


@router.get(
    "/studies/{study_id}/analyses",
    response_model=list[AnalysisOut],
    responses=_ERRORS,
    openapi_extra=perm(Permission.ANALYSES_READ),
)
def list_analyses(
    principal: Annotated[Principal, Depends(require(Permission.ANALYSES_READ))],
    platform: Annotated[Platform, Depends(get_platform)],
    study_id: str,
) -> list[AnalysisOut]:
    """Análises do estudo, mais recente primeiro (sem o resultado completo).

    Modo silencioso (REQ-195) sem ``shadow:view``: cada item vem com ``shadow: true``
    e sem hash, estado do resultado nem revisões.
    """
    visible = ai_visible(platform, principal)
    with platform.db.session() as session:
        if session.get(Study, study_id) is None:
            raise HTTPException(status_code=404, detail="estudo inexistente")
        rows = session.execute(
            select(Analysis)
            .where(Analysis.study_id == study_id)
            .order_by(Analysis.requested_at.desc())
        ).scalars()
        outs = [analysis_out(session, a, research_view=False, result=None) for a in rows]
    return outs if visible else [redact_analysis(o) for o in outs]


@router.get(
    "/analyses/{analysis_id}",
    response_model=AnalysisOut,
    responses=_ERRORS,
    openapi_extra=perm(Permission.ANALYSES_READ),
)
def get_analysis(
    principal: Annotated[Principal, Depends(require(Permission.ANALYSES_READ))],
    platform: Annotated[Platform, Depends(get_platform)],
    analysis_id: str,
    include_research: Annotated[
        bool, Query(description="Inclui achados de nível D (exige papel com modo pesquisa)")
    ] = False,
) -> AnalysisOut:
    """Estado e, quando concluída, o resultado imutável (conferido contra ``result_sha256``).

    Por padrão achados exploratórios (nível D) são omitidos e contados em
    ``hidden_research_findings``.

    Modo silencioso (REQ-195) sem ``shadow:view``: estado, entradas e tempos são
    servidos; a saída de IA não (``shadow: true``, ``result: null``). Servir um
    resultado concluído assim é auditado (``shadow.redacted``).
    """
    from radiologyai.reporting.service import load_result

    if include_research and not principal.can(Permission.RESEARCH_MODE):
        raise HTTPException(status_code=403, detail="modo pesquisa não permitido para o seu papel")
    with platform.db.session() as session:
        row = session.get(Analysis, analysis_id)
        if row is None:
            raise HTTPException(status_code=404, detail="análise inexistente")
        if not ai_visible(platform, principal):
            out = redact_analysis(analysis_out(session, row, research_view=False, result=None))
            completed = row.result_json is not None
        else:
            result = load_result(row) if row.result_json is not None else None
            return analysis_out(session, row, research_view=include_research, result=result)
    if completed:
        audit_redaction(platform, principal, analysis_id=analysis_id)
    return out


@router.post(
    "/analyses/{analysis_id}/reviews",
    response_model=ReviewOut,
    status_code=status.HTTP_201_CREATED,
    responses=_ERRORS,
    openapi_extra=perm(Permission.REVIEWS_WRITE),
)
def create_review(
    principal: Annotated[Principal, Depends(require(Permission.REVIEWS_WRITE))],
    platform: Annotated[Platform, Depends(get_platform)],
    analysis_id: str,
    body: ReviewIn,
) -> ReviewOut:
    """Aceita, rejeita ou emenda UM achado (índice no resultado completo).

    A revisão carrega o ``result_sha256`` que o médico viu; divergente => 409.
    Revisões são somente inserção: a mais recente vale; o histórico fica.
    """
    from radiologyai.persistence.audit import append_event
    from radiologyai.reporting.service import load_result

    block_ai(
        platform, principal, action="review.create", subject_type="analysis", subject_id=analysis_id
    )
    if body.action == "amend" and not (body.amended_decision or body.amended_text):
        raise HTTPException(status_code=422, detail="emenda exige amended_decision ou amended_text")
    with platform.db.session() as session:
        row = session.get(Analysis, analysis_id)
        if row is None:
            raise HTTPException(status_code=404, detail="análise inexistente")
        if row.state != "completed":
            raise HTTPException(status_code=409, detail=f"análise está {row.state!r}")
        if row.result_sha256 != body.result_sha256:
            raise HTTPException(
                status_code=409, detail="result_sha256 não corresponde ao resultado gravado"
            )
        result = load_result(row)
        if not 0 <= body.finding_idx < len(result.findings):
            raise HTTPException(status_code=404, detail="achado inexistente")
        finding = result.findings[body.finding_idx]
        if finding.visibility != "main":
            raise HTTPException(
                status_code=409,
                detail="achado exploratório (nível D) não entra no laudo e não é revisado",
            )
        review = Review(
            analysis_id=analysis_id,
            finding_idx=body.finding_idx,
            reviewer_id=principal.user_id,
            action=body.action,
            amended_decision=body.amended_decision,
            amended_text=body.amended_text,
            comment=body.comment,
            result_sha256=body.result_sha256,
        )
        session.add(review)
        session.flush()
        out = review_out(review, principal.username)
        append_event(
            None,
            "review.created",
            actor_id=principal.user_id,
            actor_username=principal.username,
            subject_type="analysis",
            subject_id=analysis_id,
            payload={
                "finding_idx": body.finding_idx,
                "condition_id": finding.condition_id,
                "action": body.action,
                "amended_decision": body.amended_decision,
                "result_sha256": body.result_sha256,
            },
            session=session,
        )
    return out


@router.get(
    "/analyses/{analysis_id}/reviews",
    response_model=list[ReviewOut],
    responses=_ERRORS,
    openapi_extra=perm(Permission.ANALYSES_READ),
)
def list_reviews(
    principal: Annotated[Principal, Depends(require(Permission.ANALYSES_READ))],
    platform: Annotated[Platform, Depends(get_platform)],
    analysis_id: str,
) -> list[ReviewOut]:
    """Histórico completo de revisões (somente inserção)."""
    block_ai(
        platform, principal, action="review.list", subject_type="analysis", subject_id=analysis_id
    )
    return list(get_analysis_reviews(platform, analysis_id))


def get_analysis_reviews(platform: Platform, analysis_id: str) -> tuple[ReviewOut, ...]:
    with platform.db.session() as session:
        row = session.get(Analysis, analysis_id)
        if row is None:
            raise HTTPException(status_code=404, detail="análise inexistente")
        return analysis_out(session, row, research_view=False, result=None).reviews


@router.get(
    "/analyses/{analysis_id}/segmentations/{seg_index}/frames/{index}",
    response_class=Response,
    responses={
        200: {"content": {"application/octet-stream": {}}, "description": FRAME_HEADERS},
        **_ERRORS,
    },
    openapi_extra=perm(Permission.ANALYSES_READ),
)
def segmentation_frame(
    principal: Annotated[Principal, Depends(require(Permission.ANALYSES_READ))],
    platform: Annotated[Platform, Depends(get_platform)],
    analysis_id: str,
    seg_index: int,
    index: int,
    max_dim: Annotated[int | None, Query(ge=16)] = None,
    include_research: Annotated[
        bool,
        Query(description="Exigido para segmentação exploratória (papel com modo pesquisa)"),
    ] = False,
) -> Response:
    """Quadro do mapa de rótulos na MESMA grade de ``/series/{id}/frames/{index}``.

    ``seg_index`` é o índice no resultado COMPLETO (``AnalysisOut.segmentation_indices``).
    Valores inteiros (``uint8``/``uint16``): 0 é fundo; o significado de cada
    índice está em ``result.segmentations[seg_index].labels``. Segmentação sem
    achado A-C ligado (exploratória) só é servida com ``include_research=true``
    a papel com modo pesquisa; caso contrário, 403.
    """
    from radiologyai.interop.images import encode_label_frame
    from radiologyai.reporting.exposure import main_exposure
    from radiologyai.reporting.service import load_result

    block_ai(
        platform,
        principal,
        action="segmentation.frame",
        subject_type="analysis",
        subject_id=analysis_id,
    )
    with platform.db.session() as session:
        row = session.get(Analysis, analysis_id)
        if row is None or row.result_json is None:
            raise HTTPException(status_code=404, detail="análise sem resultado")
        result = load_result(row)
    if not 0 <= seg_index < len(result.segmentations):
        raise HTTPException(status_code=404, detail="segmentação inexistente")
    if seg_index not in main_exposure(result).segmentations and not (
        include_research and principal.can(Permission.RESEARCH_MODE)
    ):
        raise HTTPException(
            status_code=403,
            detail="segmentação exploratória (sem achado A-C ligado) só no modo pesquisa",
        )
    seg = result.segmentations[seg_index]
    labelmap = platform.blobs.store.read_labelmap(seg.mask)
    limit = min(max_dim or platform.settings.frames_max_dim, platform.settings.frames_hard_cap)
    data = encode_label_frame(labelmap, index, limit)
    count = 1 if labelmap.ndim == 2 else int(labelmap.shape[0])
    return Response(
        content=data.data,
        media_type="application/octet-stream",
        headers=frame_headers(data, index, count),
    )
