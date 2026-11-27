"""Laudos preliminares: instantâneo, assinatura vinculada a hash, PDF, SR, SEG, FHIR."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select

from radiologyai.api.app import API_PREFIX
from radiologyai.api.deps import Principal, get_platform, perm, require
from radiologyai.api.routers.studies import _ERRORS
from radiologyai.api.schemas import (
    ExportOut,
    ExportSkipOut,
    ReportIntegrity,
    ReportOut,
    SignIn,
)
from radiologyai.persistence.models import Analysis, Report, User
from radiologyai.persistence.security import Permission
from radiologyai.reporting.content import ReportContent
from radiologyai.worker.runtime import Platform

router = APIRouter(tags=["reports"])


def report_out(platform: Platform, report_id: str) -> ReportOut:
    from radiologyai.persistence.config import server_key
    from radiologyai.reporting.service import integrity

    with platform.db.session() as session:
        report = session.get(Report, report_id)
        if report is None:
            raise HTTPException(status_code=404, detail="laudo inexistente")
        analysis = session.get(Analysis, report.analysis_id)
        assert analysis is not None
        check = integrity(report, analysis, server_key(platform.settings))
        signer = session.get(User, report.signed_by) if report.signed_by else None
        content = ReportContent.model_validate_json(report.content_json)
        base = f"{API_PREFIX}/reports/{report.id}"
        downloads = {"pdf": f"{base}/pdf", "fhir": f"{base}/fhir", "sr": f"{base}/sr"}
        if check.result_matches_hash:
            # Só segmentações ligadas a achado A-C do laudo não rejeitado; máscara
            # exploratória nunca vira SEG nem sai para o PACS.
            for item in content.segmentations:
                downloads[f"seg/{item.idx}"] = f"{base}/seg/{item.idx}"
        return ReportOut(
            id=report.id,
            analysis_id=report.analysis_id,
            created_by=report.created_by,
            created_at=report.created_at,
            report_sha256=report.report_sha256,
            result_sha256=report.result_sha256,
            signed=report.signed_at is not None,
            signed_by=report.signed_by,
            signed_by_name=signer.full_name if signer else None,
            signed_at=report.signed_at,
            signature_sha256=report.signature_sha256,
            integrity=ReportIntegrity(
                content_matches_hash=check.content_matches_hash,
                result_matches_hash=check.result_matches_hash,
                signature_valid=check.signature_valid,
            ),
            content=content,
            downloads=downloads,
        )


@router.post(
    "/analyses/{analysis_id}/reports",
    response_model=ReportOut,
    status_code=status.HTTP_201_CREATED,
    responses=_ERRORS,
    openapi_extra=perm(Permission.REPORTS_CREATE),
)
def create(
    principal: Annotated[Principal, Depends(require(Permission.REPORTS_CREATE))],
    platform: Annotated[Platform, Depends(get_platform)],
    analysis_id: str,
) -> ReportOut:
    """Instantâneo do laudo a partir do resultado imutável e das revisões atuais.

    Achados de nível D nunca entram. Mudou uma revisão? Crie um laudo novo.
    """
    from radiologyai.reporting.service import create_report

    report_id = create_report(
        platform, analysis_id, user_id=principal.user_id, username=principal.username
    )
    return report_out(platform, report_id)


@router.get(
    "/analyses/{analysis_id}/reports",
    response_model=list[ReportOut],
    responses=_ERRORS,
    openapi_extra=perm(Permission.REPORTS_READ),
)
def list_for_analysis(
    principal: Annotated[Principal, Depends(require(Permission.REPORTS_READ))],
    platform: Annotated[Platform, Depends(get_platform)],
    analysis_id: str,
) -> list[ReportOut]:
    del principal
    with platform.db.session() as session:
        ids = list(
            session.execute(
                select(Report.id)
                .where(Report.analysis_id == analysis_id)
                .order_by(Report.created_at.desc())
            ).scalars()
        )
    return [report_out(platform, rid) for rid in ids]


@router.get(
    "/reports/{report_id}",
    response_model=ReportOut,
    responses=_ERRORS,
    openapi_extra=perm(Permission.REPORTS_READ),
)
def get(
    principal: Annotated[Principal, Depends(require(Permission.REPORTS_READ))],
    platform: Annotated[Platform, Depends(get_platform)],
    report_id: str,
) -> ReportOut:
    """Laudo com verificação de integridade recalculada a cada leitura."""
    del principal
    return report_out(platform, report_id)


@router.post(
    "/reports/{report_id}/sign",
    response_model=ReportOut,
    responses=_ERRORS,
    openapi_extra=perm(Permission.REPORTS_SIGN),
)
def sign(
    principal: Annotated[Principal, Depends(require(Permission.REPORTS_SIGN))],
    platform: Annotated[Platform, Depends(get_platform)],
    report_id: str,
    body: SignIn,
) -> ReportOut:
    """Assina: exige a senha do médico e os dois hashes que ele viu (laudo e resultado).

    A senha errada é auditada (``auth.reauth_failed``) e limitada como o login:
    na quinta falha em 15 minutos as sessões do usuário são revogadas e novas
    tentativas recebem 429.
    """
    from radiologyai.persistence.users import reauthenticate
    from radiologyai.reporting.service import sign_report

    if not reauthenticate(platform.db, principal.user_id, body.password):
        raise HTTPException(status_code=401, detail="reautenticação falhou: senha incorreta")
    sign_report(
        platform,
        report_id,
        user_id=principal.user_id,
        username=principal.username,
        report_sha256=body.report_sha256,
        result_sha256=body.result_sha256,
    )
    return report_out(platform, report_id)


def _download(
    platform: Platform, report_id: str, kind: str, seg_index: int | None = None
) -> Response:
    from radiologyai.reporting.service import render_document

    data, media, name = render_document(platform, report_id, kind, seg_index=seg_index)
    return Response(
        content=data,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


@router.get(
    "/reports/{report_id}/pdf",
    response_class=Response,
    responses={200: {"content": {"application/pdf": {}}}, **_ERRORS},
    openapi_extra=perm(Permission.REPORTS_READ),
)
def pdf(
    principal: Annotated[Principal, Depends(require(Permission.REPORTS_READ))],
    platform: Annotated[Platform, Depends(get_platform)],
    report_id: str,
) -> Response:
    """PDF em português com cabeçalho de uso exclusivo em pesquisa e bloco de assinatura."""
    del principal
    return _download(platform, report_id, "pdf")


@router.get(
    "/reports/{report_id}/sr",
    response_class=Response,
    responses={200: {"content": {"application/dicom": {}}}, **_ERRORS},
    openapi_extra=perm(Permission.REPORTS_READ),
)
def sr(
    principal: Annotated[Principal, Depends(require(Permission.REPORTS_READ))],
    platform: Annotated[Platform, Depends(get_platform)],
    report_id: str,
) -> Response:
    """DICOM SR TID 1500 (Comprehensive 3D SR). 409 se o estudo não é DICOM."""
    del principal
    return _download(platform, report_id, "sr")


@router.get(
    "/reports/{report_id}/seg/{seg_index}",
    response_class=Response,
    responses={200: {"content": {"application/dicom": {}}}, **_ERRORS},
    openapi_extra=perm(Permission.REPORTS_READ),
)
def seg(
    principal: Annotated[Principal, Depends(require(Permission.REPORTS_READ))],
    platform: Annotated[Platform, Depends(get_platform)],
    report_id: str,
    seg_index: int,
) -> Response:
    """DICOM SEG (LABELMAP) da segmentação ``seg_index`` do resultado."""
    del principal
    return _download(platform, report_id, "seg", seg_index)


@router.get(
    "/reports/{report_id}/fhir",
    response_class=Response,
    responses={200: {"content": {"application/fhir+json": {}}}, **_ERRORS},
    openapi_extra=perm(Permission.REPORTS_READ),
)
def fhir(
    principal: Annotated[Principal, Depends(require(Permission.REPORTS_READ))],
    platform: Annotated[Platform, Depends(get_platform)],
    report_id: str,
) -> Response:
    """FHIR R4 ``DiagnosticReport`` com ``status: preliminary``."""
    del principal
    return _download(platform, report_id, "fhir")


@router.post(
    "/reports/{report_id}/export",
    response_model=ExportOut,
    responses=_ERRORS,
    openapi_extra=perm(Permission.DICOMWEB_EXPORT),
)
def export(
    principal: Annotated[Principal, Depends(require(Permission.DICOMWEB_EXPORT))],
    platform: Annotated[Platform, Depends(get_platform)],
    report_id: str,
) -> ExportOut:
    """Envia o SR (e os SEGs do laudo) ao PACS por STOW-RS. Só laudos ASSINADOS saem.

    Só vão os SEGs listados em ``content.segmentations`` (ligados a achado A-C
    não rejeitado). ``stored_sop_instance_uids`` lista apenas o que o servidor
    CONFIRMOU; SEG que não pôde ser gerado aparece em ``skipped``. Se o servidor
    recusar ou não confirmar algum objeto, a exportação é auditada como parcial
    e a resposta é 502 — nunca um sucesso.
    """
    import io

    import pydicom

    from radiologyai.interop.dicomweb import DicomwebError, DicomwebGateway, store_outcome
    from radiologyai.persistence.audit import append_event
    from radiologyai.reporting.dicom_sr import ReportFormatError
    from radiologyai.reporting.service import render_document

    with platform.db.session() as session:
        report = session.get(Report, report_id)
        if report is None:
            raise HTTPException(status_code=404, detail="laudo inexistente")
        if report.signed_at is None:
            raise HTTPException(status_code=409, detail="só laudos assinados são exportados")
        content = ReportContent.model_validate_json(report.content_json)
    gateway = DicomwebGateway.from_settings(platform.settings)
    documents = [render_document(platform, report_id, "sr")[0]]
    skipped: list[ExportSkipOut] = []
    for item in content.segmentations:
        try:
            documents.append(render_document(platform, report_id, "seg", seg_index=item.idx)[0])
        except ReportFormatError as exc:
            skipped.append(ExportSkipOut(seg_index=item.idx, reason=str(exc)))
    datasets = [pydicom.dcmread(io.BytesIO(d)) for d in documents]
    sent = [str(ds.SOPInstanceUID) for ds in datasets]
    response = gateway.store(datasets)
    outcome = store_outcome(response, sent)
    payload = {
        "destination": gateway.url,
        "sop_instance_uids": list(outcome.stored),
        "failed": [{"sop_instance_uid": u, "reason": r} for u, r in outcome.failed],
        "skipped_segmentations": [s.model_dump() for s in skipped],
    }
    append_event(
        platform.db,
        "report.export_partial" if outcome.failed else "report.exported",
        actor_id=principal.user_id,
        actor_username=principal.username,
        subject_type="report",
        subject_id=report_id,
        payload=payload,
    )
    if outcome.failed:
        detail = "; ".join(f"{u}: {r}" for u, r in outcome.failed)
        raise DicomwebError(
            f"exportação parcial: {len(outcome.failed)} objeto(s) não armazenado(s) em "
            f"{gateway.url} ({detail}); confirmados: {len(outcome.stored)}"
        )
    return ExportOut(
        stored_sop_instance_uids=outcome.stored,
        destination=gateway.url,
        skipped=tuple(skipped),
    )
