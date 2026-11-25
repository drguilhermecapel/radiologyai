"""Ciclo de vida do laudo: instantâneo, integridade, assinatura e documentos.

* ``create_report``: monta o conteúdo a partir do ``result_json`` IMUTÁVEL da
  análise (conferido contra ``result_sha256``) e da última revisão de cada
  achado; grava ``content_json`` e ``report_sha256``. Laudo novo = linha nova.
* ``sign_report``: exige que o signatário reenvie os DOIS hashes que viu
  (``report_sha256`` e ``result_sha256``) e a própria senha; todos os achados
  do laudo precisam estar revisados. A assinatura é um hash sobre esses dois
  valores + signatário + horário; qualquer divergência posterior a invalida.
* Documentos (PDF, SR, SEG, FHIR) são renderizações do conteúdo hasheado e da
  assinatura; são guardados como artefatos endereçados por conteúdo.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from radiologyai.errors import RadiologyAIError
from radiologyai.reporting.content import (
    ReportContent,
    ReviewInfo,
    build_report_content,
    canonical_bytes,
    sha256_hex,
    signature_digest,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from radiologyai.analysis.types import AnalysisResult
    from radiologyai.persistence.models import Analysis, Report
    from radiologyai.reporting.pdf import SignOff
    from radiologyai.worker.runtime import Platform

__all__ = [
    "Integrity",
    "ReportStateError",
    "create_report",
    "integrity",
    "latest_reviews",
    "load_result",
    "render_document",
    "sign_off",
    "sign_report",
]


class ReportStateError(RadiologyAIError):
    """Operação incompatível com o estado (análise não concluída, hash divergente...)."""


class ResultIntegrityError(ReportStateError):
    """``result_json`` não confere com ``result_sha256`` — adulteração ou corrupção."""


def load_result(analysis: Analysis) -> AnalysisResult:
    """Resultado imutável, conferido contra o hash gravado.

    Raises:
        ReportStateError: análise sem resultado.
        ResultIntegrityError: bytes não conferem com ``result_sha256``.
    """
    from radiologyai.analysis.types import AnalysisResult

    if analysis.result_json is None or analysis.result_sha256 is None:
        raise ReportStateError(f"análise {analysis.id} não tem resultado (estado {analysis.state})")
    digest = hashlib.sha256(analysis.result_json.encode("utf-8")).hexdigest()
    if digest != analysis.result_sha256:
        raise ResultIntegrityError(
            f"resultado da análise {analysis.id} não confere com result_sha256 gravado"
        )
    return AnalysisResult.model_validate_json(analysis.result_json)


def latest_reviews(session: Session, analysis_id: str) -> dict[int, ReviewInfo]:
    """Última revisão de cada achado (revisões são somente inserção)."""
    from sqlalchemy import select

    from radiologyai.persistence.models import Review, User

    rows = session.execute(
        select(Review, User)
        .join(User, User.id == Review.reviewer_id)
        .where(Review.analysis_id == analysis_id)
        .order_by(Review.created_at, Review.id)
    ).all()
    out: dict[int, ReviewInfo] = {}
    for review, user in rows:
        out[review.finding_idx] = ReviewInfo(
            action=review.action,  # type: ignore[arg-type]
            amended_decision=review.amended_decision,
            amended_text=review.amended_text,
            comment=review.comment,
            reviewer_username=user.username,
            reviewer_name=user.full_name,
            reviewed_at=review.created_at.isoformat(),
        )
    return out


def create_report(platform: Platform, analysis_id: str, *, user_id: str, username: str) -> str:
    """Cria o instantâneo do laudo; devolve o id.

    Raises:
        ReportStateError: análise inexistente ou não concluída.
    """
    from radiologyai.persistence.audit import append_event
    from radiologyai.persistence.models import Analysis, Report, Study

    catalog = platform.engine.catalog
    with platform.db.session() as session:
        analysis = session.get(Analysis, analysis_id)
        if analysis is None:
            raise ReportStateError(f"análise {analysis_id} inexistente")
        if analysis.state != "completed":
            raise ReportStateError(f"análise {analysis_id} está {analysis.state!r}")
        result = load_result(analysis)
        study = session.get(Study, analysis.study_id)
        assert study is not None
        content = build_report_content(
            result,
            result_sha256=str(analysis.result_sha256),
            analysis_id=analysis.id,
            study_id=study.id,
            study_uid=study.study_uid,
            patient_pseudo_id=study.patient_pseudo_id,
            reviews=latest_reviews(session, analysis_id),
            catalog=catalog,
            study_warnings=tuple(study.warnings or ()),
        )
        content_bytes = canonical_bytes(content.model_dump(mode="json"))
        report = Report(
            analysis_id=analysis_id,
            created_by=user_id,
            content_json=content_bytes.decode("utf-8"),
            report_sha256=sha256_hex(content_bytes),
            result_sha256=str(analysis.result_sha256),
        )
        session.add(report)
    append_event(
        platform.db,
        "report.created",
        actor_id=user_id,
        actor_username=username,
        subject_type="report",
        subject_id=report.id,
        payload={
            "analysis_id": analysis_id,
            "report_sha256": report.report_sha256,
            "result_sha256": report.result_sha256,
            "n_findings": len(content.findings),
            "unreviewed": content.unreviewed_findings,
        },
    )
    return report.id


@dataclass(frozen=True, slots=True)
class Integrity:
    content_matches_hash: bool
    result_matches_hash: bool
    signature_valid: bool | None


def integrity(report: Report, analysis: Analysis) -> Integrity:
    """Recalcula os três hashes: conteúdo, resultado e assinatura."""
    content_ok = sha256_hex(report.content_json.encode("utf-8")) == report.report_sha256
    result_ok = (
        analysis.result_json is not None
        and hashlib.sha256(analysis.result_json.encode("utf-8")).hexdigest() == report.result_sha256
        and analysis.result_sha256 == report.result_sha256
    )
    signature: bool | None = None
    if report.signed_at is not None:
        expected = signature_digest(
            report_id=report.id,
            report_sha256=report.report_sha256,
            result_sha256=report.result_sha256,
            signer_id=str(report.signed_by),
            signed_at=report.signed_at,
        )
        signature = content_ok and result_ok and expected == report.signature_sha256
    return Integrity(content_ok, result_ok, signature)


def sign_report(
    platform: Platform,
    report_id: str,
    *,
    user_id: str,
    username: str,
    report_sha256: str,
    result_sha256: str,
) -> None:
    """Assina o laudo (a senha já foi conferida por quem chama).

    Raises:
        ReportStateError: laudo inexistente, já assinado, hashes divergentes do
            que está gravado, integridade reprovada ou achados não revisados.
    """
    from radiologyai.persistence.audit import append_event
    from radiologyai.persistence.models import Analysis, Report, utcnow

    with platform.db.session() as session:
        report = session.get(Report, report_id)
        if report is None:
            raise ReportStateError(f"laudo {report_id} inexistente")
        if report.signed_at is not None:
            raise ReportStateError("laudo já assinado; crie um novo laudo para alterações")
        if report_sha256 != report.report_sha256 or result_sha256 != report.result_sha256:
            raise ReportStateError(
                "os hashes enviados não correspondem ao laudo gravado: o conteúdo que você viu "
                "não é o que seria assinado"
            )
        analysis = session.get(Analysis, report.analysis_id)
        assert analysis is not None
        check = integrity(report, analysis)
        if not (check.content_matches_hash and check.result_matches_hash):
            raise ReportStateError("integridade reprovada: conteúdo ou resultado alterado")
        content = ReportContent.model_validate_json(report.content_json)
        if content.unreviewed_findings:
            raise ReportStateError(
                f"{content.unreviewed_findings} achado(s) sem revisão; "
                "revise todos antes de assinar"
            )
        signed_at = utcnow()
        report.signed_by = user_id
        report.signed_at = signed_at
        report.signature_sha256 = signature_digest(
            report_id=report.id,
            report_sha256=report.report_sha256,
            result_sha256=report.result_sha256,
            signer_id=user_id,
            signed_at=signed_at,
        )
        sig = report.signature_sha256
    append_event(
        platform.db,
        "report.signed",
        actor_id=user_id,
        actor_username=username,
        subject_type="report",
        subject_id=report_id,
        payload={
            "report_sha256": report_sha256,
            "result_sha256": result_sha256,
            "signature_sha256": sig,
        },
    )


def sign_off(session: Session, report: Report) -> SignOff:
    from radiologyai.persistence.models import User
    from radiologyai.reporting.pdf import SignOff

    if report.signed_at is None or report.signed_by is None:
        return SignOff(report_id=report.id, report_sha256=report.report_sha256)
    user = session.get(User, report.signed_by)
    return SignOff(
        report_id=report.id,
        report_sha256=report.report_sha256,
        signer_name=user.full_name if user else None,
        signer_username=user.username if user else None,
        professional_id=user.professional_id if user else None,
        signed_at=report.signed_at.isoformat(),
        signature_sha256=report.signature_sha256,
    )


def _source_datasets(
    session: Session, platform: Platform, analysis: Analysis
) -> dict[int, list[Any]]:
    """Cabeçalhos DICOM (des-identificados) de cada imagem do exame, na ordem do volume."""
    import io

    import pydicom

    from radiologyai.persistence.models import Series

    out: dict[int, list[Any]] = {}
    for item in analysis.inputs:
        series = session.get(Series, item["series_id"])
        if series is None or series.source_format not in ("dicom", "dicom_series"):
            continue
        datasets = [
            pydicom.dcmread(io.BytesIO(platform.blobs.read(inst.artifact_sha256)))
            for inst in series.instances
        ]
        order = series.meta.get("image_meta", {}).get("sop_uids") or []
        if series.source_format == "dicom_series" and order:
            by_uid = {str(ds.SOPInstanceUID): ds for ds in datasets}
            datasets = [by_uid[uid] for uid in order if uid in by_uid]
        out[int(item["image_idx"])] = datasets
    return out


def render_document(
    platform: Platform, report_id: str, kind: str, *, seg_index: int | None = None
) -> tuple[bytes, str, str]:
    """Renderiza ``pdf`` | ``sr`` | ``seg`` | ``fhir``; guarda como artefato.

    Devolve ``(bytes, media_type, nome_de_arquivo)``.

    Raises:
        ReportStateError: laudo inexistente ou integridade reprovada.
        ReportFormatError: formato impossível para o estudo (ex.: SR sem DICOM).
    """
    from radiologyai.persistence.models import Analysis, Report
    from radiologyai.reporting.dicom_sr import ReportFormatError, build_seg, build_sr
    from radiologyai.reporting.fhir import build_diagnostic_report
    from radiologyai.reporting.pdf import render_pdf

    with platform.db.session() as session:
        report = session.get(Report, report_id)
        if report is None:
            raise ReportStateError(f"laudo {report_id} inexistente")
        analysis = session.get(Analysis, report.analysis_id)
        assert analysis is not None
        check = integrity(report, analysis)
        if not (check.content_matches_hash and check.result_matches_hash):
            raise ReportStateError("integridade reprovada: o laudo não pode ser emitido")
        if check.signature_valid is False:
            raise ReportStateError("assinatura inválida: o laudo não pode ser emitido")
        content = ReportContent.model_validate_json(report.content_json)
        sign = sign_off(session, report)
        if kind == "pdf":
            data, media, name = (
                render_pdf(content, sign),
                "application/pdf",
                f"laudo-{report.id}.pdf",
            )
            store_media = "application/pdf"
        elif kind == "fhir":
            data = build_diagnostic_report(content, sign, issued=report.created_at.isoformat())
            media, name, store_media = (
                "application/fhir+json",
                f"laudo-{report.id}.fhir.json",
                "application/json",
            )
        elif kind in ("sr", "seg"):
            sources = _source_datasets(session, platform, analysis)
            if kind == "sr":
                evidence = [ds for idx in sorted(sources) for ds in sources[idx]]
                image_sop = {
                    idx: (str(ds_list[0].SOPClassUID), str(ds_list[0].SOPInstanceUID))
                    for idx, ds_list in sources.items()
                    if len(ds_list) == 1
                }
                data = build_sr(content, sign, evidence, image_sop)
                name = f"laudo-{report.id}.sr.dcm"
            else:
                result = load_result(analysis)
                if seg_index is None or not 0 <= seg_index < len(result.segmentations):
                    raise ReportFormatError(f"segmentação {seg_index} inexistente nesta análise")
                seg = result.segmentations[seg_index]
                if seg.image_idx not in sources:
                    raise ReportFormatError("DICOM SEG exige imagem de origem DICOM")
                prov = result.provenance[seg.provenance_idx]
                data = build_seg(
                    report_id=report.id,
                    seg_name=seg.name,
                    model_id=prov.model_id,
                    model_version=prov.model_version,
                    labelmap=platform.blobs.store.read_labelmap(seg.mask),
                    labels={i: (seg.label_keys[i], seg.labels[i].meaning) for i in seg.labels},
                    source_images=sources[seg.image_idx],
                )
                name = f"laudo-{report.id}.seg{seg_index}.dcm"
            media, store_media = "application/dicom", "application/dicom"
        else:
            raise ReportFormatError(f"formato desconhecido {kind!r}")
        platform.blobs.put(session, data, store_media, kind=f"report_{kind}")
    return data, media, name
