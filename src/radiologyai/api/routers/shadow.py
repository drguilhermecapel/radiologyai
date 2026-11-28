"""Modo silencioso (REQ-195): exportação para a comparação posterior com o laudo do radiologista.

Só a equipe do piloto (``shadow:view``) exporta. O CSV não tem dado pessoal: o
estudo aparece pelo id opaco da plataforma (aleatório) e pelo Study Instance UID
JÁ pseudonimizado na ingestão (HMAC com ``MEDAI_DEIDENT_KEY``) — quem guarda a
chave no hospital re-deriva o mesmo pseudônimo a partir do UID original para
ligar a linha ao laudo. Sem nome, identificador do paciente, data de nascimento,
acesso (accession), descrição livre ou pixel. Cada exportação é auditada na
cadeia (``shadow.exported``, com o sha256 do arquivo).
"""

from __future__ import annotations

import csv
import hashlib
import io
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Response

from radiologyai.api.deps import Principal, get_platform, perm, require
from radiologyai.api.routers.studies import _ERRORS
from radiologyai.persistence.security import Permission
from radiologyai.worker.runtime import Platform

router = APIRouter(prefix="/shadow", tags=["shadow"])

#: Colunas do CSV, nesta ordem (contrato com a análise do piloto; nenhuma é PHI).
COLUMNS: tuple[str, ...] = (
    "study_id",
    "study_uid_pseudo",
    "analysis_id",
    "exam_type",
    "requested_at",
    "finished_at",
    "result_sha256",
    "result_status",
    "code_version",
    "finding_idx",
    "model_id",
    "model_version",
    "manifest_sha256",
    "condition_id",
    "label",
    "kind",
    "evidence_level",
    "visibility",
    "raw_score",
    "score_kind",
    "probability",
    "decision",
    "category",
    "abstention_reason",
)


def _iso(value: Any) -> str:
    return value.isoformat() if value is not None else ""


def _rows(platform: Platform) -> tuple[list[dict[str, Any]], int]:
    """Uma linha por achado (todos os níveis); análise sem achado vira uma linha sem achado."""
    from sqlalchemy import select

    from radiologyai.persistence.models import Analysis, Study
    from radiologyai.reporting.service import load_result

    rows: list[dict[str, Any]] = []
    n_analyses = 0
    with platform.db.session() as session:
        found = session.execute(
            select(Analysis, Study.study_uid)
            .join(Study, Study.id == Analysis.study_id)
            .where(Analysis.state == "completed", Analysis.result_json.is_not(None))
            .order_by(Analysis.finished_at, Analysis.id)
        ).all()
        for analysis, study_uid in found:
            result = load_result(analysis)  # confere o sha256 do resultado gravado
            n_analyses += 1
            base = {
                "study_id": analysis.study_id,
                "study_uid_pseudo": study_uid or "",
                "analysis_id": analysis.id,
                "exam_type": analysis.exam_type,
                "requested_at": _iso(analysis.requested_at),
                "finished_at": _iso(analysis.finished_at),
                "result_sha256": analysis.result_sha256,
                "result_status": result.status,
                "code_version": analysis.code_version,
            }
            if not result.findings:
                rows.append(base)
                continue
            for idx, f in enumerate(result.findings):
                prov = result.provenance[f.provenance_idx]
                rows.append(
                    {
                        **base,
                        "finding_idx": idx,
                        "model_id": prov.model_id,
                        "model_version": prov.model_version,
                        "manifest_sha256": prov.manifest_sha256,
                        "condition_id": f.condition_id,
                        "label": f.label,
                        "kind": f.kind,
                        "evidence_level": f.evidence_level.value,
                        "visibility": f.visibility,
                        "raw_score": "" if f.raw_score is None else repr(f.raw_score),
                        "score_kind": str(f.score_kind),
                        "probability": "" if f.probability is None else repr(f.probability),
                        "decision": f.decision,
                        "category": f.category or "",
                        "abstention_reason": (
                            f.abstention_reason.value if f.abstention_reason else ""
                        ),
                    }
                )
    return rows, n_analyses


@router.get(
    "/export",
    response_class=Response,
    responses={200: {"content": {"text/csv": {}}}, **_ERRORS},
    openapi_extra=perm(Permission.SHADOW_VIEW),
)
def export_csv(
    principal: Annotated[Principal, Depends(require(Permission.SHADOW_VIEW))],
    platform: Annotated[Platform, Depends(get_platform)],
) -> Response:
    """CSV sem PHI de toda análise concluída, um achado por linha (REQ-195), auditado.

    Funciona com o modo silencioso ligado ou não. Colunas: ver ``COLUMNS``
    (ids opacos/pseudonimizados, ``result_sha256``, modelo e versão,
    ``condition_id``, nível de evidência, escore bruto, decisão e tempos). A
    exportação é registrada na cadeia de auditoria ANTES de ser entregue.
    """
    from radiologyai.persistence.audit import append_event

    rows, n_analyses = _rows(platform)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n", restval="")
    writer.writeheader()
    writer.writerows(rows)
    data = buf.getvalue().encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    append_event(
        platform.db,
        "shadow.exported",
        actor_id=principal.user_id,
        actor_username=principal.username,
        subject_type="shadow_export",
        subject_id=digest,
        payload={
            "csv_sha256": digest,
            "rows": len(rows),
            "analyses": n_analyses,
            "columns": list(COLUMNS),
            "shadow_mode": bool(platform.settings.shadow_mode),
        },
    )
    return Response(
        content=data,
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": 'attachment; filename="shadow-export.csv"',
            "X-Content-SHA256": digest,
        },
    )
