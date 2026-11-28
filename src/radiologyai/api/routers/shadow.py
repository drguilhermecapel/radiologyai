"""Modo silencioso (REQ-195): exportação para a comparação posterior com o laudo do radiologista.

Só a equipe do piloto (``shadow:view``) exporta, e só com o modo silencioso LIGADO
(desligado => 409, ``error: "ShadowModeOffError"``): fora do piloto não há exportação.
O recorte é por ``requested_at`` da análise, ``since <= requested_at < until`` (ISO 8601
com fuso), e fica registrado no evento de auditoria.

O CSV é **pseudonimizado (dado pessoal pela LGPD); sem identificador direto, texto livre
ou pixel**. O estudo aparece pelo id opaco da plataforma (aleatório) e pelo Study
Instance UID já pseudonimizado na ingestão (HMAC com ``MEDAI_DEIDENT_KEY``) — quem
guarda a chave no hospital re-deriva o mesmo pseudônimo a partir do UID original para
ligar a linha ao laudo; por isso é dado pseudonimizado (LGPD, art. 13, § 4º), não
anonimizado, e os horários exatos são quasi-identificadores. Sem nome, identificador do
paciente, data de nascimento, acesso (accession), descrição livre ou pixel. Cada
exportação é auditada na cadeia (``shadow.exported``, com o sha256 do arquivo e o recorte).

Suspensões de decisão (REQ-193) valem na exportação: achado de trio suspenso — mesmo de
análise gravada antes da suspensão — sai com ``decision = score_only`` e o motivo em
``decision_held`` (vazio quando não há suspensão). O escore bruto é mantido.
"""

from __future__ import annotations

import csv
import hashlib
import io
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from radiologyai.api.deps import Principal, get_platform, perm, require
from radiologyai.api.routers.studies import _ERRORS
from radiologyai.persistence.security import Permission
from radiologyai.worker.runtime import Platform

if TYPE_CHECKING:
    from radiologyai.analysis.evidence_policy import HoldLookup

router = APIRouter(prefix="/shadow", tags=["shadow"])

#: Colunas do CSV, nesta ordem (contrato com a análise do piloto; nenhum identificador
#: direto nem texto livre — o conjunto é pseudonimizado, dado pessoal pela LGPD).
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
    "decision_held",
)


def _iso(value: Any) -> str:
    return value.isoformat() if value is not None else ""


def _rows(
    platform: Platform,
    held: HoldLookup,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Uma linha por achado (todos os níveis); análise sem achado vira uma linha sem achado.

    Só análises com ``since <= requested_at < until`` (limites ausentes = abertos).
    """
    from sqlalchemy import select

    from radiologyai.analysis.evidence_policy import apply_hold, hold_note
    from radiologyai.persistence.models import Analysis, Study
    from radiologyai.reporting.service import load_result

    rows: list[dict[str, Any]] = []
    n_analyses = 0
    query = (
        select(Analysis, Study.study_uid)
        .join(Study, Study.id == Analysis.study_id)
        .where(Analysis.state == "completed", Analysis.result_json.is_not(None))
    )
    if since is not None:
        query = query.where(Analysis.requested_at >= since)
    if until is not None:
        query = query.where(Analysis.requested_at < until)
    with platform.db.session() as session:
        found = session.execute(query.order_by(Analysis.finished_at, Analysis.id)).all()
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
            for idx, stored in enumerate(result.findings):
                prov = result.provenance[stored.provenance_idx]
                hold = held(stored.condition_id, prov.model_id, stored.label)
                f = stored if hold is None else apply_hold(stored, hold)
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
                        "decision_held": (
                            ""
                            if hold is None or stored.decision == "abstained"
                            else hold_note(hold, stored.evidence_level)
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
    since: Annotated[
        datetime | None,
        Query(description="Início (inclusivo) do recorte em requested_at; ISO 8601 com fuso"),
    ] = None,
    until: Annotated[
        datetime | None,
        Query(description="Fim (exclusivo) do recorte em requested_at; ISO 8601 com fuso"),
    ] = None,
) -> Response:
    """CSV pseudonimizado (dado pessoal pela LGPD) do piloto, um achado por linha (REQ-195).

    Sem identificador direto, texto livre ou pixel. Exige o modo silencioso ligado
    (desligado => 409 ``ShadowModeOffError``). ``since``/``until`` recortam por
    ``requested_at`` (``since`` inclusivo, ``until`` exclusivo; com fuso horário).
    Colunas: ver ``COLUMNS`` (ids opacos/pseudonimizados, ``result_sha256``, modelo e
    versão, ``condition_id``, nível de evidência, escore bruto, decisão — ``score_only``
    com o motivo em ``decision_held`` quando a decisão do trio está suspensa (REQ-193) — e
    tempos). A exportação, com o recorte, é registrada na cadeia de auditoria ANTES de ser
    entregue.
    """
    from radiologyai.api.shadow import ShadowModeOffError, shadow_on
    from radiologyai.persistence.audit import append_event
    from radiologyai.reporting.service import held_fn

    if not shadow_on(platform):
        raise ShadowModeOffError(
            "exportação do piloto só com o modo silencioso ligado (MEDAI_SHADOW_MODE=true)"
        )
    bounds: dict[str, datetime | None] = {"since": since, "until": until}
    for name, value in bounds.items():
        if value is not None and value.tzinfo is None:
            raise HTTPException(
                status_code=422, detail=f"{name} precisa de fuso horário (ex.: ...Z ou -03:00)"
            )
    since = since.astimezone(UTC) if since is not None else None
    until = until.astimezone(UTC) if until is not None else None
    if since is not None and until is not None and since >= until:
        raise HTTPException(status_code=422, detail="since deve ser anterior a until")
    rows, n_analyses = _rows(platform, held_fn(platform), since=since, until=until)
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
            "shadow_mode": True,
            "filter": {
                "field": "requested_at",
                "since": since.isoformat() if since is not None else None,
                "until": until.isoformat() if until is not None else None,
            },
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
