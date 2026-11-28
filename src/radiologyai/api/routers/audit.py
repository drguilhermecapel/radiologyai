"""Trilha de auditoria: leitura paginada e verificação da cadeia de hashes (administrador).

Modo silencioso (REQ-195): o administrador tem ``audit:read`` mas não ``shadow:view``.
Campos de carga que derivam da saída de IA (contagens de achados, estado do
resultado, estado por modelo, decisões de revisão...) são servidos como ``None``
a quem não pode ver a saída de IA, e a carga ganha ``shadow_redacted`` com os
nomes dos campos retidos. O ``record_hash`` do evento redigido (e o
``prev_hash`` do evento seguinte, que é o mesmo valor) é servido vazio: é o
sha256 do JSON canônico e, com os demais campos visíveis, permitiria recuperar
por força bruta um valor redigido de domínio pequeno (ex.: ``n_findings``).

A redação é só na leitura: o evento gravado e os hashes no banco não mudam e a
cadeia continua verificável (``/audit/verify`` recalcula a partir do banco, não
da resposta redigida).
"""

from __future__ import annotations

import json
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select

from radiologyai.api.deps import Principal, get_platform, perm, require
from radiologyai.api.routers.studies import _ERRORS
from radiologyai.api.schemas import AuditEventOut, AuditListOut, AuditVerifyOut
from radiologyai.persistence.models import AuditEvent
from radiologyai.persistence.security import Permission
from radiologyai.worker.runtime import Platform

router = APIRouter(prefix="/audit", tags=["audit"])
_P = Permission.AUDIT_READ

#: Campos de carga derivados da saída de IA, por tipo de evento (REQ-195).
#: ``result_sha256`` também sai: a análise redigida não o expõe (``redact_analysis``).
AI_DERIVED_FIELDS: dict[str, frozenset[str]] = {
    "analysis.completed": frozenset({"n_findings", "status", "runs", "result_sha256"}),
    # A mensagem de erro do pipeline pode citar abstenção/saída de modelo.
    "analysis.failed": frozenset({"error"}),
    "report.created": frozenset({"n_findings", "unreviewed", "demoted", "result_sha256"}),
    "report.signed": frozenset({"result_sha256"}),
    "review.created": frozenset(
        {"finding_idx", "condition_id", "action", "amended_decision", "result_sha256"}
    ),
    "legacy.analyze": frozenset({"n_findings", "abstained"}),
    # Uma linha do CSV por achado: o total de linhas é contagem de achados.
    "shadow.exported": frozenset({"rows"}),
}

#: Chave acrescentada à carga servida redigida (nunca gravada).
REDACTED_KEY = "shadow_redacted"

#: Valor servido no lugar de um hash retido (o esquema exige texto; vazio = retido).
WITHHELD_HASH = ""


def redact_payload(event: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Cópia da carga sem os campos derivados da IA (``None``) e com ``shadow_redacted``.

    Eventos sem campo de IA voltam iguais (mesmo objeto); a carga original não é alterada.
    """
    hidden = sorted(k for k in AI_DERIVED_FIELDS.get(event, frozenset()) if k in payload)
    if not hidden:
        return payload
    out = {k: (None if k in hidden else v) for k, v in payload.items()}
    out[REDACTED_KEY] = hidden
    return out


@router.get("", response_model=AuditListOut, responses=_ERRORS, openapi_extra=perm(_P))
def events(
    principal: Annotated[Principal, Depends(require(_P))],
    platform: Annotated[Platform, Depends(get_platform)],
    event: Annotated[str | None, Query(max_length=64)] = None,
    subject_id: Annotated[str | None, Query(max_length=64)] = None,
    after_seq: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> AuditListOut:
    """Eventos em ordem de sequência; reconstrói decisões por ``subject_id``."""
    from radiologyai.api.shadow import ai_visible

    redact = not ai_visible(platform, principal)
    query = select(AuditEvent).where(AuditEvent.seq > after_seq)
    if event:
        query = query.where(AuditEvent.event == event)
    if subject_id:
        query = query.where(AuditEvent.subject_id == subject_id)
    with platform.db.session() as session:
        rows = list(session.execute(query.order_by(AuditEvent.seq).limit(limit)).scalars())
        payloads = {r.seq: json.loads(r.payload_json) for r in rows}
        served = {
            r.seq: redact_payload(r.event, payloads[r.seq]) if redact else payloads[r.seq]
            for r in rows
        }
        # Eventos cujo hash é retido: os redigidos desta página e, para o ``prev_hash``,
        # os predecessores redigidos que ficaram fora dela (filtro/paginação).
        withheld = {seq for seq, p in served.items() if REDACTED_KEY in p}
        if redact:
            outside = {r.seq - 1 for r in rows} - set(served)
            if outside:
                for prev in session.execute(
                    select(AuditEvent).where(AuditEvent.seq.in_(outside))
                ).scalars():
                    if REDACTED_KEY in redact_payload(prev.event, json.loads(prev.payload_json)):
                        withheld.add(prev.seq)
        items = tuple(
            AuditEventOut(
                seq=r.seq,
                timestamp_utc=r.timestamp_utc,
                event=r.event,
                actor_id=r.actor_id,
                actor_username=r.actor_username,
                subject_type=r.subject_type,
                subject_id=r.subject_id,
                payload=served[r.seq],
                prev_hash=WITHHELD_HASH if r.seq - 1 in withheld else r.prev_hash,
                record_hash=WITHHELD_HASH if r.seq in withheld else r.record_hash,
            )
            for r in rows
        )
    return AuditListOut(items=items, next_after_seq=items[-1].seq if len(items) == limit else None)


@router.get("/verify", response_model=AuditVerifyOut, responses=_ERRORS, openapi_extra=perm(_P))
def verify(
    principal: Annotated[Principal, Depends(require(_P))],
    platform: Annotated[Platform, Depends(get_platform)],
) -> AuditVerifyOut:
    """Recalcula a cadeia inteira; aponta o primeiro evento adulterado ou elo quebrado.

    Compara também com o último checkpoint externo (fora do banco):
    eventos recentes apagados reprovam a verificação.
    """
    from radiologyai.api.shadow import ai_visible
    from radiologyai.persistence.audit import verify_chain

    last_hash = None
    with platform.db.session() as session:
        status = verify_chain(session)
        if not ai_visible(platform, principal):
            # Modo silencioso: o último hash é o ``record_hash`` do último evento; se este
            # foi redigido na leitura, o hash também é retido (ver o módulo).
            last = session.execute(
                select(AuditEvent).order_by(AuditEvent.seq.desc()).limit(1)
            ).scalar_one_or_none()
            if last is not None and REDACTED_KEY in redact_payload(
                last.event, json.loads(last.payload_json)
            ):
                last_hash = WITHHELD_HASH
    return AuditVerifyOut(
        ok=status.ok,
        events=status.events,
        last_hash=status.last_hash if last_hash is None else last_hash,
        broken_at_seq=status.broken_at_seq,
        detail=status.detail,
        anchored_seq=status.anchored_seq,
    )
