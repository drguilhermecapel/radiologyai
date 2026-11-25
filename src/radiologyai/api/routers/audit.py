"""Trilha de auditoria: leitura paginada e verificação da cadeia de hashes (administrador)."""

from __future__ import annotations

import json
from typing import Annotated

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
    del principal
    query = select(AuditEvent).where(AuditEvent.seq > after_seq)
    if event:
        query = query.where(AuditEvent.event == event)
    if subject_id:
        query = query.where(AuditEvent.subject_id == subject_id)
    with platform.db.session() as session:
        rows = list(session.execute(query.order_by(AuditEvent.seq).limit(limit)).scalars())
        items = tuple(
            AuditEventOut(
                seq=r.seq,
                timestamp_utc=r.timestamp_utc,
                event=r.event,
                actor_id=r.actor_id,
                actor_username=r.actor_username,
                subject_type=r.subject_type,
                subject_id=r.subject_id,
                payload=json.loads(r.payload_json),
                prev_hash=r.prev_hash,
                record_hash=r.record_hash,
            )
            for r in rows
        )
    return AuditListOut(items=items, next_after_seq=items[-1].seq if len(items) == limit else None)


@router.get("/verify", response_model=AuditVerifyOut, responses=_ERRORS, openapi_extra=perm(_P))
def verify(
    principal: Annotated[Principal, Depends(require(_P))],
    platform: Annotated[Platform, Depends(get_platform)],
) -> AuditVerifyOut:
    """Recalcula a cadeia inteira; aponta o primeiro evento adulterado ou elo quebrado."""
    from radiologyai.persistence.audit import verify_chain

    del principal
    with platform.db.session() as session:
        status = verify_chain(session)
    return AuditVerifyOut(
        ok=status.ok,
        events=status.events,
        last_hash=status.last_hash,
        broken_at_seq=status.broken_at_seq,
        detail=status.detail,
    )
