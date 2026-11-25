"""Trilha de auditoria no banco: append-only, encadeada por hash.

Porta para o banco a semântica de :mod:`radiologyai.audit.trail` (REQ-070,
REQ-071): cada evento carrega o hash do anterior; alterar, remover ou
reordenar um evento passado invalida a cadeia a partir dele, e
:func:`verify_chain` aponta a posição exata.

* O hash cobre ``seq``, horário, evento, ator, sujeito, carga e ``prev_hash``
  em JSON canônico (chaves ordenadas, sem espaços) — o mesmo esquema da trilha
  em arquivo, portanto reprodutível em outra máquina.
* ``prev_hash`` é ``UNIQUE``: dois processos que tentem estender o mesmo elo
  não produzem bifurcação; um deles falha e tenta de novo.
* A carga NUNCA contém identificador direto de paciente nem pixel; só ids
  opacos, hashes e decisões. :func:`append_event` recusa chaves suspeitas.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from radiologyai.audit.trail import GENESIS_HASH, AuditError

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from radiologyai.persistence.db import Database
    from radiologyai.persistence.models import AuditEvent

__all__ = ["ChainStatus", "append_event", "canonical_hash", "verify_chain"]

_LOCK = threading.Lock()
_FORBIDDEN_PAYLOAD_KEYS = frozenset(
    {"patientname", "patient_name", "patientid", "patient_id", "patientbirthdate", "pixeldata"}
)


def canonical_hash(fields: dict[str, Any]) -> str:
    """sha256 do JSON canônico (chaves ordenadas, separadores compactos, UTF-8)."""
    canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _fields(event: AuditEvent) -> dict[str, Any]:
    return {
        "seq": event.seq,
        "timestamp_utc": event.timestamp_utc,
        "event": event.event,
        "actor_id": event.actor_id,
        "actor_username": event.actor_username,
        "subject_type": event.subject_type,
        "subject_id": event.subject_id,
        "payload": json.loads(event.payload_json),
        "prev_hash": event.prev_hash,
    }


def _check_payload(payload: dict[str, Any]) -> None:
    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, sub in value.items():
                if str(key).replace("-", "_").lower() in _FORBIDDEN_PAYLOAD_KEYS:
                    raise AuditError(f"carga de auditoria não pode conter {key!r} (dado pessoal)")
                walk(sub)
        elif isinstance(value, (list, tuple)):
            for sub in value:
                walk(sub)

    walk(payload)


def append_event(
    db: Database,
    event: str,
    *,
    actor_id: str | None = None,
    actor_username: str | None = None,
    subject_type: str | None = None,
    subject_id: str | None = None,
    payload: dict[str, Any] | None = None,
    retries: int = 5,
) -> str:
    """Acrescenta um evento numa transação própria; devolve o ``record_hash``.

    Raises:
        AuditError: carga com dado pessoal ou cadeia disputada além de ``retries``.
    """
    from sqlalchemy import select
    from sqlalchemy.exc import IntegrityError

    from radiologyai.persistence.models import AuditEvent

    body = payload or {}
    _check_payload(body)
    payload_json = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    for _ in range(max(1, retries)):
        with _LOCK:
            try:
                with db.session() as session:
                    last = session.execute(
                        select(AuditEvent).order_by(AuditEvent.seq.desc()).limit(1)
                    ).scalar_one_or_none()
                    row = AuditEvent(
                        seq=1 if last is None else last.seq + 1,
                        timestamp_utc=datetime.now(UTC).isoformat(),
                        event=event,
                        actor_id=actor_id,
                        actor_username=actor_username,
                        subject_type=subject_type,
                        subject_id=subject_id,
                        payload_json=payload_json,
                        prev_hash=GENESIS_HASH if last is None else last.record_hash,
                        record_hash="",
                    )
                    row.record_hash = canonical_hash(_fields(row))
                    session.add(row)
                return row.record_hash
            except IntegrityError:
                continue
    raise AuditError("não foi possível encadear o evento de auditoria (disputa persistente)")


@dataclass(frozen=True, slots=True)
class ChainStatus:
    """Resultado da verificação da cadeia."""

    ok: bool
    events: int
    last_hash: str
    broken_at_seq: int | None = None
    detail: str | None = None


def verify_chain(session: Session) -> ChainStatus:
    """Recalcula a cadeia inteira; aponta o primeiro elo quebrado ou adulterado."""
    from sqlalchemy import select

    from radiologyai.persistence.models import AuditEvent

    prev = GENESIS_HASH
    expected_seq = 1
    count = 0
    for row in session.execute(select(AuditEvent).order_by(AuditEvent.seq)).scalars():
        count += 1
        if row.seq != expected_seq:
            return ChainStatus(
                False, count, prev, row.seq, f"sequência pula de {expected_seq - 1} para {row.seq}"
            )
        if row.prev_hash != prev:
            return ChainStatus(
                False,
                count,
                prev,
                row.seq,
                f"elo quebrado: prev_hash não corresponde a {prev[:12]}…",
            )
        try:
            recomputed = canonical_hash(_fields(row))
        except json.JSONDecodeError:
            return ChainStatus(False, count, prev, row.seq, "carga corrompida (JSON inválido)")
        if recomputed != row.record_hash:
            return ChainStatus(
                False,
                count,
                prev,
                row.seq,
                f"evento adulterado: hash gravado {row.record_hash[:12]}…, "
                f"recalculado {recomputed[:12]}…",
            )
        prev = row.record_hash
        expected_seq += 1
    return ChainStatus(True, count, prev)
