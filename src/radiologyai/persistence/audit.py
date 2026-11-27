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
* A cabeça da cadeia é ancorada FORA do banco: a cada evento confirmado, uma
  linha ``{seq, record_hash}`` com HMAC da chave do servidor é acrescentada a um
  arquivo de checkpoints. :func:`verify_chain` exige que o banco ainda contenha
  o último checkpoint — apagar os eventos mais recentes deixa de passar.
* Ação e auditoria são ATÔMICAS: com ``session=``, o evento entra na MESMA
  transação da mudança de estado (assinatura, revisão, papel...), com a cabeça
  da cadeia lida sob trava de escrita do banco; ou as duas gravam, ou nenhuma.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import weakref
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from radiologyai.audit.trail import GENESIS_HASH, AuditError

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine
    from sqlalchemy.orm import Session

    from radiologyai.persistence.db import Database
    from radiologyai.persistence.models import AuditEvent

__all__ = [
    "AnchorConfig",
    "ChainStatus",
    "append_event",
    "canonical_hash",
    "configure_anchor",
    "read_anchor",
    "verify_chain",
]

_LOCK = threading.Lock()
_FORBIDDEN_PAYLOAD_KEYS = frozenset(
    {"patientname", "patient_name", "patientid", "patient_id", "patientbirthdate", "pixeldata"}
)


@dataclass(frozen=True, slots=True)
class AnchorConfig:
    """Arquivo de checkpoints da cabeça da cadeia (fora do banco) e a chave do HMAC."""

    path: Path
    key: bytes


_ANCHORS: weakref.WeakKeyDictionary[Any, AnchorConfig] = weakref.WeakKeyDictionary()


def configure_anchor(db: Database, path: Path, key: bytes) -> None:
    """Liga a ancoragem externa para os eventos gravados neste banco."""
    _ANCHORS[db.engine] = AnchorConfig(Path(path), key)


def _anchor_for(bind: Engine | Any) -> AnchorConfig | None:
    engine = getattr(bind, "engine", bind)
    return _ANCHORS.get(engine)


def _anchor_line(cfg: AnchorConfig, seq: int, record_hash: str) -> bytes:
    body = {"seq": seq, "record_hash": record_hash}
    mac = hmac.new(cfg.key, json.dumps(body, sort_keys=True).encode(), hashlib.sha256)
    return (json.dumps({**body, "mac": mac.hexdigest()}, sort_keys=True) + "\n").encode()


def _write_anchor(cfg: AnchorConfig, seq: int, record_hash: str) -> None:
    """Acrescenta um checkpoint (escrita única com O_APPEND: linhas não se misturam)."""
    cfg.path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(cfg.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        os.write(fd, _anchor_line(cfg, seq, record_hash))
    finally:
        os.close(fd)


def read_anchor(cfg: AnchorConfig) -> tuple[int, str] | None:
    """Último checkpoint válido ``(seq, record_hash)``; ``None`` sem arquivo ou vazio.

    Raises:
        AuditError: alguma linha com HMAC inválido (arquivo de checkpoints adulterado).
    """
    if not cfg.path.is_file():
        return None
    best: tuple[int, str] | None = None
    for n, raw in enumerate(cfg.path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
            seq, record_hash, mac = int(item["seq"]), str(item["record_hash"]), str(item["mac"])
        except (ValueError, KeyError, TypeError) as exc:
            raise AuditError(f"checkpoint externo ilegível na linha {n}") from exc
        expected = json.loads(_anchor_line(cfg, seq, record_hash))["mac"]
        if not hmac.compare_digest(expected, mac):
            raise AuditError(f"checkpoint externo com HMAC inválido na linha {n}")
        if best is None or seq > best[0]:
            best = (seq, record_hash)
    return best


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


_ADVISORY_KEY = 0x52414441  # "RADA": trava consultiva da cabeça da cadeia (Postgres)


def _lock_chain(session: Session) -> None:
    """Trava de escrita da cabeça da cadeia até o fim da transação da sessão.

    Postgres: trava consultiva de transação. SQLite: um ``UPDATE`` nulo abre a
    transação de escrita (trava RESERVED), de modo que nenhum outro processo
    grava entre a leitura da cabeça e o ``COMMIT``.
    """
    from sqlalchemy import text

    dialect = session.get_bind().dialect.name
    if dialect == "postgresql":
        session.execute(text("SELECT pg_advisory_xact_lock(:k)"), {"k": _ADVISORY_KEY})
    elif dialect == "sqlite":
        session.execute(text("UPDATE audit_events SET seq = seq WHERE 1 = 0"))


def _append_in(
    session: Session,
    event: str,
    payload_json: str,
    *,
    actor_id: str | None,
    actor_username: str | None,
    subject_type: str | None,
    subject_id: str | None,
) -> str:
    from sqlalchemy import select

    from radiologyai.persistence.models import AuditEvent

    _lock_chain(session)
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
    session.flush()
    anchor = _anchor_for(session.get_bind())
    if anchor is not None:
        from sqlalchemy import event as sa_event

        seq, digest = row.seq, row.record_hash

        def _on_commit(_session: Session) -> None:
            _write_anchor(anchor, seq, digest)

        sa_event.listen(session, "after_commit", _on_commit, once=True)
    return row.record_hash


def append_event(
    db: Database | None,
    event: str,
    *,
    actor_id: str | None = None,
    actor_username: str | None = None,
    subject_type: str | None = None,
    subject_id: str | None = None,
    payload: dict[str, Any] | None = None,
    retries: int = 5,
    session: Session | None = None,
) -> str:
    """Acrescenta um evento; devolve o ``record_hash``.

    Com ``session``, o evento entra na transação dessa sessão (atômico com a
    ação que ela grava; disputa da cadeia aborta a transação inteira). Sem
    ``session``, abre uma transação própria e tenta de novo em disputa.

    Raises:
        AuditError: carga com dado pessoal ou cadeia disputada além de ``retries``.
        sqlalchemy.exc.IntegrityError: com ``session``, elo disputado (a
            transação do chamador é desfeita).
    """
    from sqlalchemy.exc import IntegrityError

    body = payload or {}
    _check_payload(body)
    payload_json = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    fields: dict[str, Any] = {
        "actor_id": actor_id,
        "actor_username": actor_username,
        "subject_type": subject_type,
        "subject_id": subject_id,
    }
    if session is not None:
        return _append_in(session, event, payload_json, **fields)
    if db is None:
        raise AuditError("append_event exige db ou session")
    for _ in range(max(1, retries)):
        with _LOCK:
            try:
                with db.session() as own:
                    return _append_in(own, event, payload_json, **fields)
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
    anchored_seq: int | None = None


def verify_chain(session: Session, anchor: AnchorConfig | None = None) -> ChainStatus:
    """Recalcula a cadeia inteira; aponta o primeiro elo quebrado ou adulterado.

    Com checkpoint externo (``anchor`` ou o configurado para o banco), exige que
    o evento do último checkpoint exista com o mesmo hash: cadeia encurtada
    (eventos recentes apagados) ou reescrita é reprovada.
    """
    from sqlalchemy import select

    from radiologyai.persistence.models import AuditEvent

    cfg = anchor or _anchor_for(session.get_bind())
    anchored: tuple[int, str] | None = None
    if cfg is not None:
        try:
            anchored = read_anchor(cfg)
        except AuditError as exc:
            return ChainStatus(False, 0, GENESIS_HASH, None, str(exc))
    anchored_hash: str | None = None
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
        if anchored is not None and row.seq == anchored[0]:
            anchored_hash = row.record_hash
        prev = row.record_hash
        expected_seq += 1
    if anchored is not None:
        seq, digest = anchored
        if anchored_hash is None:
            return ChainStatus(
                False,
                count,
                prev,
                seq,
                f"cadeia encurtada: o checkpoint externo registra o evento {seq}, "
                f"mas o banco termina em {expected_seq - 1}",
                seq,
            )
        if anchored_hash != digest:
            return ChainStatus(
                False, count, prev, seq, f"evento {seq} difere do checkpoint externo", seq
            )
    return ChainStatus(True, count, prev, anchored_seq=None if anchored is None else anchored[0])
