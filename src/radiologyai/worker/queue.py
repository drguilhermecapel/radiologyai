"""Fila de análises no banco e o worker em processo (concorrência 1).

Estados: ``queued`` -> ``running`` -> ``completed`` | ``failed``. A reivindicação
é um ``UPDATE ... WHERE state = 'queued'`` condicional: só um worker vence,
mesmo com vários processos no mesmo banco. Reexecutar é criar outra análise.

Ao subir, análises que ficaram ``running`` com worker MORTO viram ``failed``
com o motivo — nunca são retomadas nem completadas em silêncio. Worker vivo
(outro processo no mesmo banco) não é tocado: o id do worker é
``host:pid:token``; no mesmo host, morto = pid inexistente (ou token que não
está ativo neste processo); noutro host, morto = execução além de
``worker_lease_seconds``. Id em outro formato (legado) é tratado como morto.
"""

from __future__ import annotations

import logging
import os
import secrets
import socket
import threading
import traceback
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from radiologyai.worker.runtime import Platform

__all__ = [
    "Worker",
    "claim_next",
    "fail",
    "new_worker_id",
    "recover_interrupted",
    "run_one",
    "worker_alive",
]

_LIVE_LOCK = threading.Lock()
_LIVE_TOKENS: set[str] = set()


def _host() -> str:
    return socket.gethostname()[:40]


def new_worker_id() -> str:
    """``host:pid:token`` (até 64 caracteres) de um worker deste processo."""
    return f"{_host()}:{os.getpid()}:{secrets.token_hex(6)}"


def _set_live(worker_id: str, alive: bool) -> None:
    with _LIVE_LOCK:
        if alive:
            _LIVE_TOKENS.add(worker_id)
        else:
            _LIVE_TOKENS.discard(worker_id)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def worker_alive(
    worker_id: str | None, started_at: datetime | None, *, lease_seconds: float, now: datetime
) -> bool:
    """O worker que reivindicou a análise ainda pode estar rodando?"""
    if not worker_id:
        return False
    parts = worker_id.split(":")
    if len(parts) != 3 or not parts[1].isdigit():
        return False  # formato legado: sem como provar que vive
    host, pid = parts[0], int(parts[1])
    if host == _host():
        if pid == os.getpid():
            with _LIVE_LOCK:
                return worker_id in _LIVE_TOKENS
        return _pid_alive(pid)
    if started_at is None:
        return False
    return now - started_at < timedelta(seconds=lease_seconds)


log = logging.getLogger("radiologyai.worker")

INTERRUPTED = (
    "interrompida: o processo foi reiniciado durante a execução; "
    "nenhum resultado foi gravado. Crie uma nova análise para reexecutar."
)


def claim_next(platform: Platform, worker_id: str) -> str | None:
    """Reivindica a análise ``queued`` mais antiga; ``None`` se a fila está vazia."""
    from sqlalchemy import select, update

    from radiologyai.persistence.models import Analysis, utcnow

    with platform.db.session() as session:
        candidate = session.execute(
            select(Analysis.id)
            .where(Analysis.state == "queued")
            .order_by(Analysis.requested_at, Analysis.id)
            .limit(1)
        ).scalar_one_or_none()
        if candidate is None:
            return None
        claimed = session.execute(
            update(Analysis)
            .where(Analysis.id == candidate, Analysis.state == "queued")
            .values(state="running", started_at=utcnow(), worker_id=worker_id)
        )
        return candidate if claimed.rowcount == 1 else None  # type: ignore[attr-defined]


def _fail_in(
    session: Session, analysis_id: str, error: str, *, only_worker: str | None = None
) -> bool:
    """``UPDATE`` condicional para ``failed`` + auditoria na mesma transação."""
    from sqlalchemy import update

    from radiologyai.persistence.audit import append_event
    from radiologyai.persistence.models import Analysis, utcnow

    conditions = [Analysis.id == analysis_id, Analysis.state.in_(("queued", "running"))]
    if only_worker is not None:
        conditions.append(Analysis.worker_id == only_worker)
    done = session.execute(
        update(Analysis)
        .where(*conditions)
        .values(state="failed", error=error[:4000], finished_at=utcnow())
        .execution_options(synchronize_session=False)
    )
    if done.rowcount != 1:  # type: ignore[attr-defined]
        return False
    append_event(
        None,
        "analysis.failed",
        actor_username="worker",
        subject_type="analysis",
        subject_id=analysis_id,
        payload={"error": error[:500]},
        session=session,
    )
    return True


def fail(platform: Platform, analysis_id: str, error: str) -> None:
    """Marca a análise como ``failed`` com o motivo e audita (atômico; nada se já terminou)."""
    with platform.db.session() as session:
        _fail_in(session, analysis_id, error)


def recover_interrupted(platform: Platform) -> int:
    """Análises ``running`` de workers MORTOS viram ``failed``; devolve quantas."""
    from sqlalchemy import select

    from radiologyai.persistence.models import Analysis, utcnow

    now = utcnow()
    lease = platform.settings.worker_lease_seconds
    with platform.db.session() as session:
        rows = session.execute(
            select(Analysis.id, Analysis.worker_id, Analysis.started_at).where(
                Analysis.state == "running"
            )
        ).all()
    dead = [
        (analysis_id, worker_id)
        for analysis_id, worker_id, started in rows
        if not worker_alive(worker_id, started, lease_seconds=lease, now=now)
    ]
    count = 0
    for analysis_id, worker_id in dead:
        with platform.db.session() as session:
            # Condicional ao MESMO worker: se outro reivindicou entretanto, nada muda.
            if worker_id is None:
                count += _fail_in(session, analysis_id, INTERRUPTED)
            else:
                count += _fail_in(session, analysis_id, INTERRUPTED, only_worker=worker_id)
    return count


def run_one(platform: Platform, worker_id: str | None = None) -> str | None:
    """Reivindica e executa uma análise; devolve o id processado (ou ``None``)."""
    from radiologyai.worker.runner import execute, persist_result

    wid = worker_id or new_worker_id()
    _set_live(wid, True)
    try:
        analysis_id = claim_next(platform, wid)
        if analysis_id is None:
            return None
        try:
            result = execute(platform, analysis_id)
            persist_result(platform, analysis_id, result, worker_id=wid)
        except Exception as exc:  # noqa: BLE001 - falha registrada com motivo, nunca substituída
            log.error("análise %s falhou: %s", analysis_id, traceback.format_exc(limit=5))
            fail(platform, analysis_id, f"{type(exc).__name__}: {exc}")
        return analysis_id
    finally:
        if worker_id is None:
            _set_live(wid, False)


class Worker:
    """Thread única que consome a fila; ``notify()`` acorda sem esperar o intervalo."""

    def __init__(self, platform: Platform) -> None:
        self.platform = platform
        self.worker_id = new_worker_id()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        _set_live(self.worker_id, True)
        recovered = recover_interrupted(self.platform)
        if recovered:
            log.warning("%d análise(s) interrompida(s) marcadas como falhas", recovered)
        self._thread = threading.Thread(target=self._loop, name="radiologyai-worker", daemon=True)
        self._thread.start()

    def notify(self) -> None:
        self._wake.set()

    def stop(self, timeout: float = 30.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)
        if not self.alive:
            _set_live(self.worker_id, False)

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                processed = run_one(self.platform, self.worker_id)
            except Exception:  # noqa: BLE001 - erro de infraestrutura: registra e continua
                log.exception("erro no laço do worker")
                processed = None
            if processed is None:
                self._wake.wait(self.platform.settings.worker_poll_seconds)
                self._wake.clear()
