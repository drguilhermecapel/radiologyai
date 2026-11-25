"""Fila de análises no banco e o worker em processo (concorrência 1).

Estados: ``queued`` -> ``running`` -> ``completed`` | ``failed``. A reivindicação
é um ``UPDATE ... WHERE state = 'queued'`` condicional: só um worker vence,
mesmo com vários processos no mesmo banco. Reexecutar é criar outra análise.

Ao subir, análises que ficaram ``running`` (processo morreu no meio) viram
``failed`` com o motivo — nunca são retomadas nem completadas em silêncio.
"""

from __future__ import annotations

import logging
import os
import threading
import traceback
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from radiologyai.worker.runtime import Platform

__all__ = ["Worker", "claim_next", "fail", "recover_interrupted", "run_one"]

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


def fail(platform: Platform, analysis_id: str, error: str) -> None:
    """Marca a análise como ``failed`` com o motivo e audita."""
    from radiologyai.persistence.audit import append_event
    from radiologyai.persistence.models import Analysis, utcnow

    with platform.db.session() as session:
        row = session.get(Analysis, analysis_id)
        if row is None or row.state in ("completed", "failed"):
            return
        row.state = "failed"
        row.error = error[:4000]
        row.finished_at = utcnow()
    append_event(
        platform.db,
        "analysis.failed",
        actor_username="worker",
        subject_type="analysis",
        subject_id=analysis_id,
        payload={"error": error[:500]},
    )


def recover_interrupted(platform: Platform) -> int:
    """Análises ``running`` órfãs viram ``failed``; devolve quantas."""
    from sqlalchemy import select

    from radiologyai.persistence.models import Analysis

    with platform.db.session() as session:
        ids = list(
            session.execute(select(Analysis.id).where(Analysis.state == "running")).scalars()
        )
    for analysis_id in ids:
        fail(platform, analysis_id, INTERRUPTED)
    return len(ids)


def run_one(platform: Platform, worker_id: str = "inline") -> str | None:
    """Reivindica e executa uma análise; devolve o id processado (ou ``None``)."""
    from radiologyai.worker.runner import execute, persist_result

    analysis_id = claim_next(platform, worker_id)
    if analysis_id is None:
        return None
    try:
        result = execute(platform, analysis_id)
        persist_result(platform, analysis_id, result)
    except Exception as exc:  # noqa: BLE001 - falha registrada com motivo, nunca substituída
        log.error("análise %s falhou: %s", analysis_id, traceback.format_exc(limit=5))
        fail(platform, analysis_id, f"{type(exc).__name__}: {exc}")
    return analysis_id


class Worker:
    """Thread única que consome a fila; ``notify()`` acorda sem esperar o intervalo."""

    def __init__(self, platform: Platform) -> None:
        self.platform = platform
        self.worker_id = f"pid{os.getpid()}-{id(self):x}"
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
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
