"""Saúde do serviço: ``/healthz`` (vivo), ``/readyz`` (pronto) e o estado declarado."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Request, Response

from radiologyai import __version__, selftest
from radiologyai.api.app import DISCLAIMER
from radiologyai.api.deps import PUBLIC, get_platform, perm
from radiologyai.api.schemas import HealthOut, HealthzOut, ReadyzOut, StatusOut
from radiologyai.worker.runtime import Platform

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthzOut, openapi_extra=perm(PUBLIC))
def healthz() -> HealthzOut:
    """Processo vivo. Não consulta dependências."""
    return HealthzOut(status="ok")


@router.get(
    "/readyz",
    response_model=ReadyzOut,
    responses={503: {"model": ReadyzOut}},
    openapi_extra=perm(PUBLIC),
)
def readyz(
    request: Request, response: Response, platform: Annotated[Platform, Depends(get_platform)]
) -> ReadyzOut:
    """Pronto para servir: banco na revisão esperada, repositório gravável, worker vivo."""
    import tempfile

    detail: list[str] = []
    try:
        revision = platform.db.revision()
    except Exception as exc:  # noqa: BLE001 - reportado no corpo, nunca escondido
        revision = None
        detail.append(f"banco inacessível: {type(exc).__name__}")
    expected = platform.db.head()
    if revision != expected:
        detail.append(f"banco na revisão {revision!r}, esperado {expected!r}")
    try:
        with tempfile.NamedTemporaryFile(dir=platform.blobs.store.base_dir):
            writable = True
    except OSError:
        writable = False
        detail.append("repositório de artefatos não gravável")
    worker = getattr(request.app.state, "worker", None)
    worker_state: Literal["running", "stopped", "disabled"]
    if not request.app.state.worker_enabled:
        worker_state = "disabled"
    elif worker is not None and worker.alive:
        worker_state = "running"
    else:
        worker_state = "stopped"
        detail.append("worker parado")
    if not platform.settings.deident_key:
        detail.append("MEDAI_DEIDENT_KEY ausente: ingestão de DICOM recusada")
    ready = revision == expected and writable and worker_state != "stopped"
    if not ready:
        response.status_code = 503
    return ReadyzOut(
        ready=ready,
        database_revision=revision,
        expected_revision=expected,
        worker=worker_state,
        store_writable=writable,
        deident_key_configured=bool(platform.settings.deident_key),
        detail=tuple(detail),
    )


@router.get("/health", response_model=HealthOut, openapi_extra=perm(PUBLIC))
def health() -> HealthOut:
    """Estado do serviço e disponibilidade dos subsistemas opcionais."""
    return HealthOut(
        status="ok",
        version=__version__,
        disclaimer=DISCLAIMER,
        subsystems={k: bool(v) for k, v in dict(selftest()["optional"]).items()},  # type: ignore[call-overload]
    )


@router.get("/status", response_model=StatusOut, openapi_extra=perm(PUBLIC))
def status() -> StatusOut:
    """Estado detalhado, incluindo modalidades e sua maturidade."""
    from radiologyai.modalities import available_modalities

    seen = {p.code: p.describe() for p in available_modalities().values()}
    return StatusOut(
        version=__version__,
        modalities=[seen[c] for c in sorted(seen)],
        clinically_validated=False,
        regulatory_status="não registrado em nenhuma agência",
        disclaimer=DISCLAIMER,
    )
