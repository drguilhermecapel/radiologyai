"""API REST — fábrica de aplicação FastAPI (``/api/v1``).

Princípios:

* ``create_app()`` é fábrica: banco migrado (Alembic) na construção, worker em
  processo iniciado no ``lifespan``. Nenhum estado global.
* Autenticação por token opaco revogável; RBAC por rota (ver
  :data:`radiologyai.persistence.security.PERMISSIONS`). Só saúde, login e
  métricas medidas são públicas. Sem credencial padrão.
* CORS fechado por padrão (``MEDAI_CORS_ORIGINS``); corpo de requisição
  limitado durante o recebimento (413 antes de gravar em disco).
* Nenhuma resposta contém número que um modelo real não calculou: o resultado
  da análise é o do pipeline, gravado imutável com o seu sha256.
* O servidor do v1 fabricava métricas (``y_true = np.array([1])  # Mock ground
  truth``); ``/metrics`` aqui só serve o que está em ``artifacts/eval/``.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import AsyncIterator, Awaitable, Callable, MutableMapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from radiologyai import __version__
from radiologyai.audit.trail import AuditTrail
from radiologyai.errors import BackendUnavailableError

if TYPE_CHECKING:
    from fastapi import FastAPI

    from radiologyai.persistence.config import PlatformSettings
    from radiologyai.worker.runtime import AnalysisEngine, Platform

__all__ = ["API_PREFIX", "DISCLAIMER", "app_info", "create_app", "require_fastapi"]

API_PREFIX = "/api/v1"

DISCLAIMER = (
    "Software de pesquisa. NÃO é dispositivo médico. Nenhum modelo validado "
    "clinicamente. Toda saída é rascunho e requer revisão e validação médica. "
    "O sistema não afirma normalidade."
)

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


def require_fastapi() -> None:
    """Exige os extras [api] e [platform]. Sem eles, erro explícito — nunca servidor degradado."""
    missing = [
        n
        for n in ("fastapi", "starlette", "multipart", "sqlalchemy", "alembic", "argon2")
        if importlib.util.find_spec(n) is None
    ]
    if missing:
        raise BackendUnavailableError(
            f"API indisponível: faltam {missing}. "
            "Instale com: pip install 'radiologyai[api,platform]'"
        )


class BodySizeLimit:
    """Recusa (413) corpos acima do limite ENQUANTO são recebidos, antes de chegar ao disco."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        for name, value in scope.get("headers") or []:
            if name == b"content-length" and value.isdigit() and int(value) > self.max_bytes:
                await _send_413(send, self.max_bytes)
                return
        received = 0
        started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _TooLargeError
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _TooLargeError:
            if not started:
                await _send_413(send, self.max_bytes)


class _TooLargeError(Exception):
    pass


async def _send_413(send: Send, limit: int) -> None:
    import json

    body = json.dumps(
        {"detail": f"corpo da requisição excede {limit} bytes", "error": "UploadTooLargeError"},
        ensure_ascii=False,
    ).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def _error_status() -> list[tuple[type[Exception], int]]:
    """Mapeamento exceção de domínio -> HTTP (mais específica primeiro)."""
    from radiologyai.api.shadow import ShadowModeError
    from radiologyai.errors import (
        ConfigurationError,
        ExamTypeUnknownError,
        IngestionError,
        ModelUnavailableError,
        OutOfScopeError,
    )
    from radiologyai.interop.dicomweb import DicomwebError, ImportTooLargeError
    from radiologyai.interop.images import FrameRangeError
    from radiologyai.interop.ingest import (
        DeidentKeyMissingError,
        IngestRejectedError,
        UploadTooLargeError,
    )
    from radiologyai.persistence.security import WeakPasswordError
    from radiologyai.persistence.users import (
        AuthenticationError,
        LoginThrottledError,
        UserExistsError,
    )
    from radiologyai.reporting.dicom_sr import ReportFormatError
    from radiologyai.reporting.service import ReportStateError, ResultIntegrityError
    from radiologyai.worker.jobs import NotFoundError

    return [
        (ShadowModeError, 403),
        (NotFoundError, 404),
        (FrameRangeError, 404),
        (LoginThrottledError, 429),
        (AuthenticationError, 401),
        (UserExistsError, 409),
        (WeakPasswordError, 422),
        (ExamTypeUnknownError, 409),
        (OutOfScopeError, 422),
        (UploadTooLargeError, 413),
        (IngestRejectedError, 422),
        (DeidentKeyMissingError, 503),
        (ResultIntegrityError, 500),
        (ReportStateError, 409),
        (ReportFormatError, 409),
        (ImportTooLargeError, 413),
        (DicomwebError, 502),
        (ModelUnavailableError, 503),
        (BackendUnavailableError, 503),
        (ConfigurationError, 503),
        (IngestionError, 422),
    ]


def _resolve_audit_trail(
    audit_trail: AuditTrail | None,
    audit_log_path: str | Path | None,
) -> AuditTrail | None:
    """Cópia JSONL OPCIONAL da rota legada ``/analyze``.

    A auditoria de registro é a cadeia do banco (``legacy.analyze``); esta
    cópia só existe se um caminho for configurado — nunca num diretório
    temporário que some no reinício.
    """
    if audit_trail is not None:
        return audit_trail
    resolved_path = audit_log_path or os.environ.get("RADIOLOGYAI_AUDIT_LOG_PATH")
    if resolved_path:
        return AuditTrail(Path(resolved_path))
    return None


def create_app(
    *,
    artifacts_dir: str | None = None,
    audit_trail: AuditTrail | None = None,
    audit_log_path: str | Path | None = None,
    settings: PlatformSettings | None = None,
    platform: Platform | None = None,
    engine: AnalysisEngine | None = None,
    start_worker: bool | None = None,
) -> FastAPI:
    """Constrói a aplicação.

    Args:
        artifacts_dir: raiz de ``artifacts/eval`` (métricas medidas).
        settings: configuração da plataforma (padrão: variáveis ``MEDAI_*``).
        platform: plataforma pronta (testes); ignora ``settings``.
        engine: motor de análise (registro + catálogo); padrão: o do pacote, sob demanda.
        start_worker: sobe o worker em processo no ``lifespan`` (padrão: configuração).
        audit_trail: trilha JSONL da rota legada ``/analyze`` (testes).
        audit_log_path: caminho dessa trilha (ou ``RADIOLOGYAI_AUDIT_LOG_PATH``).
    """
    require_fastapi()

    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    from radiologyai.api.routers import (
        analyses,
        analyze,
        artifacts,
        audit,
        auth,
        catalog,
        dicomweb,
        health,
        metrics,
        models,
        reports,
        shadow,
        studies,
        users,
    )
    from radiologyai.errors import RadiologyAIError
    from radiologyai.persistence.config import get_platform_settings
    from radiologyai.worker.runtime import Platform

    if platform is None:
        platform = Platform.create(settings or get_platform_settings(), engine=engine)
    platform.db.migrate()
    run_worker = platform.settings.worker_enabled if start_worker is None else start_worker

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        from radiologyai.worker.queue import Worker

        worker = None
        if run_worker:
            worker = Worker(platform)
            worker.start()
        app.state.worker = worker
        try:
            yield
        finally:
            if worker is not None:
                worker.stop()
            loaded = platform.engine_if_loaded()
            if loaded is not None:
                loaded.registry.unload_all()

    app = FastAPI(
        title="RadiologyAI",
        version=__version__,
        description=DISCLAIMER,
        docs_url=f"{API_PREFIX}/docs",
        openapi_url=f"{API_PREFIX}/openapi.json",
        lifespan=lifespan,
    )
    app.state.platform = platform
    app.state.worker = None
    app.state.worker_enabled = run_worker
    app.state.artifacts_dir = artifacts_dir or str(platform.settings.artifacts_eval_dir)
    app.state.audit_trail = _resolve_audit_trail(audit_trail, audit_log_path)

    if platform.settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(platform.settings.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "PUT", "PATCH"],
            allow_headers=["Authorization", "Content-Type"],
            expose_headers=[
                "X-Frame-Rows",
                "X-Frame-Columns",
                "X-Frame-Channels",
                "X-Frame-Dtype",
                "X-Frame-Scale",
                "X-Frame-Index",
                "X-Frame-Count",
                "X-Rescale-Slope",
                "X-Rescale-Intercept",
            ],
        )

    @app.middleware("http")
    async def security_headers(request: Request, call_next: Any) -> Any:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Cache-Control", "no-store")
        return response

    mapping = _error_status()

    @app.exception_handler(RadiologyAIError)
    async def domain_error(request: Request, exc: RadiologyAIError) -> JSONResponse:
        del request
        code = next((c for cls, c in mapping if isinstance(exc, cls)), 500)
        headers = {"WWW-Authenticate": "Bearer"} if code == 401 else None
        return JSONResponse(
            status_code=code,
            content={"detail": str(exc), "error": type(exc).__name__},
            headers=headers,
        )

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        import logging

        logging.getLogger("radiologyai.api").error(
            "erro inesperado em %s %s", request.method, request.url.path, exc_info=exc
        )
        return JSONResponse(
            status_code=500,
            content={
                "detail": "erro interno; nenhum resultado foi produzido no lugar",
                "error": type(exc).__name__,
            },
        )

    for router in (
        health.router,
        metrics.router,
        auth.router,
        users.router,
        studies.router,
        analyses.router,
        artifacts.router,
        reports.router,
        models.router,
        catalog.router,
        audit.router,
        dicomweb.router,
        shadow.router,
        analyze.router,
    ):
        app.include_router(router, prefix=API_PREFIX)

    # Limite de corpo aplicado ao app inteiro, durante o recebimento.
    app.add_middleware(BodySizeLimit, max_bytes=platform.settings.max_upload_bytes)
    return app


def app_info() -> dict[str, Any]:
    return {"name": "RadiologyAI", "version": __version__, "disclaimer": DISCLAIMER}
