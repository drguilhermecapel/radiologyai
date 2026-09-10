"""API REST — fábrica de aplicação FastAPI.

Preserva a forma de URL de ``legacy/src/medai_fastapi_server.py`` e o contrato
de ``docs/OPENAPI_SPEC.yaml``, mas **não** o comportamento: o servidor do v1
fabricava métricas de validação clínica a cada requisição
(``y_true = np.array([1])  # Mock ground truth``) e as devolvia ao cliente como
``clinical_metrics``. Aqui esse caminho não existe.

Diferenças estruturais em relação ao v1:

- ``create_app()`` é fábrica, não ``app = FastAPI()`` em nível de módulo. Torna
  a aplicação testável e configurável sem variável global.
- Segredos vêm exclusivamente do ambiente.
- ``/api/v1/metrics`` devolve o que foi **medido** em ``artifacts/eval/``, ou
  declara que não há medição. Nunca um número inventado.
- ``/api/v1/analyze`` aplica o gate de escopo antes de qualquer modelo e falha
  fechada quando não há pesos.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any

from radiologyai import __version__
from radiologyai.errors import BackendUnavailableError

if TYPE_CHECKING:
    from fastapi import FastAPI

API_PREFIX = "/api/v1"

DISCLAIMER = (
    "Software de pesquisa. NÃO é dispositivo médico. Nenhum modelo validado "
    "clinicamente. Toda saída é rascunho e requer revisão e validação médica. "
    "O sistema não afirma normalidade."
)


def require_fastapi() -> None:
    """Exige o extra [api]. Sem ele, erro explícito — nunca servidor degradado."""
    missing = [n for n in ("fastapi", "starlette") if importlib.util.find_spec(n) is None]
    if missing:
        raise BackendUnavailableError(
            f"API indisponível: faltam {missing}. Instale com: pip install 'radiologyai[api]'"
        )


def create_app(*, artifacts_dir: str | None = None) -> FastAPI:
    """Constrói a aplicação.

    Args:
        artifacts_dir: raiz de ``artifacts/eval``. As métricas servidas vêm daqui.
    """
    require_fastapi()

    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    from radiologyai.api.routers import analyze, health, metrics, models

    app = FastAPI(
        title="RadiologyAI",
        version=__version__,
        description=DISCLAIMER,
        docs_url=f"{API_PREFIX}/docs",
        openapi_url=f"{API_PREFIX}/openapi.json",
    )
    app.state.artifacts_dir = artifacts_dir or "artifacts/eval"

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    for router in (health.router, models.router, metrics.router, analyze.router):
        app.include_router(router, prefix=API_PREFIX)

    return app


def app_info() -> dict[str, Any]:
    return {"name": "RadiologyAI", "version": __version__, "disclaimer": DISCLAIMER}
