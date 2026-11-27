"""Exportação do contrato OpenAPI gerado pela própria aplicação.

``docs/OPENAPI_SPEC.yaml`` é SEMPRE gerado daqui (``radiologyai openapi``) e
versionado; um teste compara o arquivo com o gerado, de modo que a
documentação não pode divergir do código.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import FastAPI

__all__ = ["build_spec_app", "export_openapi"]


def export_openapi(app: FastAPI) -> str:
    """YAML determinístico (chaves ordenadas, UTF-8) do esquema OpenAPI da aplicação."""
    import yaml

    return yaml.safe_dump(
        _stable_reason_phrases(app.openapi()),
        sort_keys=True,
        allow_unicode=True,
        width=100,
        default_flow_style=False,
    )


# Frases de status que mudaram entre versões do Python (http.HTTPStatus): o 3.13
# adotou os nomes do RFC 9110 (413, 414, 416, 422). Sem normalizar, o contrato versionado
# dependeria da versão do interpretador que o gerou.
_RFC9110_PHRASES = {
    "413": "Content Too Large",
    "414": "URI Too Long",
    "416": "Range Not Satisfiable",
    "422": "Unprocessable Content",
}


def _stable_reason_phrases(spec: dict[str, Any]) -> dict[str, Any]:
    for operations in spec.get("paths", {}).values():
        for operation in operations.values():
            if not isinstance(operation, dict):
                continue
            for code, response in operation.get("responses", {}).items():
                if code in _RFC9110_PHRASES and isinstance(response, dict):
                    response["description"] = _RFC9110_PHRASES[code]
    return spec


def build_spec_app() -> FastAPI:
    """Aplicação descartável (SQLite temporário, sem worker) só para gerar o esquema."""
    from radiologyai.api.app import create_app
    from radiologyai.persistence.config import PlatformSettings
    from radiologyai.worker.runtime import Platform

    tmp = tempfile.mkdtemp(prefix="radiologyai-openapi-")
    settings = PlatformSettings(
        database_url=f"sqlite:///{tmp}/spec.db",
        store_dir=Path(tmp) / "store",
        worker_enabled=False,
    )
    return create_app(platform=Platform.create(settings), start_worker=False)
