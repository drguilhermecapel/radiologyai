"""Artefatos endereçados por conteúdo (mapas de explicação PNG, máscaras NPZ, documentos)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Response

from radiologyai.api.deps import Principal, get_platform, perm, require
from radiologyai.api.routers.studies import _ERRORS
from radiologyai.api.shadow import AI_ARTIFACT_KINDS, block_ai
from radiologyai.persistence.models import Artifact
from radiologyai.persistence.security import Permission
from radiologyai.worker.runtime import Platform

router = APIRouter(tags=["artifacts"])


@router.get(
    "/artifacts/{sha256}",
    response_class=Response,
    responses={
        200: {
            "content": {"image/png": {}, "application/x-npz": {}, "application/octet-stream": {}},
            "description": "Bytes conferidos contra o sha256 antes do envio.",
        },
        **_ERRORS,
    },
    openapi_extra=perm(Permission.ANALYSES_READ),
)
def artifact(
    principal: Annotated[Principal, Depends(require(Permission.ANALYSES_READ))],
    platform: Annotated[Platform, Depends(get_platform)],
    sha256: str = Path(pattern=r"^[0-9a-f]{64}$"),
) -> Response:
    """Artefato indexado (só os registrados no banco); hash conferido na leitura.

    Modo silencioso (REQ-195): mapa de explicação e máscara (saída de IA) => 403
    ``ShadowModeError`` para quem não tem ``shadow:view``.
    """
    with platform.db.session() as session:
        row = session.get(Artifact, sha256)
        if row is None:
            raise HTTPException(status_code=404, detail="artefato inexistente")
        media = row.media_type
        kind = row.kind
    if kind in AI_ARTIFACT_KINDS:
        block_ai(
            platform, principal, action="artifact.read", subject_type="artifact", subject_id=sha256
        )
    data = platform.blobs.read(sha256)
    return Response(
        content=data,
        media_type=media,
        headers={"ETag": f'"{sha256}"', "Cache-Control": "private, max-age=31536000, immutable"},
    )
