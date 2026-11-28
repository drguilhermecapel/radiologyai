"""Artefatos endereçados por conteúdo com índice no banco.

Envolve :class:`radiologyai.analysis.artifacts.ArtifactStore` (arquivos
``<aa>/<sha256><ext>``, leitura com conferência de hash) e mantém a tabela
``artifacts`` com tipo de mídia, tamanho e finalidade. O mesmo conteúdo é
sempre o mesmo artefato.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from radiologyai.analysis.artifacts import ArtifactStore
from radiologyai.analysis.types import ArtifactRef

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

__all__ = ["BlobStore"]


class BlobStore:
    """Repositório de artefatos + registro na tabela ``artifacts``."""

    def __init__(self, base_dir: Path) -> None:
        self.store = ArtifactStore(Path(base_dir))

    def put(self, session: Session, data: bytes, media_type: str, kind: str) -> ArtifactRef:
        """Grava (idempotente) e indexa; devolve a referência."""
        ref = self.store.put_bytes(data, media_type)
        self.register(session, ref, kind)
        return ref

    def register(self, session: Session, ref: ArtifactRef, kind: str) -> None:
        """Indexa um artefato já gravado no repositório (ex.: máscara do pipeline)."""
        from radiologyai.persistence.models import Artifact

        if session.get(Artifact, ref.sha256) is not None:
            return
        size = self.store.path(ref).stat().st_size
        session.add(
            Artifact(
                sha256=ref.sha256,
                media_type=ref.media_type,
                relpath=ref.relpath,
                size_bytes=size,
                kind=kind,
            )
        )
        session.flush()

    def path(self, sha256: str) -> Path:
        return self.store.path(sha256)

    def read(self, sha256: str) -> bytes:
        """Bytes conferidos contra o sha256 (corrupção em disco levanta)."""
        return self.store.read_bytes(sha256)
