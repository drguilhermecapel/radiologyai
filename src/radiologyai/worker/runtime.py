"""Estado compartilhado da plataforma (API + worker)."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from radiologyai.analysis.policy import SafetyPolicy
    from radiologyai.analysis.registry import Registry
    from radiologyai.catalog.catalog import Catalog
    from radiologyai.interop.images import ImageCache
    from radiologyai.persistence.blobs import BlobStore
    from radiologyai.persistence.config import PlatformSettings
    from radiologyai.persistence.db import Database

__all__ = ["AnalysisEngine", "Platform"]


@dataclass
class AnalysisEngine:
    """O que o pipeline precisa: registro de modelos, catálogo e política de decisão.

    ``policy=None`` faz o pipeline usar ``ManifestOnlyPolicy`` (pontos do manifesto
    só nos níveis B e D; A/C ficam ``score_only``). :meth:`default` usa a
    ``EvidencePolicy``: achados A/B/C decidem só com os pontos do registro de
    evidência que sustenta o nível. Sem calibração ajustada, nenhuma
    probabilidade é reportada.
    """

    registry: Registry
    catalog: Catalog
    policy: SafetyPolicy | None = None

    @classmethod
    def default(cls) -> AnalysisEngine:
        """Registro dos manifestos do pacote, catálogo distribuído e política de evidência.

        Raises:
            EvidenceError: arquivo de ``evidence/`` inválido (falha alta, nunca decide
                com evidência corrompida).
        """
        from radiologyai.analysis.evidence_policy import EvidencePolicy
        from radiologyai.analysis.registry import Registry
        from radiologyai.catalog.catalog import Catalog

        catalog = Catalog.load()
        return cls(registry=Registry(), catalog=catalog, policy=EvidencePolicy.load(catalog))


@dataclass
class Platform:
    """Banco, repositório de artefatos, cache de imagens e motor (carregado sob demanda)."""

    settings: PlatformSettings
    db: Database
    blobs: BlobStore
    cache: ImageCache
    _engine: AnalysisEngine | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @classmethod
    def create(
        cls, settings: PlatformSettings, *, engine: AnalysisEngine | None = None
    ) -> Platform:
        from radiologyai.interop.images import ImageCache
        from radiologyai.persistence.audit import configure_anchor
        from radiologyai.persistence.blobs import BlobStore
        from radiologyai.persistence.config import server_key
        from radiologyai.persistence.db import Database

        platform = cls(
            settings=settings,
            db=Database(settings.database_url),
            blobs=BlobStore(settings.store_dir),
            cache=ImageCache(settings.decoded_cache_series),
        )
        platform._engine = engine
        key = server_key(settings)
        if key is not None:
            configure_anchor(
                platform.db,
                settings.audit_anchor_path or settings.store_dir / "audit-anchor.jsonl",
                key,
            )
        return platform

    @property
    def engine(self) -> AnalysisEngine:
        """Motor de análise; construído na primeira chamada (manifestos + catálogo)."""
        with self._lock:
            if self._engine is None:
                self._engine = AnalysisEngine.default()
            return self._engine

    def engine_if_loaded(self) -> AnalysisEngine | None:
        return self._engine
