"""Configuração da plataforma (API, banco, fila, interoperabilidade).

Separada de :class:`radiologyai.config.settings.Settings` para não mexer no
contrato do núcleo; usa o mesmo prefixo ``MEDAI_``. Nenhum segredo tem valor
padrão: sem ``MEDAI_DEIDENT_KEY`` a ingestão de DICOM é recusada, e não há
credencial inicial de administrador (o primeiro é criado pela CLI).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["PlatformSettings", "get_platform_settings", "server_key"]


class PlatformSettings(BaseSettings):
    """Variáveis ``MEDAI_*`` da plataforma."""

    model_config = SettingsConfigDict(env_prefix="MEDAI_", env_file=".env", extra="ignore")

    database_url: str = Field(
        default="sqlite:///./data/platform/radiologyai.db",
        description="URL SQLAlchemy (MEDAI_DATABASE_URL). SQLite por padrão; Postgres pela URL.",
    )
    store_dir: Path = Field(
        default=Path("./artifacts/store"),
        description="Repositório de artefatos endereçado por conteúdo (MEDAI_STORE_DIR).",
    )
    deident_key: str | None = Field(
        default=None, description="Chave HMAC da des-identificação (MEDAI_DEIDENT_KEY)."
    )
    signing_key: str | None = Field(
        default=None,
        description="Chave HMAC da assinatura de laudos e dos checkpoints da auditoria "
        "(MEDAI_SIGNING_KEY). Ausente: derivada de MEDAI_DEIDENT_KEY; sem nenhuma, "
        "laudos não são assinados.",
    )
    audit_anchor_path: Path | None = Field(
        default=None,
        description="Arquivo de checkpoints da cabeça da cadeia de auditoria, FORA do banco "
        "(MEDAI_AUDIT_ANCHOR_PATH; padrão: <store_dir>/audit-anchor.jsonl).",
    )
    session_ttl_hours: float = Field(default=12.0, gt=0, le=24 * 30)
    password_min_length: int = Field(default=12, ge=8)
    max_upload_bytes: int = Field(default=2 * 1024**3, ge=1024)
    max_zip_members: int = Field(default=5000, ge=1)
    max_uncompressed_bytes: int = Field(default=4 * 1024**3, ge=1024)
    frames_max_dim: int = Field(default=1024, ge=64, description="Lado máximo padrão do quadro.")
    frames_hard_cap: int = Field(default=4096, ge=64, description="Lado máximo aceito no pedido.")
    decoded_cache_series: int = Field(default=2, ge=0, description="Séries decodificadas em LRU.")
    cors_origins: tuple[str, ...] = Field(
        default=(), description="Origens CORS permitidas (MEDAI_CORS_ORIGINS, JSON)."
    )
    worker_enabled: bool = Field(default=True, description="Sobe o worker em processo.")
    worker_poll_seconds: float = Field(default=1.0, gt=0)
    worker_lease_seconds: float = Field(
        default=24 * 3600,
        gt=0,
        description="Análise 'running' de worker de OUTRO host só é dada como interrompida "
        "depois deste tempo (MEDAI_WORKER_LEASE_SECONDS).",
    )
    dicomweb_url: str | None = Field(default=None, description="Base DICOMweb (Orthanc).")
    dicomweb_username: str | None = None
    dicomweb_password: str | None = None
    artifacts_eval_dir: Path = Field(
        default=Path("artifacts/eval"), description="Medições servidas por /metrics."
    )
    shadow_mode: bool = Field(
        default=False,
        description="Modo silencioso do piloto prospectivo (MEDAI_SHADOW_MODE, REQ-195): as "
        "análises rodam e são gravadas e auditadas, mas a saída de IA só é servida a quem tem "
        "'shadow:view' (equipe de pesquisa) e nada sai para o PACS.",
    )


def server_key(settings: PlatformSettings) -> bytes | None:
    """Chave de servidor (fora do banco) para HMAC de assinatura e checkpoints.

    ``MEDAI_SIGNING_KEY`` quando definida; senão, subchave derivada de
    ``MEDAI_DEIDENT_KEY`` com separação de domínio; ``None`` sem nenhuma.
    """
    import hashlib
    import hmac

    if settings.signing_key:
        return settings.signing_key.encode("utf-8")
    if settings.deident_key:
        return hmac.new(
            settings.deident_key.encode("utf-8"), b"radiologyai:signing:v1", hashlib.sha256
        ).digest()
    return None


def get_platform_settings() -> PlatformSettings:
    """Configuração corrente da plataforma."""
    return PlatformSettings()
