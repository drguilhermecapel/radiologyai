"""Configuração por ambiente.

Substitui ``medai_config.json`` do legado, que trazia caminhos absolutos de
outra máquina (``/home/ubuntu/repos/radiologyai/...``) e segredos placeholder
commitados (``"jwt_secret": "your-secret-key-here-change-in-production"``).

Nenhum segredo tem valor default. Ausente significa erro, não modo inseguro.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuração carregada de variáveis de ambiente e de ``.env``."""

    model_config = SettingsConfigDict(env_prefix="MEDAI_", env_file=".env", extra="ignore")

    data_dir: Path = Field(default=Path("./data"))
    models_dir: Path = Field(default=Path("./models"))
    artifacts_dir: Path = Field(default=Path("./artifacts"))
    logs_dir: Path = Field(default=Path("./logs"))

    deident_key: str | None = Field(
        default=None,
        description="Chave HMAC para pseudo-IDs. Obrigatória para des-identificar.",
    )

    def require_deident_key(self) -> bytes:
        """Devolve a chave HMAC ou falha alto.

        Raises:
            RuntimeError: chave não configurada. Gerar pseudo-IDs com chave
                efêmera produziria identificadores instáveis entre execuções —
                exatamente o defeito do ``hash()`` builtin no legado.
        """
        if not self.deident_key:
            raise RuntimeError(
                "MEDAI_DEIDENT_KEY não configurada. Defina-a no ambiente ou no .env "
                "(ver .env.example). Sem chave persistente os pseudo-IDs não são "
                "estáveis entre execuções."
            )
        return self.deident_key.encode("utf-8")


def get_settings() -> Settings:
    """Carrega a configuração corrente."""
    return Settings()
