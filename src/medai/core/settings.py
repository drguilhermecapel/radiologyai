"""Runtime settings, read from environment variables prefixed with ``MEDAI_``.

No secret has a default value. Paths default under ``MEDAI_HOME`` (``~/.medai``).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MEDAI_", env_file=".env", extra="ignore")

    home: Path = Field(default_factory=lambda: Path.home() / ".medai")
    weights_dir: Path | None = None  # default: <home>/weights
    store_dir: Path | None = None  # default: <home>/store (content-addressed artifacts)
    datasets_dir: Path | None = None  # default: <home>/datasets (evaluation data, never versioned)
    database_url: str | None = None  # default: sqlite:///<home>/medai.db

    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    torch_threads: int | None = None
    max_loaded_models: int = 4

    # Privacy
    deidentify_on_ingest: bool = True
    pseudonym_salt: SecretStr | None = None  # required when deidentify_on_ingest is true
    allow_network_at_inference: bool = False  # weights are fetched by `medai models pull` only

    # Product behaviour
    research_mode_default: bool = False  # level D findings are hidden unless asked for
    allow_noncommercial_models: bool = True  # models whose weights are research-only
    max_upload_mb: int = 2048

    # API
    session_ttl_minutes: int = 480
    cors_origins: list[str] = Field(default_factory=list)

    # Optional DICOMweb endpoint (e.g. Orthanc)
    dicomweb_url: str | None = None
    dicomweb_username: str | None = None
    dicomweb_password: SecretStr | None = None

    @property
    def weights_path(self) -> Path:
        return self.weights_dir or self.home / "weights"

    @property
    def store_path(self) -> Path:
        return self.store_dir or self.home / "store"

    @property
    def datasets_path(self) -> Path:
        return self.datasets_dir or self.home / "datasets"

    @property
    def db_url(self) -> str:
        return self.database_url or f"sqlite:///{self.home / 'medai.db'}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
