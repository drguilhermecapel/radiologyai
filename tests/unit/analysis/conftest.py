"""Fixtures da análise multimodelo. Dublês são injetados no teste e nunca distribuídos."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import analysis_fakes
import pytest
import yaml
from analysis_support import FAKE_MODULE, WEIGHT_BYTES, build_catalog

from radiologyai.analysis.artifacts import ArtifactStore
from radiologyai.analysis.registry import Registry
from radiologyai.catalog.catalog import Catalog
from radiologyai.config.settings import Settings


@pytest.fixture(autouse=True)
def _register_fake_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, FAKE_MODULE, analysis_fakes)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(weights_dir=tmp_path / "weights", device="cpu", max_loaded_models=8)


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "store")


RegistryFactory = Callable[..., Registry]


@pytest.fixture
def make_registry(tmp_path: Path, settings: Settings) -> RegistryFactory:
    """Registro a partir de dicts de manifesto; pesos gravados salvo ``with_weights=False``."""

    def build(*manifests: dict[str, Any], with_weights: bool = True) -> Registry:
        manifest_dir = tmp_path / "manifests"
        manifest_dir.mkdir(exist_ok=True)
        weights_dir = tmp_path / "weights"
        for data in manifests:
            (manifest_dir / f"{data['id']}.yaml").write_text(
                yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
            )
            if with_weights:
                for weight in data.get("weights", []):
                    target = weights_dir / data["id"] / weight["filename"]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(WEIGHT_BYTES)
        return Registry(manifest_dir=manifest_dir, weights_dir=weights_dir, settings=settings)

    return build


@pytest.fixture
def make_catalog() -> Callable[..., Catalog]:
    """Catálogo em que cada ``(condição, modelo, rótulo)`` recebe o nível pedido."""
    return build_catalog
