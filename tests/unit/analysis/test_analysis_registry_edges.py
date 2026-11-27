"""Bordas do registro: ids inexistentes, dispositivo, backend ausente, extração e pin.

Os adaptadores são os dublês de ``analysis_fakes`` (nunca distribuídos).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import analysis_fakes
import pytest
import yaml
from analysis_support import WEIGHT_BYTES, manifest_dict

from radiologyai.analysis.registry import Registry, _instantiate, pin, safe_extract
from radiologyai.config.settings import Settings
from radiologyai.errors import BackendUnavailableError, ManifestError, ModelUnavailableError


@pytest.mark.requirement("REQ-100")
def test_registry_unknown_ids_and_lookups(make_registry: Any) -> None:
    registry = make_registry(manifest_dict("fake-bright", "Brightness"))
    with pytest.raises(ModelUnavailableError, match="inexistente"):
        registry.manifest("nao-existe")
    with pytest.raises(ModelUnavailableError):
        registry.manifest_sha256("nao-existe")
    assert len(registry.manifest_sha256("fake-bright")) == 64
    assert registry.manifest_path("fake-bright").name == "fake-bright.yaml"
    assert registry.has("fake-bright")
    assert not registry.has("nao-existe")
    assert registry.ids(status="planned") == []


@pytest.mark.requirement("REQ-100")
def test_registry_requires_existing_manifest_dir(tmp_path: Path, settings: Settings) -> None:
    with pytest.raises(ManifestError, match="não encontrado"):
        Registry(manifest_dir=tmp_path / "nada", weights_dir=tmp_path / "w", settings=settings)


@pytest.mark.requirement("REQ-100")
def test_registry_device_resolution(
    make_registry: Any, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = make_registry(manifest_dict("fake-bright", "Brightness"))
    assert registry.resolve_device() == "cpu"
    registry.settings = settings.model_copy(update={"device": "auto"})
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda n, *a: None if n == "torch" else real(n, *a)
    )
    assert registry.resolve_device() == "cpu"


@pytest.mark.requirement("REQ-100")
def test_registry_backend_missing_becomes_unavailable(
    make_registry: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_backend(_self: Any, _weights: Any, _device: str) -> None:
        raise BackendUnavailableError("torch não instalado; instale o extra 'ml'")

    monkeypatch.setattr(analysis_fakes.Brightness, "load", no_backend)
    registry = make_registry(manifest_dict("fake-bright", "Brightness"))
    with pytest.raises(ModelUnavailableError, match="extra 'ml'"):
        registry.load("fake-bright")
    assert registry.loaded_ids() == []


@pytest.mark.requirement("REQ-100")
def test_instantiate_requires_adapter_name(make_registry: Any) -> None:
    registry = make_registry(manifest_dict("fake-bright", "Brightness"))
    manifest = registry.manifest("fake-bright").model_copy(update={"adapter": ""})
    with pytest.raises(ModelUnavailableError, match="não nomeia"):
        _instantiate(manifest)


@pytest.mark.requirement("REQ-100")
def test_safe_extract_rejects_unknown_archive_kind(tmp_path: Path) -> None:
    archive = tmp_path / "x.bin"
    archive.write_bytes(b"x")
    with pytest.raises(ManifestError, match="desconhecido"):
        safe_extract(archive, tmp_path / "out", "rar")


@pytest.mark.requirement("REQ-105")
def test_pin_rejects_manifest_without_id(tmp_path: Path) -> None:
    path = tmp_path / "m.yaml"
    path.write_text(yaml.safe_dump({"name": "sem id"}), encoding="utf-8")
    with pytest.raises(ManifestError, match="sem id"):
        pin(path, weights_dir=tmp_path / "w")


@pytest.mark.requirement("REQ-100")
def test_unload_all_clears_cache(make_registry: Any) -> None:
    registry = make_registry(manifest_dict("fake-bright", "Brightness"))
    registry.load("fake-bright")
    assert registry.loaded_ids() == ["fake-bright"]
    registry.unload_all()
    assert registry.loaded_ids() == []
    assert len(WEIGHT_BYTES) > 0
