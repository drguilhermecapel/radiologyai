"""Registro: integridade dos pesos, modelos planejados, pull/pin, extração segura, LRU."""

from __future__ import annotations

import hashlib
import io
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import analysis_fakes
import httpx
import pytest
import yaml
from analysis_support import WEIGHT_BYTES, WEIGHT_SHA, manifest_dict

from radiologyai.analysis.registry import Registry, pin, safe_extract
from radiologyai.config.settings import Settings
from radiologyai.errors import (
    IntegrityCheckError,
    ManifestError,
    ModelUnavailableError,
    WeightsIntegrityError,
)


@pytest.mark.requirement("REQ-100")
def test_load_returns_adapter_with_verified_weights(make_registry: Any) -> None:
    registry = make_registry(manifest_dict("fake-bright", "Brightness"))
    adapter = registry.load("fake-bright")
    assert isinstance(adapter, analysis_fakes.Brightness)
    assert adapter.loaded_from is not None
    assert dict(adapter.loaded_from.sha256) == {"main": WEIGHT_SHA}
    assert adapter.loaded_device == "cpu"
    assert registry.load("fake-bright") is adapter  # cache


@pytest.mark.requirement("REQ-100")
def test_missing_weights_make_the_model_unavailable(make_registry: Any) -> None:
    registry = make_registry(manifest_dict("fake-bright", "Brightness"), with_weights=False)
    with pytest.raises(ModelUnavailableError, match="pull"):
        registry.load("fake-bright")
    assert registry.weight_states("fake-bright")[0].state == "missing"


@pytest.mark.requirement("REQ-100")
def test_tampered_weights_fail_the_integrity_check(make_registry: Any) -> None:
    registry = make_registry(manifest_dict("fake-bright", "Brightness"))
    (registry.model_dir("fake-bright") / "main.bin").write_bytes(b"adulterado")
    with pytest.raises(IntegrityCheckError) as info:
        registry.load("fake-bright")
    assert isinstance(info.value, WeightsIntegrityError)
    assert isinstance(info.value, ModelUnavailableError)
    assert registry.weight_states("fake-bright")[0].state == "mismatch"


@pytest.mark.requirement("REQ-105")
def test_available_manifest_must_be_pinned(make_registry: Any) -> None:
    data = manifest_dict("fake-bright", "Brightness")
    data["weights"][0]["sha256"] = ""
    with pytest.raises(ManifestError, match="sem sha256"):
        make_registry(data)


@pytest.mark.requirement("REQ-100")
def test_planned_models_can_never_be_loaded_or_pulled(make_registry: Any) -> None:
    data = manifest_dict(
        "fake-planned",
        "Brightness",
        status="planned",
        planned_reason="licença dos pesos não declarada",
    )
    data["weights"][0]["sha256"] = ""
    registry = make_registry(data)
    with pytest.raises(ModelUnavailableError, match="licença dos pesos"):
        registry.load("fake-planned")
    with pytest.raises(ModelUnavailableError):
        registry.pull("fake-planned")


@pytest.mark.requirement("REQ-100")
def test_noncommercial_models_respect_configuration(make_registry: Any, settings: Settings) -> None:
    data = manifest_dict("fake-nc", "Brightness")
    data["license"]["commercial_use"] = "no"
    registry = make_registry(data)
    assert registry.load("fake-nc") is not None
    assert registry.provenance("fake-nc").commercial_use == "no"
    registry.unload_all()
    registry.settings = settings.model_copy(update={"allow_noncommercial_models": False})
    with pytest.raises(ModelUnavailableError, match="não comercial"):
        registry.load("fake-nc")


@pytest.mark.requirement("REQ-105")
@pytest.mark.parametrize(
    "adapter",
    [
        "os:system",
        "radiologyai.analysis.types:Finding",
        "analysis_fakes:Brightness",
        "radiologyai.analysis.adapters.x",
    ],
)
def test_adapter_must_live_in_the_adapters_package(make_registry: Any, adapter: str) -> None:
    with pytest.raises(ManifestError, match="adapter"):
        make_registry(manifest_dict("fake-x", "Brightness", adapter=adapter))


def test_adapter_class_must_be_a_model_adapter(make_registry: Any) -> None:
    registry = make_registry(manifest_dict("fake-x", "Brightness", adapter=_fake("Path")))
    with pytest.raises((ManifestError, ModelUnavailableError)):
        registry.load("fake-x")


def _fake(name: str) -> str:
    return f"radiologyai.analysis.adapters._test_fakes:{name}"


def test_missing_adapter_module_is_unavailable(make_registry: Any) -> None:
    registry = make_registry(
        manifest_dict("fake-x", "B", adapter="radiologyai.analysis.adapters.nope:B")
    )
    with pytest.raises(ModelUnavailableError, match="inexistente"):
        registry.load("fake-x")


def test_dependency_cycles_are_rejected(make_registry: Any) -> None:
    a = manifest_dict("fake-a", "Brightness", depends_on=["fake-b"])
    b = manifest_dict("fake-b", "Brightness", depends_on=["fake-a"])
    with pytest.raises(ManifestError, match="ciclo"):
        make_registry(a, b)


def test_unknown_dependency_is_rejected(tmp_path: Path, settings: Settings) -> None:
    d = tmp_path / "m"
    d.mkdir()
    data = manifest_dict("fake-a", "Brightness", depends_on=["fake-missing"])
    (d / "fake-a.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ManifestError, match="desconhecido"):
        Registry(manifest_dir=d, weights_dir=tmp_path / "w", settings=settings)


def test_execution_order_is_topological(make_registry: Any) -> None:
    registry = make_registry(
        manifest_dict("fake-a", "Brightness"),
        manifest_dict("fake-b", "Brightness", depends_on=["fake-a"]),
        manifest_dict("fake-c", "Brightness", depends_on=["fake-b", "fake-a"]),
    )
    assert registry.execution_order(["fake-c"]) == ["fake-a", "fake-b", "fake-c"]


def test_file_name_must_match_id(tmp_path: Path, settings: Settings) -> None:
    d = tmp_path / "m"
    d.mkdir()
    (d / "other.yaml").write_text(
        yaml.safe_dump(manifest_dict("fake-a", "Brightness")), encoding="utf-8"
    )
    with pytest.raises(ManifestError, match="nome do arquivo"):
        Registry(manifest_dir=d, weights_dir=tmp_path / "w", settings=settings)


def test_lru_evicts_and_unloads(settings: Settings, make_registry: Any) -> None:
    registry = make_registry(
        manifest_dict("fake-a", "Brightness"), manifest_dict("fake-b", "Brightness")
    )
    registry.settings = settings.model_copy(update={"max_loaded_models": 1})
    first = registry.load("fake-a")
    registry.load("fake-b")
    assert registry.loaded_ids() == ["fake-b"]
    assert first.unloaded is True  # type: ignore[attr-defined]


def test_provenance_carries_hashes_and_licence(make_registry: Any) -> None:
    registry = make_registry(manifest_dict("fake-bright", "Brightness"))
    prov = registry.provenance("fake-bright")
    assert prov.weights_sha256 == {"main": WEIGHT_SHA}
    assert len(prov.manifest_sha256) == 64
    assert prov.license_code == "Apache-2.0"
    assert prov.runtime["device"] == "cpu"


# --------------------------------------------------------------------------- #
# pull / pin
# --------------------------------------------------------------------------- #
def _client(payloads: dict[str, bytes]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        body = payloads.get(str(request.url))
        if body is None:
            return httpx.Response(404)
        return httpx.Response(200, content=body)

    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.mark.requirement("REQ-100")
def test_pull_downloads_and_verifies(make_registry: Any) -> None:
    registry = make_registry(manifest_dict("fake-bright", "Brightness"), with_weights=False)
    url = "https://weights.invalid/fake-bright.bin"
    result = registry.pull("fake-bright", client=_client({url: WEIGHT_BYTES}))
    assert result == {"main": (WEIGHT_SHA, len(WEIGHT_BYTES))}
    assert (registry.model_dir("fake-bright") / "main.bin").read_bytes() == WEIGHT_BYTES
    assert registry.load("fake-bright") is not None


@pytest.mark.requirement("REQ-100")
def test_pull_rejects_and_deletes_tampered_download(make_registry: Any) -> None:
    registry = make_registry(manifest_dict("fake-bright", "Brightness"), with_weights=False)
    url = "https://weights.invalid/fake-bright.bin"
    with pytest.raises(IntegrityCheckError):
        registry.pull("fake-bright", client=_client({url: b"outra coisa"}))
    leftovers = list(registry.model_dir("fake-bright").iterdir())
    assert leftovers == []


@pytest.mark.requirement("REQ-100")
def test_pin_writes_real_hash_from_bytes(tmp_path: Path) -> None:
    data = manifest_dict("fake-bright", "Brightness")
    data["weights"][0]["sha256"] = ""
    data["weights"][0]["bytes"] = 0
    path = tmp_path / "fake-bright.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    url = "https://weights.invalid/fake-bright.bin"
    pins = pin(path, tmp_path / "w", client=_client({url: WEIGHT_BYTES}))
    assert pins == {"main": (WEIGHT_SHA, len(WEIGHT_BYTES))}
    written = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert written["weights"][0]["sha256"] == WEIGHT_SHA
    assert written["weights"][0]["bytes"] == len(WEIGHT_BYTES)


def test_pin_from_local_file_without_network(tmp_path: Path) -> None:
    data = manifest_dict("fake-bright", "Brightness")
    data["weights"][0]["sha256"] = ""
    path = tmp_path / "fake-bright.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    local = tmp_path / "w" / "fake-bright" / "main.bin"
    local.parent.mkdir(parents=True)
    local.write_bytes(WEIGHT_BYTES)
    assert pin(path, tmp_path / "w", download=False)["main"][0] == WEIGHT_SHA


def test_pin_refuses_to_overwrite_a_different_pinned_hash(tmp_path: Path) -> None:
    data = manifest_dict("fake-bright", "Brightness")
    path = tmp_path / "fake-bright.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    local = tmp_path / "w" / "fake-bright" / "main.bin"
    local.parent.mkdir(parents=True)
    local.write_bytes(b"outros bytes")
    with pytest.raises(IntegrityCheckError):
        pin(path, tmp_path / "w", download=False)


def test_pin_without_file_and_without_download_fails(tmp_path: Path) -> None:
    data = manifest_dict("fake-bright", "Brightness")
    data["weights"][0]["sha256"] = ""
    path = tmp_path / "fake-bright.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(ModelUnavailableError, match="ausente"):
        pin(path, tmp_path / "w", download=False)


# --------------------------------------------------------------------------- #
# arquivos compactados
# --------------------------------------------------------------------------- #
def _archive_manifest(kind: str, payload: bytes) -> dict[str, Any]:
    data = manifest_dict("fake-arch", "Brightness")
    data["weights"][0].update(
        {
            "filename": f"main.{kind}",
            "extract": kind,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
    )
    return data


def _zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    return buf.getvalue()


def test_zip_weights_are_extracted_after_verification(make_registry: Any) -> None:
    payload = _zip({"model/weights.pt": b"abc"})
    registry = make_registry(_archive_manifest("zip", payload), with_weights=False)
    target = registry.model_dir("fake-arch") / "main.zip"
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)
    resolved = registry.verify("fake-arch")
    assert (resolved.path("main") / "model" / "weights.pt").read_bytes() == b"abc"


@pytest.mark.requirement("REQ-107")
def test_zip_traversal_is_refused(tmp_path: Path) -> None:
    archive = tmp_path / "evil.zip"
    archive.write_bytes(_zip({"../../escape.txt": b"x"}))
    with pytest.raises(IntegrityCheckError, match="escapa"):
        safe_extract(archive, tmp_path / "out", "zip")
    assert not (tmp_path / "escape.txt").exists()


@pytest.mark.requirement("REQ-107")
def test_tar_links_and_absolute_paths_are_refused(tmp_path: Path) -> None:
    link_tar = tmp_path / "link.tar"
    with tarfile.open(link_tar, "w") as tf:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tf.addfile(info)
    with pytest.raises(IntegrityCheckError, match="link"):
        safe_extract(link_tar, tmp_path / "out1", "tar")

    abs_tar = tmp_path / "abs.tar"
    with tarfile.open(abs_tar, "w") as tf:
        info = tarfile.TarInfo("/tmp/absolute.txt")
        info.size = 1
        tf.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(IntegrityCheckError):
        safe_extract(abs_tar, tmp_path / "out2", "tar")


def test_package_manifests_are_valid(settings: Settings) -> None:
    """Todo manifesto distribuído carrega, e todo disponível está fixado."""
    registry = Registry(settings=settings)
    for manifest in registry.manifests():
        if manifest.status == "available":
            assert manifest.is_pinned, manifest.id
        else:
            assert manifest.planned_reason
