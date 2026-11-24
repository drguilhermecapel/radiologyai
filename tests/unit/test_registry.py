from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from medai.core.errors import IntegrityError, ManifestError, ModelUnavailable
from medai.core.settings import Settings
from medai.models.manifest import ModelManifest
from medai.models.registry import Registry, write_pins
from tests.conftest import WEIGHT_BYTES, WEIGHT_SHA, manifest_dict
from tests.support import fakes


def test_load_returns_adapter_with_verified_weights(make_registry: Any) -> None:
    registry = make_registry(manifest_dict("fake-bright", "Brightness"))
    adapter = registry.load("fake-bright")
    assert isinstance(adapter, fakes.Brightness)
    assert adapter.loaded_from is not None
    assert adapter.loaded_from.sha256 == {"main": WEIGHT_SHA}
    assert registry.load("fake-bright") is adapter  # cached


def test_missing_weights_make_the_model_unavailable(make_registry: Any) -> None:
    registry = make_registry(manifest_dict("fake-bright", "Brightness"), with_weights=False)
    with pytest.raises(ModelUnavailable, match="medai models pull"):
        registry.load("fake-bright")


def test_tampered_weights_fail_the_integrity_check(make_registry: Any) -> None:
    registry = make_registry(manifest_dict("fake-bright", "Brightness"))
    (registry.model_dir("fake-bright") / "main.bin").write_bytes(b"tampered")
    with pytest.raises(IntegrityError):
        registry.load("fake-bright")
    assert registry.weight_states("fake-bright")[0].state == "mismatch"


def test_unpinned_weights_cannot_be_loaded(make_registry: Any) -> None:
    data = manifest_dict("fake-bright", "Brightness")
    data["weights"][0]["sha256"] = ""
    registry = make_registry(data)
    with pytest.raises(ModelUnavailable, match="no pinned sha256"):
        registry.load("fake-bright")


def test_planned_models_can_never_be_loaded(make_registry: Any) -> None:
    data = manifest_dict(
        "fake-planned", "Brightness", status="planned", planned_reason="licence not stated"
    )
    registry = make_registry(data)
    with pytest.raises(ModelUnavailable, match="licence not stated"):
        registry.load("fake-planned")
    with pytest.raises(ModelUnavailable):
        registry.pull("fake-planned")


def test_noncommercial_models_respect_configuration(make_registry: Any, settings: Settings) -> None:
    data = manifest_dict("fake-nc", "Brightness")
    data["license"]["commercial_use"] = "no"
    registry = make_registry(data)
    assert registry.load("fake-nc") is not None
    registry.unload_all()
    registry.settings = settings.model_copy(update={"allow_noncommercial_models": False})
    with pytest.raises(ModelUnavailable, match="non-commercial"):
        registry.load("fake-nc")


@pytest.mark.parametrize(
    "adapter",
    [
        "os:system",
        "medai.core.types:Finding",
        "tests.support.fakes:Brightness",
        "medai.models.adapters.x",
    ],
)
def test_adapter_must_live_in_the_adapters_package(adapter: str) -> None:
    with pytest.raises(ValueError, match="adapter"):
        ModelManifest.model_validate(manifest_dict("fake-evil", "Brightness", adapter=adapter))


def test_manifest_rejects_bad_hash_unknown_fields_and_bad_points() -> None:
    bad_hash = manifest_dict("fake-a", "Brightness")
    bad_hash["weights"][0]["sha256"] = "a3b4g9h0"  # not hex: the legacy registry had these
    with pytest.raises(ValueError, match="sha256"):
        ModelManifest.model_validate(bad_hash)
    with pytest.raises(ValueError, match="accuracy"):
        ModelManifest.model_validate(manifest_dict("fake-a", "Brightness", accuracy=0.95))
    point = {
        "label": "nope",
        "point": {
            "name": "p",
            "role": "balanced",
            "threshold": 0.5,
            "on": "sigmoid",
            "source": "t",
        },
    }
    with pytest.raises(ValueError, match="unknown label"):
        ModelManifest.model_validate(
            manifest_dict("fake-a", "Brightness", operating_points=[point])
        )
    with pytest.raises(ValueError, match="research"):
        ModelManifest.model_validate(manifest_dict("fake-a", "Brightness", task="zero_shot"))


def test_dependency_order_and_cycles(make_registry: Any, tmp_path: Path) -> None:
    seg = manifest_dict("fake-segmenter", "Segmenter", labels=[])
    derived = manifest_dict("fake-derived", "Derived", labels=[], depends_on=["fake-segmenter"])
    registry = make_registry(seg, derived)
    assert registry.execution_order(["fake-derived"]) == ["fake-segmenter", "fake-derived"]

    a = manifest_dict("fake-a", "Brightness", depends_on=["fake-b"])
    b = manifest_dict("fake-b", "Brightness", depends_on=["fake-a"])
    cyc = tmp_path / "cyc"
    cyc.mkdir()
    for data in (a, b):
        (cyc / f"{data['id']}.yaml").write_text(yaml.safe_dump(data))
    with pytest.raises(ManifestError, match="cycle"):
        Registry(manifest_dir=cyc, weights_dir=tmp_path / "w")


def _client(payload: bytes) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=payload))
    )


def test_pull_verifies_the_download(make_registry: Any) -> None:
    registry = make_registry(manifest_dict("fake-bright", "Brightness"), with_weights=False)
    assert registry.pull("fake-bright", client=_client(WEIGHT_BYTES)) == {
        "main": (WEIGHT_SHA, len(WEIGHT_BYTES))
    }
    assert registry.load("fake-bright") is not None


def test_pull_rejects_and_deletes_a_wrong_download(make_registry: Any) -> None:
    registry = make_registry(manifest_dict("fake-bright", "Brightness"), with_weights=False)
    with pytest.raises(IntegrityError):
        registry.pull("fake-bright", client=_client(b"something else"))
    assert not (registry.model_dir("fake-bright") / "main.bin").exists()


def test_pull_refuses_unpinned_unless_pinning_then_writes_real_hash(make_registry: Any) -> None:
    data = manifest_dict("fake-bright", "Brightness")
    data["weights"][0].update(sha256="", bytes=0)
    registry = make_registry(data, with_weights=False)
    with pytest.raises(ManifestError, match="not pinned"):
        registry.pull("fake-bright", client=_client(WEIGHT_BYTES))
    pins = registry.pull("fake-bright", pin=True, client=_client(WEIGHT_BYTES))
    write_pins(registry.manifest_path("fake-bright"), pins)
    reloaded = Registry(
        manifest_dir=registry.manifest_dir,
        weights_dir=registry.weights_dir,
        settings=registry.settings,
    )
    assert reloaded.manifest("fake-bright").weights[0].sha256 == WEIGHT_SHA
    assert reloaded.load("fake-bright") is not None


def _zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, payload in members.items():
            zf.writestr(name, payload)
    return buf.getvalue()


def test_archives_are_extracted_safely(make_registry: Any, tmp_path: Path) -> None:
    import hashlib

    good = _zip({"model/weights.bin": b"abc"})
    data = manifest_dict("fake-zip", "Brightness")
    data["weights"][0].update(
        filename="w.zip", extract="zip", sha256=hashlib.sha256(good).hexdigest(), bytes=len(good)
    )
    registry = make_registry(data, with_weights=False)
    registry.pull("fake-zip", client=_client(good))
    resolved = registry.verify("fake-zip")
    assert (resolved.files["main"] / "model" / "weights.bin").read_bytes() == b"abc"

    evil = _zip({"../../escape.txt": b"x"})
    data2 = manifest_dict("fake-evil-zip", "Brightness")
    data2["weights"][0].update(
        filename="w.zip", extract="zip", sha256=hashlib.sha256(evil).hexdigest(), bytes=len(evil)
    )
    registry2 = make_registry(data2, with_weights=False)
    registry2.pull("fake-evil-zip", client=_client(evil))
    with pytest.raises(IntegrityError, match="escapes"):
        registry2.verify("fake-evil-zip")
    assert not (tmp_path / "escape.txt").exists()
