"""Shared fixtures. Fake adapters are injected at test time and never shipped."""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from medai.catalog.catalog import Catalog
from medai.catalog.schema import Condition, LevelEntry, LevelsFile
from medai.core.artifacts import ArtifactStore
from medai.core.settings import Settings
from medai.core.types import CodedConcept, EvidenceLevel, ExamType
from medai.models.registry import Registry
from tests.support import fakes

FAKE_MODULE = "medai.models.adapters._test_fakes"
WEIGHT_BYTES = b"not-a-real-model: integrity-check fixture\n"
WEIGHT_SHA = hashlib.sha256(WEIGHT_BYTES).hexdigest()


@pytest.fixture(autouse=True)
def _register_fake_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, FAKE_MODULE, fakes)


def manifest_dict(model_id: str, adapter_class: str, **overrides: Any) -> dict[str, Any]:
    code = {"scheme": "99MEDAI", "value": "x", "meaning": "x"}
    data: dict[str, Any] = {
        "id": model_id,
        "version": "0.0-test",
        "name": model_id,
        "status": "available",
        "adapter": f"{FAKE_MODULE}:{adapter_class}",
        "task": "classification",
        "capabilities": ["classify"],
        "summary": "test double",
        "weights": [
            {
                "key": "main",
                "url": f"https://weights.invalid/{model_id}.bin",
                "filename": "main.bin",
                "sha256": WEIGHT_SHA,
                "bytes": len(WEIGHT_BYTES),
            }
        ],
        "license": {"code": "test", "weights": "test", "commercial_use": "yes"},
        "applicability": {
            "exam_types": [ExamType.CXR.value],
            "modalities": ["DX", "CR", "MG"],
            "image_kinds": ["projection"],
        },
        "labels": [
            {
                "key": "bright",
                "source_label": "Bright",
                "condition_id": "cxr.bright",
                "code": code,
            }
        ],
    }
    data.update(overrides)
    return data


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(home=tmp_path / "home", device="cpu", max_loaded_models=8)


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "store")


RegistryFactory = Callable[..., Registry]


@pytest.fixture
def make_registry(tmp_path: Path, settings: Settings) -> RegistryFactory:
    """Build a registry from manifest dicts; weights are written unless ``with_weights=False``."""

    def build(*manifests: dict[str, Any], with_weights: bool = True) -> Registry:
        manifest_dir = tmp_path / "manifests"
        manifest_dir.mkdir(exist_ok=True)
        weights_dir = tmp_path / "weights"
        for data in manifests:
            (manifest_dir / f"{data['id']}.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
            if with_weights:
                for weight in data.get("weights", []):
                    target = weights_dir / data["id"] / weight["filename"]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(WEIGHT_BYTES)
        return Registry(manifest_dir=manifest_dir, weights_dir=weights_dir, settings=settings)

    return build


def condition(condition_id: str) -> Condition:
    return Condition(
        id=condition_id,
        name_pt=condition_id,
        name_en=condition_id,
        group="cxr",
        exam_types=[ExamType.CXR],
        system="test",
        code=CodedConcept(scheme="99MEDAI", value=condition_id, meaning=condition_id),
    )


@pytest.fixture
def make_catalog() -> Callable[..., Catalog]:
    """Catalog where each ``(condition, model, label)`` gets the requested level."""

    def build(levels: dict[tuple[str, str, str], EvidenceLevel] | None = None) -> Catalog:
        levels = levels or {}
        conditions = [condition(cid) for cid in sorted({key[0] for key in levels})]
        entries = [
            LevelEntry(
                condition_id=cid,
                model_id=mid,
                label=label,
                level=level,
                visibility="research" if level is EvidenceLevel.D else "main",
                basis=[] if level is EvidenceLevel.D else ["test-record"],
            )
            for (cid, mid, label), level in levels.items()
        ]
        return Catalog(
            conditions, LevelsFile(protocol_id="test", evidence_digest="0" * 64, entries=entries)
        )

    return build
