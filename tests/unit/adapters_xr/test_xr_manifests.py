"""Manifestos do front de RX: fixados, rótulos = cabeças treinadas, pontos com origem."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

from radiologyai.analysis.manifest import ModelManifest, load_manifest
from radiologyai.analysis.registry import PACKAGE_MANIFEST_DIR

HAS_XRV = importlib.util.find_spec("torchxrayvision") is not None
OWN = sorted(
    [
        *PACKAGE_MANIFEST_DIR.glob("xrv-*.yaml"),
        *PACKAGE_MANIFEST_DIR.glob("xr-probe-*.yaml"),
        *PACKAGE_MANIFEST_DIR.glob("chexzero.yaml"),
    ]
)


def _load(path: Path) -> ModelManifest:
    return load_manifest(path)[0]


def test_front_has_manifests() -> None:
    names = {p.stem for p in OWN}
    assert {"xrv-densenet121-res224-pc", "xrv-view-xinario", "xrv-pspnet-chestxdet"} <= names


@pytest.mark.parametrize("path", OWN, ids=lambda p: p.stem)
@pytest.mark.requirement("REQ-127")
def test_available_manifests_are_pinned(path: Path) -> None:
    manifest = _load(path)
    if manifest.status == "available":
        assert manifest.is_pinned
        assert all(len(w.sha256) == 64 and w.bytes > 0 for w in manifest.weights)


@pytest.mark.parametrize("path", OWN, ids=lambda p: p.stem)
def test_operating_points_have_sources(path: Path) -> None:
    for op in _load(path).operating_points:
        assert op.point.source.strip()


@pytest.mark.skipif(not HAS_XRV, reason="torchxrayvision ausente (extra ml)")
@pytest.mark.parametrize(
    "path",
    [p for p in OWN if p.stem.startswith(("xrv-densenet", "xrv-resnet"))],
    ids=lambda p: p.stem,
)
@pytest.mark.requirement("REQ-121")
def test_xrv_labels_are_trained_heads_with_library_thresholds(path: Path) -> None:
    from radiologyai.analysis.adapters.xr_xrv import trained_heads

    manifest = _load(path)
    heads = trained_heads(str(manifest.config["library_weights"]))
    assert {spec.source_label for spec in manifest.labels} == set(heads)
    for spec in manifest.labels:
        assert spec.score_kind == "logit"
        thr = heads[spec.source_label][1]
        points = {p.on: p for p in manifest.points_for(spec.key)}
        assert points["sigmoid"].threshold == pytest.approx(thr, abs=1e-7)
        assert points["logit"].threshold == pytest.approx(math.log(thr / (1 - thr)), abs=1e-6)
        assert "torchxrayvision op_threshs" in points["sigmoid"].source


@pytest.mark.requirement("REQ-126")
def test_zero_shot_is_research_only_and_noncommercial() -> None:
    manifest = _load(PACKAGE_MANIFEST_DIR / "chexzero.yaml")
    assert manifest.task == "zero_shot"
    assert manifest.effective_route == "research"
    assert manifest.visibility_default == "research"
    assert manifest.license.commercial_use == "no"
    assert not manifest.operating_points


@pytest.mark.requirement("REQ-124")
def test_ctr_guideline_point_is_on_pa_only_measurement() -> None:
    manifest = _load(PACKAGE_MANIFEST_DIR / "xrv-pspnet-chestxdet.yaml")
    spec = manifest.label("cardiomegaly_ctr")
    assert spec.kind == "derived"
    assert spec.measurement == "cardiothoracic_ratio_pa"
    (point,) = manifest.points_for("cardiomegaly_ctr")
    assert point.role == "guideline"
    assert point.threshold == 0.5
    assert "10.3390/jcm10092016" in point.source


def test_package_registry_loads_with_front_manifests(tmp_path: Path) -> None:
    from radiologyai.analysis.registry import Registry
    from radiologyai.config.settings import Settings

    registry = Registry(weights_dir=tmp_path, settings=Settings(weights_dir=tmp_path))
    ids = set(registry.ids())
    assert {p.stem for p in OWN} <= ids
    order = registry.execution_order(["xrv-densenet121-res224-pc"])
    assert order.index("xrv-view-xinario") < order.index("xrv-densenet121-res224-pc")
