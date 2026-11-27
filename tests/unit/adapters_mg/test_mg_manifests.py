"""Manifestos de mamografia: sem limiar do cartão, rotas, Mirai planejado (REQ-143)."""

from __future__ import annotations

from pathlib import Path

import pytest

from radiologyai.analysis.manifest import ModelManifest, load_manifest
from radiologyai.errors import ManifestError

MANIFESTS = Path(__file__).resolve().parents[3] / "src/radiologyai/analysis/manifests"


def _load(model_id: str) -> ModelManifest:
    manifest, _ = load_manifest(MANIFESTS / f"{model_id}.yaml")
    return manifest


@pytest.mark.requirement("REQ-143")
def test_mg_mammoscreen_manifest() -> None:
    m = _load("mammoscreen")
    assert m.status == "available"
    assert m.is_pinned
    assert m.depends_on == ("mammo-crop",)
    assert m.operating_points == ()  # limiares do cartão NÃO entram
    assert {spec.key for spec in m.labels} == {
        "malignancy_left",
        "malignancy_right",
        "density_left",
        "density_right",
    }
    for side in ("left", "right"):
        assert m.label(f"malignancy_{side}").score_kind == "sigmoid"
        assert m.label(f"malignancy_{side}").condition_id == "mg.breast_cancer"
        dens = m.label(f"density_{side}")
        assert (dens.kind, dens.score_kind, dens.classes) == (
            "categorical",
            "softmax",
            tuple("ABCD"),
        )
    assert "cbis-ddsm" in m.training_data


@pytest.mark.requirement("REQ-143")
def test_mg_crop_and_monai_manifests() -> None:
    crop = _load("mammo-crop")
    assert crop.effective_route == "dependency"
    assert crop.labels == ()
    monai = _load("monai-breast-density")
    assert monai.effective_route == "research"
    assert monai.visibility_default == "research"
    assert monai.label("density").score_kind == "sigmoid"
    for m in (crop, monai):
        assert m.is_pinned
        assert m.license.code == "Apache-2.0"


@pytest.mark.requirement("REQ-143")
def test_mg_mirai_is_planned_with_reason() -> None:
    m = _load("mirai")
    assert m.status == "planned"
    assert m.adapter is None
    assert m.effective_route == "none"
    assert m.planned_reason is not None
    assert "onconet" in m.planned_reason
    assert "dcmj2pnm" in m.planned_reason


@pytest.mark.requirement("REQ-143")
def test_mg_mammoscreen_adapter_validates_manifest() -> None:
    from radiologyai.analysis.adapters.mg_mammoscreen import MammoscreenAdapter

    good = _load("mammoscreen")
    MammoscreenAdapter(good)
    no_dep = good.model_copy(update={"depends_on": ()})
    with pytest.raises(ManifestError, match="mammo-crop"):
        MammoscreenAdapter(no_dep)
    fewer = good.model_copy(update={"labels": good.labels[:2]})
    with pytest.raises(ManifestError, match="rótulos"):
        MammoscreenAdapter(fewer)


@pytest.mark.requirement("REQ-143")
def test_mg_monai_adapter_validates_manifest() -> None:
    from radiologyai.analysis.adapters.mg_monai_density import MonaiBreastDensityAdapter

    good = _load("monai-breast-density")
    MonaiBreastDensityAdapter(good)
    spec = good.label("density").model_copy(update={"score_kind": "softmax"})
    with pytest.raises(ManifestError, match="sigmoid"):
        MonaiBreastDensityAdapter(good.model_copy(update={"labels": (spec,)}))


@pytest.mark.requirement("REQ-143")
def test_mg_routing_derives_mg_models() -> None:
    from radiologyai.analysis.registry import Registry
    from radiologyai.analysis.routing import load_routing
    from radiologyai.catalog.schema import ExamType

    table = load_routing(Registry(manifest_dir=MANIFESTS, weights_dir=Path("/nonexistent")))
    main = table.models_for(ExamType.MG, include_research=False)
    research = table.models_for(ExamType.MG, include_research=True)
    assert "mammoscreen" in main
    assert "monai-breast-density" not in main
    assert "monai-breast-density" in research
    assert "mirai" not in research
