"""Manifestos de TC/RM: válidos, pinados, limiares com origem, planejados com motivo."""

from __future__ import annotations

from pathlib import Path

import pytest

from radiologyai.analysis.adapters.ct_biomarkers import BIOMARKERS, CTBiomarkerAdapter
from radiologyai.analysis.adapters.ct_totalseg import TotalSegConfig
from radiologyai.analysis.manifest import load_manifest
from radiologyai.analysis.registry import PACKAGE_MANIFEST_DIR, Registry
from radiologyai.config.settings import Settings

MINE = sorted(
    [
        *PACKAGE_MANIFEST_DIR.glob("totalseg-*.yaml"),
        PACKAGE_MANIFEST_DIR / "ct-biomarkers.yaml",
        PACKAGE_MANIFEST_DIR / "monai-brats.yaml",
        PACKAGE_MANIFEST_DIR / "lst-ai.yaml",
    ]
)
CHECKPOINTS = Path("/root/medai-cache/weights/totalseg/checkpoints.sha256")


def _load(name: str):
    return load_manifest(PACKAGE_MANIFEST_DIR / f"{name}.yaml")[0]


def test_expected_manifests_exist():
    names = {p.stem for p in MINE}
    assert {
        "totalseg-total-3mm",
        "totalseg-cerebral-bleed",
        "ct-biomarkers",
        "monai-brats",
        "lst-ai",
        "totalseg-total-mr",
    } <= names


@pytest.mark.requirement("REQ-130")
@pytest.mark.parametrize("path", MINE, ids=lambda p: p.stem)
def test_manifest_valid_and_pinned(path: Path):
    manifest, _ = load_manifest(path)
    if manifest.status == "available":
        assert manifest.is_pinned
        if manifest.id.startswith("totalseg-"):
            cfg = TotalSegConfig.model_validate(manifest.config)
            keys = {w.key for w in manifest.weights}
            for spec in cfg.datasets:
                assert {spec.checkpoint, spec.plans, spec.dataset} <= keys
            assert manifest.license.commercial_use == "yes"
            assert any("not a medical device" in lim for lim in manifest.limitations)


@pytest.mark.requirement("REQ-134")
def test_every_derived_label_has_sourced_operating_point():
    for path in MINE:
        manifest, _ = load_manifest(path)
        for label in manifest.labels:
            points = manifest.points_for(label.key)
            assert points, f"{manifest.id}:{label.key} sem ponto de operação"
            for point in points:
                assert (
                    "doi:" in point.source
                    or "USPSTF" in point.source
                    or "PRÉ-ESPECIFICADA" in point.source
                ), point.source


@pytest.mark.requirement("REQ-134")
def test_biomarker_thresholds_match_verified_evidence():
    manifest = _load("ct-biomarkers")
    got = {
        (op.label, op.point.role): (op.point.threshold, op.point.direction)
        for op in manifest.operating_points
    }
    assert got == {
        ("osteoporosis_l1", "rule_in"): (110.0, "le"),
        ("osteoporosis_l1", "rule_out"): (160.0, "le"),
        ("steatosis_unenhanced", "guideline"): (40.0, "le"),
        ("steatosis_liver_spleen", "balanced"): (-7.0, "le"),
        ("steatosis_portal_venous", "rule_in"): (80.0, "le"),
        ("steatosis_portal_venous", "rule_out"): (90.0, "le"),
        ("emphysema_laa950", "guideline"): (5.0, "ge"),
        ("abdominal_aortic_aneurysm", "guideline"): (30.0, "ge"),
        ("ascending_aorta_dilation", "guideline"): (40.0, "ge"),
        ("splenomegaly", "guideline"): (314.47, "ge"),
    }
    emitted = {key for keys in BIOMARKERS.values() for key in keys}
    for label in manifest.labels:
        assert label.measurement in emitted
    CTBiomarkerAdapter(manifest)  # config coerente com o adaptador


@pytest.mark.requirement("REQ-139")
@pytest.mark.parametrize("name", ["monai-brats", "lst-ai"])
def test_mr_models_planned_with_reason(name):
    manifest = _load(name)
    assert manifest.status == "planned"
    assert manifest.planned_reason
    assert "Não roda de verdade" in manifest.planned_reason
    assert manifest.adapter is None
    assert manifest.weights == ()


@pytest.mark.requirement("REQ-139")
def test_total_mr_measures_complete_organs_only():
    cfg = TotalSegConfig.model_validate(_load("totalseg-total-mr").config)
    assert cfg.volumes
    assert all(v.require_complete and v.absent == "abstain" for v in cfg.volumes)


def test_registry_loads_with_dependencies(tmp_path):
    registry = Registry(weights_dir=tmp_path, settings=Settings(weights_dir=tmp_path))
    order = registry.execution_order(["ct-biomarkers", "totalseg-ventricle-parts"])
    assert order.index("totalseg-total-3mm") < order.index("ct-biomarkers")


@pytest.mark.requirement("REQ-130")
@pytest.mark.skipif(not CHECKPOINTS.is_file(), reason="cache de pesos do TotalSegmentator ausente")
def test_pinned_checkpoints_match_independent_download_record():
    """O sha256 fixado confere com o registro feito no download (checkpoints.sha256)."""
    recorded = {}
    for line in CHECKPOINTS.read_text().splitlines():
        digest, rel = line.split(maxsplit=1)
        recorded[rel.lstrip("./").split("/")[0]] = digest
    checked = 0
    for path in MINE:
        manifest, _ = load_manifest(path)
        for w in manifest.weights:
            folder = w.filename.split("__")[0]
            if w.filename.endswith("checkpoint_final.pth") and folder in recorded:
                assert w.sha256 == recorded[folder], w.filename
                checked += 1
    assert checked >= 10
