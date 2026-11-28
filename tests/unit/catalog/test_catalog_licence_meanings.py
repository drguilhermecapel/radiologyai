"""Uso comercial coerente com os dados de treino e significado honesto dos níveis."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from radiologyai.catalog.schema import LEVEL_MEANING_PT, EvidenceLevel
from radiologyai.evidence.leakage import NONCOMMERCIAL_TERMS

MANIFESTS = Path(__file__).resolve().parents[3] / "src" / "radiologyai" / "analysis" / "manifests"


def _manifests() -> list[dict]:
    return [yaml.safe_load(p.read_text(encoding="utf-8")) for p in sorted(MANIFESTS.glob("*.yaml"))]


@pytest.mark.requirement("REQ-188")
def test_noncommercial_training_data_means_commercial_use_no():
    offenders = [
        (
            m["id"],
            m["license"]["commercial_use"],
            sorted(set(m["training_data"]) & set(NONCOMMERCIAL_TERMS)),
        )
        for m in _manifests()
        if set(m.get("training_data") or ()) & set(NONCOMMERCIAL_TERMS)
        and m["license"]["commercial_use"] != "no"
    ]
    assert offenders == []


@pytest.mark.requirement("REQ-188")
@pytest.mark.parametrize(
    "model_id",
    ["xrv-ae-ood", "medsam-vit-base", "xrv-densenet121-res224-mimic_ch", "chexzero"],
)
def test_reviewed_manifests_are_non_commercial(model_id):
    data = yaml.safe_load((MANIFESTS / f"{model_id}.yaml").read_text(encoding="utf-8"))
    assert data["license"]["commercial_use"] == "no"


@pytest.mark.requirement("REQ-189")
def test_level_meanings_never_say_validated():
    assert set(LEVEL_MEANING_PT) == {lv.value for lv in EvidenceLevel}
    for text in LEVEL_MEANING_PT.values():
        assert not re.search(r"validad|valida[çc][ãa]o", text, re.I), text


@pytest.mark.requirement("REQ-189")
def test_level_c_meaning_covers_every_route_to_c():
    c = LEVEL_MEANING_PT["C"]
    assert "hold-out interno" in c
    assert "externo" in c
    assert "seleção do modelo" in c
    assert "padrão de" in c
