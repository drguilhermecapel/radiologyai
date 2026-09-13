"""Alinhamento de rótulos — o mapeamento perigoso do legado não pode voltar."""

from __future__ import annotations

from radiologyai.modalities.xr.labels import (
    NIH_CXR14_LABELS,
    XRV_NOT_IN_NIH,
    XRV_PATHOLOGIES,
    xrv_to_nih_indices,
)


def test_all_nih_labels_map_to_xrv():
    mapping = xrv_to_nih_indices()
    assert set(mapping) == set(NIH_CXR14_LABELS)


def test_mapping_indices_are_valid():
    for label, idx in xrv_to_nih_indices().items():
        assert XRV_PATHOLOGIES[idx] == label


def test_no_collapsing_of_distinct_findings():
    """O legado mapeava Pneumothorax->pneumonia e Cardiomegaly->normal.

    Cada achado deve ter índice próprio: nenhum colapso é permitido.
    """
    indices = list(xrv_to_nih_indices().values())
    assert len(indices) == len(set(indices))


def test_pneumothorax_is_its_own_finding():
    mapping = xrv_to_nih_indices()
    assert XRV_PATHOLOGIES[mapping["Pneumothorax"]] == "Pneumothorax"
    assert mapping["Pneumothorax"] != mapping["Pneumonia"]


def test_cardiomegaly_is_not_normal():
    mapping = xrv_to_nih_indices()
    assert XRV_PATHOLOGIES[mapping["Cardiomegaly"]] == "Cardiomegaly"


def test_unmapped_pathologies_are_declared_not_dropped():
    for label in XRV_NOT_IN_NIH:
        assert label in XRV_PATHOLOGIES
        assert label not in NIH_CXR14_LABELS
