"""Pré-condições de aquisição: kernel, fase de contraste, kVp, espessura, idade."""

from __future__ import annotations

import pytest
from ctmr_support import ct_meta, meta

from radiologyai.analysis.biomarkers import acquisition as acq
from radiologyai.analysis.types import AbstentionReason


@pytest.mark.requirement("REQ-133")
@pytest.mark.parametrize(
    ("kernel", "expected"),
    [
        ("STANDARD", "smooth"),
        ("SOFT", "smooth"),
        ("B30f", "smooth"),
        ("Br40d\\3", "smooth"),
        ("I31f", "smooth"),
        ("B70f", "sharp"),
        ("Br64", "sharp"),
        ("LUNG", "sharp"),
        ("BONE", "sharp"),
        ("FC07", "smooth"),
        ("FC52", "sharp"),
        ("FC20", "unknown"),
        ("C", "smooth"),
        ("L", "sharp"),
        ("YA", "sharp"),
        ("", "unknown"),
        (None, "unknown"),
        ("KERNEL_X", "unknown"),
    ],
)
def test_classify_kernel(kernel, expected):
    assert acq.classify_kernel(kernel) == expected


@pytest.mark.requirement("REQ-133")
@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, "unknown"),  # tag ausente NÃO prova exame sem contraste
        ({"contrast_agent": ""}, "unknown"),
        ({"contrast_agent": "IOPAMIDOL"}, "contrast_unknown_phase"),
        ({"contrast_phase": "none"}, "none"),
        ({"contrast_phase": "none", "contrast_agent": "OMNIPAQUE"}, "unknown"),
        ({"contrast_phase": "portal_venous", "contrast_agent": "X"}, "portal_venous"),
        ({"contrast_phase": "unknown"}, "unknown"),
        ({"contrast_agent": "W/O_IV_CONTRAST"}, "none_by_tag"),
        ({"contrast_agent": "NONE"}, "none_by_tag"),
        ({"contrast_agent": "SEM CONTRASTE"}, "none_by_tag"),
        ({"contrast_agent": "OMNIPAQUE 350"}, "contrast_unknown_phase"),
        ({"contrast_agent": "NON-IONIC CONTRAST"}, "contrast_unknown_phase"),
        ({"contrast_phase": "none", "contrast_agent": "W/O_IV_CONTRAST"}, "none"),
    ],
)
def test_contrast_phase_only_when_confirmed(overrides, expected):
    assert acq.contrast_phase(meta(**overrides)) == expected


@pytest.mark.requirement("REQ-133")
def test_gates():
    assert acq.require_kvp(ct_meta(kvp=120.0), 120.0).ok
    gate = acq.require_kvp(ct_meta(kvp=80.0), 120.0)
    assert not gate.ok
    assert gate.reason is AbstentionReason.ACQUISITION
    assert not acq.require_thin_slices(ct_meta(slice_thickness_mm=None), 1.5).ok
    assert acq.require_thin_slices(ct_meta(slice_thickness_mm=1.5), 1.5).ok
    assert acq.require_adult(ct_meta(patient_age_years=None)).reason is AbstentionReason.POPULATION
    assert acq.require_smooth_kernel(ct_meta(convolution_kernel="B30f")).ok
    assert not acq.require_phase(ct_meta(contrast_phase=None), "none").ok
    # tag que nega contraste não substitui a confirmação exigida pela fonte
    tag_only = ct_meta(contrast_phase=None, contrast_agent="W/O_IV_CONTRAST")
    assert not acq.require_phase(tag_only, "none").ok


@pytest.mark.requirement("REQ-133")
def test_facts_are_strings_and_mark_absent():
    facts = acq.acquisition_facts(meta())
    assert facts["kvp"] == "ausente"
    assert facts["contrast_phase"] == "unknown"
    assert all(isinstance(v, str) for v in facts.values())
