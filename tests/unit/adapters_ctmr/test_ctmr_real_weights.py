"""Execução REAL do TotalSegmentator (pesos verificados) em amostras públicas.

Pula sem o pacote, sem os pesos pinados em ``MEDAI_CTMR_WEIGHTS`` (padrão
``/root/medai-cache/weights/registry``) ou sem as amostras. Rode sob a trava
pesada: ``/root/medai-cache/bin/heavy .venv/bin/python -m pytest -m slow ...``.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
from cache_paths import MEDAI_CACHE

from radiologyai.analysis.base import ModelInputs
from radiologyai.analysis.registry import Registry
from radiologyai.analysis.types import Modality
from radiologyai.catalog.schema import ExamType
from radiologyai.config.settings import Settings

WEIGHTS = Path(os.environ.get("MEDAI_CTMR_WEIGHTS", str(MEDAI_CACHE / "weights/registry")))
SAMPLES = MEDAI_CACHE / "datasets/samples-ct/totalseg-tests"
CT = SAMPLES / "example_ct_sm.nii.gz"
MR = SAMPLES / "example_mr_sm.nii.gz"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        importlib.util.find_spec("totalsegmentator") is None, reason="TotalSegmentator ausente"
    ),
    pytest.mark.skipif(
        not (WEIGHTS / "totalseg-total-3mm").is_dir(), reason="pesos pinados ausentes"
    ),
    pytest.mark.skipif(not CT.is_file() or not MR.is_file(), reason="amostras ausentes"),
]


@pytest.fixture(scope="module")
def registry():
    reg = Registry(weights_dir=WEIGHTS, settings=Settings(weights_dir=WEIGHTS, device="cpu"))
    yield reg
    reg.unload_all()


@pytest.mark.requirement("REQ-130")
@pytest.mark.requirement("REQ-131")
def test_total_3mm_and_biomarkers_on_real_ct(registry):
    from radiologyai.analysis.imaging import load_nifti

    image = load_nifti(CT, modality=Modality.CT, value_range=(-1024.0, 3071.0))
    total = registry.load("totalseg-total-3mm")
    inputs = ModelInputs(images=(image,), exam_type=ExamType.CT_ABDOMEN)
    assert total.applicability(inputs).applicable
    out = total.predict(inputs)
    (labelmap,) = out.labelmaps
    assert labelmap.array.shape == image.pixels.shape
    present = {labelmap.labels[int(i)] for i in np.unique(labelmap.array) if i}
    assert len(present) >= 10
    assert out.extras["runner"]["totalsegmentator_version"] == "2.18.0"

    bio = registry.load("ct-biomarkers")
    binputs = ModelInputs(
        images=(image,), exam_type=ExamType.CT_ABDOMEN, upstream={"totalseg-total-3mm": out}
    )
    assert bio.applicability(binputs).applicable
    measures = {m.key: m for m in bio.predict(binputs).measurements}
    # amostra NIfTI sem metadados DICOM: tudo que exige fase/kVp/idade se abstém
    for key in ("l1_trabecular_hu", "liver_attenuation_unenhanced_hu", "lung_laa950_percent"):
        assert measures[key].value is None
        assert measures[key].abstention is not None
    for m in measures.values():
        assert (m.value is None) != (m.abstention is None)


@pytest.mark.requirement("REQ-139")
def test_total_mr_on_real_mr(registry):
    from radiologyai.analysis.imaging import load_nifti

    image = load_nifti(MR, modality=Modality.MR)
    adapter = registry.load("totalseg-total-mr")
    inputs = ModelInputs(images=(image,), exam_type=ExamType.MR_BODY)
    assert adapter.applicability(inputs).applicable
    out = adapter.predict(inputs)
    assert out.labelmaps[0].array.shape == image.pixels.shape
    for m in out.measurements:
        assert (m.value is None) != (m.abstention is None)
