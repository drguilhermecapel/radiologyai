"""Mamografia REAL (amostra CMMD, CC BY 4.0) com pesos reais; pula sem dados/pesos.

Rode sob o lock pesado::

    /root/medai-cache/bin/heavy .venv/bin/python -m pytest tests/unit/adapters_mg -m slow
"""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path
from typing import Any

import pytest
from mg_support import CMMD, WEIGHTS

from radiologyai.analysis.routing import role_of

PATIENT = CMMD / "D2-0424"  # 4 incidências; mama esquerda maligna por biópsia
HAVE_DATA = PATIENT.is_dir()
HAVE_MODELS = (
    HAVE_DATA
    and all((WEIGHTS / m).is_dir() for m in ("mammoscreen", "mammo-crop", "monai-breast-density"))
    and all(importlib.util.find_spec(m) for m in ("torch", "timm", "cv2", "torchvision"))
)


@pytest.mark.requirement("REQ-141")
@pytest.mark.skipif(not HAVE_DATA, reason="amostra CMMD ausente")
def test_mg_cmmd_header_parsing() -> None:
    from radiologyai.modalities.mg.loader import load_mammogram

    images = [load_mammogram(f) for f in sorted(PATIENT.glob("*.dcm"))]
    roles = sorted(r for r in (role_of(im) for im in images) if r)
    assert roles == ["L-CC", "L-MLO", "R-CC", "R-MLO"]
    for image in images:
        voi = image.extra["dicom_voi"]
        assert voi["function"] == "SIGMOID"  # CMMD grava janela sigmoide
        assert voi["window_center"]
        assert voi["window_width"]
        assert image.extra.get("laterality_source") is None  # veio de ImageLaterality


@pytest.mark.requirement("REQ-143")
@pytest.mark.slow
@pytest.mark.skipif(not HAVE_MODELS, reason="pesos MG ou amostra CMMD ausentes")
def test_mg_cmmd_pipeline_real_models() -> None:
    import torch

    from radiologyai.analysis.artifacts import ArtifactStore
    from radiologyai.analysis.pipeline import analyze
    from radiologyai.analysis.registry import Registry
    from radiologyai.analysis.types import ExamInputs
    from radiologyai.catalog.catalog import Catalog
    from radiologyai.catalog.schema import ExamType
    from radiologyai.modalities.mg.loader import load_mammogram

    torch.set_num_threads(4)
    images = tuple(load_mammogram(f) for f in sorted(PATIENT.glob("*.dcm")))
    with tempfile.TemporaryDirectory() as tmp:
        result = analyze(
            ExamInputs(images=images, exam_type=ExamType.MG),
            registry=Registry(weights_dir=WEIGHTS),
            catalog=Catalog.load(),
            store=ArtifactStore(Path(tmp)),
            model_ids=["mammoscreen", "monai-breast-density"],
            include_research=True,
        )
    status = {r.model_id: r.status for r in result.runs}
    assert status == {
        "mammo-crop": "completed",
        "mammoscreen": "completed",
        "monai-breast-density": "completed",
    }, result.runs
    by_label: dict[str, Any] = {f.label: f for f in result.findings}
    assert set(by_label) == {
        "malignancy_left",
        "malignancy_right",
        "density_left",
        "density_right",
        "density",
    }
    for side in ("left", "right"):
        f = by_label[f"malignancy_{side}"]
        assert 0.0 <= f.raw_score <= 1.0
        assert f.decision == "score_only"  # sem ponto de operação com origem
        d = by_label[f"density_{side}"]
        assert d.category in ("A", "B", "C", "D")
        assert sum(d.class_scores.values()) == pytest.approx(1.0, abs=1e-4)
    assert by_label["density"].category in ("A", "B", "C", "D")
