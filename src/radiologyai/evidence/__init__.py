"""Fábrica de evidência: medição -> registro com proveniência -> ``levels.json``.

Importa apenas pydantic e o catálogo; numpy, pandas e yaml são carregados sob
demanda dentro das funções.
"""

from __future__ import annotations

from radiologyai.evidence.leakage import DATASETS, assess_leakage, lineage
from radiologyai.evidence.levels import build_levels, check_levels, render_levels
from radiologyai.evidence.measure import (
    MeasuredEvidence,
    MeasurementContext,
    judge_agreement,
    judge_detection,
    judge_discrimination,
    judge_segmentation,
    measure_agreement,
    measure_binary,
    measure_detection,
    measure_segmentation,
)
from radiologyai.evidence.store import load_records, write_measured

__all__ = [
    "DATASETS",
    "MeasuredEvidence",
    "MeasurementContext",
    "assess_leakage",
    "build_levels",
    "check_levels",
    "judge_agreement",
    "judge_detection",
    "judge_discrimination",
    "judge_segmentation",
    "lineage",
    "load_records",
    "measure_agreement",
    "measure_binary",
    "measure_detection",
    "measure_segmentation",
    "render_levels",
    "write_measured",
]
