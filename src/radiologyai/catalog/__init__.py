"""Catálogo de condições com nível de evidência calculado (A-D).

Este subpacote é leve: importa apenas pydantic. YAML é carregado sob demanda,
dentro das funções de leitura, para que ``import radiologyai.catalog`` não traga
``yaml``, ``numpy`` nem ``torch`` para a memória.
"""

from __future__ import annotations

from radiologyai.catalog.catalog import Catalog, load_protocol
from radiologyai.catalog.schema import (
    CodedConcept,
    Condition,
    ConditionSource,
    EvidenceLevel,
    EvidenceRecord,
    EvidenceSource,
    LevelEntry,
    LevelsFile,
    MeasurementProvenance,
    Metric,
    Protocol,
    Visibility,
    compute_level,
    visibility_for,
)

__all__ = [
    "Catalog",
    "CodedConcept",
    "Condition",
    "ConditionSource",
    "EvidenceLevel",
    "EvidenceRecord",
    "EvidenceSource",
    "LevelEntry",
    "LevelsFile",
    "MeasurementProvenance",
    "Metric",
    "Protocol",
    "Visibility",
    "compute_level",
    "load_protocol",
    "visibility_for",
]
