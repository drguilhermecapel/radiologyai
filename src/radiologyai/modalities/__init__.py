"""Plugins de modalidade — o ponto de extensão da plataforma (ROADMAP §4.1)."""

from __future__ import annotations

from radiologyai.modalities.base import (
    Dimensionality,
    ModalityPlugin,
    UnimplementedModality,
    ValidationReport,
)
from radiologyai.modalities.registry import available_modalities, resolve

__all__ = [
    "Dimensionality",
    "ModalityPlugin",
    "UnimplementedModality",
    "ValidationReport",
    "available_modalities",
    "resolve",
]
