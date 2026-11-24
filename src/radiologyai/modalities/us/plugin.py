"""Plugin de US — declarado, sem modelo treinado ou validado.

Existe para que o contrato multimodal seja provado desde a Fase 1 e para que o
caminho 'cine' exista no sistema de tipos. Nenhuma predição é produzida.
"""

from __future__ import annotations

from typing import ClassVar

from radiologyai.modalities.base import Dimensionality, UnimplementedModality


class USPlugin(UnimplementedModality):
    """US — pendente das fases descritas em ROADMAP.md §5."""

    code: ClassVar[str] = "US"
    accepted_modalities: ClassVar[frozenset[str]] = frozenset({"US"})
    dimensionality: ClassVar[Dimensionality] = "cine"
