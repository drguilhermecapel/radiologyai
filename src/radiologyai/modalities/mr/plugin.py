"""Plugin de MR — declarado, sem modelo treinado ou validado.

Existe para que o contrato multimodal seja provado desde a Fase 1 e para que o
caminho '3d' exista no sistema de tipos. Nenhuma predição é produzida.
"""

from __future__ import annotations

from typing import ClassVar

from radiologyai.modalities.base import Dimensionality, UnimplementedModality


class MRPlugin(UnimplementedModality):
    """MR — pendente das fases descritas em ROADMAP.md §5."""

    code: ClassVar[str] = "MR"
    accepted_modalities: ClassVar[frozenset[str]] = frozenset({"MR"})
    dimensionality: ClassVar[Dimensionality] = "3d"
