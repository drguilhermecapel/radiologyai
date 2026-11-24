"""Modalidade MG (mamografia digital 2D): escopo, VOI e incidência/lateralidade.

``import`` deste pacote é leve (sem numpy/pydicom); :mod:`.loader` e :mod:`.voi`
importam o necessário dentro das funções.
"""

from __future__ import annotations

from radiologyai.modalities.mg.plugin import MGPlugin

__all__ = ["MGPlugin"]
