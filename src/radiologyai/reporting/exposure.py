"""Quais segmentações e medidas podem acompanhar um conjunto de achados.

Segmentações e medidas não têm nível de evidência próprio: o nível é da
condição que um achado decide. Uma máscara ou medida só pode sair do modo
pesquisa (visão padrão, laudo, PDF/SR/SEG/FHIR, exportação ao PACS) quando
está LIGADA a um achado elegível (níveis A-C):

* medida: algum achado elegível do mesmo modelo a referencia
  (``Finding.measurement_ref``);
* segmentação: é origem (``Measurement.derived_from``) de uma medida elegível,
  OU todas as chaves de rótulo dela (``label_keys``) são rótulos de achados
  elegíveis do mesmo modelo.

Qualquer outra máscara ou medida — de modelo só com rótulos de nível D, de
modelo sem rótulo no catálogo, de rótulo misto — é exploratória. Na dúvida, a
regra esconde: nunca promove saída exploratória.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from radiologyai.analysis.types import AnalysisResult

__all__ = ["Exposure", "exposed_artifacts", "main_exposure"]


@dataclass(frozen=True, slots=True)
class Exposure:
    """Índices (no resultado COMPLETO) de segmentações e medidas ligadas aos achados dados."""

    segmentations: frozenset[int]
    measurements: frozenset[int]


def exposed_artifacts(result: AnalysisResult, finding_indices: Iterable[int]) -> Exposure:
    """Segmentações e medidas ligadas aos achados ``finding_indices`` (índices no resultado)."""
    findings = [result.findings[i] for i in finding_indices if 0 <= i < len(result.findings)]
    refs = {(f.provenance_idx, f.measurement_ref) for f in findings if f.measurement_ref}
    labels_by_prov: dict[int, set[str]] = {}
    for f in findings:
        labels_by_prov.setdefault(f.provenance_idx, set()).add(f.label)

    measurements = frozenset(
        i for i, m in enumerate(result.measurements) if (m.provenance_idx, m.name) in refs
    )
    sources = {
        (result.measurements[i].provenance_idx, seg_id)
        for i in measurements
        for seg_id in result.measurements[i].derived_from
    }
    segmentations: set[int] = set()
    for i, seg in enumerate(result.segmentations):
        if (seg.provenance_idx, seg.id) in sources or (seg.provenance_idx, seg.name) in sources:
            segmentations.add(i)
            continue
        keys = set(seg.label_keys.values())
        if keys and keys <= labels_by_prov.get(seg.provenance_idx, set()):
            segmentations.add(i)
    return Exposure(frozenset(segmentations), measurements)


def main_exposure(result: AnalysisResult) -> Exposure:
    """Segmentações e medidas da visão principal: ligadas a achados de visibilidade ``main``."""
    return exposed_artifacts(
        result, (i for i, f in enumerate(result.findings) if f.visibility == "main")
    )
