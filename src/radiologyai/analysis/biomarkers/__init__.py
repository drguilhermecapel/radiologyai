"""Biomarcadores quantitativos derivados de máscaras de segmentação reais + HU/geometria.

Cada função recebe a imagem real (HU em TC) e máscaras produzidas por um modelo
real (TotalSegmentator) e devolve :class:`~radiologyai.analysis.base.RawMeasurement`
— com valor OU com motivo de abstenção, nunca um valor inventado.

Regras impostas aqui (ver ``docs/regulatory/06-requirements/REQ-133.md`` a ``REQ-136``):

* pré-condições de aquisição lidas do DICOM (fase de contraste, kVp, kernel,
  espessura) são conferidas ANTES do cálculo; fase desconhecida onde importa
  ⇒ abstenção ``ACQUISITION``;
* estrutura cortada pelo campo de visão ⇒ abstenção (volume/diâmetro seria um
  limite inferior, não uma medida);
* limiares de decisão NÃO vivem aqui: são pontos de operação com origem nos
  manifestos (``ct-biomarkers.yaml``), citados de
  ``/root/medai-cache/research/evidence-ct-mr.verified.md``.

O pacote não importa numpy/scipy no nível de módulo.
"""

from __future__ import annotations

__all__: list[str] = []
