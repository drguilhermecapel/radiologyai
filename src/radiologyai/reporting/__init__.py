"""Laudo estruturado PRELIMINAR de pesquisa — nunca laudo diagnóstico.

* :mod:`radiologyai.reporting.content` — conteúdo canônico (hash ``report_sha256``)
  montado do resultado imutável e das revisões do médico; nível D excluído.
* :mod:`radiologyai.reporting.pdf` — PDF em português (reportlab).
* :mod:`radiologyai.reporting.dicom_sr` — DICOM SR TID 1500 e SEG (highdicom).
* :mod:`radiologyai.reporting.fhir` — ``DiagnosticReport`` FHIR R4, ``preliminary``.

Requer o extra ``platform``.
"""

from __future__ import annotations

__all__: list[str] = []
