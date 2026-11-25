"""Interoperabilidade: ingestão de arquivos, DICOMweb (Orthanc) e leitura de imagens.

* :mod:`radiologyai.interop.ingest` — envio de DICOM/zip/NIfTI/PNG/JPEG com
  des-identificação na entrada, proteção contra zip-slip e limites de tamanho.
* :mod:`radiologyai.interop.images` — reconstrução de ``ImageData`` a partir do
  repositório e extração de quadros para o visualizador.
* :mod:`radiologyai.interop.dicomweb` — QIDO/WADO/STOW via ``dicomweb-client``.
"""

from __future__ import annotations

__all__: list[str] = []
