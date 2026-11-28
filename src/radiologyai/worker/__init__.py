"""Fila de análises no banco e worker em processo (concorrência 1).

* :mod:`radiologyai.worker.runtime` — ``Platform``: banco, artefatos, cache de
  imagens e motor de análise (registro + catálogo + política) compartilhados
  pela API e pelo worker.
* :mod:`radiologyai.worker.queue` — reivindicação atômica de tarefas, recuperação
  de tarefas interrompidas e a thread do worker.
* :mod:`radiologyai.worker.runner` — executa UMA análise com
  :func:`radiologyai.analysis.pipeline.analyze` e grava o resultado imutável.
"""

from __future__ import annotations

__all__: list[str] = []
